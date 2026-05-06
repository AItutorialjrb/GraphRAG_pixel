#!/usr/bin/env python3
"""
GraphRAG Pixel chat.py

Compatible with current RAG_pixel structure:

  data/index/chunks.json
  data/index/bm25.pkl
  data/index/dense.index
  data/graph_clean/graph_clean.graphml
  data/graph_clean/entity_to_chunks_clean.json

Examples:

  python chat.py \
    --query "What does HV-MAPS stand for?" \
    --mode hybrid \
    --top_k 10 \
    --json \
    --no_generate

  python chat.py \
    --query "What does LGAD stand for?" \
    --mode graph \
    --top_k 10 \
    --json \
    --no_generate

  python chat.py \
    --query "What is Time-over-Threshold used for?" \
    --mode agentic_graph_rag \
    --top_k 10 \
    --json \
    --no_generate
"""

import argparse
import json
import math
import pickle
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx


try:
    import numpy as np
except Exception:
    np = None

try:
    import faiss
except Exception:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None

try:
    from transformers import pipeline
except Exception:
    pipeline = None


DEFAULT_CHUNKS = Path("data/index/chunks.json")
DEFAULT_BM25 = Path("data/index/bm25.pkl")
DEFAULT_DENSE_INDEX = Path("data/index/dense.index")

DEFAULT_GRAPHML = Path("data/graph_clean/graph_clean.graphml")
DEFAULT_ENTITY_CHUNKS = Path("data/graph_clean/entity_to_chunks_clean.json")

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


DOMAIN_ALIASES = {
    "hv maps": ["HVMAPS", "HV-CMOS", "MAPS", "High Voltage Monolithic Active Pixel Sensors"],
    "hv-maps": ["HVMAPS", "HV-CMOS", "MAPS", "High Voltage Monolithic Active Pixel Sensors"],
    "hvmaps": ["HVMAPS", "HV-CMOS", "MAPS", "High Voltage Monolithic Active Pixel Sensors"],
    "high voltage monolithic active pixel sensors": ["HVMAPS", "HV-CMOS", "MAPS"],

    "hv cmos": ["HV-CMOS", "HVMAPS", "CMOS", "MAPS"],
    "hv-cmos": ["HV-CMOS", "HVMAPS", "CMOS", "MAPS"],
    "hvcmos": ["HV-CMOS", "HVMAPS", "CMOS", "MAPS"],

    "maps": ["MAPS", "HVMAPS", "HV-CMOS", "DMAPS", "Monolithic Active Pixel Sensors"],
    "dmaps": ["DMAPS", "MAPS", "HV-CMOS", "Depleted Monolithic Active Pixel Sensors"],

    "lgad": ["LGAD", "Low Gain Avalanche Detector", "Low-Gain Avalanche Detector", "AC-LGAD", "DC-LGAD", "TI-LGAD"],
    "low gain avalanche detector": ["LGAD", "AC-LGAD", "DC-LGAD", "TI-LGAD"],
    "low-gain avalanche detector": ["LGAD", "AC-LGAD", "DC-LGAD", "TI-LGAD"],

    "time over threshold": ["ToT", "Time-over-Threshold", "charge", "signal amplitude"],
    "time-over-threshold": ["ToT", "Time-over-Threshold", "charge", "signal amplitude"],
    "tot": ["ToT", "Time-over-Threshold", "charge", "signal amplitude"],

    "time of arrival": ["ToA", "Time-of-Arrival", "timing"],
    "time-of-arrival": ["ToA", "Time-of-Arrival", "timing"],
    "toa": ["ToA", "timing"],

    "timepix3": ["Timepix3", "Timepix", "pixel detector"],
    "timepix": ["Timepix", "Timepix3", "pixel detector"],
    "mightypix": ["MightyPix", "HVMAPS", "HV-CMOS", "MAPS"],
}


STOPWORDS = {
    "what", "is", "are", "was", "were", "does", "do", "did",
    "the", "a", "an", "of", "to", "for", "in", "on", "with",
    "and", "or", "between", "compare", "explain", "describe",
    "why", "how", "used", "use", "using", "relation", "relationship",
}


def norm_text(s: Any) -> str:
    s = "" if s is None else str(s)
    s = s.replace("–", "-").replace("—", "-")
    s = s.lower()
    s = re.sub(r"[_/]+", " ", s)
    s = re.sub(r"[^a-z0-9\-\.\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def compact(s: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", norm_text(s))


def tokenize(s: str) -> List[str]:
    toks = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9\-\+\.]{1,}", s.lower())
    return [t for t in toks if t not in STOPWORDS]


def expand_query(query: str) -> str:
    qn = norm_text(query)
    qc = compact(query)

    extra = []

    for key, vals in DOMAIN_ALIASES.items():
        if key in qn or compact(key) in qc:
            extra.extend(vals)

    for tok in tokenize(query):
        tc = compact(tok)
        if tc in DOMAIN_ALIASES:
            extra.extend(DOMAIN_ALIASES[tc])

    if extra:
        return query + " " + " ".join(sorted(set(extra)))

    return query


def load_chunks_file(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            for key in ["chunks", "documents", "items", "data"]:
                if key in data and isinstance(data[key], list):
                    return data[key]

        raise ValueError(f"Unsupported JSON chunk format: {path}")

    if path.suffix == ".jsonl":
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    raise ValueError(f"Unsupported chunk file type: {path}")


def get_text(row: Dict[str, Any]) -> str:
    for key in ["text", "content", "chunk_text", "body", "page_content", "snippet"]:
        if key in row and row[key]:
            return str(row[key])
    return json.dumps(row, ensure_ascii=False)


def get_chunk_id(row: Dict[str, Any], fallback: int) -> str:
    for key in ["chunk_id", "id", "doc_id", "document_id", "cid"]:
        if key in row and row[key] not in (None, ""):
            return str(row[key])
    return str(fallback)


def get_source(row: Dict[str, Any]) -> str:
    for key in ["source", "file", "path", "pdf", "paper", "url"]:
        if key in row and row[key]:
            return str(row[key])

    meta = row.get("metadata", {})
    if isinstance(meta, dict):
        for key in ["source", "file", "path", "pdf", "paper", "url"]:
            if key in meta and meta[key]:
                return str(meta[key])

    return ""


def get_title(row: Dict[str, Any]) -> str:
    for key in ["title", "paper_title", "name"]:
        if key in row and row[key]:
            return str(row[key])

    meta = row.get("metadata", {})
    if isinstance(meta, dict):
        for key in ["title", "paper_title", "name"]:
            if key in meta and meta[key]:
                return str(meta[key])

    src = get_source(row)
    return Path(src).stem if src else ""


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


class GraphRAGPixel:
    def __init__(
        self,
        chunks_path: Path = DEFAULT_CHUNKS,
        bm25_path: Path = DEFAULT_BM25,
        dense_index_path: Path = DEFAULT_DENSE_INDEX,
        graphml_path: Path = DEFAULT_GRAPHML,
        entity_chunks_path: Path = DEFAULT_ENTITY_CHUNKS,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        llm_model_name: str = DEFAULT_LLM_MODEL,
        load_llm: bool = False,
        verbose: bool = False,
    ):
        self.chunks_path = chunks_path
        self.bm25_path = bm25_path
        self.dense_index_path = dense_index_path
        self.graphml_path = graphml_path
        self.entity_chunks_path = entity_chunks_path
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name
        self.verbose = verbose

        self.chunks = self._load_chunks()
        self.chunk_by_id = {str(c["chunk_id"]): c for c in self.chunks}
        self.texts = [c["text"] for c in self.chunks]

        self.bm25 = self._load_or_build_bm25()

        self.embedding_model = None
        self.dense_index = None
        self._load_dense_backend()

        self.graph = self._load_graph()
        self.entity_to_chunks = self._load_entity_chunks()
        self.node_lookup = {compact(n): n for n in self.graph.nodes()}

        self.llm = None
        if load_llm:
            self.llm = self._load_llm()

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, file=sys.stderr)

    def _load_chunks(self) -> List[Dict[str, Any]]:
        rows = load_chunks_file(self.chunks_path)

        if not rows:
            alt_paths = [
                Path("data/index/chunks.json"),
                Path("data/chunks.json"),
                Path("data/chunks.jsonl"),
                Path("data/chunks/chunks.jsonl"),
                Path("data/processed/chunks.jsonl"),
                Path("data/corpus/chunks.jsonl"),
            ]

            for path in alt_paths:
                rows = load_chunks_file(path)
                if rows:
                    self.chunks_path = path
                    break

        chunks = []
        for i, row in enumerate(rows):
            chunks.append({
                "chunk_id": get_chunk_id(row, i),
                "doc_id": str(row.get("doc_id", "")),
                "title": get_title(row),
                "source": get_source(row),
                "text": get_text(row),
                "metadata": row.get("metadata", {}),
            })

        if not chunks:
            raise FileNotFoundError(
                "No chunks loaded. Expected data/index/chunks.json or equivalent."
            )

        self._log(f"Loaded {len(chunks)} chunks from {self.chunks_path}")
        return chunks

    def _load_or_build_bm25(self):
        if self.bm25_path.exists():
            try:
                with open(self.bm25_path, "rb") as f:
                    obj = pickle.load(f)
                self._log(f"Loaded BM25 from {self.bm25_path}")
                return obj
            except Exception as e:
                self._log(f"Could not load BM25 pickle, rebuilding. Error: {e}")

        tokenized = [tokenize(t) for t in self.texts]

        if BM25Okapi is not None:
            self._log("Built BM25Okapi in memory")
            return BM25Okapi(tokenized)

        self._log("Built simple token index in memory")
        return tokenized

    def _load_dense_backend(self) -> None:
        if SentenceTransformer is None or np is None:
            self._log("Dense backend unavailable: sentence_transformers or numpy missing")
            return

        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            self._log(f"Loaded embedding model {self.embedding_model_name}")
        except Exception as e:
            self.embedding_model = None
            self._log(f"Could not load embedding model: {e}")
            return

        if faiss is not None and self.dense_index_path.exists():
            try:
                self.dense_index = faiss.read_index(str(self.dense_index_path))
                self._log(f"Loaded FAISS index {self.dense_index_path}")
            except Exception as e:
                self.dense_index = None
                self._log(f"Could not load FAISS index: {e}")

    def _load_graph(self) -> nx.Graph:
        if not self.graphml_path.exists():
            self._log(f"Graph not found: {self.graphml_path}")
            return nx.Graph()

        G = nx.read_graphml(self.graphml_path)

        for _, _, d in G.edges(data=True):
            d["weight"] = safe_float(d.get("weight", 1.0), 1.0)

        self._log(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

    def _load_entity_chunks(self) -> Dict[str, Set[str]]:
        if not self.entity_chunks_path.exists():
            self._log(f"entity_to_chunks not found: {self.entity_chunks_path}")
            return {}

        with open(self.entity_chunks_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        out = {}
        for ent, chunks in raw.items():
            if isinstance(chunks, list):
                out[str(ent)] = set(str(x) for x in chunks)

        self._log(f"Loaded entity_to_chunks: {len(out)} entities")
        return out

    def _load_llm(self):
        if pipeline is None:
            self._log("transformers pipeline unavailable")
            return None

        try:
            return pipeline(
                "text-generation",
                model=self.llm_model_name,
                device_map="auto",
                max_new_tokens=512,
            )
        except Exception as e:
            self._log(f"Could not load LLM: {e}")
            return None

    def _result_from_chunk(self, chunk: Dict[str, Any], score: float, rank: int, retriever: str) -> Dict[str, Any]:
        return {
            "rank": int(rank),
            "chunk_id": str(chunk["chunk_id"]),
            "doc_id": str(chunk.get("doc_id", "")),
            "title": chunk.get("title", ""),
            "source": chunk.get("source", ""),
            "score": float(score),
            "retriever": retriever,
            "text": chunk.get("text", ""),
        }

    def bm25_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        expanded = expand_query(query)
        q_tokens = tokenize(expanded)

        if hasattr(self.bm25, "get_scores"):
            scores = self.bm25.get_scores(q_tokens)
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            return [
                self._result_from_chunk(self.chunks[i], float(scores[i]), rank, "bm25")
                for rank, i in enumerate(ranked, start=1)
            ]

        q_count = Counter(q_tokens)
        scores = []
        for toks in self.bm25:
            c = Counter(toks)
            s = sum(min(c[t], q_count[t]) for t in q_count)
            scores.append(float(s))

        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            self._result_from_chunk(self.chunks[i], float(scores[i]), rank, "bm25_simple")
            for rank, i in enumerate(ranked, start=1)
        ]

    def dense_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if self.embedding_model is None:
            return []

        expanded = expand_query(query)
        emb = self.embedding_model.encode([expanded], normalize_embeddings=True)

        if self.dense_index is not None and faiss is not None and np is not None:
            q = np.asarray(emb, dtype="float32")
            scores, idxs = self.dense_index.search(q, top_k)

            out = []
            for rank, (idx, score) in enumerate(zip(idxs[0], scores[0]), start=1):
                idx = int(idx)
                if idx < 0 or idx >= len(self.chunks):
                    continue
                out.append(self._result_from_chunk(self.chunks[idx], float(score), rank, "dense"))
            return out

        if np is None:
            return []

        doc_emb = self.embedding_model.encode(self.texts, normalize_embeddings=True)
        scores = np.dot(np.asarray(doc_emb), np.asarray(emb[0]))
        ranked = np.argsort(scores)[::-1][:top_k]

        return [
            self._result_from_chunk(self.chunks[int(i)], float(scores[int(i)]), rank, "dense_bruteforce")
            for rank, i in enumerate(ranked, start=1)
        ]

    def hybrid_search(self, query: str, top_k: int = 10, candidate_k: int = 80) -> List[Dict[str, Any]]:
        bm = self.bm25_search(query, candidate_k)
        de = self.dense_search(query, candidate_k)

        scores = defaultdict(float)
        payloads = {}

        for r in bm:
            cid = r["chunk_id"]
            payloads[cid] = r
            scores[cid] += 0.60 / max(r["rank"], 1)

        for r in de:
            cid = r["chunk_id"]
            payloads[cid] = r
            scores[cid] += 0.40 / max(r["rank"], 1)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        out = []
        for rank, (cid, score) in enumerate(ranked, start=1):
            r = dict(payloads[cid])
            r["rank"] = rank
            r["score"] = float(score)
            r["retriever"] = "hybrid_rrf"
            out.append(r)

        return out

    def match_query_nodes(self, query: str, max_nodes: int = 8) -> List[str]:
        if self.graph.number_of_nodes() == 0:
            return []

        expanded = expand_query(query)
        phrases = [query, expanded]

        phrases.extend(tokenize(expanded))

        qn = norm_text(expanded)
        qc = compact(expanded)

        for key, vals in DOMAIN_ALIASES.items():
            if key in qn or compact(key) in qc:
                phrases.extend(vals)

        candidates: List[Tuple[float, str]] = []
        seen_phrases = set()

        for phrase in phrases:
            pc = compact(phrase)
            if not pc or pc in seen_phrases:
                continue
            seen_phrases.add(pc)

            if pc in self.node_lookup:
                node = self.node_lookup[pc]
                score = 1000.0 + self.graph.degree(node, weight="weight")
                candidates.append((score, node))

            for node in self.graph.nodes():
                nc = compact(node)
                if not nc:
                    continue

                if pc == nc:
                    score = 1000.0 + self.graph.degree(node, weight="weight")
                    candidates.append((score, node))
                elif len(pc) >= 4 and (pc in nc or nc in pc):
                    score = 100.0 + self.graph.degree(node, weight="weight")
                    candidates.append((score, node))

        best = []
        used = set()

        for _, node in sorted(candidates, key=lambda x: x[0], reverse=True):
            if node not in used:
                best.append(node)
                used.add(node)
            if len(best) >= max_nodes:
                break

        return best

    def graph_search(
        self,
        query: str,
        top_k: int = 10,
        candidate_k: int = 80,
        use_paths: bool = False,
        agentic: bool = False,
    ) -> Dict[str, Any]:
        seed_nodes = self.match_query_nodes(query, max_nodes=8)

        expanded_nodes = set(seed_nodes)
        paths = []

        for node in seed_nodes:
            if node not in self.graph:
                continue

            neighbours = sorted(
                self.graph.neighbors(node),
                key=lambda nb: safe_float(self.graph[node][nb].get("weight", 1.0), 1.0),
                reverse=True,
            )

            n1 = 12 if agentic else 8

            for nb in neighbours[:n1]:
                expanded_nodes.add(nb)
                if use_paths or agentic:
                    paths.append([node, nb])

                if agentic and nb in self.graph:
                    nb2s = sorted(
                        self.graph.neighbors(nb),
                        key=lambda nb2: safe_float(self.graph[nb][nb2].get("weight", 1.0), 1.0),
                        reverse=True,
                    )
                    for nb2 in nb2s[:3]:
                        expanded_nodes.add(nb2)
                        paths.append([node, nb, nb2])

        chunk_scores = defaultdict(float)

        for node in expanded_nodes:
            chunks = self.entity_to_chunks.get(node, set())
            node_weight = 1.0

            if node in seed_nodes:
                node_weight += 3.0

            if node in self.graph:
                node_weight += math.log1p(self.graph.degree(node, weight="weight")) * 0.05

            for cid in chunks:
                chunk_scores[str(cid)] += node_weight

        hybrid = self.hybrid_search(query, top_k=candidate_k, candidate_k=max(candidate_k, 80))

        for r in hybrid:
            chunk_scores[r["chunk_id"]] += 1.0 / max(r["rank"], 1)

        ranked = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        used = set()

        for cid, score in ranked:
            chunk = self.chunk_by_id.get(str(cid))
            if not chunk:
                continue
            results.append(self._result_from_chunk(chunk, float(score), len(results) + 1, "graph"))
            used.add(str(cid))
            if len(results) >= top_k:
                break

        if len(results) < top_k:
            for r in hybrid:
                if r["chunk_id"] not in used:
                    rr = dict(r)
                    rr["rank"] = len(results) + 1
                    rr["retriever"] = "graph_fallback_hybrid"
                    results.append(rr)
                    used.add(rr["chunk_id"])
                if len(results) >= top_k:
                    break

        return {
            "retrieved": results,
            "seed_nodes": seed_nodes,
            "expanded_nodes": sorted(expanded_nodes),
            "paths": paths,
        }

    def retrieve(self, query: str, mode: str, top_k: int, candidate_k: int = 80) -> Dict[str, Any]:
        t0 = time.time()

        mode = mode.lower()

        if mode == "bm25":
            retrieved = self.bm25_search(query, top_k)
            extra = {}

        elif mode == "dense":
            retrieved = self.dense_search(query, top_k)
            extra = {}

        elif mode == "hybrid":
            retrieved = self.hybrid_search(query, top_k, candidate_k)
            extra = {}

        elif mode in {"graph", "graph_rag"}:
            obj = self.graph_search(query, top_k, candidate_k, use_paths=False, agentic=False)
            retrieved = obj["retrieved"]
            extra = obj

        elif mode in {"graph_path", "graph_path_rag"}:
            obj = self.graph_search(query, top_k, candidate_k, use_paths=True, agentic=False)
            retrieved = obj["retrieved"]
            extra = obj

        elif mode in {"agentic_graph", "agentic_graph_rag"}:
            obj = self.graph_search(query, top_k, candidate_k, use_paths=True, agentic=True)
            retrieved = obj["retrieved"]
            extra = obj

        else:
            raise ValueError(f"Unknown mode: {mode}")

        latency = time.time() - t0
        unique_sources = len(set(r.get("source", "") for r in retrieved if r.get("source", "")))

        metrics = {
            "retrieval_latency_s": latency,
            "generation_latency_s": 0.0,
            "unique_sources": unique_sources,
            "seed_node_count": len(extra.get("seed_nodes", [])) if extra else 0,
            "expanded_node_count": len(extra.get("expanded_nodes", [])) if extra else 0,
            "graph_hit_fraction": 1.0 if extra.get("expanded_nodes") else 0.0,
            "path_hit_fraction": 1.0 if extra.get("paths") else 0.0,
        }

        return {
            "query": query,
            "expanded_query": expand_query(query),
            "mode": mode,
            "retrieved": retrieved,
            "seed_nodes": extra.get("seed_nodes", []),
            "expanded_nodes": extra.get("expanded_nodes", []),
            "paths": extra.get("paths", []),
            "metrics": metrics,
        }

    def build_prompt(self, query: str, retrieved: List[Dict[str, Any]]) -> str:
        evidence = []

        for r in retrieved[:8]:
            evidence.append(
                f"[chunk_id={r['chunk_id']}; source={r.get('title') or r.get('source')}]\n"
                f"{r.get('text', '')[:1800]}"
            )

        return (
            "You are answering questions about silicon pixel detectors and detector instrumentation.\n"
            "Use only the evidence below. If the evidence is insufficient, say so.\n"
            "Cite chunk ids explicitly.\n\n"
            f"Question:\n{query}\n\n"
            f"Evidence:\n\n" + "\n\n".join(evidence) + "\n\nAnswer:"
        )

    def generate_answer(self, query: str, retrieved: List[Dict[str, Any]]) -> Tuple[str, float]:
        if self.llm is None:
            return "", 0.0

        prompt = self.build_prompt(query, retrieved)

        t0 = time.time()

        try:
            out = self.llm(
                prompt,
                max_new_tokens=384,
                do_sample=False,
                return_full_text=False,
            )
            latency = time.time() - t0

            if isinstance(out, list) and out:
                return str(out[0].get("generated_text", "")).strip(), latency

            return str(out).strip(), latency

        except Exception as e:
            return f"[generation failed: {e}]", time.time() - t0

    def ask(
        self,
        query: str,
        mode: str,
        top_k: int,
        candidate_k: int,
        generate: bool,
    ) -> Dict[str, Any]:
        payload = self.retrieve(query, mode=mode, top_k=top_k, candidate_k=candidate_k)

        if generate:
            answer, gen_latency = self.generate_answer(query, payload["retrieved"])
            payload["answer"] = answer
            payload["metrics"]["generation_latency_s"] = gen_latency
        else:
            payload["answer"] = ""

        return payload


def print_human(payload: Dict[str, Any]) -> None:
    print("\nQuery:", payload["query"])
    print("Mode:", payload["mode"])
    print("Expanded query:", payload.get("expanded_query", ""))

    if payload.get("seed_nodes"):
        print("\nSeed nodes:", ", ".join(payload["seed_nodes"]))

    if payload.get("expanded_nodes"):
        print("Expanded nodes:", ", ".join(payload["expanded_nodes"][:40]))

    print("\nRetrieved evidence:")

    for r in payload["retrieved"]:
        print(f"\n[{r['rank']}] score={r['score']:.4f} chunk_id={r['chunk_id']}")
        print("source:", r.get("title") or r.get("source"))
        print(r.get("text", "")[:700].replace("\n", " "))

    if payload.get("answer"):
        print("\nAnswer:\n", payload["answer"])

    print("\nMetrics:")
    print(json.dumps(payload.get("metrics", {}), indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--query", required=True)
    ap.add_argument(
        "--mode",
        default="hybrid",
        choices=[
            "bm25",
            "dense",
            "hybrid",
            "graph",
            "graph_rag",
            "graph_path",
            "graph_path_rag",
            "agentic_graph",
            "agentic_graph_rag",
        ],
    )

    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--candidate_k", type=int, default=80)

    ap.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    ap.add_argument("--bm25", type=Path, default=DEFAULT_BM25)
    ap.add_argument("--dense_index", type=Path, default=DEFAULT_DENSE_INDEX)

    ap.add_argument("--graphml", type=Path, default=DEFAULT_GRAPHML)
    ap.add_argument("--entity_chunks", type=Path, default=DEFAULT_ENTITY_CHUNKS)

    ap.add_argument("--embedding_model", default=DEFAULT_EMBEDDING_MODEL)
    ap.add_argument("--llm_model", default=DEFAULT_LLM_MODEL)

    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no_generate", action="store_true")
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    rag = GraphRAGPixel(
        chunks_path=args.chunks,
        bm25_path=args.bm25,
        dense_index_path=args.dense_index,
        graphml_path=args.graphml,
        entity_chunks_path=args.entity_chunks,
        embedding_model_name=args.embedding_model,
        llm_model_name=args.llm_model,
        load_llm=not args.no_generate,
        verbose=args.verbose,
    )

    payload = rag.ask(
        query=args.query,
        mode=args.mode,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        generate=not args.no_generate,
    )

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print_human(payload)


if __name__ == "__main__":
    main()
