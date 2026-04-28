import os
import re
import json
import math
import csv
import time
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv

# ============================================================
# 0. PATH / ENV
# ============================================================

BASE_DIR = Path("/eos/home-r/rjiang/RAG_pixel")
ENV_PATH = BASE_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

DATA_DIR = BASE_DIR / "data"
EVAL_DIR = DATA_DIR / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = EVAL_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BENCHMARK_PATH = EVAL_DIR / "benchmark_claims.json"
CONFIGS_PATH = EVAL_DIR / "eval_configs.json"

RESULTS_JSON_PATH = EVAL_DIR / "eval_results.json"
SUMMARY_CSV_PATH = EVAL_DIR / "eval_summary.csv"
PER_QUERY_JSONL_PATH = EVAL_DIR / "per_query_results.jsonl"

RETRIEVAL_CACHE_PATH = CACHE_DIR / "retrieval_cache.json"
ANSWER_CACHE_PATH = CACHE_DIR / "answer_cache.json"

LOCAL_EMBED_MODEL = os.getenv(
    "LOCAL_EMBED_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
).strip()

TOP_K_METRICS = [1, 3, 5]
EPS = 1e-12

# ============================================================
# 1. IMPORT CHAT.PY DIRECTLY
# ============================================================

sys.path.insert(0, str(BASE_DIR))

from chat import load_artifacts, retrieve, generate_answer_local  # noqa: E402

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


# ============================================================
# 2. DATA STRUCTURES
# ============================================================

@dataclass
class RetrievedItem:
    rank: int
    chunk_id: Optional[int]
    source: str
    text: str
    score: float
    node_id: Optional[str] = None
    community_id: Optional[str] = None
    path_nodes: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class QueryRunResult:
    query_id: str
    query_type: str
    config_name: str
    query: str
    answer: Optional[str]
    repaired_answer: Optional[str]
    retrieved: List[RetrievedItem]
    retrieval_latency_s: float
    generation_latency_s: float
    repair_latency_s: float
    metrics: Dict[str, Any]
    repair_metrics: Optional[Dict[str, Any]]
    diagnostics: Dict[str, Any]


# ============================================================
# 3. BASIC UTILS
# ============================================================

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers is not installed.")
        _embedder = SentenceTransformer(LOCAL_EMBED_MODEL)
    return _embedder


def embed_texts(texts: List[str]) -> np.ndarray:
    model = get_embedder()
    return model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    ).astype("float32")


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < EPS or nb < EPS:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", " ").replace("\ufeff", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def sent_split(text: str) -> List[str]:
    text = clean_text(text)
    if not text:
        return []
    parts = re.split(r'(?<=[\.\!\?])\s+', text)
    return [p.strip() for p in parts if p.strip()]


def tokenise(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_\-\+\.]+", text.lower())


def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def load_json(path: Path, default=None):
    if not path.exists():
        return {} if default is None else default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(rows: List[Dict[str, Any]], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_key(config_name: str, query_id: str) -> str:
    return f"{config_name}::{query_id}"


# ============================================================
# 4. RETRIEVAL METRICS
# ============================================================

def precision_at_k(retrieved: List[Any], relevant: set, k: int) -> float:
    if k <= 0:
        return 0.0
    hits = sum(1 for x in retrieved[:k] if x in relevant)
    return hits / k


def recall_at_k(retrieved: List[Any], relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for x in retrieved[:k] if x in relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved: List[Any], relevant: set) -> float:
    for i, x in enumerate(retrieved, start=1):
        if x in relevant:
            return 1.0 / i
    return 0.0


def dcg_at_k(retrieved: List[Any], relevant: set, k: int) -> float:
    score = 0.0
    for i, x in enumerate(retrieved[:k], start=1):
        rel = 1 if x in relevant else 0
        if rel:
            score += 1.0 / math.log2(i + 1)
    return score


def ndcg_at_k(retrieved: List[Any], relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg_at_k(retrieved, relevant, k) / idcg if idcg > 0 else 0.0


def average_precision_ranked(retrieved: List[Any], relevant: set) -> float:
    if not relevant:
        return 0.0
    num_hits = 0
    precisions = []
    for i, x in enumerate(retrieved, start=1):
        if x in relevant:
            num_hits += 1
            precisions.append(num_hits / i)
    if not precisions:
        return 0.0
    return sum(precisions) / len(relevant)


# ============================================================
# 5. CLAIM / ENTITY SUPPORT
# ============================================================

def extract_entities_simple(text: str) -> List[str]:
    ents = re.findall(r"\b[A-Z][A-Za-z0-9\-\+]{1,}\b", text)
    tech = re.findall(
        r"\b(?:CMOS|MAPS|DMAPS|HV-CMOS|LGAD|ToT|TDC|ASIC|3D|HL-LHC|LHC|ATLAS|CMS|Timepix|RD53)\b",
        text
    )
    ents += tech
    ents = [e.strip() for e in ents if len(e.strip()) > 1]
    return sorted(set(ents))


def entity_recall(gold_entities: List[str], contexts: List[str]) -> float:
    if not gold_entities:
        return 0.0
    ce = set()
    for c in contexts:
        ce.update(x.lower() for x in extract_entities_simple(c))
    ge = set(x.lower() for x in gold_entities)
    return len(ce & ge) / max(1, len(ge))


def claim_support_lexical(claim: str, contexts: List[str], threshold: float = 0.22) -> Tuple[bool, float]:
    claim_tokens = tokenise(claim)
    best = 0.0
    for c in contexts:
        score = jaccard(claim_tokens, tokenise(c))
        if score > best:
            best = score
    return best >= threshold, best


def context_recall_from_claims(claims: List[str], contexts: List[str]) -> Tuple[float, List[Dict[str, Any]]]:
    if not claims:
        return 0.0, []

    rows = []
    supported_count = 0
    for claim in claims:
        supported, score = claim_support_lexical(claim, contexts)
        if supported:
            supported_count += 1
        rows.append({
            "claim": claim,
            "supported": supported,
            "score": score
        })

    return supported_count / len(claims), rows


# ============================================================
# 6. ANSWER METRICS
# ============================================================

def answer_relevancy_from_embeds(question_vec: np.ndarray, answer: str) -> float:
    ans_vec = embed_texts([answer])[0]
    return cosine_sim(question_vec, ans_vec)


def answer_correctness_from_embeds(gold_answer_vec: np.ndarray, gold_answer: str, answer: str) -> Dict[str, float]:
    ans_vec = embed_texts([answer])[0]
    semantic = cosine_sim(gold_answer_vec, ans_vec)

    gold_sents = sent_split(gold_answer)
    ans_sents = sent_split(answer)

    if not gold_sents:
        factual_f1 = 0.0
    else:
        matched = 0
        for gs in gold_sents:
            gs_tokens = tokenise(gs)
            best = 0.0
            for a in ans_sents:
                best = max(best, jaccard(gs_tokens, tokenise(a)))
            if best >= 0.35:
                matched += 1

        recall = matched / max(1, len(gold_sents))
        precision = matched / max(1, len(ans_sents)) if ans_sents else 0.0
        factual_f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0

    overall = 0.75 * semantic + 0.25 * factual_f1
    return {
        "answer_correctness": overall,
        "semantic_similarity": semantic,
        "factual_f1": factual_f1
    }


def decompose_answer_into_claims(answer: str) -> List[str]:
    claims = sent_split(answer)
    claims = [c for c in claims if len(tokenise(c)) >= 4]
    return claims[:12]


def faithfulness(answer: str, contexts: List[str]) -> Tuple[float, List[Dict[str, Any]]]:
    claims = decompose_answer_into_claims(answer)
    if not claims:
        return 0.0, []

    rows = []
    supported = 0
    for claim in claims:
        ok, score = claim_support_lexical(claim, contexts, threshold=0.20)
        if ok:
            supported += 1
        rows.append({
            "claim": claim,
            "supported": ok,
            "score": score
        })

    return supported / len(claims), rows


# ============================================================
# 7. OPTIONAL FAST REPAIR
# ============================================================

def extractive_fallback_from_context(retrieved: List[RetrievedItem], max_sentences: int = 3) -> str:
    kept = []
    for item in retrieved[:3]:
        sents = sent_split(item.text)
        for s in sents:
            if len(tokenise(s)) >= 6:
                kept.append(s)
            if len(kept) >= max_sentences:
                return " ".join(kept)
    return "Insufficient evidence in the retrieved context to provide a reliable answer."


def repair_answer_fast(old_answer: str, retrieved: List[RetrievedItem]) -> str:
    contexts = [x.text for x in retrieved]
    claims = decompose_answer_into_claims(old_answer)

    if not claims:
        return extractive_fallback_from_context(retrieved)

    kept = []
    for claim in claims:
        ok, _ = claim_support_lexical(claim, contexts, threshold=0.20)
        if ok:
            kept.append(claim)

    if kept:
        repaired = " ".join(kept)
        repaired = re.sub(r"\s+", " ", repaired).strip()
        return repaired

    return extractive_fallback_from_context(retrieved)


# ============================================================
# 8. GRAPH DIAGNOSTICS
# ============================================================

def graph_diagnostics(
    retrieved: List[RetrievedItem],
    gold_sources: List[str]
) -> Dict[str, Any]:
    node_ids = [x.node_id for x in retrieved if x.node_id]
    community_ids = [x.community_id for x in retrieved if x.community_id]
    path_lengths = [len(x.path_nodes) for x in retrieved if x.path_nodes]

    source_hits = [x.source for x in retrieved if x.source in set(gold_sources)]

    return {
        "n_retrieved": len(retrieved),
        "unique_nodes": len(set(node_ids)),
        "unique_communities": len(set(community_ids)),
        "mean_path_len": float(np.mean(path_lengths)) if path_lengths else None,
        "max_path_len": int(max(path_lengths)) if path_lengths else None,
        "gold_source_hits": len(source_hits),
        "retrieved_sources": [x.source for x in retrieved],
    }


# ============================================================
# 9. CONVERSION / RETRIEVAL CORE
# ============================================================

def convert_retrieved(raw_items: List[Dict[str, Any]]) -> List[RetrievedItem]:
    rows = []
    for i, x in enumerate(raw_items, start=1):
        rows.append(
            RetrievedItem(
                rank=i,
                chunk_id=x.get("chunk_id"),
                source=x.get("source", ""),
                text=clean_text(x.get("text", "")),
                score=float(x.get("score", 0.0)),
                node_id=x.get("node_id"),
                community_id=x.get("community_id"),
                path_nodes=x.get("path_nodes"),
                metadata=x.get("metadata")
            )
        )
    return rows


def serialise_retrieval_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for i, item in enumerate(results, start=1):
        chunk = item["chunk"]
        matched_nodes = item.get("matched_nodes", [])
        out.append({
            "rank": i,
            "chunk_id": chunk.get("chunk_id"),
            "source": chunk.get("source", ""),
            "text": clean_text(chunk.get("text", "")),
            "score": float(item.get("score", 0.0)),
            "node_id": matched_nodes[0] if matched_nodes else None,
            "community_id": item.get("community_id"),
            "path_nodes": item.get("path_nodes", []),
            "metadata": {
                "title": chunk.get("title"),
                "type": chunk.get("type"),
                "ingest_method": chunk.get("ingest_method"),
                "chunk_index": chunk.get("chunk_index"),
                "dense_score": item.get("dense_score"),
                "bm25_score": item.get("bm25_score"),
                "graph_bonus": item.get("graph_bonus"),
                "matched_nodes": matched_nodes,
            }
        })
    return out


def run_retrieval_once(
    bundle,
    query: str,
    retrieval_mode: str,
    top_k: int
) -> Dict[str, Any]:
    chunk_index, chunks, bm25, graph, entity_to_chunks, node_texts, node_names, node_index = bundle

    t0 = time.perf_counter()
    results, seed_nodes, expanded_nodes = retrieve(
        retrieval_mode=retrieval_mode,
        chunk_index=chunk_index,
        bm25=bm25,
        chunks=chunks,
        graph=graph,
        entity_to_chunks=entity_to_chunks,
        node_names=node_names,
        node_index=node_index,
        query=query,
        final_k=top_k
    )
    latency = time.perf_counter() - t0

    return {
        "retrieval_latency_s": latency,
        "seed_nodes": seed_nodes,
        "expanded_nodes": expanded_nodes,
        "retrieved": serialise_retrieval_results(results),
        "raw_results": results,
    }


def generate_answer_once(
    query: str,
    raw_results: List[Dict[str, Any]]
) -> Tuple[str, float]:
    t0 = time.perf_counter()
    answer = generate_answer_local(query, raw_results, [])
    latency = time.perf_counter() - t0
    return answer, latency


# ============================================================
# 10. METRICS
# ============================================================

def compute_retrieval_metrics(
    retrieved: List[RetrievedItem],
    gold_chunk_ids: List[int],
    gold_sources: List[str]
) -> Dict[str, Any]:
    retrieved_chunk_ids = [x.chunk_id for x in retrieved if x.chunk_id is not None]
    retrieved_sources = [x.source for x in retrieved]

    gold_chunks_set = set(gold_chunk_ids)
    gold_sources_set = set(gold_sources)

    out = {}

    for k in TOP_K_METRICS:
        out[f"chunk_P@{k}"] = precision_at_k(retrieved_chunk_ids, gold_chunks_set, k)
        out[f"chunk_R@{k}"] = recall_at_k(retrieved_chunk_ids, gold_chunks_set, k)
        out[f"source_P@{k}"] = precision_at_k(retrieved_sources, gold_sources_set, k)
        out[f"source_R@{k}"] = recall_at_k(retrieved_sources, gold_sources_set, k)

    out["chunk_MRR"] = reciprocal_rank(retrieved_chunk_ids, gold_chunks_set)
    out["source_MRR"] = reciprocal_rank(retrieved_sources, gold_sources_set)
    out["chunk_NDCG@5"] = ndcg_at_k(retrieved_chunk_ids, gold_chunks_set, 5)
    out["source_NDCG@5"] = ndcg_at_k(retrieved_sources, gold_sources_set, 5)

    out["context_precision_chunk"] = average_precision_ranked(retrieved_chunk_ids, gold_chunks_set)
    out["context_precision_source"] = average_precision_ranked(retrieved_sources, gold_sources_set)

    return out


def compute_generation_metrics(
    benchmark_item: Dict[str, Any],
    answer: str,
    retrieved: List[RetrievedItem],
    question_vec: np.ndarray,
    gold_answer_vec: np.ndarray
) -> Dict[str, Any]:
    contexts = [x.text for x in retrieved]

    gold_answer = benchmark_item.get("gold_answer", "")
    claims = benchmark_item.get("claims", [])
    gold_entities = benchmark_item.get("gold_entities", [])

    context_recall_score, claim_rows = context_recall_from_claims(claims, contexts)

    if gold_entities:
        context_entity_recall_score = entity_recall(gold_entities, contexts)
    else:
        inferred_gold_entities = extract_entities_simple(gold_answer)
        context_entity_recall_score = entity_recall(inferred_gold_entities, contexts)

    answer_relevancy_score = answer_relevancy_from_embeds(question_vec, answer)
    correctness = answer_correctness_from_embeds(gold_answer_vec, gold_answer, answer)
    faithfulness_score, answer_claim_rows = faithfulness(answer, contexts)

    out = {}
    out["context_recall"] = context_recall_score
    out["context_entity_recall"] = context_entity_recall_score
    out["answer_relevancy"] = answer_relevancy_score
    out["answer_correctness"] = correctness["answer_correctness"]
    out["semantic_similarity"] = correctness["semantic_similarity"]
    out["factual_f1"] = correctness["factual_f1"]
    out["faithfulness"] = faithfulness_score
    out["_claim_support_rows"] = claim_rows
    out["_answer_claim_rows"] = answer_claim_rows
    return out


# ============================================================
# 11. SINGLE QUERY EVAL
# ============================================================

def evaluate_one_query(
    bundle,
    item: Dict[str, Any],
    config: Dict[str, Any],
    query_vec: np.ndarray,
    gold_answer_vec: np.ndarray,
    retrieval_cache: Dict[str, Any],
    answer_cache: Dict[str, Any],
    with_generation: bool,
    with_repair: bool,
    refresh_retrieval_cache: bool,
    refresh_answer_cache: bool
) -> QueryRunResult:
    query = item["query"]
    query_id = item["query_id"]
    query_type = item.get("query_type", "unspecified")
    config_name = config["name"]

    top_k = int(config.get("top_k", 6))
    retrieval_mode = config.get("retrieval_mode", "graph_rag")

    cache_key = stable_key(config_name, query_id)

    # ---------------- retrieval ----------------
    if (cache_key in retrieval_cache) and (not refresh_retrieval_cache):
        r_payload = retrieval_cache[cache_key]
        retrieved = convert_retrieved(r_payload["retrieved"])
        raw_results = None
        retrieval_latency_s = float(r_payload.get("retrieval_latency_s", 0.0))
        seed_nodes = r_payload.get("seed_nodes", {})
        expanded_nodes = r_payload.get("expanded_nodes", {})
    else:
        r_payload = run_retrieval_once(
            bundle=bundle,
            query=query,
            retrieval_mode=retrieval_mode,
            top_k=top_k
        )
        retrieved = convert_retrieved(r_payload["retrieved"])
        raw_results = r_payload["raw_results"]
        retrieval_latency_s = float(r_payload["retrieval_latency_s"])
        seed_nodes = r_payload.get("seed_nodes", {})
        expanded_nodes = r_payload.get("expanded_nodes", {})

        retrieval_cache[cache_key] = {
            "retrieval_latency_s": retrieval_latency_s,
            "seed_nodes": seed_nodes,
            "expanded_nodes": expanded_nodes,
            "retrieved": r_payload["retrieved"]
        }

    retrieval_metrics = compute_retrieval_metrics(
        retrieved=retrieved,
        gold_chunk_ids=item.get("gold_chunk_ids", []),
        gold_sources=item.get("gold_sources", [])
    )

    diagnostics = graph_diagnostics(retrieved, item.get("gold_sources", []))
    diagnostics["seed_nodes"] = seed_nodes
    diagnostics["expanded_nodes"] = expanded_nodes

    if not with_generation:
        return QueryRunResult(
            query_id=query_id,
            query_type=query_type,
            config_name=config_name,
            query=query,
            answer=None,
            repaired_answer=None,
            retrieved=retrieved,
            retrieval_latency_s=retrieval_latency_s,
            generation_latency_s=0.0,
            repair_latency_s=0.0,
            metrics=retrieval_metrics,
            repair_metrics=None,
            diagnostics=diagnostics
        )

    # ---------------- generation ----------------
    if (cache_key in answer_cache) and (not refresh_answer_cache):
        a_payload = answer_cache[cache_key]
        answer = a_payload["answer"]
        generation_latency_s = float(a_payload.get("generation_latency_s", 0.0))
    else:
        if raw_results is None:
            # rebuild raw-like results from cached retrieval
            # only used when retrieval was loaded from cache
            raw_results = []
            for r in retrieved:
                raw_results.append({
                    "score": r.score,
                    "graph_bonus": r.metadata.get("graph_bonus") if r.metadata else None,
                    "dense_score": r.metadata.get("dense_score") if r.metadata else None,
                    "bm25_score": r.metadata.get("bm25_score") if r.metadata else None,
                    "matched_nodes": r.metadata.get("matched_nodes", []) if r.metadata else [],
                    "community_id": r.community_id,
                    "path_nodes": r.path_nodes,
                    "chunk": {
                        "chunk_id": r.chunk_id,
                        "source": r.source,
                        "text": r.text,
                        "title": r.metadata.get("title") if r.metadata else None,
                        "type": r.metadata.get("type") if r.metadata else None,
                        "ingest_method": r.metadata.get("ingest_method") if r.metadata else None,
                        "chunk_index": r.metadata.get("chunk_index") if r.metadata else None,
                    }
                })

        answer, generation_latency_s = generate_answer_once(query, raw_results)
        answer_cache[cache_key] = {
            "answer": answer,
            "generation_latency_s": generation_latency_s
        }

    generation_metrics = compute_generation_metrics(
        benchmark_item=item,
        answer=answer,
        retrieved=retrieved,
        question_vec=query_vec,
        gold_answer_vec=gold_answer_vec
    )

    metrics = {}
    metrics.update(retrieval_metrics)
    metrics.update(generation_metrics)

    diagnostics["claim_support"] = generation_metrics.pop("_claim_support_rows", [])
    diagnostics["answer_claim_support"] = generation_metrics.pop("_answer_claim_rows", [])

    repaired_answer = None
    repair_metrics = None
    repair_latency_s = 0.0

    if with_repair and config.get("use_repair", False):
        if (
            metrics["faithfulness"] < config.get("repair_threshold_faithfulness", 0.75)
            or metrics["answer_correctness"] < config.get("repair_threshold_correctness", 0.70)
        ):
            t0 = time.perf_counter()
            repaired_answer = repair_answer_fast(answer, retrieved)
            repair_latency_s = time.perf_counter() - t0

            repair_metrics = {}
            repair_metrics.update(retrieval_metrics)
            repair_gen_metrics = compute_generation_metrics(
                benchmark_item=item,
                answer=repaired_answer,
                retrieved=retrieved,
                question_vec=query_vec,
                gold_answer_vec=gold_answer_vec
            )
            repair_metrics.update(repair_gen_metrics)

            diagnostics["repair_answer_claim_support"] = repair_gen_metrics.pop("_answer_claim_rows", [])
            diagnostics["repair_gain"] = {
                "delta_answer_correctness": repair_metrics["answer_correctness"] - metrics["answer_correctness"],
                "delta_faithfulness": repair_metrics["faithfulness"] - metrics["faithfulness"],
                "delta_answer_relevancy": repair_metrics["answer_relevancy"] - metrics["answer_relevancy"],
            }

    return QueryRunResult(
        query_id=query_id,
        query_type=query_type,
        config_name=config_name,
        query=query,
        answer=answer,
        repaired_answer=repaired_answer,
        retrieved=retrieved,
        retrieval_latency_s=retrieval_latency_s,
        generation_latency_s=generation_latency_s,
        repair_latency_s=repair_latency_s,
        metrics=metrics,
        repair_metrics=repair_metrics,
        diagnostics=diagnostics
    )


# ============================================================
# 12. AGGREGATION
# ============================================================

def mean_or_none(vals: List[float]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def aggregate_runs(runs: List[QueryRunResult]) -> Dict[str, Any]:
    if not runs:
        return {}

    metric_keys = list(runs[0].metrics.keys())
    agg = {"n_queries": len(runs), "metrics": {}}

    for k in metric_keys:
        agg["metrics"][k] = mean_or_none([r.metrics[k] for r in runs if k in r.metrics])

    agg["latency"] = {
        "retrieval_latency_mean_s": mean_or_none([r.retrieval_latency_s for r in runs]),
        "generation_latency_mean_s": mean_or_none([r.generation_latency_s for r in runs]),
        "repair_latency_mean_s": mean_or_none([r.repair_latency_s for r in runs if r.repair_latency_s > 0]),
    }

    repaired_runs = [r for r in runs if r.repair_metrics is not None]
    if repaired_runs:
        repair_metric_keys = list(repaired_runs[0].repair_metrics.keys())
        agg["repair_metrics"] = {}
        for k in repair_metric_keys:
            agg["repair_metrics"][k] = mean_or_none([r.repair_metrics[k] for r in repaired_runs if k in r.repair_metrics])

    return agg


def aggregate_by_type(runs: List[QueryRunResult]) -> Dict[str, Any]:
    groups: Dict[str, List[QueryRunResult]] = {}
    for r in runs:
        groups.setdefault(r.query_type, []).append(r)
    return {qtype: aggregate_runs(rr) for qtype, rr in groups.items()}


# ============================================================
# 13. OUTPUT
# ============================================================

def write_summary_csv(summary: Dict[str, Any], path: Path):
    rows = []
    for cfg_name, cfg_data in summary.items():
        overall = cfg_data["overall"]["metrics"]
        lat = cfg_data["overall"]["latency"]
        row = {
            "config": cfg_name,
            "n_queries": cfg_data["overall"]["n_queries"],
            **overall,
            **lat
        }
        rows.append(row)

    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 14. CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Fast evaluation for local GraphRAG")
    parser.add_argument("--retrieval-only", action="store_true", help="Evaluate retrieval metrics only")
    parser.add_argument("--with-repair", action="store_true", help="Enable repair ablation")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of benchmark queries")
    parser.add_argument("--config", type=str, default=None, help="Run only one config by name")
    parser.add_argument("--refresh-retrieval-cache", action="store_true", help="Ignore retrieval cache")
    parser.add_argument("--refresh-answer-cache", action="store_true", help="Ignore answer cache")
    return parser.parse_args()


# ============================================================
# 15. MAIN
# ============================================================

def main():
    args = parse_args()

    if not BENCHMARK_PATH.exists():
        raise FileNotFoundError(f"Missing benchmark file: {BENCHMARK_PATH}")
    if not CONFIGS_PATH.exists():
        raise FileNotFoundError(f"Missing config file: {CONFIGS_PATH}")

    benchmark = load_json(BENCHMARK_PATH)
    configs = load_json(CONFIGS_PATH)

    if not isinstance(benchmark, list):
        raise TypeError("benchmark_claims.json must be a list of query dictionaries.")
    if not isinstance(configs, list):
        raise TypeError("eval_configs.json must be a list of config dictionaries.")

    if args.limit is not None:
        benchmark = benchmark[:args.limit]

    if args.config is not None:
        configs = [c for c in configs if c["name"] == args.config]
        if not configs:
            raise ValueError(f"No config matched --config {args.config}")

    retrieval_cache = load_json(RETRIEVAL_CACHE_PATH, default={})
    answer_cache = load_json(ANSWER_CACHE_PATH, default={})

    print("Loading GraphRAG artifacts once...")
    bundle = load_artifacts()
    print("Artifacts loaded.")

    print("Precomputing benchmark embeddings...")
    queries = [item["query"] for item in benchmark]
    gold_answers = [item.get("gold_answer", "") for item in benchmark]
    query_vecs = embed_texts(queries)
    gold_answer_vecs = embed_texts(gold_answers)
    print("Embeddings ready.")

    with_generation = not args.retrieval_only

    all_summary = {}
    all_jsonl_rows = []

    for cfg in configs:
        cfg_name = cfg["name"]
        print(f"\n===== Evaluating config: {cfg_name} =====")

        runs: List[QueryRunResult] = []

        for idx, item in enumerate(benchmark):
            if not isinstance(item, dict):
                raise TypeError(f"Each benchmark item must be a dict, got: {type(item)}")

            print(f"[{cfg_name}] {item['query_id']} :: {item['query']}")
            run = evaluate_one_query(
                bundle=bundle,
                item=item,
                config=cfg,
                query_vec=query_vecs[idx],
                gold_answer_vec=gold_answer_vecs[idx],
                retrieval_cache=retrieval_cache,
                answer_cache=answer_cache,
                with_generation=with_generation,
                with_repair=args.with_repair,
                refresh_retrieval_cache=args.refresh_retrieval_cache,
                refresh_answer_cache=args.refresh_answer_cache
            )
            runs.append(run)

            all_jsonl_rows.append({
                "query_id": run.query_id,
                "query_type": run.query_type,
                "config_name": run.config_name,
                "query": run.query,
                "answer": run.answer,
                "repaired_answer": run.repaired_answer,
                "retrieval_latency_s": run.retrieval_latency_s,
                "generation_latency_s": run.generation_latency_s,
                "repair_latency_s": run.repair_latency_s,
                "metrics": run.metrics,
                "repair_metrics": run.repair_metrics,
                "diagnostics": run.diagnostics,
                "retrieved": [asdict(x) for x in run.retrieved],
            })

        all_summary[cfg_name] = {
            "overall": aggregate_runs(runs),
            "by_query_type": aggregate_by_type(runs),
        }

    save_json(retrieval_cache, RETRIEVAL_CACHE_PATH)
    save_json(answer_cache, ANSWER_CACHE_PATH)

    save_json(all_summary, RESULTS_JSON_PATH)
    write_summary_csv(all_summary, SUMMARY_CSV_PATH)
    write_jsonl(all_jsonl_rows, PER_QUERY_JSONL_PATH)

    print("\nSaved:")
    print(f"  {RESULTS_JSON_PATH}")
    print(f"  {SUMMARY_CSV_PATH}")
    print(f"  {PER_QUERY_JSONL_PATH}")
    print(f"  {RETRIEVAL_CACHE_PATH}")
    print(f"  {ANSWER_CACHE_PATH}")


if __name__ == "__main__":
    main()
