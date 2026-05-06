#!/usr/bin/env python3
"""
evaluate.py — full evaluation pipeline for JINST-style RAG/GraphRAG paper.

Benchmark format: data/eval/benchmark_jinst.json
[
  {
    "query_id": "q001",
    "query": "What timing resolution is reported for HV-MAPS?",
    "query_type": "factoid|conceptual|comparative|multi_hop|negative",
    "answer": "Gold/reference answer written by expert.",
    "claims": ["Atomic claim 1", "Atomic claim 2"],
    "gold_sources": ["paper.pdf", "arxiv:xxxx.xxxxx"],
    "gold_chunk_ids": [12, 45],
    "gold_entities": ["HV-MAPS", "MightyPix"],
    "should_abstain": false
  }
]

Run:
  python evaluate.py --benchmark data/eval/benchmark_jinst.json --out_dir data/eval/runs/jinst_v1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from chat import (
    BASE_DIR,
    LOCAL_EMBED_MODEL,
    clean_text,
    run_single_query,
)


ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

DEFAULT_BENCHMARK = BASE_DIR / "data" / "eval" / "benchmark_jinst.json"
DEFAULT_OUT_DIR = BASE_DIR / "data" / "eval" / "runs" / time.strftime("%Y%m%d_%H%M%S")

TOP_KS = [1, 3, 5, 8]
EPS = 1e-12
_EMBEDDER = None


@dataclass
class EvalConfig:
    name: str
    mode: str
    top_k: int = 8
    candidate_k: int = 40
    use_reranker: bool = True
    generate: bool = True


def get_embedder() -> SentenceTransformer:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer(LOCAL_EMBED_MODEL)
    return _EMBEDDER


def embed(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype="float32")
    return get_embedder().encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")


def cos(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + EPS))


def tokenise(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_\-\+\.]+", text.lower())


def sent_split(text: str) -> List[str]:
    text = clean_text(text)
    if not text:
        return []
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+", text) if x.strip()]


def normalise_source(s: str) -> str:
    s = str(s or "").lower()
    s = s.replace("\\", "/")
    s = re.sub(r"\.pdf$|\.md$|\.txt$", "", s)
    return s


def source_matches(retrieved_source: str, gold_source: str) -> bool:
    r = normalise_source(retrieved_source)
    g = normalise_source(gold_source)
    return bool(g) and (g in r or r in g)


def load_benchmark(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "queries" in data:
        data = data["queries"]
    if not isinstance(data, list):
        raise ValueError("Benchmark must be a list or {'queries': [...]}")
    for i, row in enumerate(data):
        row.setdefault("query_id", f"q{i+1:04d}")
        row.setdefault("query_type", "unknown")
        row.setdefault("claims", [])
        row.setdefault("gold_sources", [])
        row.setdefault("gold_chunk_ids", [])
        row.setdefault("gold_entities", [])
        row.setdefault("should_abstain", False)
    return data


def default_configs(top_k: int, no_reranker: bool, no_generate: bool) -> List[EvalConfig]:
    return [
        EvalConfig("bm25", "bm25_only", top_k=top_k, use_reranker=False, generate=not no_generate),
        EvalConfig("dense", "vector_only", top_k=top_k, use_reranker=not no_reranker, generate=not no_generate),
        EvalConfig("hybrid", "hybrid_no_graph", top_k=top_k, use_reranker=not no_reranker, generate=not no_generate),
        EvalConfig("graph", "graph_rag", top_k=top_k, use_reranker=not no_reranker, generate=not no_generate),
        EvalConfig("graph_path", "graph_path_rag", top_k=top_k, use_reranker=not no_reranker, generate=not no_generate),
        EvalConfig("agentic_graph", "agentic_graph_rag", top_k=top_k, use_reranker=not no_reranker, generate=not no_generate),
    ]


# -----------------------------
# Retrieval metrics
# -----------------------------

def retrieved_ids(payload: Dict[str, Any]) -> List[int]:
    return [int(r["chunk_id"]) for r in payload.get("retrieved", []) if r.get("chunk_id") is not None]


def retrieved_sources(payload: Dict[str, Any]) -> List[str]:
    return [r.get("source", "") for r in payload.get("retrieved", [])]


def is_relevant_item(r: Dict[str, Any], gold_chunk_ids: set, gold_sources: List[str]) -> bool:
    cid = r.get("chunk_id")
    if cid is not None and int(cid) in gold_chunk_ids:
        return True
    src = r.get("source", "")
    return any(source_matches(src, gs) for gs in gold_sources)


def binary_relevance(payload: Dict[str, Any], row: Dict[str, Any]) -> List[int]:
    gold_chunk_ids = set(int(x) for x in row.get("gold_chunk_ids", []) if str(x).strip() != "")
    gold_sources = row.get("gold_sources", []) or []
    return [1 if is_relevant_item(r, gold_chunk_ids, gold_sources) else 0 for r in payload.get("retrieved", [])]


def precision_at(rel: List[int], k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(rel[:k]) / k


def recall_at(rel: List[int], n_gold: int, k: int) -> float:
    if n_gold <= 0:
        return 0.0
    return min(sum(rel[:k]), n_gold) / n_gold


def mrr(rel: List[int]) -> float:
    for i, x in enumerate(rel, start=1):
        if x:
            return 1.0 / i
    return 0.0


def ndcg_at(rel: List[int], k: int) -> float:
    dcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rel[:k]))
    ideal = sorted(rel, reverse=True)
    idcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(ideal[:k]))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision(rel: List[int], n_gold: int) -> float:
    if n_gold <= 0:
        return 0.0
    hits = 0
    ps = []
    for i, r in enumerate(rel, start=1):
        if r:
            hits += 1
            ps.append(hits / i)
    return sum(ps) / n_gold if ps else 0.0


def retrieval_metrics(payload: Dict[str, Any], row: Dict[str, Any]) -> Dict[str, float]:
    rel = binary_relevance(payload, row)
    n_gold = max(1, len(set(row.get("gold_chunk_ids", []))) + len(row.get("gold_sources", [])))
    out = {
        "MRR": mrr(rel),
        "MAP": average_precision(rel, n_gold),
    }
    for k in TOP_KS:
        out[f"P@{k}"] = precision_at(rel, k)
        out[f"R@{k}"] = recall_at(rel, n_gold, k)
        out[f"NDCG@{k}"] = ndcg_at(rel, k)
        out[f"Hit@{k}"] = 1.0 if any(rel[:k]) else 0.0
    return out


# -----------------------------
# Answer metrics
# -----------------------------

def semantic_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    v = embed([a, b])
    return cos(v[0], v[1])


def best_context_support(claim: str, contexts: List[str]) -> float:
    if not claim or not contexts:
        return 0.0
    texts = [claim] + contexts
    v = embed(texts)
    q = v[0]
    return max(cos(q, x) for x in v[1:]) if len(v) > 1 else 0.0


def claim_recall(claims: List[str], answer: str) -> float:
    if not claims:
        return 0.0
    v = embed([answer] + claims)
    ans = v[0]
    hits = sum(1 for cvec in v[1:] if cos(ans, cvec) >= 0.58)
    return hits / len(claims)


def context_claim_support(claims: List[str], contexts: List[str]) -> float:
    if not claims:
        return 0.0
    hits = 0
    for claim in claims:
        if best_context_support(claim, contexts) >= 0.55:
            hits += 1
    return hits / len(claims)


def answer_faithfulness(answer: str, contexts: List[str]) -> float:
    sentences = sent_split(answer)
    if not sentences:
        return 0.0
    supported = 0
    for s in sentences:
        if best_context_support(s, contexts) >= 0.55:
            supported += 1
    return supported / len(sentences)


def citation_coverage(answer: str) -> float:
    sentences = [s for s in sent_split(answer) if len(tokenise(s)) >= 5]
    if not sentences:
        return 0.0
    cited = sum(1 for s in sentences if re.search(r"\[\d+\]", s))
    return cited / len(sentences)


def hallucination_proxy(answer: str, contexts: List[str]) -> float:
    # Fraction of named/technical tokens in answer not seen in context.
    ctx = " ".join(contexts).lower()
    ents = re.findall(r"\b[A-Z][A-Za-z0-9\-]{2,}\b|\b(?:HV-CMOS|HV-MAPS|DMAPS|LGAD|ASIC|TDC|ToT|LHCb|ATLAS|CMS|Timepix3|MightyPix|RD53)\b", answer)
    ents = [e for e in ents if len(e) > 2]
    if not ents:
        return 0.0
    unsupported = sum(1 for e in ents if e.lower() not in ctx)
    return unsupported / len(ents)


def abstention_metrics(payload: Dict[str, Any], row: Dict[str, Any]) -> Dict[str, float]:
    should = bool(row.get("should_abstain", False))
    did = bool(payload.get("abstained", False))
    return {
        "should_abstain": float(should),
        "did_abstain": float(did),
        "abstention_correct": float(should == did),
        "false_answer_rate": float(should and not did),
        "over_abstention_rate": float((not should) and did),
    }


def answer_metrics(payload: Dict[str, Any], row: Dict[str, Any]) -> Dict[str, float]:
    answer = payload.get("answer", "") or ""
    contexts = [r.get("text", "") for r in payload.get("retrieved", [])]
    gold_answer = row.get("answer", "") or ""
    claims = row.get("claims", []) or []
    out = {
        "answer_similarity": semantic_similarity(answer, gold_answer) if gold_answer else 0.0,
        "claim_recall": claim_recall(claims, answer),
        "context_claim_support": context_claim_support(claims, contexts),
        "faithfulness": answer_faithfulness(answer, contexts),
        "citation_coverage": citation_coverage(answer),
        "hallucination_proxy": hallucination_proxy(answer, contexts),
        "answer_length_tokens": float(len(tokenise(answer))),
    }
    out.update(abstention_metrics(payload, row))
    return out


def graph_diagnostics(payload: Dict[str, Any]) -> Dict[str, float]:
    retrieved = payload.get("retrieved", [])
    graph_hits = sum(1 for r in retrieved if r.get("matched_nodes"))
    path_hits = sum(1 for r in retrieved if r.get("path_nodes"))
    unique_sources = len(set(r.get("source", "") for r in retrieved))
    return {
        "graph_hit_fraction": graph_hits / max(1, len(retrieved)),
        "path_hit_fraction": path_hits / max(1, len(retrieved)),
        "unique_sources": float(unique_sources),
        "seed_node_count": float(len(payload.get("seed_nodes", {}) or {})),
        "expanded_node_count": float(len(payload.get("expanded_nodes", {}) or {})),
        "retrieval_latency_s": float(payload.get("retrieval_latency_s", 0.0)),
        "generation_latency_s": float(payload.get("generation_latency_s", 0.0)),
    }


# -----------------------------
# Aggregation and reporting
# -----------------------------

def mean(xs: Sequence[float]) -> float:
    xs = [float(x) for x in xs if x is not None and not math.isnan(float(x))]
    return sum(xs) / len(xs) if xs else 0.0


def stderr(xs: Sequence[float]) -> float:
    xs = [float(x) for x in xs if x is not None and not math.isnan(float(x))]
    if len(xs) <= 1:
        return 0.0
    return statistics.stdev(xs) / math.sqrt(len(xs))


def aggregate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["config_name"], "ALL")].append(r)
        groups[(r["config_name"], r["query_type"])].append(r)

    metric_names = sorted({k for r in rows for k in r["metrics"].keys()})
    summary = []
    for (cfg, qtype), rs in sorted(groups.items()):
        out = {"config_name": cfg, "query_type": qtype, "n": len(rs)}
        for m in metric_names:
            vals = [r["metrics"].get(m, 0.0) for r in rs]
            out[m] = mean(vals)
            out[m + "_se"] = stderr(vals)
        summary.append(out)
    return summary


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def flatten_metric_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flat = []
    for r in rows:
        base = {
            "query_id": r["query_id"],
            "query_type": r["query_type"],
            "config_name": r["config_name"],
            "mode": r["mode"],
            "query": r["query"],
        }
        base.update(r["metrics"])
        flat.append(base)
    return flat


def run_eval(args) -> None:
    benchmark = load_benchmark(Path(args.benchmark))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = default_configs(args.top_k, args.no_reranker, args.no_generate)
    if args.configs:
        wanted = set(x.strip() for x in args.configs.split(",") if x.strip())
        configs = [c for c in configs if c.name in wanted or c.mode in wanted]

    all_rows = []
    payload_dir = out_dir / "payloads"
    payload_dir.mkdir(exist_ok=True)

    for cfg in configs:
        for i, row in enumerate(benchmark, start=1):
            print(f"[{cfg.name}] {i}/{len(benchmark)} {row['query_id']}: {row['query'][:90]}")
            payload_obj = run_single_query(
                query=row["query"],
                mode=cfg.mode,
                top_k=cfg.top_k,
                candidate_k=cfg.candidate_k,
                use_reranker=cfg.use_reranker,
                generate=cfg.generate,
            )
            payload = asdict(payload_obj)

            rm = retrieval_metrics(payload, row)
            am = answer_metrics(payload, row) if cfg.generate else {}
            gm = graph_diagnostics(payload)
            metrics = {}
            metrics.update(rm)
            metrics.update(am)
            metrics.update(gm)

            payload_path = payload_dir / f"{cfg.name}__{row['query_id']}.json"
            with open(payload_path, "w", encoding="utf-8") as f:
                json.dump({"benchmark": row, "payload": payload, "metrics": metrics}, f, ensure_ascii=False, indent=2)

            all_rows.append({
                "query_id": row["query_id"],
                "query_type": row.get("query_type", "unknown"),
                "config_name": cfg.name,
                "mode": cfg.mode,
                "query": row["query"],
                "payload_path": str(payload_path),
                "metrics": metrics,
            })

    summary = aggregate(all_rows)
    write_jsonl(out_dir / "per_query_results.jsonl", all_rows)
    write_csv(out_dir / "per_query_metrics.csv", flatten_metric_rows(all_rows))
    write_csv(out_dir / "summary.csv", summary)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nWrote:\n  {out_dir / 'summary.csv'}\n  {out_dir / 'per_query_metrics.csv'}\n  {out_dir / 'per_query_results.jsonl'}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", type=str, default=str(DEFAULT_BENCHMARK))
    p.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    p.add_argument("--top_k", type=int, default=8)
    p.add_argument("--configs", type=str, default="", help="Comma-separated subset, e.g. bm25,dense,hybrid,graph,agentic_graph")
    p.add_argument("--no_reranker", action="store_true")
    p.add_argument("--no_generate", action="store_true", help="Retrieval-only evaluation")
    return p


if __name__ == "__main__":
    run_eval(build_argparser().parse_args())
