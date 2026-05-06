#!/usr/bin/env python3
"""
Paper-grade evaluator for GraphRAG_pixel.

This evaluator fixes the common failure mode where all Hit@k/MRR/MAP/NDCG
become zero because the benchmark gold labels and retrieved evidence are not
in the same representation.

It supports:
  1. strict chunk/source matching
  2. soft text/entity evidence matching
  3. graph/entity/path coverage
  4. negative-query abstention
  5. robust JSON payload extraction

Typical usage, re-scoring an existing run:
  python evaluate_paper.py \
    --benchmark data/eval/benchmark_jinst_gold.json \
    --run_dir data/eval/runs/test_retrieval \
    --out_dir data/eval/runs/paper_eval_v1

Typical usage, running chat.py directly:
  python evaluate_paper.py \
    --benchmark data/eval/benchmark_jinst_gold.json \
    --chat ./chat.py \
    --configs bm25,dense,hybrid,graph,graph_path,agentic_graph \
    --top_k 10 \
    --out_dir data/eval/runs/paper_eval_v1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


# ----------------------------
# Alias normalisation
# ----------------------------

ALIASES = {
    "hv maps": "hvmaps",
    "hv-maps": "hvmaps",
    "hvmaps": "hvmaps",
    "high voltage monolithic active pixel sensor": "hvmaps",
    "high voltage monolithic active pixel sensors": "hvmaps",
    "high-voltage monolithic active pixel sensor": "hvmaps",
    "high-voltage monolithic active pixel sensors": "hvmaps",

    "hvcmos": "hv-cmos",
    "hv cmos": "hv-cmos",
    "hv-cmos": "hv-cmos",
    "high voltage cmos": "hv-cmos",

    "maps": "maps",
    "map": "maps",
    "monolithic active pixel sensor": "maps",
    "monolithic active pixel sensors": "maps",

    "dmaps": "dmaps",
    "depleted monolithic active pixel sensor": "dmaps",
    "depleted monolithic active pixel sensors": "dmaps",

    "lgad": "lgad",
    "lgads": "lgad",
    "low gain avalanche detector": "lgad",
    "low gain avalanche detectors": "lgad",
    "low-gain avalanche detector": "lgad",
    "low-gain avalanche detectors": "lgad",

    "ac lgad": "ac-lgad",
    "ac-lgad": "ac-lgad",
    "dc lgad": "dc-lgad",
    "dc-lgad": "dc-lgad",
    "ti lgad": "ti-lgad",
    "ti-lgad": "ti-lgad",

    "time over threshold": "tot",
    "time-over-threshold": "tot",
    "tot": "tot",

    "time of arrival": "toa",
    "time-of-arrival": "toa",
    "toa": "toa",

    "timepix 3": "timepix3",
    "timepix3": "timepix3",

    "mighty pix": "mightypix",
    "mightypix": "mightypix",
}

ABSTAIN_PATTERNS = [
    r"\bi do not know\b",
    r"\binsufficient evidence\b",
    r"\bnot enough evidence\b",
    r"\bcannot determine\b",
    r"\bnot provided\b",
    r"\bnot in the context\b",
    r"\bno evidence\b",
    r"\bunable to answer\b",
]


def norm_text(s: Any) -> str:
    s = "" if s is None else str(s)
    s = s.lower()
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"[_/]+", " ", s)
    s = re.sub(r"[^a-z0-9\-\.\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def compact(s: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", norm_text(s))


def canonical_phrase(s: Any) -> str:
    x = norm_text(s)
    if x in ALIASES:
        return ALIASES[x]

    c = compact(x)
    for k, v in ALIASES.items():
        if compact(k) == c:
            return v

    return c


def phrase_in_text(phrase: str, text: str) -> bool:
    if not phrase:
        return False

    p = canonical_phrase(phrase)
    t_norm = norm_text(text)
    t_compact = compact(text)

    if p in canonical_phrase(t_norm):
        return True

    if compact(phrase) and compact(phrase) in t_compact:
        return True

    # Also allow exact normalised phrase for multi-word natural answers
    pn = norm_text(phrase)
    if len(pn) >= 4 and pn in t_norm:
        return True

    return False


def is_abstention(answer: str) -> bool:
    a = norm_text(answer)
    return any(re.search(p, a) for p in ABSTAIN_PATTERNS)


# ----------------------------
# Data classes
# ----------------------------

@dataclass
class QueryGold:
    query_id: str
    query: str
    query_type: str = "unknown"
    gold_chunk_ids: Set[str] = field(default_factory=set)
    gold_sources: Set[str] = field(default_factory=set)
    gold_answers: List[str] = field(default_factory=list)
    gold_entities: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    is_negative: bool = False


@dataclass
class RetrievedItem:
    rank: int
    chunk_id: str = ""
    source: str = ""
    title: str = ""
    text: str = ""
    score: Optional[float] = None
    entities: List[str] = field(default_factory=list)


@dataclass
class EvalRow:
    query_id: str
    query_type: str
    config_name: str
    mode: str
    query: str
    n_retrieved: int
    strict_available: int
    strict_first_hit_rank: int
    soft_first_hit_rank: int
    source_first_hit_rank: int
    graph_entity_hit: float
    graph_path_hit: float
    answer_abstained: float
    should_abstain: float
    retrieval_latency_s: float
    generation_latency_s: float
    unique_sources: int
    seed_node_count: int
    expanded_node_count: int
    metrics: Dict[str, float]


# ----------------------------
# Benchmark loading
# ----------------------------

def first_present(d: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default


def as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    if isinstance(x, set):
        return list(x)
    return [x]


def load_benchmark(path: Path) -> Dict[str, QueryGold]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "questions" in data:
            records = data["questions"]
        elif "queries" in data:
            records = data["queries"]
        elif "items" in data:
            records = data["items"]
        else:
            records = list(data.values())
    else:
        records = data

    out: Dict[str, QueryGold] = {}

    for i, r in enumerate(records):
        qid = str(first_present(r, ["query_id", "id", "qid"], f"q{i:04d}"))
        query = str(first_present(r, ["query", "question", "prompt"], ""))
        qtype = str(first_present(r, ["query_type", "type", "category"], "unknown"))

        gold_chunk_ids = set(str(x) for x in as_list(first_present(
            r, ["gold_chunk_ids", "gold_chunks", "relevant_chunk_ids", "gold_ids"], []
        )))

        gold_sources = set(str(x) for x in as_list(first_present(
            r, ["gold_sources", "relevant_sources", "gold_source", "sources"], []
        )))

        gold_answers = [str(x) for x in as_list(first_present(
            r, ["gold_answers", "answers", "answer", "expected_answer"], []
        )) if str(x).strip()]

        gold_entities = [str(x) for x in as_list(first_present(
            r, ["gold_entities", "entities", "answer_entities", "expected_entities"], []
        )) if str(x).strip()]

        aliases = [str(x) for x in as_list(first_present(
            r, ["aliases", "gold_aliases", "entity_aliases"], []
        )) if str(x).strip()]

        is_negative = bool(first_present(r, ["is_negative", "negative", "should_abstain"], False))
        if qtype.lower() in {"negative", "out_of_scope", "unanswerable"}:
            is_negative = True

        out[qid] = QueryGold(
            query_id=qid,
            query=query,
            query_type=qtype,
            gold_chunk_ids=gold_chunk_ids,
            gold_sources=gold_sources,
            gold_answers=gold_answers,
            gold_entities=gold_entities,
            aliases=aliases,
            is_negative=is_negative,
        )

    return out


# ----------------------------
# Payload extraction
# ----------------------------

RETRIEVAL_KEYS = [
    "retrieved", "retrieved_chunks", "results", "contexts", "context",
    "evidence", "evidence_chunks", "ranked_chunks", "chunks",
    "top_chunks", "documents", "docs",
]

TEXT_KEYS = ["text", "content", "chunk_text", "body", "page_content", "context", "snippet"]
ID_KEYS = ["chunk_id", "id", "doc_id", "document_id", "chunk", "cid"]
SOURCE_KEYS = ["source", "file", "path", "pdf", "paper", "title", "url", "metadata.source"]
TITLE_KEYS = ["title", "paper_title", "name"]
SCORE_KEYS = ["score", "retrieval_score", "rank_score", "similarity", "bm25_score"]


def get_nested(d: Dict[str, Any], dotted: str) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def value_from_keys(d: Dict[str, Any], keys: Sequence[str]) -> Any:
    for k in keys:
        if "." in k:
            v = get_nested(d, k)
        else:
            v = d.get(k)
        if v not in (None, ""):
            return v
    return ""


def flatten_candidate_lists(obj: Any) -> List[Any]:
    """
    Recursively find likely retrieved-evidence lists.
    Avoid returning arbitrary short scalar lists such as entity tokens.
    """
    found: List[Any] = []

    if isinstance(obj, dict):
        for k in RETRIEVAL_KEYS:
            if k in obj and isinstance(obj[k], list):
                found.extend(obj[k])

        # Some payloads wrap result under data/output/response
        for k in ["data", "output", "response", "result", "payload"]:
            if k in obj:
                found.extend(flatten_candidate_lists(obj[k]))

    elif isinstance(obj, list):
        # If list looks like retrieved items, return it.
        if obj and all(isinstance(x, (dict, str)) for x in obj):
            dict_like = [x for x in obj if isinstance(x, dict)]
            if dict_like:
                has_textish = any(any(key in d for key in TEXT_KEYS + ID_KEYS + SOURCE_KEYS) for d in dict_like)
                if has_textish:
                    found.extend(obj)
        else:
            for x in obj:
                found.extend(flatten_candidate_lists(x))

    return found


def item_to_retrieved(raw: Any, rank: int) -> RetrievedItem:
    if isinstance(raw, str):
        return RetrievedItem(rank=rank, text=raw)

    if not isinstance(raw, dict):
        return RetrievedItem(rank=rank, text=str(raw))

    metadata = raw.get("metadata", {})
    merged = dict(raw)
    if isinstance(metadata, dict):
        for k, v in metadata.items():
            merged.setdefault(k, v)

    text = str(value_from_keys(merged, TEXT_KEYS) or "")
    chunk_id = str(value_from_keys(merged, ID_KEYS) or "")
    source = str(value_from_keys(merged, SOURCE_KEYS) or "")
    title = str(value_from_keys(merged, TITLE_KEYS) or "")

    score_val = value_from_keys(merged, SCORE_KEYS)
    try:
        score = float(score_val) if score_val not in ("", None) else None
    except Exception:
        score = None

    ents = []
    for ek in ["entities", "entity", "nodes", "seed_nodes", "expanded_nodes"]:
        v = merged.get(ek)
        if isinstance(v, list):
            ents.extend(str(x) for x in v)
        elif isinstance(v, str):
            ents.append(v)

    # If no text was found, keep a compact serialisation so soft matching can still work.
    if not text:
        try:
            text = json.dumps(raw, ensure_ascii=False)
        except Exception:
            text = str(raw)

    return RetrievedItem(
        rank=rank,
        chunk_id=chunk_id,
        source=source,
        title=title,
        text=text,
        score=score,
        entities=ents,
    )


def extract_retrieved(payload: Dict[str, Any], top_k: int) -> List[RetrievedItem]:
    raw_items = flatten_candidate_lists(payload)

    # Deduplicate by chunk_id if available, otherwise source+text prefix.
    out: List[RetrievedItem] = []
    seen: Set[str] = set()

    for raw in raw_items:
        item = item_to_retrieved(raw, rank=len(out) + 1)
        key = item.chunk_id or (item.source + "::" + compact(item.text[:160]))
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= top_k:
            break

    return out


def extract_answer(payload: Dict[str, Any]) -> str:
    for k in ["answer", "generated_answer", "response", "final_answer", "output_text"]:
        v = payload.get(k)
        if isinstance(v, str):
            return v
    # Sometimes nested
    for k in ["output", "result", "data"]:
        v = payload.get(k)
        if isinstance(v, dict):
            a = extract_answer(v)
            if a:
                return a
    return ""


def extract_graph_strings(payload: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
    seeds: List[str] = []
    expanded: List[str] = []
    paths: List[str] = []

    def walk(x: Any):
        nonlocal seeds, expanded, paths
        if isinstance(x, dict):
            for k, v in x.items():
                lk = k.lower()
                if lk in {"seed_nodes", "seeds", "query_nodes"}:
                    seeds.extend(str(z) for z in as_list(v))
                elif lk in {"expanded_nodes", "nodes", "graph_nodes"}:
                    expanded.extend(str(z) for z in as_list(v))
                elif lk in {"paths", "evidence_paths", "graph_paths"}:
                    if isinstance(v, list):
                        for p in v:
                            if isinstance(p, list):
                                paths.append(" ".join(str(z) for z in p))
                            else:
                                paths.append(str(p))
                    else:
                        paths.append(str(v))
                else:
                    walk(v)
        elif isinstance(x, list):
            for y in x:
                walk(y)

    walk(payload)
    return seeds, expanded, paths


# ----------------------------
# Metrics
# ----------------------------

def source_match(retrieved_source: str, gold_sources: Set[str]) -> bool:
    if not retrieved_source or not gold_sources:
        return False
    rs = compact(Path(retrieved_source).name) or compact(retrieved_source)
    rfull = compact(retrieved_source)

    for g in gold_sources:
        gs = compact(Path(str(g)).name) or compact(g)
        gfull = compact(g)
        if not gs:
            continue
        if gs in rfull or rs in gfull or gs == rs:
            return True
    return False


def chunk_match(chunk_id: str, gold_chunk_ids: Set[str]) -> bool:
    if not chunk_id or not gold_chunk_ids:
        return False
    c = str(chunk_id)
    return c in gold_chunk_ids or compact(c) in {compact(g) for g in gold_chunk_ids}


def soft_evidence_match(item: RetrievedItem, gold: QueryGold) -> bool:
    text = " ".join([item.text, item.title, item.source] + item.entities)
    phrases = []
    phrases.extend(gold.gold_answers)
    phrases.extend(gold.gold_entities)
    phrases.extend(gold.aliases)

    # If the benchmark has no soft gold, this metric is not available.
    if not phrases:
        return False

    return any(phrase_in_text(p, text) for p in phrases)


def graph_entity_match(strings: Sequence[str], gold: QueryGold) -> bool:
    phrases = []
    phrases.extend(gold.gold_entities)
    phrases.extend(gold.aliases)
    phrases.extend(gold.gold_answers)

    if not phrases:
        return False

    haystack = " ".join(str(x) for x in strings)
    return any(phrase_in_text(p, haystack) for p in phrases)


def relevance_vector(items: List[RetrievedItem], gold: QueryGold, mode: str) -> List[int]:
    rel = []
    for it in items:
        ok = False
        if mode == "strict":
            ok = chunk_match(it.chunk_id, gold.gold_chunk_ids) or source_match(it.source, gold.gold_sources)
        elif mode == "chunk":
            ok = chunk_match(it.chunk_id, gold.gold_chunk_ids)
        elif mode == "source":
            ok = source_match(it.source, gold.gold_sources)
        elif mode == "soft":
            ok = soft_evidence_match(it, gold)
        elif mode == "paper":
            ok = (
                chunk_match(it.chunk_id, gold.gold_chunk_ids)
                or source_match(it.source, gold.gold_sources)
                or soft_evidence_match(it, gold)
            )
        else:
            raise ValueError(mode)
        rel.append(1 if ok else 0)
    return rel


def first_hit_rank(rel: Sequence[int]) -> int:
    for i, x in enumerate(rel, start=1):
        if x:
            return i
    return 0


def precision_at(rel: Sequence[int], k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(rel[:k]) / float(k)


def recall_at(rel: Sequence[int], k: int, n_gold: int) -> float:
    if n_gold <= 0:
        return float("nan")
    return min(sum(rel[:k]), n_gold) / float(n_gold)


def hit_at(rel: Sequence[int], k: int) -> float:
    return 1.0 if any(rel[:k]) else 0.0


def reciprocal_rank(rel: Sequence[int]) -> float:
    r = first_hit_rank(rel)
    return 1.0 / r if r > 0 else 0.0


def average_precision(rel: Sequence[int], n_gold: int) -> float:
    if n_gold <= 0:
        return float("nan")
    hits = 0
    score = 0.0
    for i, x in enumerate(rel, start=1):
        if x:
            hits += 1
            score += hits / i
    return score / min(n_gold, max(1, sum(rel))) if hits else 0.0


def dcg(rel: Sequence[int], k: int) -> float:
    s = 0.0
    for i, x in enumerate(rel[:k], start=1):
        if x:
            s += 1.0 / math.log2(i + 1)
    return s


def ndcg_at(rel: Sequence[int], k: int, n_gold: int) -> float:
    if n_gold <= 0:
        return float("nan")
    ideal_hits = min(n_gold, k)
    ideal = dcg([1] * ideal_hits, k)
    if ideal <= 0:
        return 0.0
    return dcg(rel, k) / ideal


def n_gold_for_mode(gold: QueryGold, mode: str) -> int:
    if mode == "chunk":
        return len(gold.gold_chunk_ids)
    if mode == "source":
        return len(gold.gold_sources)
    if mode == "soft":
        return max(1, len(gold.gold_answers) + len(gold.gold_entities) + len(gold.aliases))
    if mode in {"strict", "paper"}:
        n = len(gold.gold_chunk_ids) + len(gold.gold_sources)
        if mode == "paper":
            n += len(gold.gold_answers) + len(gold.gold_entities) + len(gold.aliases)
        return max(1, n)
    return 0


def metrics_from_rel(rel: Sequence[int], n_gold: int, ks: Sequence[int], prefix: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    out[f"{prefix}_MRR"] = reciprocal_rank(rel)
    out[f"{prefix}_MAP"] = average_precision(rel, n_gold)
    for k in ks:
        out[f"{prefix}_Hit@{k}"] = hit_at(rel, k)
        out[f"{prefix}_P@{k}"] = precision_at(rel, k)
        out[f"{prefix}_R@{k}"] = recall_at(rel, k, n_gold)
        out[f"{prefix}_NDCG@{k}"] = ndcg_at(rel, k, n_gold)
    return out


def safe_mean(xs: Iterable[float]) -> float:
    vals = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def safe_se(xs: Iterable[float]) -> float:
    vals = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if len(vals) <= 1:
        return 0.0 if vals else float("nan")
    m = sum(vals) / len(vals)
    var = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
    return math.sqrt(var / len(vals))


# ----------------------------
# Running or re-scoring
# ----------------------------

CONFIG_TO_MODE = {
    "bm25": "bm25",
    "dense": "dense",
    "hybrid": "hybrid",
    "graph": "graph_rag",
    "graph_path": "graph_path_rag",
    "agentic_graph": "agentic_graph_rag",
    "graph_rag": "graph_rag",
    "graph_path_rag": "graph_path_rag",
    "agentic_graph_rag": "agentic_graph_rag",
}


def run_chat(chat: Path, query: str, mode: str, top_k: int) -> Tuple[Dict[str, Any], float]:
    cmd_variants = [
        [sys.executable, str(chat), "--query", query, "--mode", mode, "--top_k", str(top_k), "--json", "--no_generate"],
        [sys.executable, str(chat), "--query", query, "--mode", mode, "--top_k", str(top_k), "--json"],
        [sys.executable, str(chat), "--query", query, "--config", mode, "--top_k", str(top_k), "--json", "--no_generate"],
    ]

    last_err = ""
    for cmd in cmd_variants:
        t0 = time.time()
        try:
            p = subprocess.run(cmd, check=False, text=True, capture_output=True)
            latency = time.time() - t0
            if p.returncode != 0:
                last_err = p.stderr[-2000:]
                continue

            txt = p.stdout.strip()
            # Last JSON object in stdout
            start = txt.rfind("{")
            if start >= 0:
                try:
                    return json.loads(txt[start:]), latency
                except Exception:
                    pass

            return json.loads(txt), latency
        except Exception as e:
            last_err = str(e)

    raise RuntimeError(f"Could not run chat.py for mode={mode}. Last error:\n{last_err}")


def load_existing_run(run_dir: Path) -> List[Dict[str, Any]]:
    path = run_dir / "per_query_results.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_payload(payload_path: str, run_dir: Path) -> Dict[str, Any]:
    p = Path(payload_path)
    if not p.is_absolute():
        # Try relative to current cwd, then relative to run_dir parent hierarchy.
        candidates = [
            Path.cwd() / p,
            run_dir / p,
            run_dir.parent.parent.parent / p,
            run_dir.parent.parent / p,
        ]
    else:
        candidates = [p]

    for c in candidates:
        if c.exists():
            with open(c, "r", encoding="utf-8") as f:
                return json.load(f)

    return {}


def evaluate_one(
    gold: QueryGold,
    config_name: str,
    mode: str,
    payload: Dict[str, Any],
    top_k: int,
    ks: Sequence[int],
    retrieval_latency_override: Optional[float] = None,
) -> EvalRow:
    items = extract_retrieved(payload, top_k=top_k)
    answer = extract_answer(payload)
    seeds, expanded, paths = extract_graph_strings(payload)

    strict_available = int(bool(gold.gold_chunk_ids or gold.gold_sources))
    soft_available = int(bool(gold.gold_answers or gold.gold_entities or gold.aliases))

    all_metrics: Dict[str, float] = {}

    for metric_mode, prefix in [
        ("chunk", "chunk"),
        ("source", "source"),
        ("strict", "strict"),
        ("soft", "soft"),
        ("paper", "paper"),
    ]:
        n_gold = n_gold_for_mode(gold, metric_mode)
        available = True

        if metric_mode == "chunk" and not gold.gold_chunk_ids:
            available = False
        if metric_mode == "source" and not gold.gold_sources:
            available = False
        if metric_mode == "strict" and not (gold.gold_chunk_ids or gold.gold_sources):
            available = False
        if metric_mode == "soft" and not (gold.gold_answers or gold.gold_entities or gold.aliases):
            available = False

        if available:
            rel = relevance_vector(items, gold, metric_mode)
            all_metrics.update(metrics_from_rel(rel, n_gold=n_gold, ks=ks, prefix=prefix))
        else:
            for k in ks:
                all_metrics[f"{prefix}_Hit@{k}"] = float("nan")
                all_metrics[f"{prefix}_P@{k}"] = float("nan")
                all_metrics[f"{prefix}_R@{k}"] = float("nan")
                all_metrics[f"{prefix}_NDCG@{k}"] = float("nan")
            all_metrics[f"{prefix}_MRR"] = float("nan")
            all_metrics[f"{prefix}_MAP"] = float("nan")

    strict_rel = relevance_vector(items, gold, "strict") if strict_available else []
    soft_rel = relevance_vector(items, gold, "soft") if soft_available else []
    source_rel = relevance_vector(items, gold, "source") if gold.gold_sources else []

    graph_hit = 1.0 if graph_entity_match(seeds + expanded, gold) else 0.0
    path_hit = 1.0 if graph_entity_match(paths, gold) else 0.0

    # If payload already contains graph metrics, keep the stronger signal.
    existing_metrics = payload.get("metrics", {})
    if isinstance(existing_metrics, dict):
        graph_hit = max(graph_hit, float(existing_metrics.get("graph_hit_fraction", 0.0) or 0.0))
        path_hit = max(path_hit, float(existing_metrics.get("path_hit_fraction", 0.0) or 0.0))

    unique_sources = len(set(it.source for it in items if it.source))
    retrieval_latency = retrieval_latency_override
    if retrieval_latency is None:
        retrieval_latency = float(existing_metrics.get("retrieval_latency_s", 0.0) or payload.get("retrieval_latency_s", 0.0) or 0.0)

    gen_latency = float(existing_metrics.get("generation_latency_s", 0.0) or payload.get("generation_latency_s", 0.0) or 0.0)

    seed_n = len(set(seeds))
    exp_n = len(set(expanded))
    if isinstance(existing_metrics, dict):
        seed_n = max(seed_n, int(existing_metrics.get("seed_node_count", 0.0) or 0))
        exp_n = max(exp_n, int(existing_metrics.get("expanded_node_count", 0.0) or 0))

    return EvalRow(
        query_id=gold.query_id,
        query_type=gold.query_type,
        config_name=config_name,
        mode=mode,
        query=gold.query,
        n_retrieved=len(items),
        strict_available=strict_available,
        strict_first_hit_rank=first_hit_rank(strict_rel) if strict_available else 0,
        soft_first_hit_rank=first_hit_rank(soft_rel) if soft_available else 0,
        source_first_hit_rank=first_hit_rank(source_rel) if gold.gold_sources else 0,
        graph_entity_hit=graph_hit,
        graph_path_hit=path_hit,
        answer_abstained=1.0 if is_abstention(answer) else 0.0,
        should_abstain=1.0 if gold.is_negative else 0.0,
        retrieval_latency_s=float(retrieval_latency),
        generation_latency_s=gen_latency,
        unique_sources=unique_sources,
        seed_node_count=seed_n,
        expanded_node_count=exp_n,
        metrics=all_metrics,
    )


# ----------------------------
# Writing outputs
# ----------------------------

def row_to_flat_dict(row: EvalRow) -> Dict[str, Any]:
    d = {
        "query_id": row.query_id,
        "query_type": row.query_type,
        "config_name": row.config_name,
        "mode": row.mode,
        "query": row.query,
        "n_retrieved": row.n_retrieved,
        "strict_available": row.strict_available,
        "strict_first_hit_rank": row.strict_first_hit_rank,
        "soft_first_hit_rank": row.soft_first_hit_rank,
        "source_first_hit_rank": row.source_first_hit_rank,
        "graph_entity_hit": row.graph_entity_hit,
        "graph_path_hit": row.graph_path_hit,
        "answer_abstained": row.answer_abstained,
        "should_abstain": row.should_abstain,
        "retrieval_latency_s": row.retrieval_latency_s,
        "generation_latency_s": row.generation_latency_s,
        "unique_sources": row.unique_sources,
        "seed_node_count": row.seed_node_count,
        "expanded_node_count": row.expanded_node_count,
    }
    d.update(row.metrics)
    return d


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)

    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def summarise(rows: List[EvalRow]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[EvalRow]] = defaultdict(list)
    for r in rows:
        groups[(r.config_name, "ALL")].append(r)
        groups[(r.config_name, r.query_type)].append(r)

    out = []
    for (cfg, qtype), rs in sorted(groups.items()):
        flat = [row_to_flat_dict(r) for r in rs]

        metric_keys = []
        for r in flat:
            for k, v in r.items():
                if isinstance(v, (int, float)) and k not in {"should_abstain"}:
                    metric_keys.append(k)
        metric_keys = sorted(set(metric_keys))

        rec: Dict[str, Any] = {
            "config_name": cfg,
            "query_type": qtype,
            "n": len(rs),
            "n_strict_available": sum(r.strict_available for r in rs),
            "n_negative": sum(1 for r in rs if r.should_abstain > 0),
        }

        for k in metric_keys:
            vals = [float(r[k]) for r in flat if k in r and isinstance(r[k], (int, float))]
            rec[k] = safe_mean(vals)
            rec[k + "_se"] = safe_se(vals)

        # Negative abstention accuracy, only over negative rows.
        neg = [r for r in rs if r.should_abstain > 0]
        if neg:
            rec["negative_abstention_accuracy"] = safe_mean(r.answer_abstained for r in neg)
        else:
            rec["negative_abstention_accuracy"] = float("nan")

        out.append(rec)

    return out


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--run_dir", type=Path, default=None, help="Existing run with per_query_results.jsonl and payloads.")
    ap.add_argument("--chat", type=Path, default=None, help="Path to chat.py if running fresh retrieval.")
    ap.add_argument("--configs", default="bm25,dense,hybrid,graph,graph_path,agentic_graph")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--ks", default="1,3,5,8,10")
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    golds = load_benchmark(args.benchmark)
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    configs = [x.strip() for x in args.configs.split(",") if x.strip()]

    rows: List[EvalRow] = []

    if args.run_dir is not None:
        old_rows = load_existing_run(args.run_dir)

        for old in old_rows:
            qid = str(old.get("query_id", ""))
            if qid not in golds:
                continue

            cfg = str(old.get("config_name", "unknown"))
            mode = str(old.get("mode", CONFIG_TO_MODE.get(cfg, cfg)))
            payload = {}

            payload_path = old.get("payload_path", "")
            if payload_path:
                payload = load_payload(payload_path, args.run_dir)

            # Fallback: if payload missing, use old row itself. This cannot recover
            # retrieved text, but preserves graph/latency metrics.
            if not payload:
                payload = old

            if isinstance(old.get("metrics"), dict):
                payload.setdefault("metrics", old["metrics"])

            rows.append(evaluate_one(
                gold=golds[qid],
                config_name=cfg,
                mode=mode,
                payload=payload,
                top_k=args.top_k,
                ks=ks,
                retrieval_latency_override=None,
            ))

    elif args.chat is not None:
        for cfg in configs:
            mode = CONFIG_TO_MODE.get(cfg, cfg)
            for qid, gold in golds.items():
                print(f"[{cfg}] {qid}: {gold.query}")
                payload, lat = run_chat(args.chat, gold.query, mode, args.top_k)
                rows.append(evaluate_one(
                    gold=gold,
                    config_name=cfg,
                    mode=mode,
                    payload=payload,
                    top_k=args.top_k,
                    ks=ks,
                    retrieval_latency_override=lat,
                ))
    else:
        raise SystemExit("Provide either --run_dir or --chat.")

    flat_rows = [row_to_flat_dict(r) for r in rows]
    summary = summarise(rows)

    write_csv(out_dir / "per_query_metrics_paper.csv", flat_rows)
    write_jsonl(out_dir / "per_query_metrics_paper.jsonl", flat_rows)
    write_csv(out_dir / "summary_paper.csv", summary)

    with open(out_dir / "summary_paper.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Audit file: tells you whether benchmark is ready for strict publication metrics.
    audit = {
        "n_queries": len(golds),
        "n_with_gold_chunk_ids": sum(1 for g in golds.values() if g.gold_chunk_ids),
        "n_with_gold_sources": sum(1 for g in golds.values() if g.gold_sources),
        "n_with_soft_gold": sum(1 for g in golds.values() if g.gold_answers or g.gold_entities or g.aliases),
        "n_negative": sum(1 for g in golds.values() if g.is_negative),
        "warning": (
            "For a paper, strict Hit@k/MRR/NDCG should only be reported if "
            "n_with_gold_chunk_ids or n_with_gold_sources is large enough. "
            "Soft and graph metrics are valid only if gold_answers/gold_entities/aliases are curated."
        ),
    }

    with open(out_dir / "benchmark_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    print("Saved:", out_dir / "summary_paper.csv")
    print("Saved:", out_dir / "per_query_metrics_paper.csv")
    print("Saved:", out_dir / "benchmark_audit.json")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
