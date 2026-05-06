#!/usr/bin/env python3
import argparse
import csv
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path


ABSTAIN_PATTERNS = [
    "insufficient evidence",
    "not enough evidence",
    "cannot determine",
    "not provided",
    "not in the context",
    "no evidence",
    "unable to answer",
    "i do not know",
]


ALIASES = {
    "hv maps": "hvmaps",
    "hv-maps": "hvmaps",
    "hvmaps": "hvmaps",
    "high voltage monolithic active pixel sensors": "hvmaps",
    "high-voltage monolithic active pixel sensors": "hvmaps",

    "hv cmos": "hv-cmos",
    "hv-cmos": "hv-cmos",
    "hvcmos": "hv-cmos",

    "maps": "maps",
    "monolithic active pixel sensors": "maps",

    "dmaps": "dmaps",
    "depleted monolithic active pixel sensors": "dmaps",

    "lgad": "lgad",
    "lgads": "lgad",
    "low gain avalanche detector": "lgad",
    "low-gain avalanche detector": "lgad",

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
}


def norm(s):
    s = "" if s is None else str(s)
    s = s.lower()
    s = s.replace("–", "-").replace("—", "-")
    for ch in "_/()[]{}:;,.|":
        s = s.replace(ch, " ")
    return " ".join(s.split())


def compact(s):
    return "".join(ch for ch in norm(s) if ch.isalnum())


def canonical(s):
    x = norm(s)
    if x in ALIASES:
        return ALIASES[x]
    cx = compact(x)
    for k, v in ALIASES.items():
        if compact(k) == cx:
            return v
    return cx


def phrase_match(phrase, text):
    if not phrase or not text:
        return False

    p1 = norm(phrase)
    t1 = norm(text)

    if len(p1) >= 4 and p1 in t1:
        return True

    p2 = canonical(phrase)
    t2 = compact(text)

    if len(p2) >= 3 and p2 in t2:
        return True

    return False


def source_match(src, gold_sources):
    if not src or not gold_sources:
        return False

    s1 = compact(Path(str(src)).name)
    s2 = compact(str(src))

    for g in gold_sources:
        g1 = compact(Path(str(g)).name)
        g2 = compact(str(g))
        if not g1 and not g2:
            continue
        if g1 and (g1 in s2 or s1 in g2 or g1 == s1):
            return True

    return False


def chunk_match(cid, gold_chunk_ids):
    if cid is None:
        return False
    c = str(cid)
    return c in set(str(x) for x in gold_chunk_ids)


def is_abstain(answer):
    a = norm(answer)
    return any(p in a for p in ABSTAIN_PATTERNS)


def load_benchmark(path):
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, dict):
        qs = data.get("questions", data.get("queries", data.get("items", list(data.values()))))
    else:
        qs = data

    out = []
    for i, q in enumerate(qs):
        qid = str(q.get("query_id", q.get("id", f"q{i:04d}")))
        out.append({
            "query_id": qid,
            "query_type": q.get("query_type", q.get("type", "unknown")),
            "query": q.get("query", q.get("question", "")),
            "answer": q.get("answer", ""),
            "claims": q.get("claims", []),
            "gold_entities": q.get("gold_entities", []),
            "gold_chunk_ids": [str(x) for x in q.get("gold_chunk_ids", [])],
            "gold_sources": [str(x) for x in q.get("gold_sources", [])],
            "should_abstain": bool(q.get("should_abstain", q.get("is_negative", False))),
        })
    return out


def run_chat(chat_py, query, mode, top_k, candidate_k, generate):
    cmd = [
        sys.executable, str(chat_py),
        "--query", query,
        "--mode", mode,
        "--top_k", str(top_k),
        "--candidate_k", str(candidate_k),
        "--json",
    ]

    if not generate:
        cmd.append("--no_generate")

    t0 = time.time()
    p = subprocess.run(cmd, text=True, capture_output=True)
    latency = time.time() - t0

    if p.returncode != 0:
        raise RuntimeError(p.stderr)

    text = p.stdout.strip()

    try:
        payload = json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise RuntimeError("Cannot parse JSON from chat.py output:\n" + text[:1000])
        payload = json.loads(text[start:end + 1])

    return payload, latency


def load_existing_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_retrieved(payload):
    for key in ["retrieved", "results", "contexts", "evidence", "chunks", "documents"]:
        if key in payload and isinstance(payload[key], list):
            return payload[key]
    return []


def item_text(item):
    if isinstance(item, str):
        return item

    parts = []
    for key in ["text", "content", "chunk_text", "body", "snippet", "title", "source"]:
        if key in item and item[key]:
            parts.append(str(item[key]))
    return "\n".join(parts)


def item_chunk_id(item):
    if not isinstance(item, dict):
        return ""
    for key in ["chunk_id", "id", "doc_id", "cid"]:
        if key in item:
            return str(item[key])
    return ""


def item_source(item):
    if not isinstance(item, dict):
        return ""
    for key in ["source", "file", "path", "pdf", "paper", "url", "title"]:
        if key in item and item[key]:
            return str(item[key])
    return ""


def first_hit_rank(flags):
    for i, x in enumerate(flags, start=1):
        if x:
            return i
    return 0


def hit_at(flags, k):
    return 1.0 if any(flags[:k]) else 0.0


def precision_at(flags, k):
    return sum(flags[:k]) / float(k)


def recall_at(flags, k, n_gold):
    if n_gold <= 0:
        return float("nan")
    return min(sum(flags[:k]), n_gold) / float(n_gold)


def mrr(flags):
    r = first_hit_rank(flags)
    return 1.0 / r if r else 0.0


def average_precision(flags, n_gold):
    if n_gold <= 0:
        return float("nan")
    hits = 0
    score = 0.0
    for i, f in enumerate(flags, start=1):
        if f:
            hits += 1
            score += hits / float(i)
    if hits == 0:
        return 0.0
    return score / float(min(n_gold, hits))


def dcg(flags, k):
    out = 0.0
    for i, f in enumerate(flags[:k], start=1):
        if f:
            out += 1.0 / math.log2(i + 1)
    return out


def ndcg_at(flags, k, n_gold):
    if n_gold <= 0:
        return float("nan")
    ideal_n = min(k, n_gold)
    ideal = dcg([1] * ideal_n, k)
    if ideal <= 0:
        return 0.0
    return dcg(flags, k) / ideal


def metric_pack(flags, n_gold, ks, prefix):
    d = {}
    d[f"{prefix}_MRR"] = mrr(flags)
    d[f"{prefix}_MAP"] = average_precision(flags, n_gold)
    for k in ks:
        d[f"{prefix}_Hit@{k}"] = hit_at(flags, k)
        d[f"{prefix}_P@{k}"] = precision_at(flags, k)
        d[f"{prefix}_R@{k}"] = recall_at(flags, k, n_gold)
        d[f"{prefix}_NDCG@{k}"] = ndcg_at(flags, k, n_gold)
    return d


def graph_strings(payload):
    xs = []
    for key in ["seed_nodes", "expanded_nodes", "paths"]:
        val = payload.get(key, [])
        if isinstance(val, list):
            for v in val:
                if isinstance(v, list):
                    xs.append(" ".join(str(z) for z in v))
                else:
                    xs.append(str(v))
    return xs


def evaluate_payload(q, payload, mode, config_name, top_k, ks, wall_latency):
    retrieved = get_retrieved(payload)[:top_k]

    gold_chunk_ids = q["gold_chunk_ids"]
    gold_sources = q["gold_sources"]
    soft_terms = []
    soft_terms.extend(q.get("gold_entities", []))
    soft_terms.extend(q.get("claims", []))
    if q.get("answer"):
        soft_terms.append(q["answer"])

    strict_flags = []
    source_flags = []
    chunk_flags = []
    soft_flags = []
    paper_flags = []

    for item in retrieved:
        cid = item_chunk_id(item)
        src = item_source(item)
        txt = item_text(item)

        c_ok = chunk_match(cid, gold_chunk_ids)
        s_ok = source_match(src, gold_sources)
        soft_ok = any(phrase_match(t, txt) for t in soft_terms)

        chunk_flags.append(1 if c_ok else 0)
        source_flags.append(1 if s_ok else 0)
        strict_flags.append(1 if (c_ok or s_ok) else 0)
        soft_flags.append(1 if soft_ok else 0)
        paper_flags.append(1 if (c_ok or s_ok or soft_ok) else 0)

    while len(strict_flags) < top_k:
        strict_flags.append(0)
        source_flags.append(0)
        chunk_flags.append(0)
        soft_flags.append(0)
        paper_flags.append(0)

    row = {
        "query_id": q["query_id"],
        "query_type": q["query_type"],
        "query": q["query"],
        "config_name": config_name,
        "mode": mode,
        "n_retrieved": len(retrieved),
        "n_gold_chunks": len(gold_chunk_ids),
        "n_gold_sources": len(gold_sources),
        "n_soft_terms": len(soft_terms),
        "should_abstain": 1 if q.get("should_abstain") else 0,
        "wall_latency_s": wall_latency,
    }

    strict_available = bool(gold_chunk_ids or gold_sources)
    soft_available = bool(soft_terms)

    if strict_available:
        row.update(metric_pack(strict_flags, max(1, len(gold_chunk_ids) + len(gold_sources)), ks, "strict"))
    else:
        row.update({f"strict_MRR": float("nan"), f"strict_MAP": float("nan")})
        for k in ks:
            row[f"strict_Hit@{k}"] = float("nan")
            row[f"strict_P@{k}"] = float("nan")
            row[f"strict_R@{k}"] = float("nan")
            row[f"strict_NDCG@{k}"] = float("nan")

    if soft_available:
        row.update(metric_pack(soft_flags, max(1, len(soft_terms)), ks, "soft"))
    else:
        row.update({f"soft_MRR": float("nan"), f"soft_MAP": float("nan")})
        for k in ks:
            row[f"soft_Hit@{k}"] = float("nan")
            row[f"soft_P@{k}"] = float("nan")
            row[f"soft_R@{k}"] = float("nan")
            row[f"soft_NDCG@{k}"] = float("nan")

    row.update(metric_pack(paper_flags, max(1, len(gold_chunk_ids) + len(gold_sources) + len(soft_terms)), ks, "paper"))

    gtxt = " ".join(graph_strings(payload))
    row["graph_entity_hit"] = 1.0 if any(phrase_match(t, gtxt) for t in soft_terms) else 0.0
    row["path_entity_hit"] = row["graph_entity_hit"] if payload.get("paths") else 0.0

    answer = payload.get("answer", "")
    row["answer_abstained"] = 1.0 if is_abstain(answer) else 0.0

    metrics = payload.get("metrics", {})
    if isinstance(metrics, dict):
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                row[k] = v

    row["unique_sources_observed"] = len(set(item_source(x) for x in retrieved if item_source(x)))

    return row


def mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and not math.isnan(x)]
    if not xs:
        return float("nan")
    return sum(xs) / len(xs)


def se(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and not math.isnan(x)]
    if len(xs) <= 1:
        return 0.0 if xs else float("nan")
    m = mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var / len(xs))


def summarise(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["config_name"], "ALL")].append(r)
        groups[(r["config_name"], r["query_type"])].append(r)

    out = []
    for (cfg, qt), rs in sorted(groups.items()):
        rec = {
            "config_name": cfg,
            "query_type": qt,
            "n": len(rs),
            "n_with_gold_chunks": sum(1 for r in rs if r["n_gold_chunks"] > 0),
            "n_with_gold_sources": sum(1 for r in rs if r["n_gold_sources"] > 0),
            "n_with_soft_terms": sum(1 for r in rs if r["n_soft_terms"] > 0),
            "n_negative": sum(1 for r in rs if r["should_abstain"] == 1),
        }

        keys = sorted(set(k for r in rs for k, v in r.items() if isinstance(v, (int, float))))
        for k in keys:
            if k in {"should_abstain"}:
                continue
            vals = [r[k] for r in rs if k in r and isinstance(r[k], (int, float))]
            rec[k] = mean(vals)
            rec[k + "_se"] = se(vals)

        negs = [r for r in rs if r["should_abstain"] == 1]
        if negs:
            rec["negative_abstention_accuracy"] = mean([r["answer_abstained"] for r in negs])
        else:
            rec["negative_abstention_accuracy"] = float("nan")

        out.append(rec)

    return out


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--chat", default="./chat.py")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--configs", default="bm25,dense,hybrid,graph,graph_path,agentic_graph")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--candidate_k", type=int, default=80)
    ap.add_argument("--ks", default="1,3,5,8,10")
    ap.add_argument("--generate", action="store_true")
    args = ap.parse_args()

    benchmark = load_benchmark(Path(args.benchmark))
    configs = [x.strip() for x in args.configs.split(",") if x.strip()]
    ks = [int(x) for x in args.ks.split(",") if x.strip()]

    mode_map = {
        "bm25": "bm25",
        "dense": "dense",
        "hybrid": "hybrid",
        "graph": "graph",
        "graph_path": "graph_path",
        "agentic_graph": "agentic_graph",
        "graph_rag": "graph_rag",
        "graph_path_rag": "graph_path_rag",
        "agentic_graph_rag": "agentic_graph_rag",
    }

    rows = []
    payload_dir = Path(args.out_dir) / "payloads"
    payload_dir.mkdir(parents=True, exist_ok=True)

    for cfg in configs:
        mode = mode_map[cfg]
        for i, q in enumerate(benchmark, start=1):
            print(f"[{cfg}] {i}/{len(benchmark)} {q['query_id']}: {q['query']}")

            payload, wall_latency = run_chat(
                chat_py=Path(args.chat),
                query=q["query"],
                mode=mode,
                top_k=args.top_k,
                candidate_k=args.candidate_k,
                generate=args.generate,
            )

            payload_path = payload_dir / f"{cfg}__{q['query_id']}.json"
            with open(payload_path, "w") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

            row = evaluate_payload(
                q=q,
                payload=payload,
                mode=mode,
                config_name=cfg,
                top_k=args.top_k,
                ks=ks,
                wall_latency=wall_latency,
            )
            row["payload_path"] = str(payload_path)
            rows.append(row)

    summary = summarise(rows)

    out_dir = Path(args.out_dir)
    write_csv(out_dir / "per_query_metrics.csv", rows)
    write_jsonl(out_dir / "per_query_metrics.jsonl", rows)
    write_csv(out_dir / "summary.csv", summary)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    audit = {
        "n_queries": len(benchmark),
        "n_with_gold_chunk_ids": sum(1 for q in benchmark if q["gold_chunk_ids"]),
        "n_with_gold_sources": sum(1 for q in benchmark if q["gold_sources"]),
        "n_with_soft_terms": sum(1 for q in benchmark if q["gold_entities"] or q["claims"] or q["answer"]),
        "n_negative": sum(1 for q in benchmark if q["should_abstain"]),
        "recommendation": "For JINST, report strict metrics only after gold_chunk_ids/gold_sources are manually curated. Soft and graph metrics can be reported as complementary evidence coverage metrics.",
    }

    with open(out_dir / "benchmark_audit.json", "w") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    print("Saved", out_dir / "summary.csv")
    print("Saved", out_dir / "per_query_metrics.csv")
    print("Saved", out_dir / "benchmark_audit.json")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
