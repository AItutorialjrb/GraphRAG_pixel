#!/usr/bin/env python3

import argparse
import json
import subprocess
from pathlib import Path


ALIASES = {
    "HV-MAPS": [
        "High-Voltage Monolithic Active Pixel Sensors",
        "High Voltage Monolithic Active Pixel Sensors",
        "HV-CMOS",
        "HVCMOS",
        "High-Voltage CMOS Active Pixel Sensor",
    ],
    "HVMAPS": [
        "High-Voltage Monolithic Active Pixel Sensors",
        "High Voltage Monolithic Active Pixel Sensors",
        "HV-CMOS",
        "HVCMOS",
    ],
    "HV-CMOS": [
        "High-Voltage CMOS",
        "High Voltage CMOS",
        "HVCMOS",
        "High-Voltage CMOS Active Pixel Sensor",
    ],
    "LGAD": [
        "Low-Gain Avalanche Detector",
        "Low Gain Avalanche Detector",
    ],
    "AC-LGAD": [
        "AC-coupled LGAD",
        "AC coupled LGAD",
    ],
    "DC-LGAD": [
        "DC-coupled LGAD",
        "DC coupled LGAD",
    ],
    "TI-LGAD": [
        "Trench-Isolated LGAD",
        "Trench Isolated LGAD",
    ],
    "ToT": [
        "Time-over-Threshold",
        "Time over Threshold",
    ],
    "ToA": [
        "Time-of-Arrival",
        "Time of Arrival",
    ],
    "TDC": [
        "Time-to-Digital Converter",
        "Time to Digital Converter",
    ],
    "MAPS": [
        "Monolithic Active Pixel Sensors",
        "Monolithic Active Pixel Sensor",
    ],
    "DMAPS": [
        "Depleted Monolithic Active Pixel Sensors",
        "Depleted MAPS",
    ],
    "TCAD": [
        "Technology Computer-Aided Design",
        "Technology Computer Aided Design",
    ],
    "ASIC": [
        "Application-Specific Integrated Circuit",
        "Application Specific Integrated Circuit",
    ],
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def get_items(obj):
    if isinstance(obj, list):
        return obj
    return obj.get("items") or obj.get("questions") or obj.get("data")


def get_qid(item, i):
    return item.get("query_id") or item.get("id") or item.get("qid") or f"q{i:03d}"


def clean_query(s):
    return " ".join(str(s or "").split())


def expanded_queries(item, max_queries):
    q = clean_query(item.get("query") or item.get("question") or "")
    ans = clean_query(item.get("answer") or "")
    ents = item.get("gold_entities", []) or []

    queries = []

    def add(x):
        x = clean_query(x)
        if x:
            queries.append(x)

    add(q)

    q_lower = q.lower()
    is_definition = (
        "stand for" in q_lower
        or "stands for" in q_lower
        or q_lower.startswith("what is ")
        or q_lower.startswith("what are ")
        or q_lower.startswith("what does ")
    )

    if is_definition:
        for e in ents:
            add(f"{e} definition")
            add(f"{e} stands for")
            add(f"{e} acronym")
            for a in ALIASES.get(e, []):
                add(f"{e} {a}")

    for e in ents:
        add(e)
        for a in ALIASES.get(e, []):
            add(a)

    if ans:
        add(ans)

    out = []
    seen = set()
    for x in queries:
        key = x.lower()
        if key not in seen:
            out.append(x)
            seen.add(key)

    return out[:max_queries]


def parse_chat_stdout(stdout, stderr, query, returncode):
    try:
        return json.loads(stdout)
    except Exception:
        pass

    try:
        s = stdout.strip()
        start = s.find("{")
        if start < 0:
            raise ValueError("No JSON object found in stdout")

        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(s[start:])
        return obj
    except Exception as e:
        print("WARNING: could not parse JSON for query:", query)
        print("returncode:", returncode)
        print("parse error:", e)
        print("STDERR tail:")
        print(stderr[-1000:])
        print("STDOUT head:")
        print(stdout[:1000])
        print("STDOUT tail:")
        print(stdout[-1000:])
        return None


def run_chat(chat, query, mode, top_k, candidate_k):
    cmd = [
        "python",
        chat,
        "--query",
        query,
        "--mode",
        mode,
        "--top_k",
        str(top_k),
        "--candidate_k",
        str(candidate_k),
        "--json",
        "--no_generate",
    ]

    r = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    obj = parse_chat_stdout(r.stdout, r.stderr, query, r.returncode)
    if obj is None:
        return []

    hits = (
        obj.get("retrieved")
        or obj.get("contexts")
        or obj.get("evidence")
        or obj.get("results")
        or obj.get("chunks")
        or obj.get("hits")
        or []
    )

    return hits


def normalise_candidate(c):
    meta = c.get("metadata", {}) if isinstance(c.get("metadata"), dict) else {}

    chunk_id = (
        c.get("chunk_id")
        or c.get("id")
        or meta.get("chunk_id")
        or meta.get("id")
    )

    return {
        "gold": False,
        "rank": c.get("rank"),
        "chunk_id": str(chunk_id),
        "doc_id": str(c.get("doc_id") or meta.get("doc_id") or ""),
        "source": c.get("source") or meta.get("source"),
        "title": c.get("title") or meta.get("title"),
        "score": c.get("score"),
        "retriever": c.get("retriever"),
        "text": c.get("text") or c.get("content") or c.get("chunk_text") or "",
    }


def write_md(review, out_md, max_chars=2500):
    with open(out_md, "w", encoding="utf-8") as f:
        for item in review:
            f.write(f"# {item['query_id']} | {item['query']}\n\n")
            f.write(f"Answer: {item.get('answer')}\n\n")
            f.write(f"Gold entities: {item.get('gold_entities')}\n\n")
            f.write("Expanded queries:\n")
            for q in item.get("expanded_queries", []):
                f.write(f"- {q}\n")
            f.write("\n")

            for c in item["candidates"]:
                f.write(f"## Rank {c['rank']}\n\n")
                f.write(f"- gold: `{c['gold']}`\n")
                f.write(f"- chunk_id: `{c['chunk_id']}`\n")
                f.write(f"- source: `{c.get('source')}`\n")
                f.write(f"- title: `{c.get('title')}`\n")
                f.write(f"- score: `{c.get('score')}`\n")
                f.write(f"- retriever: `{c.get('retriever')}`\n")
                f.write(f"- matched_queries: `{c.get('matched_queries', [])}`\n\n")
                f.write("```text\n")
                f.write((c.get("text") or "")[:max_chars])
                f.write("\n```\n\n")


def load_existing_review(path):
    p = Path(path)
    if not p.exists():
        return [], set()

    try:
        data = load_json(path)
    except Exception:
        return [], set()

    done = set()
    for i, item in enumerate(data):
        qid = item.get("query_id") or item.get("id") or f"q{i:03d}"
        done.add(qid)

    return data, done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--chat", default="./chat.py")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--mode", default="hybrid")
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--candidate_k", type=int, default=60)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max_expanded_queries", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    benchmark = load_json(args.benchmark)
    items = get_items(benchmark)

    if items is None:
        raise RuntimeError("Could not find benchmark items.")

    if args.start:
        items = items[args.start:]

    if args.limit is not None:
        items = items[:args.limit]

    if args.resume:
        review, done_qids = load_existing_review(args.out_json)
    else:
        review, done_qids = [], set()

    for i, item in enumerate(items):
        global_i = args.start + i
        qid = get_qid(item, global_i)
        q = clean_query(item.get("query") or item.get("question") or "")

        if qid in done_qids:
            print(f"{qid}: already done, skipping")
            continue

        used_queries = expanded_queries(item, args.max_expanded_queries)
        merged = {}

        for qq in used_queries:
            hits = run_chat(
                chat=args.chat,
                query=qq,
                mode=args.mode,
                top_k=args.top_k,
                candidate_k=args.candidate_k,
            )

            for h in hits:
                c = normalise_candidate(h)
                cid = c.get("chunk_id")

                if cid is None or cid == "None":
                    continue

                if cid not in merged:
                    c["matched_queries"] = [qq]
                    merged[cid] = c
                else:
                    merged[cid]["matched_queries"].append(qq)

        candidates = list(merged.values())[:args.top_k]

        for rank, c in enumerate(candidates, start=1):
            c["rank"] = rank

        review_item = {
            "query_id": qid,
            "query_type": item.get("query_type"),
            "query": q,
            "answer": item.get("answer"),
            "claims": item.get("claims", []),
            "gold_entities": item.get("gold_entities", []),
            "should_abstain": item.get("should_abstain", False),
            "expanded_queries": used_queries,
            "candidates": candidates,
        }

        review.append(review_item)

        save_json(review, args.out_json)
        write_md(review, args.out_md)

        print(f"{qid}: {len(candidates)} candidates from {len(used_queries)} expanded queries")

    save_json(review, args.out_json)
    write_md(review, args.out_md)

    print("Wrote:", args.out_json)
    print("Wrote:", args.out_md)
    print("Total review items:", len(review))


if __name__ == "__main__":
    main()
