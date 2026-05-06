#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_benchmark(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        qs = data.get("questions", data.get("queries", data.get("items", list(data.values()))))
    else:
        qs = data
    return qs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--review", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    qs = load_benchmark(Path(args.benchmark))

    with open(args.review) as f:
        review = json.load(f)

    by_id = {str(r["query_id"]): r for r in review}

    for q in qs:
        qid = str(q.get("query_id", q.get("id", "")))
        r = by_id.get(qid)
        if not r:
            continue

        gold_chunk_ids = set(str(x) for x in q.get("gold_chunk_ids", []))
        gold_sources = set(str(x) for x in q.get("gold_sources", []))

        for c in r.get("candidates", []):
            if c.get("gold") is True:
                if c.get("chunk_id"):
                    gold_chunk_ids.add(str(c["chunk_id"]))
                if c.get("source"):
                    gold_sources.add(str(c["source"]))

        q["gold_chunk_ids"] = sorted(gold_chunk_ids)
        q["gold_sources"] = sorted(gold_sources)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(qs, f, indent=2, ensure_ascii=False)

    n_chunk = sum(bool(q.get("gold_chunk_ids")) for q in qs)
    n_source = sum(bool(q.get("gold_sources")) for q in qs)

    print("Saved", args.out)
    print("questions with gold_chunk_ids:", n_chunk)
    print("questions with gold_sources:", n_source)


if __name__ == "__main__":
    main()
