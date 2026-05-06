#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def f(x):
    try:
        return float(x)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()

    run = Path(args.run_dir)

    with open(run / "benchmark_audit.json") as f:
        audit = json.load(f)

    print("AUDIT")
    print(json.dumps(audit, indent=2))

    rows = read_csv(run / "summary.csv")

    print("\nSUMMARY ALL")
    for r in rows:
        if r["query_type"] != "ALL":
            continue

        keys = [
            "config_name",
            "n",
            "n_with_gold_chunks",
            "n_with_gold_sources",
            "soft_Hit@5",
            "soft_MRR",
            "paper_Hit@5",
            "paper_MRR",
            "strict_Hit@5",
            "strict_MRR",
            "graph_entity_hit",
            "path_entity_hit",
            "retrieval_latency_s",
            "unique_sources_observed",
        ]

        print("\n" + r["config_name"])
        for k in keys:
            if k in r:
                print(f"  {k}: {r[k]}")

    perq = read_csv(run / "per_query_metrics.csv")

    zero_soft = [
        r for r in perq
        if r.get("soft_Hit@5", "") not in ("", "nan", "NaN")
        and f(r.get("soft_Hit@5")) == 0.0
    ]

    print("\nSoft Hit@5 failures:", len(zero_soft))
    for r in zero_soft[:10]:
        print(r["config_name"], r["query_id"], r["query"])


if __name__ == "__main__":
    main()
