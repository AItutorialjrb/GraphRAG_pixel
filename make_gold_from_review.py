import argparse
import json
from pathlib import Path

def get_items(obj):
    if isinstance(obj, list):
        return obj
    return obj.get("items") or obj.get("questions") or obj.get("data")

def qid(x, i):
    return x.get("query_id") or x.get("id") or x.get("qid") or f"q{i:03d}"

ap = argparse.ArgumentParser()
ap.add_argument("--benchmark", required=True)
ap.add_argument("--review", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

bench = json.load(open(args.benchmark))
review = json.load(open(args.review))

bench_items = get_items(bench)
review_items = get_items(review)
rmap = {qid(x, i): x for i, x in enumerate(review_items)}

for i, x in enumerate(bench_items):
    q = qid(x, i)
    r = rmap.get(q)

    gold_chunk_ids = []
    gold_sources = []

    if r:
        for c in r.get("candidates", []):
            if c.get("gold"):
                gold_chunk_ids.append(str(c["chunk_id"]))
                if c.get("source"):
                    gold_sources.append(c["source"])

    x["gold_chunk_ids"] = sorted(set(gold_chunk_ids), key=lambda z: int(z) if z.isdigit() else z)
    x["gold_sources"] = sorted(set(gold_sources))

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(bench, open(args.out, "w"), indent=2, ensure_ascii=False)

print("wrote", args.out)
print("questions:", len(bench_items))
print("with gold:", sum(bool(x.get("gold_chunk_ids")) for x in bench_items))
