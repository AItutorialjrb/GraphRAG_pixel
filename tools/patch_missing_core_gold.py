#!/usr/bin/env python3

import json
import re
from pathlib import Path

inp = "data/eval/gold_review_final_auto.json"
out = "data/eval/gold_review_final_auto_patched.json"

d = json.load(open(inp, "r", encoding="utf-8"))

def norm(s):
    s = str(s or "").lower()
    s = s.replace("‐", "-").replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s)
    return s

def score(qid, text):
    t = norm(text)

    rules = {
        "fac001": [
            "hv-maps high-voltage monolithic active pixel sensors",
            "hv-maps high voltage monolithic active pixel sensors",
            "high-voltage monolithic active pixel sensors",
            "high voltage monolithic active pixel sensors",
        ],
        "fac002": [
            "lgad low-gain avalanche detector",
            "lgad low gain avalanche detector",
            "low-gain avalanche detector",
            "low gain avalanche detector",
        ],
        "fac003": [
            "time-over-threshold", "time over threshold", "tot",
            "charge information", "charge", "amplitude"
        ],
        "fac004": [
            "time-of-arrival", "time of arrival", "toa",
            "timestamp", "arrival time", "hit time"
        ],
        "con001": [
            "hv-maps", "hv-cmos", "high-voltage cmos",
            "high-rate", "tracking", "thin", "monolithic", "radiation"
        ],
        "com001": [
            "monolithic", "hybrid pixel", "bump bonding",
            "material budget", "readout electronics", "sensitive volume"
        ],
        "com002": [
            "hv-maps", "lgad", "timing", "gain", "avalanche",
            "monolithic", "charge collection"
        ],
        "com003": [
            "dc-lgad", "ac-lgad", "ac-coupled", "dc-coupled",
            "segmentation", "fill factor", "readout"
        ],
        "com005": [
            "time-of-arrival", "time over threshold", "toa", "tot",
            "timing", "charge", "threshold"
        ],
        "mul001": [
            "depletion depth", "electric field", "timing resolution",
            "charge collection", "drift", "bias voltage"
        ],
        "mul004": [
            "graph", "entity", "detector", "performance",
            "technology", "retrieval"
        ],
        "mul006": [
            "tot", "time-over-threshold", "time over threshold",
            "calibration", "charge", "spatial resolution", "cluster"
        ],
    }

    if qid not in rules:
        return 0

    s = sum(1 for w in rules[qid] if w in t)

    # avoid pure bibliography/reference chunks unless they directly define the acronym
    ref_like = (
        ("references" in t and t.count("doi") >= 2)
        or t.count("http") >= 4
        or t.count("doi") >= 4
    )
    if ref_like and qid not in {"fac001", "fac002"}:
        s -= 2

    return s

patched = {}

for item in d:
    qid = item["query_id"]

    if item.get("should_abstain") or qid.startswith("neg"):
        continue

    # keep existing golds
    if any(c.get("gold") for c in item.get("candidates", [])):
        continue

    scored = []
    for c in item.get("candidates", []):
        s = score(qid, c.get("text", ""))
        if s > 0:
            scored.append((s, c))

    scored.sort(key=lambda x: (-x[0], x[1].get("rank", 999)))

    # Keep 1-3 best candidates.
    for s, c in scored[:3]:
        c["gold"] = True
        patched.setdefault(qid, []).append((c["chunk_id"], s, c.get("rank")))

json.dump(d, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

print("wrote", out)
print("patched queries:", len(patched))
for qid, vals in patched.items():
    print(qid, vals)

print("queries with gold:", sum(any(c.get("gold") for c in x.get("candidates", [])) for x in d), "/", len(d))

print("\nStill missing non-negative queries:")
for x in d:
    if x.get("should_abstain") or x["query_id"].startswith("neg"):
        continue
    if not any(c.get("gold") for c in x.get("candidates", [])):
        print(x["query_id"], x["query"])
