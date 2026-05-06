#!/usr/bin/env python3
import json
from pathlib import Path

IN = Path("data/eval/benchmark_jinst.json")
OUT = Path("data/eval/benchmark_jinst_soft_gold.json")

SOFT_GOLD = {
    "fac001": {
        "answer": "HV-MAPS stands for High-Voltage Monolithic Active Pixel Sensors.",
        "claims": ["HV-MAPS stands for High-Voltage Monolithic Active Pixel Sensors."],
        "gold_entities": ["HV-MAPS", "HVMAPS", "High-Voltage Monolithic Active Pixel Sensors", "HV-CMOS"],
    },
    "fac002": {
        "answer": "LGAD stands for Low-Gain Avalanche Detector.",
        "claims": ["LGAD stands for Low-Gain Avalanche Detector."],
        "gold_entities": ["LGAD", "Low-Gain Avalanche Detector", "Low Gain Avalanche Detector"],
    },
    "fac003": {
        "answer": "Time-over-Threshold is used as a proxy for deposited charge or signal amplitude in pixel detector readout.",
        "claims": ["Time-over-Threshold provides charge or amplitude information in addition to timing."],
        "gold_entities": ["Time-over-Threshold", "ToT", "charge", "signal amplitude", "pixel detector readout"],
    },
}

BAD_ENTITY_WORDS = {
    "what", "is", "are", "was", "were", "used", "use", "for", "how", "why",
    "does", "do", "did", "the", "a", "an", "in", "on", "of", "to", "with",
    "compare", "between", "and",
}

def clean_entities(ents):
    out = []
    for e in ents or []:
        s = str(e).strip()
        if not s:
            continue
        if s.lower() in BAD_ENTITY_WORDS:
            continue
        if len(s) < 3:
            continue
        out.append(s)
    return sorted(set(out))

with open(IN) as f:
    data = json.load(f)

if isinstance(data, dict):
    qs = data.get("questions", data.get("queries", data.get("items", list(data.values()))))
else:
    qs = data

for q in qs:
    qid = q.get("query_id") or q.get("id")
    if qid in SOFT_GOLD:
        q.update(SOFT_GOLD[qid])
    else:
        q["gold_entities"] = clean_entities(q.get("gold_entities", []))

with open(OUT, "w") as f:
    json.dump(qs, f, indent=2, ensure_ascii=False)

print("Saved", OUT)
