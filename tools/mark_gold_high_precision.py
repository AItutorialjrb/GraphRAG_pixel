#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path


def norm(s):
    s = str(s or "").lower()
    s = s.replace("‐", "-").replace("–", "-").replace("—", "-")
    s = re.sub(r"[^a-z0-9+\-]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def compact(s):
    return re.sub(r"[^a-z0-9]+", "", norm(s))


def phrase_hit(phrase, text):
    return norm(phrase) in norm(text) or compact(phrase) in compact(text)


def high_precision_gold(item, cand):
    if item.get("should_abstain", False):
        return False

    q = item.get("query", "")
    answer = item.get("answer", "")
    claims = item.get("claims", []) or []
    text = cand.get("text", "")

    if "reference answer to be finalised" in norm(answer):
        return False

    for claim in claims:
        if claim and phrase_hit(claim, text):
            return True

    if answer and len(answer) < 300 and phrase_hit(answer, text):
        return True

    qn = norm(q)

    if "hv-maps" in qn or "hvmaps" in qn:
        return any(
            phrase_hit(p, text)
            for p in [
                "HV-MAPS High-Voltage Monolithic Active Pixel Sensors",
                "HV-MAPS High Voltage Monolithic Active Pixel Sensors",
                "High-Voltage Monolithic Active Pixel Sensors",
                "High Voltage Monolithic Active Pixel Sensors",
            ]
        )

    if "lgad" in qn and "stand" in qn:
        return any(
            phrase_hit(p, text)
            for p in [
                "LGAD Low-Gain Avalanche Detector",
                "LGAD Low Gain Avalanche Detector",
                "Low-Gain Avalanche Detector",
                "Low Gain Avalanche Detector",
            ]
        )

    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = json.load(open(args.review, "r", encoding="utf-8"))

    n_gold = 0
    n_queries = 0

    for item in data:
        has_gold = False
        for cand in item.get("candidates", []):
            cand["gold"] = bool(high_precision_gold(item, cand))
            if cand["gold"]:
                has_gold = True
                n_gold += 1
        if has_gold:
            n_queries += 1

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(data, open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print("Wrote:", args.out)
    print("gold candidates:", n_gold)
    print("queries with >=1 gold:", n_queries)


if __name__ == "__main__":
    main()
