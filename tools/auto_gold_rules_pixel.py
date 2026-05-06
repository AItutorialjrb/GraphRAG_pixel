#!/usr/bin/env python3

import json
import re
from pathlib import Path


INP = "data/eval/gold_review_clean.json"
OUT = "data/eval/gold_review_final_auto.json"


def norm(s):
    s = str(s or "").lower()
    s = s.replace("‐", "-").replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s)
    return s


def hit_any(t, words):
    return any(w.lower() in t for w in words)


def score_candidate(qid, q, text):
    t = norm(text)
    qn = norm(q)

    # negative questions: no gold
    if qid.startswith("neg"):
        return 0

    rules = {
        "fac005": ["1 mev neq", "n eq", "neq cm", "fluence", "proton fluence"],
        "fac006": ["discriminator", "threshold", "comparator", "hit"],
        "fac007": ["depletion depth", "depleted region", "full depletion", "depletion voltage"],
        "fac008": ["beam test", "test beam", "device under test", "dut", "telescope", "performance"],
        "fac009": ["hit detection efficiency", "detection efficiency", "hit efficiency", "ratio", "tracks"],
        "fac010": ["equivalent noise charge", "enc", "noise", "electron"],
        "fac011": ["bump bond", "bump bonding", "bonded", "readout chip"],
        "fac012": ["threshold scan", "s-curve", "threshold", "injected charge"],

        "con002": ["drift", "charge collection", "timing", "fast"],
        "con003": ["material budget", "vertex", "multiple scattering", "tracking"],
        "con004": ["reverse bias", "bias voltage", "depletion", "charge collection"],
        "con005": ["monolithic", "integration", "readout electronics", "bump bonding", "complexity"],
        "con006": ["in-time efficiency", "lhc", "bunch crossing", "25 ns", "timing"],
        "con007": ["dense retrieval", "terminology", "semantic", "acronym"],
        "con008": ["bm25", "technical", "terminology", "exact", "keyword"],
        "con009": ["abstention", "unanswerable", "scientific", "hallucination"],
        "con010": ["retrieval", "generation", "evaluate", "evidence"],
        "con011": ["local", "deployment", "collaboration", "privacy", "data"],
        "con012": ["graph", "path", "evidence", "inspect", "trace"],

        "com004": ["spad", "linear", "lidar", "photon", "silicon sensor"],
        "com006": ["bm25", "dense retrieval", "semantic", "keyword"],
        "com007": ["hybrid", "graphrag", "graph", "retrieval"],
        "com008": ["front-side", "back-side", "illumination", "bsi", "fsi"],
        "com009": ["drift", "diffusion", "charge collection"],
        "com010": ["precision", "recall", "p@k", "r@k"],
        "com011": ["citation", "faithfulness", "coverage", "evidence"],
        "com012": ["reranking", "graph expansion", "graph", "rerank"],

        "mul002": ["irradiation", "efficiency", "timing", "fluence", "radiation"],
        "mul003": ["threshold", "noise", "hit efficiency", "detection efficiency"],
        "mul004": ["graph", "detector technology", "performance metric", "entity"],
        "mul005": ["material budget", "vertexing", "multiple scattering", "impact parameter"],
        "mul007": ["leakage current", "cooling", "noise", "irradiation"],
        "mul008": ["graph noise", "hallucinated", "detector comparison", "spurious"],
        "mul009": ["negative questions", "unanswerable", "weakness", "generation"],
        "mul010": ["retrieval failure", "generation failure", "benchmark", "evidence"],
        "mul011": ["geometry", "electric field", "charge collection", "sensor"],
        "mul012": ["agentic", "routing", "multi-hop", "detector questions"],
    }

    score = 0

    # If answer/claims are real, use them strongly.
    # Placeholder answers are ignored.
    if qid not in rules:
        return 0

    for w in rules[qid]:
        if w in t:
            score += 1

    # Penalise pure references/bibliography chunks unless they are definition questions.
    ref_like = (
        ("references" in t and t.count("doi") >= 2)
        or t.count("http") >= 4
        or t.count("doi") >= 4
    )
    if ref_like and not qid.startswith("fac"):
        score -= 2

    # Additional query-specific stricter logic.
    if qid == "fac006" and not ("discriminator" in t or "comparator" in t):
        score = 0
    if qid == "fac010" and not ("enc" in t or "equivalent noise charge" in t or "noise" in t):
        score = 0
    if qid == "fac011" and not ("bump bond" in t or "bump bonding" in t):
        score = 0
    if qid == "fac012" and not ("threshold" in t and ("scan" in t or "s-curve" in t or "injected charge" in t)):
        score = 0

    return score


def main():
    d = json.load(open(INP, "r", encoding="utf-8"))

    for item in d:
        qid = item["query_id"]
        q = item["query"]

        # reset non-negative gold first, keep negative empty
        for c in item.get("candidates", []):
            c["gold"] = False

        if item.get("should_abstain") or qid.startswith("neg"):
            continue

        scored = []
        for c in item.get("candidates", []):
            s = score_candidate(qid, q, c.get("text", ""))
            if s > 0:
                scored.append((s, c))

        scored.sort(key=lambda x: (-x[0], x[1].get("rank", 999)))

        # Keep at most 3 gold chunks per query.
        for s, c in scored[:3]:
            c["gold"] = True

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    json.dump(d, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    n_gold_q = sum(any(c.get("gold") for c in x.get("candidates", [])) for x in d)
    n_gold_c = sum(c.get("gold") for x in d for c in x.get("candidates", []))
    print("wrote", OUT)
    print("queries with gold:", n_gold_q, "/", len(d))
    print("gold candidates:", n_gold_c)

    print("\nMissing non-negative queries:")
    for x in d:
        if x.get("should_abstain") or x["query_id"].startswith("neg"):
            continue
        if not any(c.get("gold") for c in x.get("candidates", [])):
            print(x["query_id"], x["query"])


if __name__ == "__main__":
    main()
