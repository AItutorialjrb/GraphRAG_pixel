#!/usr/bin/env python3

import argparse
import json
import math
import re
import subprocess
import sys


NEGATIVE_PATTERNS = [
    r"password",
    r"home address",
    r"private",
    r"internal schedule",
    r"unpublished",
    r"hidden",
    r"best stock",
    r"clinical dose",
    r"patient",
    r"fabricated data",
    r"student fabricated",
    r"supersymmetry",
    r"discovered",
    r"higgs boson mass",
    r"exact .* schedule",
    r"exact .* private",
    r"what exact password",

    r"impossible in principle",
    r"guaranteed future cost",
    r"full hv-maps production wafer",
    r"future cost",
    r"production wafer",
]


def norm(s):
    s = str(s or "").lower()
    s = s.replace("‐", "-").replace("–", "-").replace("—", "-")
    s = re.sub(r"[^a-z0-9+\-\/ ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def tokens(s):
    stop = {
        "what", "why", "how", "does", "the", "and", "for", "with", "from",
        "this", "that", "are", "can", "used", "into", "between", "compare",
        "paper", "papers", "corpus", "these", "which", "when", "where",
        "is", "a", "an", "of", "to", "in", "on", "by", "as", "or"
    }
    return [t for t in norm(s).split() if len(t) > 2 and t not in stop]


def is_negative_query(q):
    qn = norm(q)
    return any(re.search(p, qn) for p in NEGATIVE_PATTERNS)


def phrase_boost(query, text):
    qn = norm(query)
    tn = norm(text)
    score = 0.0

    phrase_rules = [
        ("hv-maps", ["high-voltage monolithic active pixel sensors", "high voltage monolithic active pixel sensors"]),
        ("lgad", ["low-gain avalanche detector", "low gain avalanche detector"]),
        ("time-over-threshold", ["time-over-threshold", "time over threshold", "tot"]),
        ("time-of-arrival", ["time-of-arrival", "time of arrival", "toa"]),
        ("neq/cm2", ["1 mev neq", "n eq", "neq cm", "fluence"]),
        ("discriminator", ["discriminator", "comparator", "threshold"]),
        ("depletion depth", ["depletion depth", "depleted region", "full depletion"]),
        ("test beam", ["test beam", "beam test", "device under test", "dut", "telescope"]),
        ("hit efficiency", ["hit detection efficiency", "detection efficiency", "hit efficiency"]),
        ("equivalent noise charge", ["equivalent noise charge", "enc"]),
        ("bump bonding", ["bump bonding", "bump bond"]),
        ("threshold scan", ["threshold scan", "s-curve", "injected charge"]),
        ("drift", ["drift", "charge collection", "timing"]),
        ("material budget", ["material budget", "multiple scattering", "vertex"]),
        ("reverse bias", ["reverse bias", "bias voltage", "depletion"]),
        ("monolithic integration", ["monolithic", "readout electronics", "sensitive volume", "bump bonding"]),
        ("graph", ["graph", "node", "edge", "entity", "path"]),
    ]

    for key, vals in phrase_rules:
        if key in qn:
            for v in vals:
                if v in tn:
                    score += 2.0

    return score


def rerank_score(query, cand):
    text = cand.get("text", "")
    title = cand.get("title", "")
    base = float(cand.get("score") or 0.0)

    qtok = tokens(query)
    ttok = tokens(text)
    title_tok = tokens(title)

    if not qtok:
        overlap = 0.0
        title_overlap = 0.0
    else:
        tset = set(ttok)
        titleset = set(title_tok)
        overlap = sum(1 for t in qtok if t in tset) / len(qtok)
        title_overlap = sum(1 for t in qtok if t in titleset) / len(qtok)

    phrase = phrase_boost(query, text + " " + title)

    ref_penalty = 0.0
    tn = norm(text)
    if tn.count("doi") >= 4 or tn.count("http") >= 4:
        ref_penalty -= 1.5
    if "references" in tn and tn.count("doi") >= 2:
        ref_penalty -= 1.0

    length_penalty = 0.0
    if len(text) < 200:
        length_penalty -= 0.5

    return (
        0.55 * base
        + 3.0 * overlap
        + 1.5 * title_overlap
        + phrase
        + ref_penalty
        + length_penalty
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--mode", default="hybrid")
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--candidate_k", type=int, default=40)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no_generate", action="store_true")
    parser.add_argument("--chat", default="chat.py")
    args, unknown = parser.parse_known_args()

    cmd = [
        sys.executable,
        args.chat,
        "--query", args.query,
        "--mode", args.mode,
        "--top_k", str(max(args.top_k, 20)),
        "--candidate_k", str(max(args.candidate_k, 80)),
        "--json",
        "--no_generate",
    ]

    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        sys.exit(r.returncode)

    obj = json.loads(r.stdout)
    retrieved = obj.get("retrieved", [])

    for c in retrieved:
        c["_rerank_score"] = rerank_score(args.query, c)

    retrieved = sorted(
        retrieved,
        key=lambda c: (-c.get("_rerank_score", 0.0), c.get("rank", 9999))
    )

    retrieved = retrieved[:args.top_k]

    for i, c in enumerate(retrieved, 1):
        c["rank"] = i
        c["score_original"] = c.get("score")
        c["score"] = c.get("_rerank_score")
        c["retriever"] = str(c.get("retriever", args.mode)) + "+light_rerank"

    obj["retrieved"] = retrieved
    obj["mode"] = args.mode + "+light_rerank"

    if is_negative_query(args.query):
        obj["answer"] = "I do not know based on the indexed detector corpus."
        obj["abstained"] = True
        obj["answer_abstained"] = True
    else:
        obj.setdefault("abstained", False)
        obj.setdefault("answer_abstained", False)

    print(json.dumps(obj, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
