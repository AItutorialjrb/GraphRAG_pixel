#!/usr/bin/env python3

import argparse
import json
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


def tokenize(text):
    text = text.lower()
    text = text.replace("‐", "-").replace("–", "-").replace("—", "-")
    text = re.sub(r"[^a-z0-9+\-]+", " ", text)
    return text.split()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/index/chunks.json")
    ap.add_argument("--out", default="data/index/bm25.pkl")
    args = ap.parse_args()

    chunks = json.load(open(args.chunks, "r", encoding="utf-8"))

    corpus_tokens = []
    for c in chunks:
        text = " ".join([
            str(c.get("title", "")),
            str(c.get("source", "")),
            str(c.get("text", "")),
        ])
        corpus_tokens.append(tokenize(text))

    bm25 = BM25Okapi(corpus_tokens)

    obj = {
        "bm25": bm25,
        "chunk_ids": [str(c["chunk_id"]) for c in chunks],
        "tokenizer": "simple_detector_tokenizer_v1",
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    backup = Path(args.out).with_suffix(".pkl.bak")
    if Path(args.out).exists():
        Path(args.out).rename(backup)
        print("Backed up old BM25 to:", backup)

    with open(args.out, "wb") as f:
        pickle.dump(obj, f)

    print("Wrote:", args.out)
    print("n_chunks:", len(chunks))

    # sanity check
    q = "High-Voltage Monolithic Active Pixel Sensors"
    scores = bm25.get_scores(tokenize(q))
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10]

    print("\nSanity check:", q)
    for r, i in enumerate(top, start=1):
        c = chunks[i]
        print(r, "chunk_id=", c["chunk_id"], "score=", float(scores[i]), "title=", c.get("title", "")[:80])


if __name__ == "__main__":
    main()
