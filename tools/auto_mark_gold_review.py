#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path


STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "being", "been", "by", "as", "at",
    "from", "that", "this", "these", "those", "it", "its", "into", "using",
    "used", "use", "does", "do", "what", "which", "how", "why", "when",
    "where", "stand", "stands"
}


def norm(s):
    s = s or ""
    s = s.lower()
    s = s.replace("‐", "-").replace("-", "-").replace("–", "-").replace("—", "-")
    s = re.sub(r"[^a-z0-9+\-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokens(s):
    return [
        t for t in norm(s).split()
        if len(t) > 2 and t not in STOPWORDS
    ]


def phrase_in_text(phrase, text):
    p = norm(phrase)
    t = norm(text)
    if not p:
        return False
    return p in t


def token_overlap_score(answer, text):
    a = set(tokens(answer))
    if not a:
        return 0.0
    t = set(tokens(text))
    return len(a & t) / len(a)


def entity_hits(entities, text):
    hits = []
    for e in entities or []:
        if phrase_in_text(e, text):
            hits.append(e)
    return hits


def is_definition_question(q):
    qn = norm(q)
    return (
        "stand for" in qn
        or "stands for" in qn
        or "what is" in qn
        or "what does" in qn
    )


def candidate_score(item, cand):
    q = item.get("query", "")
    answer = item.get("answer", "")
    claims = item.get("claims", []) or []
    entities = item.get("gold_entities", []) or []
    text = cand.get("text", "")

    score = 0
    reasons = []

    # Strongest evidence: full claim appears in candidate text.
    for claim in claims:
        if phrase_in_text(claim, text):
            score += 5
            reasons.append(f"claim_exact:{claim[:80]}")

    # Answer phrase appears directly.
    if phrase_in_text(answer, text):
        score += 4
        reasons.append("answer_exact")

    # Entity support.
    hits = entity_hits(entities, text)
    if hits:
        score += min(len(hits), 4)
        reasons.append("entity_hits:" + ",".join(hits[:5]))

    # Token-level answer coverage.
    ov = token_overlap_score(answer, text)
    if ov >= 0.75:
        score += 3
        reasons.append(f"answer_token_overlap:{ov:.2f}")
    elif ov >= 0.55:
        score += 2
        reasons.append(f"answer_token_overlap:{ov:.2f}")
    elif ov >= 0.40:
        score += 1
        reasons.append(f"answer_token_overlap:{ov:.2f}")

    # Extra caution for definition questions:
    # require at least a phrase/entity match plus reasonable answer overlap.
    if is_definition_question(q):
        if not hits and ov < 0.55:
            score -= 3
            reasons.append("definition_question_low_support")

    return score, reasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=int, default=6)
    ap.add_argument("--max_per_query", type=int, default=3)
    ap.add_argument("--apply", action="store_true",
                    help="Actually set gold=true. Without this, only adds auto_suggested fields.")
    args = ap.parse_args()

    data = json.load(open(args.review, "r", encoding="utf-8"))

    n_suggested = 0
    n_queries_with_suggestion = 0

    for item in data:
        if item.get("should_abstain", False):
            continue

        scored = []
        for cand in item.get("candidates", []):
            score, reasons = candidate_score(item, cand)
            cand["auto_gold_score"] = score
            cand["auto_gold_reasons"] = reasons
            cand["auto_suggested_gold"] = False
            scored.append((score, cand))

        scored.sort(key=lambda x: x[0], reverse=True)

        selected = 0
        for score, cand in scored:
            if score >= args.threshold and selected < args.max_per_query:
                cand["auto_suggested_gold"] = True
                if args.apply:
                    cand["gold"] = True
                selected += 1
                n_suggested += 1

        if selected:
            n_queries_with_suggestion += 1

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(data, open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print("Wrote:", args.out)
    print("Suggested gold candidates:", n_suggested)
    print("Queries with at least one suggestion:", n_queries_with_suggestion)
    print("Mode:", "APPLIED gold=true" if args.apply else "suggestion only")


if __name__ == "__main__":
    main()
