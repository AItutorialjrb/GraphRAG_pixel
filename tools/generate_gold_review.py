#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path


def load_benchmark(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("questions", data.get("queries", data.get("items", list(data.values()))))
    return data


def run_chat(chat, query, mode, top_k, candidate_k):
    cmds = [
        [sys.executable, str(chat), "--query", query, "--mode", mode, "--top_k", str(top_k), "--candidate_k", str(candidate_k), "--json", "--no_generate"],
        [sys.executable, str(chat), "--query", query, "--mode", mode, "--top_k", str(top_k), "--json", "--no_generate"],
        [sys.executable, str(chat), "--query", query, "--mode", mode, "--json", "--no_generate"],
    ]

    last_err = ""
    for cmd in cmds:
        p = subprocess.run(cmd, text=True, capture_output=True)
        if p.returncode != 0:
            last_err = p.stderr
            continue

        txt = p.stdout.strip()
        try:
            return json.loads(txt)
        except Exception:
            start = txt.find("{")
            end = txt.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(txt[start:end + 1])
                except Exception:
                    pass

    raise RuntimeError("chat.py failed for query '{}'\n{}".format(query, last_err))


def flatten_retrieved(obj):
    keys = [
        "retrieved", "retrieved_chunks", "results", "contexts", "context",
        "evidence", "evidence_chunks", "ranked_chunks", "chunks",
        "top_chunks", "documents", "docs",
    ]

    out = []

    if isinstance(obj, dict):
        for k in keys:
            v = obj.get(k)
            if isinstance(v, list):
                out.extend(v)

        for k in ["data", "output", "response", "result", "payload"]:
            if k in obj:
                out.extend(flatten_retrieved(obj[k]))

    elif isinstance(obj, list):
        for x in obj:
            if isinstance(x, dict):
                out.append(x)
            else:
                out.extend(flatten_retrieved(x))

    return out


def get_value(d, keys):
    if not isinstance(d, dict):
        return ""
    md = d.get("metadata", {})
    merged = dict(d)
    if isinstance(md, dict):
        for k, v in md.items():
            merged.setdefault(k, v)

    for k in keys:
        if k in merged and merged[k] not in (None, ""):
            return merged[k]
    return ""


def make_candidate(raw, rank):
    if isinstance(raw, str):
        return {
            "gold": False,
            "rank": rank,
            "chunk_id": "",
            "source": "",
            "title": "",
            "score": None,
            "text": raw[:2000],
        }

    text = get_value(raw, ["text", "content", "chunk_text", "body", "page_content", "snippet", "context"])
    chunk_id = get_value(raw, ["chunk_id", "id", "doc_id", "document_id", "cid"])
    source = get_value(raw, ["source", "file", "path", "pdf", "paper", "url"])
    title = get_value(raw, ["title", "paper_title", "name"])
    score = get_value(raw, ["score", "retrieval_score", "rank_score", "similarity", "bm25_score"])

    if not text:
        text = json.dumps(raw, ensure_ascii=False)[:2000]

    return {
        "gold": False,
        "rank": rank,
        "chunk_id": str(chunk_id) if chunk_id is not None else "",
        "source": str(source) if source is not None else "",
        "title": str(title) if title is not None else "",
        "score": score,
        "text": str(text)[:2500],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--chat", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--mode", default="agentic_graph_rag")
    ap.add_argument("--top_k", type=int, default=15)
    ap.add_argument("--candidate_k", type=int, default=80)
    args = ap.parse_args()

    questions = load_benchmark(Path(args.benchmark))
    review = []

    for i, q in enumerate(questions, start=1):
        qid = q.get("query_id", q.get("id", "q{:04d}".format(i)))
        query = q.get("query", q.get("question", ""))

        print("[{}/{}] {}: {}".format(i, len(questions), qid, query))

        try:
            payload = run_chat(
                chat=Path(args.chat),
                query=query,
                mode=args.mode,
                top_k=args.top_k,
                candidate_k=args.candidate_k,
            )
            retrieved = flatten_retrieved(payload)
        except Exception as e:
            print("ERROR:", e)
            retrieved = []

        candidates = []
        seen = set()
        for raw in retrieved:
            cand = make_candidate(raw, len(candidates) + 1)
            key = cand["chunk_id"] or cand["source"] + cand["text"][:80]
            if key in seen:
                continue
            seen.add(key)
            candidates.append(cand)
            if len(candidates) >= args.candidate_k:
                break

        review.append({
            "query_id": qid,
            "query_type": q.get("query_type", q.get("type", "")),
            "query": query,
            "answer": q.get("answer", ""),
            "claims": q.get("claims", []),
            "gold_entities": q.get("gold_entities", []),
            "should_abstain": q.get("should_abstain", False),
            "candidates": candidates,
        })

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)

    with open(args.out_json, "w") as f:
        json.dump(review, f, indent=2, ensure_ascii=False)

    with open(args.out_md, "w") as f:
        for item in review:
            f.write("# {} | {}\n\n".format(item["query_id"], item["query"]))
            f.write("Answer: {}\n\n".format(item.get("answer", "")))
            f.write("Gold entities: {}\n\n".format(item.get("gold_entities", [])))
            for c in item["candidates"]:
                f.write("## Rank {}\n\n".format(c["rank"]))
                f.write("- gold: `{}`\n".format(c["gold"]))
                f.write("- chunk_id: `{}`\n".format(c["chunk_id"]))
                f.write("- source: `{}`\n".format(c["source"]))
                f.write("- title: `{}`\n\n".format(c["title"]))
                f.write("```text\n{}\n```\n\n".format(c["text"][:2000]))
            f.write("\n---\n\n")

    print("Saved", args.out_json)
    print("Saved", args.out_md)


if __name__ == "__main__":
    main()
