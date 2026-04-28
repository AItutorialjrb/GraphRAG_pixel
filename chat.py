import os
import re
import json
import time
import pickle
import traceback
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

# =========================
# Set cache dirs BEFORE HF imports
# =========================
BASE_DIR = Path("/eos/home-r/rjiang/RAG_pixel")
ENV_PATH = BASE_DIR / ".env"

HF_CACHE = BASE_DIR / "hf_cache"
(HF_CACHE / "hub").mkdir(parents=True, exist_ok=True)
(HF_CACHE / "transformers").mkdir(parents=True, exist_ok=True)
(HF_CACHE / "sentence_transformers").mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE / "transformers")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(HF_CACHE / "sentence_transformers")

import faiss
import numpy as np
import networkx as nx
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# Config
# =========================
load_dotenv(dotenv_path=ENV_PATH)

LOCAL_EMBED_MODEL = os.getenv(
    "LOCAL_EMBED_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
).strip()

LOCAL_CHAT_MODEL = os.getenv(
    "LOCAL_CHAT_MODEL",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct"
).strip()

INDEX_DIR = BASE_DIR / "data" / "index"
GRAPH_DIR = BASE_DIR / "data" / "graph"

FAISS_INDEX_PATH = INDEX_DIR / "dense.index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
BM25_PATH = INDEX_DIR / "bm25.pkl"

GRAPHML_PATH = GRAPH_DIR / "graph.graphml"
ENTITY_TO_CHUNKS_PATH = GRAPH_DIR / "entity_to_chunks.json"
NODE_TEXTS_PATH = GRAPH_DIR / "node_texts.json"
NODE_FAISS_PATH = GRAPH_DIR / "node_dense.index"

EMBED_MODEL = None
CHAT_TOKENIZER = None
CHAT_MODEL = None


# =========================
# Utilities
# =========================
def tokenize_for_bm25(text: str) -> List[str]:
    text = text.lower()
    return re.findall(r"[a-zA-Z0-9_+\-\.]+", text)


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", " ").replace("\ufeff", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalise_scores(score_dict: Dict[Any, float]) -> Dict[Any, float]:
    if not score_dict:
        return {}
    vals = list(score_dict.values())
    vmin, vmax = min(vals), max(vals)
    if abs(vmax - vmin) < 1e-12:
        return {k: 1.0 for k in score_dict}
    return {k: (v - vmin) / (vmax - vmin) for k, v in score_dict.items()}


def get_local_embed_model() -> SentenceTransformer:
    global EMBED_MODEL
    if EMBED_MODEL is None:
        print(f"Loading local embedding model: {LOCAL_EMBED_MODEL}")
        EMBED_MODEL = SentenceTransformer(LOCAL_EMBED_MODEL)
    return EMBED_MODEL


def get_local_chat_model():
    global CHAT_TOKENIZER, CHAT_MODEL
    if CHAT_TOKENIZER is None or CHAT_MODEL is None:
        print(f"Loading local chat model: {LOCAL_CHAT_MODEL}")
        CHAT_TOKENIZER = AutoTokenizer.from_pretrained(LOCAL_CHAT_MODEL)
        CHAT_MODEL = AutoModelForCausalLM.from_pretrained(
            LOCAL_CHAT_MODEL,
            dtype="auto",
            device_map="auto"
        )
    return CHAT_TOKENIZER, CHAT_MODEL


def get_query_embedding(query: str) -> np.ndarray:
    model = get_local_embed_model()
    vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    ).astype("float32")
    return vec


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_int(x):
    try:
        return int(x)
    except Exception:
        return None


# =========================
# Artifact loading
# =========================
def load_artifacts():
    required = [
        FAISS_INDEX_PATH, CHUNKS_PATH, BM25_PATH,
        GRAPHML_PATH, ENTITY_TO_CHUNKS_PATH, NODE_TEXTS_PATH, NODE_FAISS_PATH
    ]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"Missing required artifact: {p}")

    chunk_index = faiss.read_index(str(FAISS_INDEX_PATH))

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    with open(BM25_PATH, "rb") as f:
        bm25_data = pickle.load(f)
    bm25 = bm25_data["bm25"]

    graph = nx.read_graphml(str(GRAPHML_PATH))
    entity_to_chunks = load_json(ENTITY_TO_CHUNKS_PATH)
    node_texts = load_json(NODE_TEXTS_PATH)
    node_index = faiss.read_index(str(NODE_FAISS_PATH))

    node_names = list(node_texts.keys())
    if len(node_names) != node_index.ntotal:
        raise ValueError(
            f"Node order mismatch: node_names={len(node_names)} vs node_index={node_index.ntotal}"
        )

    return chunk_index, chunks, bm25, graph, entity_to_chunks, node_texts, node_names, node_index


# =========================
# Retrieval primitives
# =========================
def dense_search(index, chunks: List[Dict], query: str, top_k: int = 10):
    qvec = get_query_embedding(query)
    scores, indices = index.search(qvec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        results.append((float(score), chunks[idx]))
    return results


def bm25_search(bm25, chunks: List[Dict], query: str, top_k: int = 10):
    q_tokens = tokenize_for_bm25(query)
    scores = bm25.get_scores(q_tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append((float(scores[idx]), chunks[int(idx)]))
    return results


def dense_node_search(node_index, node_names: List[str], query: str, top_k: int = 6):
    qvec = get_query_embedding(query)
    scores, indices = node_index.search(qvec, top_k)

    out = {}
    for idx, score in zip(indices[0], scores[0]):
        if idx < 0:
            continue
        if idx >= len(node_names):
            continue
        out[node_names[idx]] = float(score)
    return out


def exact_node_matches(query: str, node_names: List[str], max_hits: int = 6):
    q = query.lower()
    q_tokens = set(tokenize_for_bm25(q))
    hits = {}

    for name in node_names:
        nl = name.lower()
        if nl in q or q in nl:
            hits[name] = 1.0
            continue

        n_tokens = set(tokenize_for_bm25(nl))
        overlap = len(q_tokens & n_tokens)
        if overlap >= 2:
            hits[name] = min(0.95, overlap / max(2, len(n_tokens)))

    items = sorted(hits.items(), key=lambda x: x[1], reverse=True)[:max_hits]
    return dict(items)


def expand_graph_nodes(G: nx.Graph, seed_nodes: Dict[str, float], neighbours_per_node: int = 3):
    expanded = dict(seed_nodes)
    parent_map = {}

    for node, base_score in seed_nodes.items():
        parent_map[node] = [node]
        if node not in G:
            continue

        neighs = []
        for nb in G.neighbors(node):
            w = float(G[node][nb].get("weight", 1.0))
            neighs.append((nb, w))

        neighs.sort(key=lambda x: x[1], reverse=True)
        for nb, w in neighs[:neighbours_per_node]:
            bonus = 0.15 + 0.05 * min(w, 5.0)
            new_score = base_score * bonus
            if new_score > expanded.get(nb, 0.0):
                expanded[nb] = new_score
                parent_map[nb] = [node, nb]

    return expanded, parent_map


# =========================
# Retrieval modes
# =========================
def rrf_merge(
    dense_results: List[Tuple[float, Dict]],
    bm25_results: List[Tuple[float, Dict]]
) -> Dict[int, Dict[str, Any]]:
    merged = {}

    for rank, (score, chunk) in enumerate(dense_results, start=1):
        cid = int(chunk["chunk_id"])
        merged[cid] = {
            "rrf_score": 1.0 / (60 + rank),
            "dense_score": score,
            "bm25_score": None,
            "graph_bonus": 0.0,
            "chunk": chunk,
            "matched_nodes": [],
            "community_id": None,
            "path_nodes": []
        }

    for rank, (score, chunk) in enumerate(bm25_results, start=1):
        cid = int(chunk["chunk_id"])
        if cid not in merged:
            merged[cid] = {
                "rrf_score": 1.0 / (60 + rank),
                "dense_score": None,
                "bm25_score": score,
                "graph_bonus": 0.0,
                "chunk": chunk,
                "matched_nodes": [],
                "community_id": None,
                "path_nodes": []
            }
        else:
            merged[cid]["rrf_score"] += 1.0 / (60 + rank)
            merged[cid]["bm25_score"] = score

    return merged


def finalise_ranked_results(ranked: List[Dict[str, Any]], final_k: int = 6):
    final_results = []
    per_doc_count = {}
    seen = set()

    for item in ranked:
        chunk = item["chunk"]
        source = chunk["source"]
        key = (source, chunk.get("chunk_index"))

        if key in seen:
            continue
        seen.add(key)

        per_doc_count.setdefault(source, 0)
        if per_doc_count[source] >= 3:
            continue

        per_doc_count[source] += 1
        final_results.append(item)

        if len(final_results) >= final_k:
            break

    return final_results


def vector_only_search(chunk_index, chunks: List[Dict], query: str, final_k: int = 6):
    dense_results = dense_search(chunk_index, chunks, query, top_k=max(12, final_k * 2))
    merged = {}
    for rank, (score, chunk) in enumerate(dense_results, start=1):
        cid = int(chunk["chunk_id"])
        merged[cid] = {
            "score": float(score),
            "rrf_score": 1.0 / (60 + rank),
            "dense_score": float(score),
            "bm25_score": None,
            "graph_bonus": 0.0,
            "chunk": chunk,
            "matched_nodes": [],
            "community_id": None,
            "path_nodes": []
        }
    ranked = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
    return finalise_ranked_results(ranked, final_k=final_k), {}, {}


def bm25_only_search(bm25, chunks: List[Dict], query: str, final_k: int = 6):
    sparse_results = bm25_search(bm25, chunks, query, top_k=max(12, final_k * 2))
    merged = {}
    for rank, (score, chunk) in enumerate(sparse_results, start=1):
        cid = int(chunk["chunk_id"])
        merged[cid] = {
            "score": float(score),
            "rrf_score": 1.0 / (60 + rank),
            "dense_score": None,
            "bm25_score": float(score),
            "graph_bonus": 0.0,
            "chunk": chunk,
            "matched_nodes": [],
            "community_id": None,
            "path_nodes": []
        }
    ranked = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
    return finalise_ranked_results(ranked, final_k=final_k), {}, {}


def hybrid_no_graph_search(chunk_index, bm25, chunks: List[Dict], query: str, final_k: int = 6):
    dense_results = dense_search(chunk_index, chunks, query, top_k=12)
    bm25_results = bm25_search(bm25, chunks, query, top_k=12)

    merged = rrf_merge(dense_results, bm25_results)
    for cid, item in merged.items():
        item["score"] = item["rrf_score"]

    ranked = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
    return finalise_ranked_results(ranked, final_k=final_k), {}, {}


def hybrid_graphrag_search(
    chunk_index,
    bm25,
    chunks: List[Dict],
    graph,
    entity_to_chunks,
    node_names,
    node_index,
    query: str,
    top_k_dense: int = 12,
    top_k_bm25: int = 12,
    top_k_nodes: int = 6,
    final_k: int = 6
):
    dense_results = dense_search(chunk_index, chunks, query, top_k=top_k_dense)
    bm25_results = bm25_search(bm25, chunks, query, top_k=top_k_bm25)
    merged = rrf_merge(dense_results, bm25_results)

    dense_node_scores = dense_node_search(node_index, node_names, query, top_k=top_k_nodes * 2)
    exact_node_scores = exact_node_matches(query, node_names, max_hits=top_k_nodes * 2)

    seed_nodes = defaultdict(float)
    for n, s in dense_node_scores.items():
        seed_nodes[n] += 0.7 * s
    for n, s in exact_node_scores.items():
        seed_nodes[n] += 1.0 * s

    seed_nodes = dict(sorted(seed_nodes.items(), key=lambda x: x[1], reverse=True)[:top_k_nodes])
    expanded_nodes, parent_map = expand_graph_nodes(graph, seed_nodes, neighbours_per_node=3)

    graph_chunk_bonus = defaultdict(float)
    chunk_nodes_map = defaultdict(list)
    chunk_path_map = {}

    for node, s in expanded_nodes.items():
        cids = entity_to_chunks.get(node, [])
        for cid in cids:
            icid = int(cid)
            graph_chunk_bonus[icid] += s
            chunk_nodes_map[icid].append(node)
            if icid not in chunk_path_map:
                chunk_path_map[icid] = parent_map.get(node, [node])

    graph_chunk_bonus = normalise_scores(graph_chunk_bonus)

    for cid, bonus in graph_chunk_bonus.items():
        if cid not in merged:
            chunk = chunks[cid]
            merged[cid] = {
                "rrf_score": 0.0,
                "dense_score": None,
                "bm25_score": None,
                "graph_bonus": bonus,
                "chunk": chunk,
                "matched_nodes": chunk_nodes_map.get(cid, []),
                "community_id": None,
                "path_nodes": chunk_path_map.get(cid, [])
            }
        else:
            merged[cid]["graph_bonus"] = bonus
            merged[cid]["matched_nodes"] = chunk_nodes_map.get(cid, [])
            merged[cid]["path_nodes"] = chunk_path_map.get(cid, [])

    for cid, item in merged.items():
        matched_nodes = item.get("matched_nodes", [])
        if matched_nodes:
            item["community_id"] = matched_nodes[0]
        item["score"] = 0.8 * item["rrf_score"] + 0.2 * item["graph_bonus"]

    ranked = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
    return finalise_ranked_results(ranked, final_k=final_k), seed_nodes, expanded_nodes


def retrieve(
    retrieval_mode: str,
    chunk_index,
    bm25,
    chunks,
    graph,
    entity_to_chunks,
    node_names,
    node_index,
    query: str,
    final_k: int
):
    retrieval_mode = retrieval_mode.strip().lower()

    if retrieval_mode == "vector_only":
        return vector_only_search(chunk_index, chunks, query, final_k=final_k)

    if retrieval_mode == "bm25_only":
        return bm25_only_search(bm25, chunks, query, final_k=final_k)

    if retrieval_mode == "hybrid_no_graph":
        return hybrid_no_graph_search(chunk_index, bm25, chunks, query, final_k=final_k)

    if retrieval_mode == "graph_rag":
        return hybrid_graphrag_search(
            chunk_index=chunk_index,
            bm25=bm25,
            chunks=chunks,
            graph=graph,
            entity_to_chunks=entity_to_chunks,
            node_names=node_names,
            node_index=node_index,
            query=query,
            top_k_dense=max(12, final_k * 2),
            top_k_bm25=max(12, final_k * 2),
            top_k_nodes=6,
            final_k=final_k
        )

    raise ValueError(f"Unknown retrieval_mode: {retrieval_mode}")


# =========================
# Prompt building
# =========================
def build_context(results: List[Dict], max_chars_per_chunk: int = 1600) -> str:
    blocks = []
    for i, item in enumerate(results, start=1):
        chunk = item["chunk"]
        text = chunk["text"][:max_chars_per_chunk]
        block = (
            f"[Context {i}]\n"
            f"Title: {chunk.get('title', 'N/A')}\n"
            f"Source: {chunk.get('source', 'N/A')}\n"
            f"Type: {chunk.get('type', 'N/A')}\n"
            f"Ingest method: {chunk.get('ingest_method', 'N/A')}\n"
            f"Chunk index: {chunk.get('chunk_index', 'N/A')}\n"
            f"Text:\n{text}\n"
        )
        blocks.append(block)

    return "\n" + ("\n" + "=" * 80 + "\n").join(blocks)


def build_messages(query: str, results: List[Dict], chat_history: List[Dict]) -> List[Dict]:
    context = build_context(results)

    system_prompt = (
        "You are a careful research assistant for a local GraphRAG system. "
        "Use the retrieved context as the main evidence base. "
        "If the context is insufficient, say so clearly. "
        "Do not fabricate citations. "
        "When making factual claims from context, cite them in square brackets "
        "using the title or source shown in the context. "
        "Prefer concise, technically accurate answers."
    )

    messages = [{"role": "system", "content": system_prompt}]

    if chat_history:
        messages.extend(chat_history[-8:])

    user_prompt = (
        f"User question:\n{query}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Answer the question using the retrieved context. "
        "If multiple contexts contribute, synthesise them clearly."
    )

    messages.append({"role": "user", "content": user_prompt})
    return messages


# =========================
# Local generation
# =========================
def generate_answer_local(query: str, results: List[Dict], chat_history: List[Dict]) -> str:
    tokenizer, model = get_local_chat_model()
    messages = build_messages(query, results, chat_history)

    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        prompt_parts = []
        for m in messages:
            prompt_parts.append(f"{m['role'].upper()}:\n{m['content']}")
        prompt_parts.append("ASSISTANT:\n")
        prompt = "\n\n".join(prompt_parts)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=220,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = output[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()

    if not answer:
        return "[ERROR] Local model returned empty text"
    return answer


# =========================
# JSON adaptation for evaluate.py
# =========================
def serialise_retrieved(results: List[Dict]) -> List[Dict[str, Any]]:
    out = []
    for item in results:
        chunk = item["chunk"]
        matched_nodes = item.get("matched_nodes", [])
        out.append({
            "chunk_id": safe_int(chunk.get("chunk_id")),
            "source": chunk.get("source", ""),
            "text": clean_text(chunk.get("text", "")),
            "score": float(item.get("score", 0.0)),
            "node_id": matched_nodes[0] if matched_nodes else None,
            "community_id": item.get("community_id"),
            "path_nodes": item.get("path_nodes", []),
            "metadata": {
                "title": chunk.get("title"),
                "type": chunk.get("type"),
                "ingest_method": chunk.get("ingest_method"),
                "chunk_index": chunk.get("chunk_index"),
                "dense_score": item.get("dense_score"),
                "bm25_score": item.get("bm25_score"),
                "graph_bonus": item.get("graph_bonus"),
                "matched_nodes": matched_nodes,
            }
        })
    return out


# =========================
# Display
# =========================
def print_retrieval(results: List[Dict], seed_nodes=None, expanded_nodes=None):
    print("\nTop retrieved chunks:\n")
    for i, item in enumerate(results, start=1):
        chunk = item["chunk"]
        preview = chunk["text"][:220].replace("\n", " ")
        print(f"[{i}] score={item['score']:.4f} | graph_bonus={item.get('graph_bonus', 0.0):.4f}")
        print(f"    title  : {chunk.get('title', 'N/A')}")
        print(f"    source : {chunk.get('source', 'N/A')}")
        print(f"    type   : {chunk.get('type', 'N/A')}")
        print(f"    ingest : {chunk.get('ingest_method', 'N/A')}")
        print(f"    chunk  : {chunk.get('chunk_index', 'N/A')}")
        if item.get("matched_nodes"):
            print(f"    nodes  : {', '.join(item['matched_nodes'][:6])}")
        if item.get("path_nodes"):
            print(f"    path   : {' -> '.join(item['path_nodes'])}")
        print(f"    preview: {preview}...")
        print()

    if seed_nodes:
        print("Top seed graph nodes:")
        for n, s in list(seed_nodes.items())[:8]:
            print(f"    {n}: {s:.4f}")
        print()

    if expanded_nodes:
        print("Expanded graph nodes:")
        for n, s in list(sorted(expanded_nodes.items(), key=lambda x: x[1], reverse=True))[:12]:
            print(f"    {n}: {s:.4f}")
        print()


# =========================
# Core single-run function
# =========================
def run_single_query(
    query: str,
    retrieval_mode: str,
    top_k: int,
    chat_history: Optional[List[Dict[str, str]]] = None,
    suppress_stdout: bool = False
) -> Dict[str, Any]:
    if chat_history is None:
        chat_history = []

    chunk_index, chunks, bm25, graph, entity_to_chunks, node_texts, node_names, node_index = load_artifacts()

    t_ret_0 = time.perf_counter()
    results, seed_nodes, expanded_nodes = retrieve(
        retrieval_mode=retrieval_mode,
        chunk_index=chunk_index,
        bm25=bm25,
        chunks=chunks,
        graph=graph,
        entity_to_chunks=entity_to_chunks,
        node_names=node_names,
        node_index=node_index,
        query=query,
        final_k=top_k
    )
    retrieval_latency_s = time.perf_counter() - t_ret_0

    if not suppress_stdout:
        print_retrieval(results, seed_nodes=seed_nodes, expanded_nodes=expanded_nodes)

    t_gen_0 = time.perf_counter()
    answer = generate_answer_local(query, results, chat_history)
    generation_latency_s = time.perf_counter() - t_gen_0

    payload = {
        "answer": answer,
        "retrieval_mode": retrieval_mode,
        "top_k": top_k,
        "retrieval_latency_s": retrieval_latency_s,
        "generation_latency_s": generation_latency_s,
        "seed_nodes": seed_nodes,
        "expanded_nodes": expanded_nodes,
        "retrieved": serialise_retrieved(results)
    }
    return payload


# =========================
# CLI
# =========================
def build_argparser():
    parser = argparse.ArgumentParser(description="Local GraphRAG chat for RAG_pixel")
    parser.add_argument("--query", type=str, default=None, help="Single query mode")
    parser.add_argument(
        "--retrieval_mode",
        type=str,
        default="graph_rag",
        choices=["graph_rag", "vector_only", "bm25_only", "hybrid_no_graph"],
        help="Retrieval mode"
    )
    parser.add_argument("--top_k", type=int, default=6, help="Final number of retrieved chunks")
    parser.add_argument("--json", action="store_true", help="Return JSON for evaluator")
    parser.add_argument(
        "--eval_config_json",
        type=str,
        default=None,
        help="Optional extra config JSON string from evaluate.py"
    )
    return parser


# =========================
# Main
# =========================
def interactive_loop():
    print("Loading GraphRAG artifacts...")
    _ = load_artifacts()
    print("Artifacts loaded.")
    print(f"Using local chat model: {LOCAL_CHAT_MODEL}")
    print("Type your question. Type 'exit' or 'quit' to stop.\n")

    chat_history = []

    while True:
        try:
            query = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not query:
            continue

        if query.lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        try:
            payload = run_single_query(
                query=query,
                retrieval_mode="graph_rag",
                top_k=6,
                chat_history=chat_history,
                suppress_stdout=False
            )

            print("Assistant>\n")
            print(payload["answer"])
            print()

            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": payload["answer"]})

        except Exception:
            print("[ERROR] Full traceback below:")
            traceback.print_exc()


def main():
    parser = build_argparser()
    args = parser.parse_args()

    if args.query is None:
        interactive_loop()
        return

    try:
        extra_cfg = {}
        if args.eval_config_json:
            try:
                extra_cfg = json.loads(args.eval_config_json)
            except Exception:
                extra_cfg = {}

        top_k = int(extra_cfg.get("top_k", args.top_k))
        retrieval_mode = extra_cfg.get("retrieval_mode", args.retrieval_mode)

        payload = run_single_query(
            query=args.query,
            retrieval_mode=retrieval_mode,
            top_k=top_k,
            chat_history=[],
            suppress_stdout=args.json
        )

        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print("\nAssistant>\n")
            print(payload["answer"])

    except Exception as e:
        if args.json:
            err = {
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            print(json.dumps(err, ensure_ascii=False, indent=2))
        else:
            print("[ERROR] Full traceback below:")
            traceback.print_exc()


if __name__ == "__main__":
    main()
