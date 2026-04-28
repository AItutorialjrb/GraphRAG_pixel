import os
import re
import json
import math
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Set
from collections import Counter, defaultdict

import faiss
import numpy as np
import networkx as nx
from tqdm import tqdm
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


# =========================
# Config
# =========================
BASE_DIR = Path("/eos/home-r/rjiang/RAG_pixel")
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

LOCAL_EMBED_MODEL = os.getenv(
    "LOCAL_EMBED_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

OCR_DIR = BASE_DIR / "main_markdown"
PDF_DIR = BASE_DIR / "main_pdf"

DATA_DIR = BASE_DIR / "data"
INDEX_DIR = DATA_DIR / "index"
GRAPH_DIR = DATA_DIR / "graph"

INDEX_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# Existing RAG artifacts
FAISS_INDEX_PATH = INDEX_DIR / "dense.index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
BM25_PATH = INDEX_DIR / "bm25.pkl"
BUILD_REPORT_PATH = INDEX_DIR / "build_report.json"

# GraphRAG artifacts
GRAPHML_PATH = GRAPH_DIR / "graph.graphml"
GRAPH_JSON_PATH = GRAPH_DIR / "graph.json"
ENTITY_TO_CHUNKS_PATH = GRAPH_DIR / "entity_to_chunks.json"
NODE_TEXTS_PATH = GRAPH_DIR / "node_texts.json"
NODE_EMB_PATH = GRAPH_DIR / "node_embeddings.npy"
NODE_FAISS_PATH = GRAPH_DIR / "node_dense.index"
GRAPH_REPORT_PATH = GRAPH_DIR / "graph_report.json"

CHUNK_SIZE = 2500
CHUNK_OVERLAP = 250
EMBED_BATCH_SIZE = 64

# Entity extraction
MIN_ENTITY_LEN = 2
MAX_ENTITY_WORDS = 5
MIN_ENTITY_FREQ = 2
MAX_ENTITIES_PER_CHUNK = 25


# =========================
# Utilities
# =========================
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = text.replace("\ufeff", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalise_stem(name: str) -> str:
    stem = Path(name).stem.lower()
    stem = re.sub(r"[^a-z0-9]+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_")
    return stem


def extract_text_from_pdf(pdf_path: Path) -> str:
    try:
        reader = PdfReader(str(pdf_path))
        pages = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                pages.append(t)
        return clean_text("\n".join(pages))
    except Exception as e:
        print(f"[WARN] Failed to read PDF {pdf_path.name}: {e}")
        return ""


def load_ocr_docs() -> Tuple[List[Dict], Set[str]]:
    docs = []
    used_stems = set()

    if not OCR_DIR.exists():
        print(f"[WARN] OCR directory does not exist: {OCR_DIR}")
        return docs, used_stems

    ocr_files = list(OCR_DIR.rglob("*.md")) + list(OCR_DIR.rglob("*.txt"))
    ocr_files = sorted(set(ocr_files))

    print(f"Found {len(ocr_files)} OCR text files in {OCR_DIR}")

    for path in ocr_files:
        try:
            txt = clean_text(path.read_text(encoding="utf-8", errors="ignore"))
            if len(txt) < 50:
                print(f"[WARN] OCR text too short, skipped: {path.name}")
                continue

            stem = normalise_stem(path.name)
            used_stems.add(stem)

            docs.append({
                "source": str(path),
                "title": path.stem,
                "text": txt,
                "type": path.suffix.lstrip("."),
                "ingest_method": "ocr_text"
            })
        except Exception as e:
            print(f"[WARN] Failed to read OCR file {path}: {e}")

    return docs, used_stems


def load_fallback_pdfs(already_used_stems: Set[str]) -> List[Dict]:
    docs = []

    if not PDF_DIR.exists():
        print(f"[WARN] PDF directory does not exist: {PDF_DIR}")
        return docs

    pdf_files = sorted(PDF_DIR.rglob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs in {PDF_DIR}")

    for path in pdf_files:
        stem = normalise_stem(path.name)
        if stem in already_used_stems:
            continue

        txt = extract_text_from_pdf(path)
        if len(txt) < 50:
            print(f"[WARN] PDF text too short, skipped: {path.name}")
            continue

        docs.append({
            "source": str(path),
            "title": path.stem,
            "text": txt,
            "type": "pdf",
            "ingest_method": "pdf_fallback"
        })

    return docs


def deduplicate_docs(docs: List[Dict]) -> List[Dict]:
    unique_docs = []
    seen = set()

    for d in docs:
        key = (d["title"], len(d["text"]), d.get("ingest_method", "unknown"))
        if key in seen:
            continue
        seen.add(key)
        unique_docs.append(d)

    return unique_docs


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(end - overlap, start + 1)

    return chunks


def build_chunks(docs: List[Dict]) -> List[Dict]:
    all_chunks = []

    for doc_id, doc in enumerate(docs):
        chunks = chunk_text(doc["text"])
        for i, c in enumerate(chunks):
            all_chunks.append({
                "chunk_id": len(all_chunks),
                "doc_id": doc_id,
                "title": doc["title"],
                "source": doc["source"],
                "type": doc["type"],
                "chunk_index": i,
                "ingest_method": doc.get("ingest_method", "unknown"),
                "text": c
            })

    return all_chunks


def tokenize_for_bm25(text: str) -> List[str]:
    text = text.lower()
    return re.findall(r"[a-zA-Z0-9_+\-\.]+", text)


def get_local_embeddings(
    texts: List[str],
    model_name: str = LOCAL_EMBED_MODEL,
    batch_size: int = EMBED_BATCH_SIZE
) -> np.ndarray:
    print(f"Loading local embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    vectors = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i + batch_size]
        batch_vecs = model.encode(
            batch,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        vectors.append(batch_vecs)

    return np.vstack(vectors).astype("float32")


# =========================
# Lightweight entity extraction
# =========================
PHYSICS_STOPWORDS = {
    "introduction", "conclusion", "results", "discussion", "references",
    "figure", "table", "section", "equation", "analysis", "measurement",
    "study", "paper", "event", "events", "data", "using", "used",
    "from", "with", "into", "that", "this", "these", "those", "such",
    "model", "models", "method", "methods", "system", "systems",
    "detector", "detectors", "experiment", "experiments"
}


def normalise_entity(entity: str) -> str:
    entity = entity.strip()
    entity = re.sub(r"\s+", " ", entity)
    entity = entity.strip(" ,;:.()[]{}")
    return entity


def extract_candidate_entities(text: str) -> List[str]:
    candidates = []

    # Acronyms / common HEP tokens
    acronym_pattern = r"\b(?:[A-Z][A-Z0-9+\-]{1,}|[A-Z][a-zA-Z]*\d+[a-zA-Z0-9]*)\b"
    for m in re.finditer(acronym_pattern, text):
        ent = normalise_entity(m.group(0))
        if len(ent) >= MIN_ENTITY_LEN:
            candidates.append(ent)

    # Multi-word title-case phrases
    phrase_pattern = r"\b(?:[A-Z][a-z]+(?:[-/][A-Z]?[a-z]+)?(?:\s+|$)){1," + str(MAX_ENTITY_WORDS) + r"}"
    for m in re.finditer(phrase_pattern, text):
        ent = normalise_entity(m.group(0))
        words = ent.split()
        if 1 <= len(words) <= MAX_ENTITY_WORDS:
            candidates.append(ent)

    # Physics-ish tokens with Greek-like / detector naming patterns
    keyword_pattern = r"\b(?:silicon pixel|pixel detector|HV-MAPS|LGAD|Timepix3|Muon Collider|LHC|CMS|ATLAS|HL-LHC|VBS|SMEFT|EFT|aTGC|nTGC|aQGC|Zγ|WZγ|ZZγ|WWZ)\b"
    for m in re.finditer(keyword_pattern, text, flags=re.IGNORECASE):
        ent = normalise_entity(m.group(0))
        candidates.append(ent)

    cleaned = []
    for ent in candidates:
        ent_l = ent.lower()
        if ent_l in PHYSICS_STOPWORDS:
            continue
        if len(ent_l) <= 1:
            continue
        cleaned.append(ent)

    return cleaned


def filter_entities_global(chunks: List[Dict]) -> Tuple[List[List[str]], Counter]:
    global_counter = Counter()
    chunk_entities_raw = []

    print("Extracting entity candidates...")
    for chunk in tqdm(chunks, desc="Entity candidates"):
        ents = extract_candidate_entities(chunk["text"])
        uniq = list(dict.fromkeys(ents))
        chunk_entities_raw.append(uniq)
        global_counter.update([e.lower() for e in uniq])

    chunk_entities_filtered = []
    for ents in chunk_entities_raw:
        kept = []
        for e in ents:
            if global_counter[e.lower()] >= MIN_ENTITY_FREQ:
                kept.append(e)
        kept = kept[:MAX_ENTITIES_PER_CHUNK]
        chunk_entities_filtered.append(kept)

    return chunk_entities_filtered, global_counter


# =========================
# Graph build
# =========================
def build_entity_graph(chunks: List[Dict], chunk_entities: List[List[str]]) -> Tuple[nx.Graph, Dict[str, List[int]], Dict[str, str]]:
    G = nx.Graph()
    entity_to_chunks = defaultdict(list)
    canonical_map = {}

    def canonical(e: str) -> str:
        key = e.lower()
        if key not in canonical_map:
            canonical_map[key] = e
        return canonical_map[key]

    print("Building co-occurrence graph...")
    for chunk, entities in tqdm(list(zip(chunks, chunk_entities)), total=len(chunks), desc="Graph edges"):
        norm_entities = [canonical(e) for e in entities]
        norm_entities = list(dict.fromkeys(norm_entities))

        # node attributes
        for ent in norm_entities:
            if not G.has_node(ent):
                G.add_node(ent, label=ent, frequency=0)
            G.nodes[ent]["frequency"] += 1
            entity_to_chunks[ent].append(chunk["chunk_id"])

        # co-occurrence edges
        for i in range(len(norm_entities)):
            for j in range(i + 1, len(norm_entities)):
                a = norm_entities[i]
                b = norm_entities[j]
                if G.has_edge(a, b):
                    G[a][b]["weight"] += 1
                else:
                    G.add_edge(a, b, weight=1)

    # deduplicate chunk lists
    entity_to_chunks = {k: sorted(set(v)) for k, v in entity_to_chunks.items()}

    return G, entity_to_chunks, canonical_map


def build_node_texts(G: nx.Graph, entity_to_chunks: Dict[str, List[int]], chunks: List[Dict]) -> Dict[str, str]:
    chunk_lookup = {c["chunk_id"]: c for c in chunks}
    node_texts = {}

    print("Building node texts...")
    for node in tqdm(G.nodes(), desc="Node texts"):
        chunk_ids = entity_to_chunks.get(node, [])[:8]
        supporting = []
        for cid in chunk_ids:
            txt = chunk_lookup[cid]["text"]
            supporting.append(txt[:700])
        node_text = f"Entity: {node}\n\n" + "\n\n---\n\n".join(supporting)
        node_texts[node] = node_text

    return node_texts


def save_graph_json(G: nx.Graph, path: Path) -> None:
    graph_obj = {
        "nodes": [
            {"id": n, **dict(G.nodes[n])}
            for n in G.nodes()
        ],
        "edges": [
            {"source": u, "target": v, **dict(G[u][v])}
            for u, v in G.edges()
        ]
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(graph_obj, f, ensure_ascii=False, indent=2)


# =========================
# Main
# =========================
def main():
    print("Loading OCR documents first...")
    ocr_docs, used_stems = load_ocr_docs()
    print(f"Loaded {len(ocr_docs)} OCR documents.")

    print("Loading fallback PDFs...")
    pdf_docs = load_fallback_pdfs(used_stems)
    print(f"Loaded {len(pdf_docs)} fallback PDF documents.")

    docs = deduplicate_docs(ocr_docs + pdf_docs)
    print(f"Total documents for indexing: {len(docs)}")

    if not docs:
        print("No usable documents found in main_markdown/ or main_pdf/.")
        return

    n_ocr = sum(1 for d in docs if d["ingest_method"] == "ocr_text")
    n_pdf = sum(1 for d in docs if d["ingest_method"] == "pdf_fallback")
    print(f"Using {n_ocr} OCR docs and {n_pdf} PDF fallback docs.")

    print("Chunking documents...")
    chunks = build_chunks(docs)
    print(f"Built {len(chunks)} chunks.")

    if not chunks:
        print("No chunks were created.")
        return

    texts = [c["text"] for c in chunks]

    # ---- Hybrid RAG part ----
    print("Building BM25 index...")
    tokenized_corpus = [tokenize_for_bm25(t) for t in texts]
    bm25 = BM25Okapi(tokenized_corpus)

    print("Computing chunk embeddings...")
    embeddings = get_local_embeddings(texts)

    dim = embeddings.shape[1]
    print(f"Embedding dimension: {dim}")

    print("Building chunk FAISS index...")
    chunk_index = faiss.IndexFlatIP(dim)
    chunk_index.add(embeddings)

    # ---- GraphRAG part ----
    chunk_entities, global_counter = filter_entities_global(chunks)
    G, entity_to_chunks, canonical_map = build_entity_graph(chunks, chunk_entities)

    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    node_texts = build_node_texts(G, entity_to_chunks, chunks)
    node_names = list(G.nodes())
    node_corpus = [node_texts[n] for n in node_names]

    print("Computing node embeddings...")
    node_embs = get_local_embeddings(node_corpus)

    print("Building node FAISS index...")
    node_index = faiss.IndexFlatIP(node_embs.shape[1])
    node_index.add(node_embs)

    # ---- Save everything ----
    print("Saving hybrid RAG artifacts...")
    faiss.write_index(chunk_index, str(FAISS_INDEX_PATH))

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    with open(BM25_PATH, "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "tokenized_corpus": tokenized_corpus
        }, f)

    print("Saving graph artifacts...")
    nx.write_graphml(G, str(GRAPHML_PATH))
    save_graph_json(G, GRAPH_JSON_PATH)

    with open(ENTITY_TO_CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(entity_to_chunks, f, ensure_ascii=False, indent=2)

    with open(NODE_TEXTS_PATH, "w", encoding="utf-8") as f:
        json.dump(node_texts, f, ensure_ascii=False, indent=2)

    np.save(NODE_EMB_PATH, node_embs)
    faiss.write_index(node_index, str(NODE_FAISS_PATH))

    build_report = {
        "base_dir": str(BASE_DIR),
        "local_embed_model": LOCAL_EMBED_MODEL,
        "n_docs_total": len(docs),
        "n_docs_ocr": n_ocr,
        "n_docs_pdf_fallback": n_pdf,
        "n_chunks_total": len(chunks),
        "n_graph_nodes": G.number_of_nodes(),
        "n_graph_edges": G.number_of_edges(),
        "chunk_faiss_index_path": str(FAISS_INDEX_PATH),
        "chunks_path": str(CHUNKS_PATH),
        "bm25_path": str(BM25_PATH),
        "graphml_path": str(GRAPHML_PATH),
        "graph_json_path": str(GRAPH_JSON_PATH),
        "entity_to_chunks_path": str(ENTITY_TO_CHUNKS_PATH),
        "node_texts_path": str(NODE_TEXTS_PATH),
        "node_emb_path": str(NODE_EMB_PATH),
        "node_faiss_path": str(NODE_FAISS_PATH),
    }

    with open(BUILD_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(build_report, f, ensure_ascii=False, indent=2)

    with open(GRAPH_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(build_report, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(f"Chunk FAISS index: {FAISS_INDEX_PATH}")
    print(f"Chunks metadata: {CHUNKS_PATH}")
    print(f"BM25 index: {BM25_PATH}")
    print(f"GraphML: {GRAPHML_PATH}")
    print(f"Graph JSON: {GRAPH_JSON_PATH}")
    print(f"Entity-to-chunks: {ENTITY_TO_CHUNKS_PATH}")
    print(f"Node texts: {NODE_TEXTS_PATH}")
    print(f"Node embeddings: {NODE_EMB_PATH}")
    print(f"Node FAISS index: {NODE_FAISS_PATH}")


if __name__ == "__main__":
    main()
