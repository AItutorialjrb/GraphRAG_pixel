#!/usr/bin/env python3
"""
Post-process GraphRAG entity graph without modifying build_graphrag.py.

Input:
  data/graph/graph.graphml
  data/graph/graph.json
  data/graph/entity_to_chunks.json

Output:
  data/graph_clean/graph_clean.graphml
  data/graph_clean/graph_clean.json
  data/graph_clean/entity_to_chunks_clean.json
  data/graph_clean/entity_alias_map.json
  data/graph_clean/removed_entities.json
"""

from pathlib import Path
import json
import re
from collections import defaultdict

import networkx as nx


IN_DIR = Path("data/graph")
OUT_DIR = Path("data/graph_clean")

GRAPHML_IN = IN_DIR / "graph.graphml"
GRAPH_JSON_IN = IN_DIR / "graph.json"
ENTITY_CHUNKS_IN = IN_DIR / "entity_to_chunks.json"


STOP_ENTITIES = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "from",
    "with", "without", "by", "as", "at", "is", "are", "was", "were", "be",
    "been", "being", "this", "that", "these", "those", "we", "our", "they",
    "it", "its", "their", "there", "here", "following", "corresponding",
    "thus", "therefore", "however", "also", "using", "used", "shown",
    "figure", "fig", "table", "section", "abstract", "introduction",
    "conclusion", "references", "appendix", "supplementary",
    "university", "institute", "collaboration", "author", "authors",
    "april", "january", "february", "march", "may", "june", "july",
    "august", "september", "october", "november", "december",
    "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x",
}


DOMAIN_KEEP = {
    "hvmaps", "hv-maps", "hv maps", "hvcmos", "hv-cmos", "hv cmos",
    "dmaps", "maps", "cmos", "lgad", "ti-lgad", "ac-lgad", "dc-lgad",
    "timepix", "timepix3", "mightypix", "mighty pix", "atlas", "cms",
    "lhcb", "alice", "itk", "hgtd", "velo", "sensor", "sensors",
    "pixel", "pixels", "detector", "detectors", "tracking", "tracker",
    "timing", "resolution", "efficiency", "threshold", "depletion",
    "radiation", "irradiation", "fluence", "charge", "noise",
    "monolithic", "active pixel sensor", "monolithic active pixel sensor",
    "monolithic active pixel sensors",
    "depleted monolithic active pixel sensors",
    "high voltage monolithic active pixel sensors",
    "high-voltage monolithic active pixel sensors",
}


CANONICAL_ALIASES = {
    "hv maps": "HVMAPS",
    "hv-maps": "HVMAPS",
    "hvmaps": "HVMAPS",
    "high voltage monolithic active pixel sensors": "HVMAPS",
    "high-voltage monolithic active pixel sensors": "HVMAPS",
    "high voltage monolithic active pixel sensor": "HVMAPS",
    "high-voltage monolithic active pixel sensor": "HVMAPS",

    "hv cmos": "HV-CMOS",
    "hvcmos": "HV-CMOS",
    "hv-cmos": "HV-CMOS",

    "depleted monolithic active pixel sensors": "DMAPS",
    "depleted monolithic active pixel sensor": "DMAPS",
    "dmaps": "DMAPS",

    "monolithic active pixel sensors": "MAPS",
    "monolithic active pixel sensor": "MAPS",
    "maps": "MAPS",

    "low gain avalanche detector": "LGAD",
    "low-gain avalanche detector": "LGAD",
    "low gain avalanche detectors": "LGAD",
    "low-gain avalanche detectors": "LGAD",
    "lgads": "LGAD",
    "lgad": "LGAD",

    "timepix 3": "Timepix3",
    "timepix3": "Timepix3",
    "timepix": "Timepix",

    "mighty pix": "MightyPix",
    "mightypix": "MightyPix",
    "mightypix1": "MightyPix1",
    "mightypix2": "MightyPix2",
}


def normalise_space(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"[_]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def norm_key(s: str) -> str:
    s = normalise_space(s).lower()
    s = s.strip(" -–—_:;,.()[]{}")
    s = re.sub(r"\s+", " ", s)
    return s


def compact_key(s: str) -> str:
    s = norm_key(s)
    s = s.replace("-", "")
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def canonicalise(entity: str) -> str:
    e = normalise_space(entity)
    k = norm_key(e)

    if k in CANONICAL_ALIASES:
        return CANONICAL_ALIASES[k]

    ck = compact_key(k)
    for alias, canonical in CANONICAL_ALIASES.items():
        if compact_key(alias) == ck:
            return canonical

    # Remove trailing punctuation/dashes
    e = e.strip(" -–—_:;,.")
    return e


def is_bad_entity(entity: str) -> bool:
    raw = str(entity)
    e = normalise_space(raw)
    k = norm_key(e)

    if not e:
        return True

    # Keep important detector terms even if short
    if k in DOMAIN_KEEP or compact_key(k) in {compact_key(x) for x in DOMAIN_KEEP}:
        return False

    if k in STOP_ENTITIES:
        return True

    if len(e) < 3:
        return True

    if len(e) > 80:
        return True

    if re.fullmatch(r"[0-9]+", k):
        return True

    if re.fullmatch(r"[0-9]+(\.[0-9]+)+", k):
        return True

    if re.match(r"^(http|www|doi|arxiv)", k):
        return True

    # Mostly digits
    n_digit = sum(ch.isdigit() for ch in e)
    if n_digit / max(len(e), 1) > 0.35:
        return True

    # Not enough letters
    n_alpha = sum(ch.isalpha() for ch in e)
    if n_alpha < 3:
        return True

    # Equation fragments or units only
    if re.fullmatch(r"[a-z]?[0-9\.\+\-\*/\^\=\(\)\[\]\{\}\s]+", k):
        return True

    # Very generic one-word capitalised junk often produced by PDF extraction
    generic_bad = {
        "result", "results", "method", "methods", "data", "analysis",
        "study", "studies", "work", "paper", "model", "models",
        "simulation", "simulations", "measurement", "measurements",
        "test", "tests", "sample", "samples", "case", "cases",
        "value", "values", "parameter", "parameters",
    }

    if k in generic_bad:
        return True

    return False


def merge_node_attr(old_attr, new_attr):
    out = dict(old_attr or {})
    for k, v in (new_attr or {}).items():
        if k not in out or out[k] in ("", None):
            out[k] = v
    return out


def load_entity_to_chunks(path: Path):
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading graph: {GRAPHML_IN}")
    G = nx.read_graphml(GRAPHML_IN)
    print(f"Original graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    entity_to_chunks = load_entity_to_chunks(ENTITY_CHUNKS_IN)

    alias_map = {}
    removed = {}
    clean_entities = {}

    for n in G.nodes():
        if is_bad_entity(n):
            removed[str(n)] = "bad_label"
            continue

        c = canonicalise(n)

        if is_bad_entity(c):
            removed[str(n)] = "bad_after_canonicalisation"
            continue

        alias_map[str(n)] = c
        clean_entities[str(n)] = c

    H = nx.Graph()

    # Add nodes
    for old, new in clean_entities.items():
        old_attr = dict(G.nodes[old])
        old_attr["original_labels"] = old

        if H.has_node(new):
            attr = merge_node_attr(H.nodes[new], old_attr)
            originals = set(str(attr.get("original_labels", "")).split(" || "))
            originals.add(old)
            attr["original_labels"] = " || ".join(sorted(x for x in originals if x))
            H.nodes[new].update(attr)
        else:
            H.add_node(new, **old_attr)

    # Add and merge edges
    skipped_edges = 0
    for u, v, d in G.edges(data=True):
        if u not in clean_entities or v not in clean_entities:
            skipped_edges += 1
            continue

        cu = clean_entities[u]
        cv = clean_entities[v]

        if cu == cv:
            continue

        try:
            w = float(d.get("weight", 1.0))
        except Exception:
            w = 1.0

        if H.has_edge(cu, cv):
            H[cu][cv]["weight"] = float(H[cu][cv].get("weight", 1.0)) + w
        else:
            nd = dict(d)
            nd["weight"] = w
            H.add_edge(cu, cv, **nd)

    # Remove low-degree weak junk unless domain-relevant
    to_remove = []
    for n in H.nodes():
        k = norm_key(n)
        ck = compact_key(n)
        domain = k in DOMAIN_KEEP or ck in {compact_key(x) for x in DOMAIN_KEEP}
        if not domain and H.degree(n) < 2:
            to_remove.append(n)

    H.remove_nodes_from(to_remove)

    # Rebuild clean entity_to_chunks
    clean_entity_to_chunks = defaultdict(set)

    for old_entity, chunks in entity_to_chunks.items():
        if old_entity not in clean_entities:
            continue

        new_entity = clean_entities[old_entity]
        if new_entity not in H:
            continue

        for ch in chunks:
            clean_entity_to_chunks[new_entity].add(ch)

    clean_entity_to_chunks = {
        k: sorted(v)
        for k, v in clean_entity_to_chunks.items()
        if k in H
    }

    print(f"Clean graph: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges")
    print(f"Removed entities: {len(removed)}")
    print(f"Skipped edges: {skipped_edges}")
    print(f"Removed low-degree nodes: {len(to_remove)}")

    graphml_out = OUT_DIR / "graph_clean.graphml"
    graph_json_out = OUT_DIR / "graph_clean.json"
    entity_chunks_out = OUT_DIR / "entity_to_chunks_clean.json"
    alias_out = OUT_DIR / "entity_alias_map.json"
    removed_out = OUT_DIR / "removed_entities.json"

    nx.write_graphml(H, graphml_out)

    data = {
        "nodes": [
            {
                "id": n,
                **{k: str(v) for k, v in H.nodes[n].items()},
            }
            for n in H.nodes()
        ],
        "edges": [
            {
                "source": u,
                "target": v,
                **{k: str(vv) for k, vv in d.items()},
            }
            for u, v, d in H.edges(data=True)
        ],
    }

    with open(graph_json_out, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    with open(entity_chunks_out, "w") as f:
        json.dump(clean_entity_to_chunks, f, indent=2, ensure_ascii=False)

    with open(alias_out, "w") as f:
        json.dump(alias_map, f, indent=2, ensure_ascii=False)

    with open(removed_out, "w") as f:
        json.dump(removed, f, indent=2, ensure_ascii=False)

    print(f"Saved: {graphml_out}")
    print(f"Saved: {graph_json_out}")
    print(f"Saved: {entity_chunks_out}")
    print(f"Saved: {alias_out}")
    print(f"Saved: {removed_out}")


if __name__ == "__main__":
    main()
