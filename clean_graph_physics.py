#!/usr/bin/env python3
"""
Physics-aware cleaner for GraphRAG_pixel entity graph.

Input:
  data/graph/graph.graphml
  data/graph/entity_to_chunks.json

Output:
  data/graph_clean/graph_clean.graphml
  data/graph_clean/entity_to_chunks_clean.json
  data/graph_clean/entity_alias_map.json
  data/graph_clean/removed_entities.json

Usage:
  python clean_graph_physics.py
"""

from pathlib import Path
import json
import re
from collections import defaultdict

import networkx as nx


IN_GRAPHML = Path("data/graph/graph.graphml")
IN_ENTITY_CHUNKS = Path("data/graph/entity_to_chunks.json")

OUT_DIR = Path("data/graph_clean")
OUT_GRAPHML = OUT_DIR / "graph_clean.graphml"
OUT_JSON = OUT_DIR / "graph_clean.json"
OUT_ENTITY_CHUNKS = OUT_DIR / "entity_to_chunks_clean.json"
OUT_ALIAS_MAP = OUT_DIR / "entity_alias_map.json"
OUT_REMOVED = OUT_DIR / "removed_entities.json"
OUT_REPORT = OUT_DIR / "clean_report.json"


KEEP_TERMS = {
    # detector technologies
    "CMOS", "HV-CMOS", "HVCMOS", "MAPS", "DMAPS", "HVMAPS",
    "LGAD", "AC-LGAD", "DC-LGAD", "TI-LGAD",
    "SPAD", "SiPM", "MPPC",

    # electronics / readout
    "ASIC", "TCAD", "Geant4", "ADC", "DAC", "TDC", "PLL",
    "ToT", "TOT", "ToA", "TOA", "CSA", "FE", "AFE",

    # experiments / systems often meaningful in detector papers
    "LHCb", "VELO", "ATLAS", "CMS", "ALICE", "ITk", "HGTD", "ITS3",

    # named detector/readout chips
    "Timepix", "Timepix3", "MightyPix", "MightyPix1", "MightyPix2",
    "CROCv1", "CROCv2", "RD53", "RD53A", "RD53B",
}


DROP_TERMS = {
    # table/PDF extraction junk
    "Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8",
    "URL", "DOI", "arXiv", "IEEE", "JINST",
    "Number", "All", "Any", "One", "Two", "Three", "Each",
    "After", "Since", "Before", "Following", "Corresponding",
    "Figure", "Fig", "Table", "Section", "Abstract", "References",
    "Conclusion", "Introduction", "Appendix",

    # common non-concept tokens seen in your graph
    "LUT", "VOL", "Map", "High", "Low", "Run", "Active", "Charged",
    "MONOLITH", "MONOLITHIC", "Dipartimento", "Development",
    "Photomultipliers", "Silicon Photomultipliers",

    # pronouns / glue words
    "The", "A", "An", "And", "Or", "Of", "In", "On", "For", "With",
    "Without", "By", "As", "At", "This", "That", "These", "Those",
    "We", "Our", "They", "It", "Its", "Here", "There", "Thus",
}


DOMAIN_KEYWORDS = [
    "pixel", "pixels", "sensor", "sensors", "detector", "detectors",
    "silicon", "semiconductor", "monolithic", "active pixel",
    "depleted", "depletion", "substrate", "epitaxial", "epilayer",
    "electrode", "column", "bias", "voltage",

    "cmos", "hv-cmos", "hvcmos", "maps", "dmaps", "hvmaps",
    "lgad", "spad", "sipm", "asic", "tcad", "geant4",

    "readout", "frontend", "front-end", "amplifier", "threshold",
    "noise", "charge", "carrier", "drift", "diffusion", "collection",
    "efficiency", "resolution", "timing", "time resolution",
    "time-over-threshold", "time over threshold", "time-of-arrival",
    "time of arrival", "tot", "toa", "tdc", "pll", "adc", "dac",

    "radiation", "irradiation", "fluence", "dose", "tid",
    "neutron", "proton", "mrad", "neq",

    "tracking", "tracker", "vertex", "velo", "itk", "hgtd", "its3",
    "timepix", "mightypix",
]


CANONICAL_ALIASES = {
    # HV-MAPS
    "hv maps": "HVMAPS",
    "hv-maps": "HVMAPS",
    "hvmaps": "HVMAPS",
    "high voltage monolithic active pixel sensor": "HVMAPS",
    "high voltage monolithic active pixel sensors": "HVMAPS",
    "high-voltage monolithic active pixel sensor": "HVMAPS",
    "high-voltage monolithic active pixel sensors": "HVMAPS",

    # HV-CMOS
    "hv cmos": "HV-CMOS",
    "hv-cmos": "HV-CMOS",
    "hvcmos": "HV-CMOS",
    "high voltage cmos": "HV-CMOS",
    "high-voltage cmos": "HV-CMOS",

    # MAPS / DMAPS
    "map": "MAPS",
    "maps": "MAPS",
    "monolithic active pixel": "MAPS",
    "monolithic active pixel sensor": "MAPS",
    "monolithic active pixel sensors": "MAPS",
    "monolithic active pixels sensors": "MAPS",

    "dmaps": "DMAPS",
    "depleted monolithic active pixel sensor": "DMAPS",
    "depleted monolithic active pixel sensors": "DMAPS",

    # LGAD
    "lgad": "LGAD",
    "lgads": "LGAD",
    "low gain avalanche detector": "LGAD",
    "low gain avalanche detectors": "LGAD",
    "low-gain avalanche detector": "LGAD",
    "low-gain avalanche detectors": "LGAD",

    "ac lgad": "AC-LGAD",
    "ac-lgad": "AC-LGAD",
    "dc lgad": "DC-LGAD",
    "dc-lgad": "DC-LGAD",
    "ti lgad": "TI-LGAD",
    "ti-lgad": "TI-LGAD",

    # timing/readout
    "time over threshold": "ToT",
    "time-over-threshold": "ToT",
    "tot": "ToT",
    "time of arrival": "ToA",
    "time-of-arrival": "ToA",
    "toa": "ToA",

    "timepix 3": "Timepix3",
    "timepix3": "Timepix3",
    "mighty pix": "MightyPix",
    "mightypix": "MightyPix",

    # common noisy variants
    "pixels": "Pixel",
    "pixel detectors": "pixel detector",
    "pixel sensors": "pixel sensor",
    "silicon pixels": "silicon pixel",
    "time resolution": "time resolution",
    "timeresolution": "time resolution",
}


def norm_space(s: str) -> str:
    s = str(s).strip()
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def norm_lower(s: str) -> str:
    return norm_space(s).lower().strip(" -–—_:;,.()[]{}")


def compact(s: str) -> str:
    s = norm_lower(s)
    s = s.replace("-", "")
    return re.sub(r"[^a-z0-9]+", "", s)


KEEP_COMPACT = {compact(x) for x in KEEP_TERMS}
DROP_COMPACT = {compact(x) for x in DROP_TERMS}
DOMAIN_COMPACT = [compact(x) for x in DOMAIN_KEYWORDS if compact(x)]


def canonicalise(entity: str) -> str:
    e = norm_space(entity).strip(" -–—_:;,.")
    k = norm_lower(e)
    c = compact(e)

    if k in CANONICAL_ALIASES:
        return CANONICAL_ALIASES[k]

    for alias, canonical in CANONICAL_ALIASES.items():
        if compact(alias) == c:
            return canonical

    # standardise common capitalisation
    if c == "cmos":
        return "CMOS"
    if c == "asic":
        return "ASIC"
    if c == "tcad":
        return "TCAD"
    if c == "geant4":
        return "Geant4"
    if c == "spad":
        return "SPAD"
    if c == "sipm":
        return "SiPM"
    if c == "tdc":
        return "TDC"
    if c == "pll":
        return "PLL"

    return e


def looks_like_noise(entity: str) -> bool:
    e = norm_space(entity)
    k = norm_lower(e)
    c = compact(e)

    if not e:
        return True

    if c in KEEP_COMPACT:
        return False

    if c in DROP_COMPACT:
        return True

    if len(e) > 90:
        return True

    if len(c) < 3:
        return True

    if re.fullmatch(r"\d+(\.\d+)*", k):
        return True

    if re.match(r"^(http|www|doi|arxiv)", k):
        return True

    if re.fullmatch(r"col\d+", c):
        return True

    # Mostly digits
    n_digit = sum(ch.isdigit() for ch in e)
    if n_digit / max(len(e), 1) > 0.35:
        return True

    # Not enough letters
    n_alpha = sum(ch.isalpha() for ch in e)
    if n_alpha < 3:
        return True

    # Single generic adjectives/nouns that are not useful concepts
    generic_bad = {
        "high", "low", "active", "charged", "map", "run", "each", "after",
        "since", "before", "following", "corresponding", "number",
        "result", "results", "method", "methods", "data", "analysis",
        "study", "studies", "work", "paper", "model", "models",
        "measurement", "measurements", "sample", "samples",
        "case", "cases", "value", "values", "parameter", "parameters",
    }
    if k in generic_bad:
        return True

    # Very short all-caps tokens are usually acronyms.
    # Keep only if whitelist/domain-relevant.
    if e.isupper() and len(c) <= 4:
        if c not in KEEP_COMPACT:
            return True

    return False


def domain_score(entity: str) -> float:
    e = norm_space(entity)
    k = norm_lower(e)
    c = compact(e)

    score = 0.0

    if c in KEEP_COMPACT:
        score += 100.0

    if c in DROP_COMPACT:
        score -= 100.0

    for kw in DOMAIN_COMPACT:
        if kw and (kw in c or c in kw):
            score += 10.0
            break

    # Multi-word technical phrases are more likely meaningful
    words = k.split()
    if 2 <= len(words) <= 6:
        score += 3.0

    # penalise generic one-word terms unless explicitly kept
    if len(words) == 1 and c not in KEEP_COMPACT:
        score -= 2.0

    # penalise very short terms unless explicitly kept
    if len(c) <= 4 and c not in KEEP_COMPACT:
        score -= 3.0

    return score


def should_keep(entity: str) -> bool:
    c = compact(entity)

    if c in KEEP_COMPACT:
        return True

    if c in DROP_COMPACT:
        return False

    if looks_like_noise(entity):
        return False

    return domain_score(entity) > 0


def merge_attrs(old_attr, new_attr):
    out = dict(old_attr or {})
    for k, v in (new_attr or {}).items():
        if k not in out or out[k] in ("", None):
            out[k] = v
    return out


def load_entity_chunks(path: Path):
    if not path.exists():
        return {}
    with open(path) as f:
        raw = json.load(f)
    return raw


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading graph: {IN_GRAPHML}")
    G = nx.read_graphml(IN_GRAPHML)
    print(f"Original graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    entity_to_chunks = load_entity_chunks(IN_ENTITY_CHUNKS)

    alias_map = {}
    removed = {}

    # First pass: decide nodes and canonical names
    old_to_new = {}

    for n in G.nodes():
        n_str = str(n)
        canon = canonicalise(n_str)

        if should_keep(canon):
            old_to_new[n_str] = canon
            alias_map[n_str] = canon
        else:
            removed[n_str] = {
                "reason": "failed_physics_clean",
                "canonical": canon,
                "domain_score": domain_score(canon),
            }

    H = nx.Graph()

    # Add merged nodes
    for old, new in old_to_new.items():
        old_attr = dict(G.nodes[old])
        old_attr["original_labels"] = old

        if H.has_node(new):
            attr = merge_attrs(H.nodes[new], old_attr)
            originals = set(str(attr.get("original_labels", "")).split(" || "))
            originals.add(old)
            attr["original_labels"] = " || ".join(sorted(x for x in originals if x))
            H.nodes[new].update(attr)
        else:
            H.add_node(new, **old_attr)

    # Add merged weighted edges
    skipped_edges = 0
    merged_self_edges = 0

    for u, v, d in G.edges(data=True):
        if u not in old_to_new or v not in old_to_new:
            skipped_edges += 1
            continue

        cu = old_to_new[u]
        cv = old_to_new[v]

        if cu == cv:
            merged_self_edges += 1
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

    # Remove remaining isolated nodes unless explicitly important
    isolated_removed = []
    for n in list(nx.isolates(H)):
        if compact(n) not in KEEP_COMPACT:
            isolated_removed.append(n)

    H.remove_nodes_from(isolated_removed)

    # Rebuild entity_to_chunks using canonical names
    clean_e2c = defaultdict(set)

    for old_entity, chunks in entity_to_chunks.items():
        old_entity = str(old_entity)

        if old_entity in old_to_new:
            new_entity = old_to_new[old_entity]
        else:
            new_entity = canonicalise(old_entity)
            if not should_keep(new_entity):
                continue

        if new_entity not in H:
            continue

        for ch in chunks:
            clean_e2c[new_entity].add(str(ch))

    clean_e2c = {
        k: sorted(v)
        for k, v in clean_e2c.items()
        if k in H
    }

    print(f"Clean graph: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges")
    print(f"Removed entities: {len(removed)}")
    print(f"Skipped edges: {skipped_edges}")
    print(f"Merged self-edges: {merged_self_edges}")
    print(f"Removed isolated nodes: {len(isolated_removed)}")

    nx.write_graphml(H, OUT_GRAPHML)

    graph_json = {
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

    with open(OUT_JSON, "w") as f:
        json.dump(graph_json, f, indent=2, ensure_ascii=False)

    with open(OUT_ENTITY_CHUNKS, "w") as f:
        json.dump(clean_e2c, f, indent=2, ensure_ascii=False)

    with open(OUT_ALIAS_MAP, "w") as f:
        json.dump(alias_map, f, indent=2, ensure_ascii=False)

    with open(OUT_REMOVED, "w") as f:
        json.dump(removed, f, indent=2, ensure_ascii=False)

    report = {
        "input_graph": str(IN_GRAPHML),
        "output_graph": str(OUT_GRAPHML),
        "original_nodes": G.number_of_nodes(),
        "original_edges": G.number_of_edges(),
        "clean_nodes": H.number_of_nodes(),
        "clean_edges": H.number_of_edges(),
        "removed_entities": len(removed),
        "skipped_edges": skipped_edges,
        "merged_self_edges": merged_self_edges,
        "removed_isolated_nodes": len(isolated_removed),
        "keep_terms": sorted(KEEP_TERMS),
        "drop_terms": sorted(DROP_TERMS),
    }

    with open(OUT_REPORT, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Saved: {OUT_GRAPHML}")
    print(f"Saved: {OUT_JSON}")
    print(f"Saved: {OUT_ENTITY_CHUNKS}")
    print(f"Saved: {OUT_ALIAS_MAP}")
    print(f"Saved: {OUT_REMOVED}")
    print(f"Saved: {OUT_REPORT}")


if __name__ == "__main__":
    main()
