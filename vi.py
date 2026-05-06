#!/usr/bin/env python3
"""
Physics-clean publication graph visualisation for GraphRAG_pixel.

Input:
  data/graph_clean/entity_to_chunks_clean.json

Output:
  figures/paper_graph_pixel.png
  figures/paper_graph_hvmaps_hvcmos.png
  figures/paper_graph_lgad_timing.png

Usage:
  python vi.py --which all
  python vi.py --which all --dense
  python vi.py --which lgad --top_n 30 --max_edges 75 --label_top_n 22
"""

from pathlib import Path
import argparse
import json
import re
import textwrap
from itertools import combinations

import networkx as nx
import matplotlib.pyplot as plt


DEFAULT_ENTITY_CHUNKS = Path("data/graph_clean/entity_to_chunks_clean.json")


# -----------------------------
# Strict physics vocabulary
# -----------------------------

CANONICAL_MAP = {
    # Pixel family
    "pixel": "Pixel",
    "pixels": "Pixel",
    "pixel detector": "Pixel detector",
    "pixel detectors": "Pixel detector",
    "pixel sensor": "Pixel sensor",
    "pixel sensors": "Pixel sensor",
    "silicon pixel": "Silicon pixel",
    "silicon pixels": "Silicon pixel",
    "silicon pixel sensor": "Silicon pixel sensor",
    "silicon pixel sensors": "Silicon pixel sensor",

    # MAPS / HV-CMOS
    "maps": "MAPS",
    "map": "MAPS",
    "monolithic active pixel": "MAPS",
    "monolithic active pixel sensor": "MAPS",
    "monolithic active pixel sensors": "MAPS",
    "monolithic active pixels sensors": "MAPS",
    "active pixel sensor": "MAPS",
    "active pixel sensors": "MAPS",

    "dmaps": "DMAPS",
    "depleted monolithic active pixel sensor": "DMAPS",
    "depleted monolithic active pixel sensors": "DMAPS",

    "hvmaps": "HVMAPS",
    "hv maps": "HVMAPS",
    "hv-maps": "HVMAPS",
    "high voltage monolithic active pixel sensor": "HVMAPS",
    "high voltage monolithic active pixel sensors": "HVMAPS",
    "high-voltage monolithic active pixel sensor": "HVMAPS",
    "high-voltage monolithic active pixel sensors": "HVMAPS",

    "hv-cmos": "HV-CMOS",
    "hv cmos": "HV-CMOS",
    "hvcmos": "HV-CMOS",
    "high voltage cmos": "HV-CMOS",
    "high-voltage cmos": "HV-CMOS",

    # LGAD
    "lgad": "LGAD",
    "lgads": "LGAD",
    "lgad1": "LGAD",
    "lgad2": "LGAD",
    "lgad 1": "LGAD",
    "lgad 2": "LGAD",
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

    # Readout / timing
    "tot": "ToT",
    "time over threshold": "ToT",
    "time-over-threshold": "ToT",
    "toa": "ToA",
    "time of arrival": "ToA",
    "time-of-arrival": "ToA",
    "tdc": "TDC",
    "pll": "PLL",
    "adc": "ADC",
    "dac": "DAC",
    "asic": "ASIC",
    "cmos": "CMOS",
    "tcad": "TCAD",
    "geant4": "Geant4",

    # detector names
    "timepix": "Timepix",
    "timepix3": "Timepix3",
    "timepix 3": "Timepix3",
    "mightypix": "MightyPix",
    "mighty pix": "MightyPix",
    "velo": "VELO",
    "itk": "ITk",
    "hgtd": "HGTD",
    "its3": "ITS3",

    # performance / physics
    "time resolution": "Time resolution",
    "timing resolution": "Time resolution",
    "spatial resolution": "Spatial resolution",
    "detection efficiency": "Detection efficiency",
    "efficiency": "Efficiency",
    "charge collection": "Charge collection",
    "charge sharing": "Charge sharing",
    "threshold": "Threshold",
    "noise": "Noise",
    "depletion": "Depletion",
    "bias voltage": "Bias voltage",
    "fluence": "Fluence",
    "radiation damage": "Radiation damage",
    "irradiation": "Irradiation",
    "radiation": "Radiation",
    "gain": "Gain",
    "avalanche gain": "Avalanche gain",
    "jitter": "Jitter",
    "signal": "Signal",
    "charge": "Charge",
    "drift": "Drift",
    "diffusion": "Diffusion",
}


ALLOWED_EXACT = {
    # central detector concepts
    "Pixel", "Pixel detector", "Pixel sensor", "Silicon pixel", "Silicon pixel sensor",
    "MAPS", "DMAPS", "HVMAPS", "HV-CMOS", "CMOS",
    "LGAD", "AC-LGAD", "DC-LGAD", "TI-LGAD",

    # readout/electronics
    "ASIC", "TCAD", "Geant4", "ADC", "DAC", "TDC", "PLL", "ToT", "ToA",
    "Timepix", "Timepix3", "MightyPix",

    # experiments/subsystems
    "VELO", "ITk", "HGTD", "ITS3",

    # performance/physics
    "Charge", "Signal", "Threshold", "Noise", "Depletion", "Bias voltage",
    "Radiation", "Irradiation", "Radiation damage", "Fluence",
    "Efficiency", "Detection efficiency",
    "Time resolution", "Timing", "Spatial resolution",
    "Charge collection", "Charge sharing",
    "Gain", "Avalanche gain", "Jitter",
    "Drift", "Diffusion", "Readout", "Front-end", "Tracking", "Tracker", "Vertex detector",
}


ALLOWED_PATTERNS = [
    r"\bpixel\b",
    r"\bpixel detector\b",
    r"\bpixel sensor\b",
    r"\bsilicon pixel\b",
    r"\bsensor\b",
    r"\bdetector\b",
    r"\bsilicon\b",
    r"\bmonolithic active pixel\b",
    r"\bmaps\b",
    r"\bdmaps\b",
    r"\bhv[- ]?cmos\b",
    r"\bhv[- ]?maps\b",
    r"\bhvmaps\b",
    r"\bcmos\b",
    r"\blgad\b",
    r"\bac[- ]?lgad\b",
    r"\bdc[- ]?lgad\b",
    r"\bti[- ]?lgad\b",
    r"\blow[- ]?gain avalanche detector\b",
    r"\breadout\b",
    r"\bfront[- ]?end\b",
    r"\bthreshold\b",
    r"\bnoise\b",
    r"\bcharge\b",
    r"\bsignal\b",
    r"\bdrift\b",
    r"\bdiffusion\b",
    r"\bdepletion\b",
    r"\bbias\b",
    r"\bvoltage\b",
    r"\bradiation\b",
    r"\birradiation\b",
    r"\bfluence\b",
    r"\bdose\b",
    r"\befficiency\b",
    r"\bresolution\b",
    r"\btiming\b",
    r"\bjitter\b",
    r"\bgain\b",
    r"\bavalanche\b",
    r"\btot\b",
    r"\btoa\b",
    r"\btdc\b",
    r"\bpll\b",
    r"\badc\b",
    r"\bdac\b",
    r"\btimepix\b",
    r"\btimepix3\b",
    r"\bmightypix\b",
    r"\bvelo\b",
    r"\bitk\b",
    r"\bhgtd\b",
    r"\bits3\b",
    r"\btcad\b",
    r"\bgeant4\b",
]


# Hard OCR / PDF / token noise.
BAD_EXACT = {
    "Let", "LET", "Tor", "TOR", "IMING", "IMIN", "IMNG",
    "VOL", "LUT", "Map", "High", "Low", "The High Gain",
    "LGAD1", "LGAD2", "LGAD 1", "LGAD 2",
    "Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8",
    "URL", "DOI", "arXiv", "IEEE", "JINST",
    "Number", "All", "Any", "One", "Two", "Three",
    "Each", "After", "Since", "Before", "Following", "Corresponding",
    "Run", "Figure", "Fig", "Table", "Section", "Abstract",
    "References", "Conclusion", "Introduction", "Appendix",
    "Distribution", "Accuracy", "Fraction",
    "Dipartimento", "Development", "Photomultipliers",
    "Silicon Photomultipliers", "Silicon Carbide",
    "Active", "Charged", "MONOLITH", "MONOLITHIC",
}


BAD_PATTERNS = [
    r"^col\d+$",
    r"^url$",
    r"^doi$",
    r"^arxiv$",
    r"^ieee$",
    r"^jinst$",
    r"^number$",
    r"^all$",
    r"^any$",
    r"^one$",
    r"^two$",
    r"^three$",
    r"^each$",
    r"^after$",
    r"^since$",
    r"^before$",
    r"^following$",
    r"^corresponding$",
    r"^run$",
    r"^let$",
    r"^tor$",
    r"^iming$",
    r"^vol$",
    r"^lut$",
    r"^map$",
    r"^high$",
    r"^low$",
    r"^active$",
    r"^charged$",
    r"^figure$",
    r"^fig$",
    r"^table$",
    r"^section$",
    r"^abstract$",
    r"^references$",
    r"^conclusion$",
    r"^introduction$",
    r"^appendix$",
    r"^distribution$",
    r"^accuracy$",
    r"^fraction$",
    r"^development$",
    r"^dipartimento$",
    r"^monolith$",
    r"^monolithic$",
    r"^lgad\d+$",
    r"^[a-zA-Z]{1,2}$",
]


GENERIC_HUBS = {
    "CERN", "LHC", "HL-LHC", "ATLAS", "CMS", "ALICE", "IEEE", "JINST",
}


FIG_CONFIGS = {
    "pixel": {
        "title": "Pixel detector concept graph",
        "out": "figures/paper_graph_pixel.png",
        "query_terms": [
            "Pixel", "Pixel detector", "Pixel sensor", "Silicon pixel",
            "MAPS", "DMAPS", "HV-CMOS", "LGAD",
        ],
        "wanted": {
            "Pixel", "Pixel detector", "Pixel sensor", "Silicon pixel", "Silicon pixel sensor",
            "MAPS", "DMAPS", "HV-CMOS", "HVMAPS", "LGAD", "CMOS", "ASIC",
            "Charge", "Charge collection", "Charge sharing", "Threshold", "Noise",
            "Depletion", "Bias voltage", "Radiation", "Irradiation", "Fluence",
            "Efficiency", "Detection efficiency", "Time resolution", "Spatial resolution",
            "Readout", "Timepix", "Timepix3", "MightyPix", "TCAD", "Geant4",
        },
        "remove": {
            "CERN", "LHC", "HL-LHC", "ATLAS", "CMS", "ALICE",
            "Pixel Center", "in-pixel", "Sensors",
        },
        "centre": {"Pixel", "Pixel detector", "Silicon pixel"},
        "top_n": 34,
        "max_edges": 88,
        "min_overlap": 1,
        "label_top_n": 28,
        "seed": 31,
    },
    "hvmaps": {
        "title": "HV-MAPS / HV-CMOS subgraph",
        "out": "figures/paper_graph_hvmaps_hvcmos.png",
        "query_terms": [
            "HVMAPS", "HV-CMOS", "MAPS", "DMAPS", "CMOS",
            "MightyPix", "Pixel detector",
        ],
        "wanted": {
            "HVMAPS", "HV-CMOS", "MAPS", "DMAPS", "CMOS", "ASIC",
            "MightyPix", "Pixel", "Pixel detector", "Pixel sensor", "Silicon pixel",
            "Readout", "Front-end", "Threshold", "Noise", "Charge", "Charge collection",
            "Depletion", "Bias voltage", "Radiation", "Irradiation", "Fluence",
            "Efficiency", "Detection efficiency", "Time resolution", "Timing",
            "Timepix", "Timepix3", "VELO", "ITk", "TCAD",
        },
        "remove": {
            "CERN", "LHC", "HL-LHC", "ATLAS", "CMS", "ALICE",
            "MONOLITH", "MONOLITHIC", "Active", "Charged",
        },
        "centre": {"HVMAPS", "HV-CMOS", "MAPS", "DMAPS"},
        "top_n": 34,
        "max_edges": 90,
        "min_overlap": 1,
        "label_top_n": 27,
        "seed": 37,
    },
    "lgad": {
        "title": "LGAD timing subgraph",
        "out": "figures/paper_graph_lgad_timing.png",
        "query_terms": [
            "LGAD", "AC-LGAD", "DC-LGAD", "TI-LGAD",
            "Low Gain Avalanche Detector", "Time resolution", "Gain",
        ],
        "wanted": {
            "LGAD", "AC-LGAD", "DC-LGAD", "TI-LGAD",
            "Gain", "Avalanche gain", "Charge", "Signal",
            "Time resolution", "Timing", "Jitter", "Efficiency",
            "Threshold", "Noise", "Depletion", "Bias voltage",
            "Radiation", "Irradiation", "Fluence",
            "Silicon pixel", "Pixel detector", "Sensor", "Detector",
            "TCAD", "Geant4",
        },
        "remove": {
            "CERN", "LHC", "HL-LHC", "ATLAS", "CMS", "ALICE",
            "Photomultipliers", "Silicon Photomultipliers", "Dipartimento",
            "CAD", "IMING", "Low", "High", "The High Gain",
        },
        "centre": {"LGAD", "AC-LGAD", "DC-LGAD", "TI-LGAD"},
        "top_n": 30,
        "max_edges": 78,
        "min_overlap": 1,
        "label_top_n": 24,
        "seed": 41,
    },
}


def norm(s):
    s = str(s).strip()
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"[_/]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def lower(s):
    return norm(s).lower().strip(" -_:;,.()[]{}")


def compact(s):
    return re.sub(r"[^a-z0-9]+", "", lower(s))


def canonical(entity):
    x = lower(entity)

    if x in CANONICAL_MAP:
        return CANONICAL_MAP[x]

    cx = compact(x)
    for k, v in CANONICAL_MAP.items():
        if compact(k) == cx:
            return v

    # Standard capitalisation for known acronyms
    acronym = {
        "cmos": "CMOS",
        "asic": "ASIC",
        "tcad": "TCAD",
        "geant4": "Geant4",
        "tdc": "TDC",
        "pll": "PLL",
        "adc": "ADC",
        "dac": "DAC",
        "tot": "ToT",
        "toa": "ToA",
    }
    if cx in acronym:
        return acronym[cx]

    return norm(entity).strip(" -_:;,.()[]{}")


def sameish(a, b):
    ca = compact(a)
    cb = compact(b)
    if not ca or not cb:
        return False
    return ca == cb or ca in cb or cb in ca


def matches_any(text, patterns):
    x = lower(text)
    return any(re.search(p, x) for p in patterns)


def is_bad_entity(entity):
    e = canonical(entity)
    x = lower(e)
    c = compact(e)

    if not e:
        return True

    if e in BAD_EXACT:
        return True

    if matches_any(e, BAD_PATTERNS):
        return True

    if re.fullmatch(r"\d+(\.\d+)*", x):
        return True

    if len(c) < 3:
        return True

    if len(e) > 90:
        return True

    # Remove OCR-looking short fragments unless explicitly allowed.
    if len(c) <= 5 and e not in ALLOWED_EXACT:
        if c not in {"cmos", "asic", "tcad", "lgad", "maps", "dmaps", "spad", "tot", "toa", "tdc", "pll", "adc", "dac"}:
            return True

    # Remove tokens with too many digits, except known detector names.
    n_digit = sum(ch.isdigit() for ch in e)
    if n_digit / max(len(e), 1) > 0.25 and e not in {"Timepix3", "RD53A", "RD53B"}:
        return True

    return False


def is_physics_entity(entity):
    e = canonical(entity)

    if e in ALLOWED_EXACT:
        return True

    if is_bad_entity(e):
        return False

    return matches_any(e, ALLOWED_PATTERNS)


def load_entity_chunks(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    merged = {}

    for ent, chunks in raw.items():
        c = canonical(ent)

        if is_bad_entity(c):
            continue

        if not is_physics_entity(c):
            continue

        if c not in merged:
            merged[c] = set()

        merged[c].update(str(x) for x in chunks)

    return merged


def filter_for_config(e2c, cfg):
    out = {}

    for e, chunks in e2c.items():
        e = canonical(e)

        if e in cfg["remove"]:
            continue

        if e in GENERIC_HUBS and e not in cfg["wanted"]:
            continue

        if e in cfg["wanted"]:
            out[e] = chunks
            continue

        # For publication figure, be strict:
        # keep only if semantically related and not a generic orphan.
        if is_physics_entity(e):
            # avoid letting all detector words enter every subgraph
            if any(sameish(e, w) for w in cfg["wanted"]):
                out[e] = chunks
            elif any(sameish(e, q) for q in cfg["query_terms"]):
                out[e] = chunks

    return out


def get_query_chunk_set(e2c, query_terms):
    matched = []

    for qt in query_terms:
        for e in e2c:
            if sameish(e, qt):
                matched.append(e)

    matched = sorted(set(matched), key=lambda e: len(e2c[e]), reverse=True)

    qset = set()
    for e in matched:
        qset |= e2c[e]

    return matched, qset


def score_entity(e, chunks, query_set, cfg):
    overlap = len(chunks & query_set)
    n_chunks = len(chunks)

    score = overlap * 10.0 + min(n_chunks, 150) * 0.035

    if e in cfg["wanted"]:
        score += 15.0

    if e in cfg["centre"]:
        score += 30.0

    # Prefer proper scientific multi-word concepts.
    if 2 <= len(e.split()) <= 5:
        score += 2.5

    # Penalise generic hubs unless explicitly central.
    if e in {"CMOS", "ASIC", "Sensor", "Detector"} and e not in cfg["centre"]:
        score -= 4.0

    return score, overlap


def build_graph(e2c_all, cfg, top_n, max_edges, min_overlap):
    e2c = filter_for_config(e2c_all, cfg)

    print("\n== {} ==".format(cfg["title"]))
    print("Candidate physics entities:", len(e2c))

    matched, qset = get_query_chunk_set(e2c, cfg["query_terms"])
    print("Matched query entities:", matched)
    print("Union query chunks:", len(qset))

    if not qset:
        raise RuntimeError("No query chunks found. Try less strict query_terms.")

    candidates = []

    for e, chunks in e2c.items():
        score, ov = score_entity(e, chunks, qset, cfg)
        if ov >= min_overlap:
            candidates.append((score, ov, len(chunks), e))

    candidates = sorted(candidates, reverse=True)

    print("Selected candidates:")
    for score, ov, n, e in candidates[:45]:
        print("{:7.2f} | ov={:3d} | n={:4d} | {}".format(score, ov, n, e))

    selected = []
    for _, _, _, e in candidates:
        if e not in selected:
            selected.append(e)
        if len(selected) >= top_n:
            break

    G = nx.Graph()

    for e in selected:
        G.add_node(e, chunk_count=len(e2c[e]))

    edges = []
    for a, b in combinations(selected, 2):
        ov = len(e2c[a] & e2c[b])
        if ov >= min_overlap:
            edges.append((a, b, ov))

    edges = sorted(edges, key=lambda x: x[2], reverse=True)[:max_edges]

    for a, b, ov in edges:
        G.add_edge(a, b, weight=float(ov))

    # Remove isolates except centre terms.
    for n in list(nx.isolates(G)):
        if n not in cfg["centre"]:
            G.remove_node(n)

    print("Final graph:", G.number_of_nodes(), "nodes,", G.number_of_edges(), "edges")
    return G


def clean_label(s):
    s = str(s).strip()

    title_case = {
        "pixel detector": "Pixel detector",
        "pixel sensor": "Pixel sensor",
        "silicon pixel": "Silicon pixel",
        "silicon pixel sensor": "Silicon pixel sensor",
        "time resolution": "Time resolution",
        "spatial resolution": "Spatial resolution",
        "bias voltage": "Bias voltage",
    }

    if lower(s) in title_case:
        s = title_case[lower(s)]

    words = s.split()
    if len(words) > 5:
        s = " ".join(words[:5]) + "..."

    return "\n".join(textwrap.wrap(s, width=17))


def is_centre(n, cfg):
    return n in cfg["centre"] or any(sameish(n, c) for c in cfg["centre"])


def node_colour(n):
    x = lower(n)

    if "lgad" in x or "gain" in x or "avalanche" in x:
        return "#fdae61"  # orange
    if "maps" in x or "cmos" in x or "mightypix" in x:
        return "#74add1"  # blue
    if any(k in x for k in ["charge", "threshold", "noise", "bias", "depletion", "radiation", "fluence"]):
        return "#abdda4"  # green
    if any(k in x for k in ["timing", "time", "resolution", "jitter", "tdc", "tot", "toa"]):
        return "#d53e4f"  # red
    if any(k in x for k in ["tcad", "geant4"]):
        return "#b2abd2"  # purple
    if any(k in x for k in ["pixel", "sensor", "detector", "silicon"]):
        return "#80cdc1"  # teal

    return "#c7eae5"


def draw_graph(G, cfg, out, label_top_n, seed, dpi):
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(13.8, 9.4))

    if G.number_of_nodes() == 1:
        pos = {list(G.nodes())[0]: (0, 0)}
    else:
        pos = nx.spring_layout(
            G,
            seed=seed,
            weight="weight",
            k=0.62,
            iterations=2200,
        )

    degree = dict(G.degree(weight="weight"))
    max_degree = max(degree.values()) if degree else 1.0
    if max_degree == 0:
        max_degree = 1.0

    node_sizes = []
    for n in G.nodes():
        base = 360 + 1900 * (degree.get(n, 0.0) / max_degree) ** 0.60
        if is_centre(n, cfg):
            base *= 1.85
        node_sizes.append(base)

    weights = [float(d.get("weight", 1.0)) for _, _, d in G.edges(data=True)]
    max_w = max(weights) if weights else 1.0

    for u, v, d in G.edges(data=True):
        w = float(d.get("weight", 1.0))
        width = 0.55 + 3.8 * (w / max_w) ** 0.72
        alpha = 0.22 + 0.30 * (w / max_w) ** 0.5
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v)],
            width=width,
            alpha=alpha,
            edge_color="#555555",
        )

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_sizes,
        node_color=[node_colour(n) for n in G.nodes()],
        edgecolors="#222222",
        linewidths=0.9,
        alpha=0.96,
    )

    label_nodes = sorted(
        G.nodes(),
        key=lambda n: (
            1 if is_centre(n, cfg) else 0,
            degree.get(n, 0.0),
        ),
        reverse=True,
    )[:label_top_n]

    labels = {n: clean_label(n) for n in label_nodes}

    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        font_size=8.3,
        font_family="DejaVu Sans",
    )

    plt.title(cfg["title"], fontsize=16, pad=12)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()

    print("Saved:", out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity_chunks", type=Path, default=DEFAULT_ENTITY_CHUNKS)
    parser.add_argument("--which", choices=["all", "pixel", "hvmaps", "lgad"], default="all")
    parser.add_argument("--top_n", type=int, default=None)
    parser.add_argument("--max_edges", type=int, default=None)
    parser.add_argument("--min_overlap", type=int, default=None)
    parser.add_argument("--label_top_n", type=int, default=None)
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--dpi", type=int, default=450)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    e2c = load_entity_chunks(args.entity_chunks)
    print("Loaded physics-clean entities:", len(e2c))

    names = ["pixel", "hvmaps", "lgad"] if args.which == "all" else [args.which]

    for name in names:
        cfg = FIG_CONFIGS[name]

        top_n = args.top_n if args.top_n is not None else cfg["top_n"]
        max_edges = args.max_edges if args.max_edges is not None else cfg["max_edges"]
        min_overlap = args.min_overlap if args.min_overlap is not None else cfg["min_overlap"]
        label_top_n = args.label_top_n if args.label_top_n is not None else cfg["label_top_n"]
        seed = args.seed if args.seed is not None else cfg["seed"]

        if args.dense:
            top_n += 6
            max_edges += 25
            label_top_n += 4
            min_overlap = 1

        G = build_graph(
            e2c_all=e2c,
            cfg=cfg,
            top_n=top_n,
            max_edges=max_edges,
            min_overlap=min_overlap,
        )

        draw_graph(
            G=G,
            cfg=cfg,
            out=Path(cfg["out"]),
            label_top_n=label_top_n,
            seed=seed,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
