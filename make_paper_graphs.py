#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import re
import textwrap
from itertools import combinations

import networkx as nx
import matplotlib.pyplot as plt


BAD_PATTERNS = [
    r"^col\d+$", r"^url$", r"^doi$", r"^arxiv$", r"^ieee$", r"^jinst$",
    r"^number$", r"^all$", r"^any$", r"^one$", r"^two$", r"^three$",
    r"^each$", r"^after$", r"^since$", r"^run$", r"^the$", r"^and$",
    r"^or$", r"^in$", r"^on$", r"^of$", r"^for$", r"^with$",
    r"^figure$", r"^table$", r"^section$", r"^abstract$", r"^references$",
    r"^distribution$", r"^accuracy$", r"^fraction$", r"^corresponding$",
    r"^following$", r"^university$", r"^institute$", r"^collaboration$",
    r"^dipartimento$", r"^development$", r"^photomultipliers$",
    r"^silicon photomultipliers$", r"^silicon carbide$",
]

GENERIC_HUBS = {
    "CERN", "LHC", "HL-LHC", "ATLAS", "CMS", "ALICE",
    "IEEE", "PDE", "Number", "URL", "All", "Any", "One", "Two",
}

FIG_CONFIGS = {
    "pixel": {
        "title": "Pixel detector concept graph",
        "out": "figures/paper_graph_pixel.png",
        "query_terms": [
            "Pixel", "pixel detector", "silicon pixel", "Pixel Sensors",
            "Pixel Detectors", "Silicon Pixel Sensor",
            "Semiconductor Pixel Detectors",
        ],
        "domain_patterns": [
            r"pixel", r"sensor", r"detector", r"silicon", r"monolithic",
            r"maps", r"dmaps", r"hv[- ]?cmos", r"hv[- ]?maps", r"hvmaps",
            r"lgad", r"charge", r"depletion", r"threshold", r"noise",
            r"radiation", r"irradiation", r"fluence", r"efficiency",
            r"resolution", r"timing", r"readout", r"vertex", r"tracking",
            r"tracker", r"timepix", r"mightypix", r"velo", r"itk", r"its3",
            r"hgtd", r"tcad", r"geant4", r"bias",
        ],
        "boost_terms": [
            "Pixel", "pixel detector", "silicon pixel", "Pixel Sensors",
            "MAPS", "DMAPS", "HV-CMOS", "HVMAPS", "LGAD",
            "Silicon", "Charge", "Depletion", "Threshold", "Noise",
            "Radiation", "Efficiency", "Resolution", "Timing",
            "Readout", "Timepix", "Timepix3", "MightyPix", "VELO",
            "ITk", "ITS3", "TCAD", "Geant4", "Bias",
        ],
        "hard_remove": {
            "Development", "Sensors", "Photomultipliers",
            "Silicon Photomultipliers", "Silicon Carbide",
            "Inner Tracking System",
        },
        "centre_keywords": ["pixel", "pixel detector", "silicon pixel"],
        "top_n": 28,
        "max_edges": 60,
        "min_overlap": 2,
        "label_top_n": 22,
    },
    "hvmaps": {
        "title": "HV-MAPS / HV-CMOS subgraph",
        "out": "figures/paper_graph_hvmaps_hvcmos.png",
        "query_terms": [
            "HVMAPS", "HV-CMOS", "HVCMOS", "MAPS", "DMAPS",
            "High Voltage", "High Voltage Monolithic",
            "Monolithic Active Pixel", "Monolithic Active Pixel Sensors",
        ],
        "domain_patterns": [
            r"hv[- ]?maps", r"hvmaps", r"hv[- ]?cmos", r"hvcmos",
            r"maps", r"dmaps", r"monolithic", r"active pixel",
            r"depleted", r"cmos", r"pixel", r"sensor", r"detector",
            r"readout", r"threshold", r"noise", r"charge", r"depletion",
            r"bias", r"radiation", r"irradiation", r"fluence",
            r"efficiency", r"timing", r"resolution", r"mightypix",
            r"timepix", r"velo", r"itk", r"tcad",
        ],
        "boost_terms": [
            "HVMAPS", "HV-CMOS", "MAPS", "DMAPS",
            "Monolithic Active Pixel", "High Voltage", "CMOS",
            "Pixel", "pixel detector", "silicon pixel", "sensor", "detector",
            "threshold", "noise", "charge", "depletion", "radiation",
            "efficiency", "timing", "resolution", "MightyPix", "VELO",
            "TCAD", "Bias", "Readout",
        ],
        "hard_remove": {
            "CMOS", "Development", "Sensors", "CERN", "LHC", "HL-LHC",
            "ATLAS", "CMS", "ALICE", "IEEE",
        },
        "centre_keywords": ["hvmaps", "hv-cmos", "maps", "dmaps"],
        "top_n": 28,
        "max_edges": 60,
        "min_overlap": 2,
        "label_top_n": 22,
    },
    "lgad": {
        "title": "LGAD timing subgraph",
        "out": "figures/paper_graph_lgad_timing.png",
        "query_terms": [
            "LGAD", "Low Gain Avalanche Detector", "Low-Gain Avalanche Detector",
            "AC-LGAD", "DC-LGAD", "TI-LGAD", "timing", "Timing",
        ],
        "domain_patterns": [
            r"lgad", r"low[- ]?gain", r"avalanche", r"timing",
            r"resolution", r"time", r"gain", r"charge", r"signal",
            r"sensor", r"detector", r"silicon", r"radiation",
            r"irradiation", r"fluence", r"efficiency", r"threshold",
            r"noise", r"depletion", r"bias", r"pixel", r"strip",
            r"ac[- ]?lgad", r"dc[- ]?lgad", r"ti[- ]?lgad",
            r"picosecond", r"jitter", r"tcad", r"geant4",
        ],
        "boost_terms": [
            "LGAD", "AC-LGAD", "DC-LGAD", "TI-LGAD",
            "Low Gain Avalanche Detector", "Timing", "Resolution",
            "Gain", "Charge", "Signal", "Jitter", "Picosecond",
            "Radiation", "Fluence", "Efficiency", "Threshold",
            "Noise", "Depletion", "Bias", "TCAD", "Time",
        ],
        "hard_remove": {
            "Dipartimento", "Sensors", "Silicon Photomultipliers",
            "Photomultipliers", "CERN", "LHC", "HL-LHC", "ATLAS",
            "CMS", "ALICE", "IEEE",
        },
        "centre_keywords": ["lgad", "ac-lgad", "dc-lgad", "ti-lgad", "timing"],
        "top_n": 28,
        "max_edges": 60,
        "min_overlap": 2,
        "label_top_n": 22,
    },
}


def norm(s):
    return re.sub(r"\s+", " ", str(s).strip().lower())


def compact(s):
    return re.sub(r"[^a-z0-9]+", "", norm(s))


def matches_any(s, patterns):
    x = norm(s)
    return any(re.search(p, x) for p in patterns)


def sameish(a, b):
    ca = compact(a)
    cb = compact(b)
    return ca == cb or ca in cb or cb in ca


def is_bad_entity(e):
    x = norm(e)

    if not x:
        return True
    if len(x) < 3 or len(x) > 90:
        return True
    if re.fullmatch(r"\d+(\.\d+)*", x):
        return True
    if matches_any(x, BAD_PATTERNS):
        return True

    n_alpha = sum(ch.isalpha() for ch in x)
    n_digit = sum(ch.isdigit() for ch in x)

    if n_alpha < 3:
        return True
    if n_digit / max(len(x), 1) > 0.35:
        return True

    return False


def is_hard_removed(e, cfg):
    for bad in cfg.get("hard_remove", set()):
        if sameish(e, bad):
            return True
    return False


def is_domain_entity(e, cfg):
    if is_hard_removed(e, cfg):
        return False

    if e in cfg["boost_terms"]:
        return True

    for term in cfg["boost_terms"]:
        if sameish(e, term):
            return True

    return matches_any(e, cfg["domain_patterns"])


def clean_label(s):
    s = re.sub(r"\s+", " ", str(s).strip())
    words = s.split()
    if len(words) > 5:
        s = " ".join(words[:5]) + "..."
    return "\n".join(textwrap.wrap(s, width=18))


def load_e2c(path):
    with open(path) as f:
        raw = json.load(f)

    return {
        e: set(map(str, chunks))
        for e, chunks in raw.items()
        if not is_bad_entity(e)
    }


def filter_for_config(e2c, cfg):
    out = {}
    for e, chunks in e2c.items():
        if e in GENERIC_HUBS and e not in cfg["boost_terms"]:
            continue
        if not is_domain_entity(e, cfg):
            continue
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

    score = overlap * 10.0 + min(n_chunks, 120) * 0.04

    for term in cfg["boost_terms"]:
        if sameish(e, term):
            score += 12.0

    words = norm(e).split()
    if 1 <= len(words) <= 5:
        score += 1.5

    if is_hard_removed(e, cfg):
        score -= 999.0

    return score, overlap


def build_graph(e2c_all, cfg, top_n, max_edges, min_overlap):
    e2c = filter_for_config(e2c_all, cfg)

    print("\n== {} ==".format(cfg["title"]))
    print("Domain entities:", len(e2c))

    matched_queries, query_set = get_query_chunk_set(e2c, cfg["query_terms"])

    print("Matched query entities:", matched_queries[:30])
    print("Union query chunks:", len(query_set))

    if not query_set:
        raise RuntimeError("No query chunks found for {}".format(cfg["title"]))

    candidates = []

    for e, chunks in e2c.items():
        score, ov = score_entity(e, chunks, query_set, cfg)
        if ov >= min_overlap:
            candidates.append((score, ov, len(chunks), e))

    candidates = sorted(candidates, reverse=True)

    print("Selected candidates:")
    for score, ov, n, e in candidates[:40]:
        print("{:7.2f} | ov={:3d} | n={:4d} | {}".format(score, ov, n, e))

    selected = []
    for _, _, _, e in candidates:
        if e not in selected and not is_hard_removed(e, cfg):
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

    G.remove_nodes_from(list(nx.isolates(G)))

    print("Final graph:", G.number_of_nodes(), "nodes,", G.number_of_edges(), "edges")

    return G


def is_centre_node(n, cfg):
    for kw in cfg["centre_keywords"]:
        if sameish(n, kw):
            return True
    return False


def draw_graph(G, cfg, out, label_top_n, seed):
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(13, 9))

    if G.number_of_nodes() == 1:
        pos = {list(G.nodes())[0]: (0, 0)}
    else:
        pos = nx.spring_layout(
            G,
            seed=seed,
            weight="weight",
            k=0.72,
            iterations=1400,
        )

    degree = dict(G.degree(weight="weight"))
    max_degree = max(degree.values()) if degree else 1.0
    if max_degree == 0:
        max_degree = 1.0

    node_sizes = []
    for n in G.nodes():
        base = 320 + 1650 * (degree.get(n, 0.0) / max_degree) ** 0.62
        if is_centre_node(n, cfg):
            base *= 1.85
        node_sizes.append(base)

    weights = [float(d.get("weight", 1.0)) for _, _, d in G.edges(data=True)]
    max_w = max(weights) if weights else 1.0

    edge_widths = [
        0.45 + 3.4 * (float(d.get("weight", 1.0)) / max_w) ** 0.70
        for _, _, d in G.edges(data=True)
    ]

    nx.draw_networkx_edges(
        G,
        pos,
        width=edge_widths,
        alpha=0.34,
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_sizes,
        node_color="#9dd9d2",
        edgecolors="black",
        linewidths=0.8,
        alpha=0.95,
    )

    label_nodes = sorted(
        G.nodes(),
        key=lambda n: (
            1 if is_centre_node(n, cfg) else 0,
            degree.get(n, 0.0),
        ),
        reverse=True,
    )[:label_top_n]

    labels = {n: clean_label(n) for n in label_nodes}

    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        font_size=8,
        font_family="DejaVu Sans",
    )

    plt.title(cfg["title"], fontsize=15)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=350, bbox_inches="tight")
    plt.close()

    print("Saved:", out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity_chunks", default="data/graph_clean/entity_to_chunks_clean.json")
    parser.add_argument("--which", choices=["all", "pixel", "hvmaps", "lgad"], default="all")
    parser.add_argument("--top_n", type=int, default=None)
    parser.add_argument("--max_edges", type=int, default=None)
    parser.add_argument("--min_overlap", type=int, default=None)
    parser.add_argument("--label_top_n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    e2c_all = load_e2c(Path(args.entity_chunks))
    print("Loaded clean entities:", len(e2c_all))

    names = ["pixel", "hvmaps", "lgad"] if args.which == "all" else [args.which]

    for name in names:
        cfg = dict(FIG_CONFIGS[name])

        top_n = args.top_n if args.top_n is not None else cfg["top_n"]
        max_edges = args.max_edges if args.max_edges is not None else cfg["max_edges"]
        min_overlap = args.min_overlap if args.min_overlap is not None else cfg["min_overlap"]
        label_top_n = args.label_top_n if args.label_top_n is not None else cfg["label_top_n"]

        G = build_graph(
            e2c_all=e2c_all,
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
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
