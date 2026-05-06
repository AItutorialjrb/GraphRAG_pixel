#!/usr/bin/env python3

from pathlib import Path
import argparse
import json
import re
import textwrap
from itertools import combinations

import networkx as nx
import matplotlib.pyplot as plt


DOMAIN_PATTERNS = [
    r"pixel",
    r"sensor",
    r"detector",
    r"silicon",
    r"cmos",
    r"maps",
    r"dmaps",
    r"hv[- ]?cmos",
    r"hv[- ]?maps",
    r"hvmaps",
    r"lgad",
    r"monolithic",
    r"active pixel",
    r"depleted",
    r"tracking",
    r"tracker",
    r"vertex",
    r"readout",
    r"asic",
    r"front[- ]?end",
    r"threshold",
    r"noise",
    r"charge",
    r"depletion",
    r"bias",
    r"radiation",
    r"irradiation",
    r"fluence",
    r"efficiency",
    r"resolution",
    r"timing",
    r"timepix",
    r"mightypix",
    r"velo",
    r"itk",
    r"its3",
    r"hgtd",
    r"tcad",
    r"geant4",
]

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
    r"^run$",
    r"^the$",
    r"^and$",
    r"^or$",
    r"^in$",
    r"^on$",
    r"^of$",
    r"^for$",
    r"^with$",
    r"^figure$",
    r"^table$",
    r"^section$",
    r"^abstract$",
    r"^references$",
    r"^distribution$",
    r"^accuracy$",
    r"^fraction$",
    r"^pde$",
    r"^cms$",
    r"^atlas$",
    r"^alice$",
    r"^cern$",
    r"^lhc$",
    r"^hl[- ]?lhc$",
]


PREFERRED_ENTITIES = {
    "Pixel",
    "Pixels",
    "pixel detector",
    "Pixel Detectors",
    "Pixel Sensors",
    "silicon pixel",
    "Silicon Pixel Sensor",
    "Semiconductor Pixel Detectors",
    "Monolithic Active Pixel",
    "Monolithic Active Pixel Sensor",
    "Monolithic Active Pixel Sensors",
    "Depleted Monolithic Active Pixel Sensors",
    "DMAPS",
    "MAPS",
    "HVMAPS",
    "HV-CMOS",
    "CMOS",
    "ASIC",
    "TCAD",
    "Geant4",
    "readout",
    "threshold",
    "noise",
    "charge",
    "depletion",
    "efficiency",
    "resolution",
    "timing",
    "radiation",
    "irradiation",
    "fluence",
    "Timepix",
    "Timepix3",
    "MightyPix",
    "VELO",
    "ITk",
    "ITS3",
    "HGTD",
}


def norm(s):
    return re.sub(r"\s+", " ", str(s).strip().lower())


def compact(s):
    return re.sub(r"[^a-z0-9]+", "", norm(s))


def matches_any(s, patterns):
    x = norm(s)
    return any(re.search(p, x) for p in patterns)


def is_bad_entity(e):
    x = norm(e)

    if not x:
        return True

    if len(x) < 3 or len(x) > 80:
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


def is_domain_entity(e):
    if e in PREFERRED_ENTITIES:
        return True
    return matches_any(e, DOMAIN_PATTERNS)


def clean_label(s):
    s = re.sub(r"\s+", " ", str(s).strip())
    words = s.split()
    if len(words) > 5:
        s = " ".join(words[:5]) + "..."
    return "\n".join(textwrap.wrap(s, width=18))


def load_e2c(path):
    with open(path) as f:
        raw = json.load(f)

    e2c = {}
    for e, chunks in raw.items():
        if is_bad_entity(e):
            continue
        if not is_domain_entity(e):
            continue
        e2c[e] = set(map(str, chunks))

    return e2c


def find_entity(e2c, query):
    q = compact(query)

    for e in e2c:
        if compact(e) == q:
            return e

    hits = [e for e in e2c if q in compact(e)]
    if hits:
        return sorted(hits, key=lambda x: len(e2c[x]), reverse=True)[0]

    raise ValueError("Cannot find query entity: {}".format(query))


def domain_score(entity, overlap, n_chunks):
    score = overlap * 10 + min(n_chunks, 100) * 0.05

    e = norm(entity)

    boosts = {
        "pixel": 5,
        "sensor": 4,
        "detector": 4,
        "silicon": 4,
        "monolithic": 4,
        "cmos": 3,
        "maps": 3,
        "dmaps": 3,
        "hv": 2,
        "lgad": 3,
        "readout": 2,
        "threshold": 2,
        "noise": 2,
        "charge": 2,
        "depletion": 2,
        "radiation": 2,
        "timing": 2,
        "efficiency": 2,
        "resolution": 2,
    }

    for key, val in boosts.items():
        if key in e:
            score += val

    return score


def build_graph(e2c, query, top_n, max_edges, min_overlap):
    centre = find_entity(e2c, query)
    cset = e2c[centre]

    print("Centre:", centre)
    print("Centre chunks:", len(cset))

    candidates = []

    for e, chunks in e2c.items():
        if e == centre:
            continue

        ov = len(cset & chunks)
        if ov < min_overlap:
            continue

        score = domain_score(e, ov, len(chunks))
        candidates.append((score, ov, len(chunks), e))

    candidates = sorted(candidates, reverse=True)

    print("Selected domain entities:")
    for score, ov, n, e in candidates[:30]:
        print("{:7.2f} | ov={:3d} | n={:4d} | {}".format(score, ov, n, e))

    selected = [centre] + [e for _, _, _, e in candidates[:top_n - 1]]

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

    if centre not in G:
        G.add_node(centre)

    if G.number_of_nodes() > 1:
        for comp in nx.connected_components(G):
            if centre in comp:
                G = G.subgraph(comp).copy()
                break

    print("Final graph:", G.number_of_nodes(), "nodes,", G.number_of_edges(), "edges")
    return G


def draw(G, out, title, label_top_n):
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(13, 9))

    if G.number_of_nodes() == 1:
        pos = {list(G.nodes())[0]: (0, 0)}
    else:
        pos = nx.spring_layout(
            G,
            seed=11,
            weight="weight",
            k=0.95,
            iterations=1000,
        )

    degree = dict(G.degree(weight="weight"))
    max_degree = max(degree.values()) if degree else 1.0
    if max_degree == 0:
        max_degree = 1.0

    node_sizes = [
        350 + 1600 * (degree.get(n, 0.0) / max_degree) ** 0.65
        for n in G.nodes()
    ]

    weights = [float(d.get("weight", 1.0)) for _, _, d in G.edges(data=True)]
    max_w = max(weights) if weights else 1.0

    edge_widths = [
        0.6 + 3.0 * (float(d.get("weight", 1.0)) / max_w) ** 0.7
        for _, _, d in G.edges(data=True)
    ]

    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.32)
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
        key=lambda n: degree.get(n, 0.0),
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

    plt.title(title, fontsize=15)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity_chunks", default="data/graph_clean/entity_to_chunks_clean.json")
    parser.add_argument("--query", default="Pixel")
    parser.add_argument("--top_n", type=int, default=30)
    parser.add_argument("--max_edges", type=int, default=60)
    parser.add_argument("--min_overlap", type=int, default=1)
    parser.add_argument("--label_top_n", type=int, default=18)
    parser.add_argument("--out", default="figures/graph_pixel_domain.png")
    args = parser.parse_args()

    e2c = load_e2c(Path(args.entity_chunks))
    print("Loaded detector-domain entities:", len(e2c))

    G = build_graph(
        e2c=e2c,
        query=args.query,
        top_n=args.top_n,
        max_edges=args.max_edges,
        min_overlap=args.min_overlap,
    )

    draw(
        G,
        out=Path(args.out),
        title="Pixel detector concept graph",
        label_top_n=args.label_top_n,
    )


if __name__ == "__main__":
    main()
