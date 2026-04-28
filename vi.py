import os
os.environ["MPLCONFIGDIR"] = "/eos/home-r/rjiang/RAG_pixel/.mpl_cache"

import matplotlib
matplotlib.use("Agg")

import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

GRAPHML_PATH = Path("/eos/home-r/rjiang/RAG_pixel/data/graph/graph.graphml")

G = nx.read_graphml(GRAPHML_PATH)
print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Stronger filtering
min_freq = 8
min_weight = 3

nodes_to_keep = [
    n for n, d in G.nodes(data=True)
    if float(d.get("frequency", 0)) >= min_freq
]
H = G.subgraph(nodes_to_keep).copy()

edges_to_keep = [
    (u, v) for u, v, d in H.edges(data=True)
    if float(d.get("weight", 0)) >= min_weight
]
H = H.edge_subgraph(edges_to_keep).copy()

print(f"After thresholding: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges")

# Keep only largest connected component
if H.number_of_nodes() > 0 and H.number_of_edges() > 0:
    largest_cc = max(nx.connected_components(H), key=len)
    H = H.subgraph(largest_cc).copy()

print(f"Largest connected component: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges")

# Optional: only keep top-N nodes by frequency
top_n = 100
if H.number_of_nodes() > top_n:
    top_nodes = sorted(
        H.nodes(data=True),
        key=lambda x: float(x[1].get("frequency", 0)),
        reverse=True
    )[:top_n]
    keep = [x[0] for x in top_nodes]
    H = H.subgraph(keep).copy()

print(f"Final graph to draw: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges")

plt.figure(figsize=(16, 12))

# Much lighter layout
pos = nx.spring_layout(H, k=0.8, iterations=20, seed=42)

node_sizes = []
for n, d in H.nodes(data=True):
    freq = float(d.get("frequency", 1))
    node_sizes.append(20 + freq * 8)

edge_widths = []
for _, _, d in H.edges(data=True):
    w = float(d.get("weight", 1))
    edge_widths.append(0.2 + 0.3 * min(w, 5))

nx.draw_networkx_edges(H, pos, alpha=0.15, width=edge_widths)
nx.draw_networkx_nodes(H, pos, node_size=node_sizes, alpha=0.8)

# Only label the top 30 nodes
top_label_nodes = sorted(
    H.nodes(data=True),
    key=lambda x: float(x[1].get("frequency", 0)),
    reverse=True
)[:30]
label_dict = {n: n for n, _ in top_label_nodes}
nx.draw_networkx_labels(H, pos, labels=label_dict, font_size=8)

plt.title("Entity Co-occurrence Graph (Filtered)")
plt.axis("off")
plt.tight_layout()
plt.savefig("graph_visualisation_filtered.png", dpi=250)
print("Saved to graph_visualisation_filtered.png")
