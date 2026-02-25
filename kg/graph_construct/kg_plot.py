"""
kg_plot.py  —  Visualise KG statistics saved by kg_stat.py.
Outputs PNG files to: /home/nadun/wd/datasets/wsi_bench/kg_data/stats/
"""

import json
import os
import networkx as nx
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
STATS_DIR    = "/home/nadun/wd/datasets/wsi_bench/kg_data/stats"
CSV_PATH     = "/home/nadun/wd/datasets/wsi_bench/custom_kg/relations.csv"

BASIC_JSON   = os.path.join(STATS_DIR, "basic_stats.json")
COMP_JSON    = os.path.join(STATS_DIR, "connected_components.json")
INCID_JSON   = os.path.join(STATS_DIR, "incidence_list.json")

# ── Load stats ────────────────────────────────────────────────────────────────
with open(BASIC_JSON)  as f: basic  = json.load(f)
with open(INCID_JSON)  as f: edges  = json.load(f)

df = pd.DataFrame(edges)

# Rebuild graph
G = nx.DiGraph()
for row in edges:
    G.add_edge(row["reference_entity"], row["related_entity"], relation=row["relation_name"])

PALETTE = plt.get_cmap("tab10").colors

# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Relation type distribution (horizontal bar)
# ─────────────────────────────────────────────────────────────────────────────
rel_counts = basic["relation_type_counts"]
labels = list(rel_counts.keys())
values = list(rel_counts.values())

fig, ax = plt.subplots(figsize=(9, 4))
colors = [PALETTE[i % len(PALETTE)] for i in range(len(labels))]
bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.6)
ax.bar_label(bars, padding=4, fontsize=10)
ax.set_xlabel("Number of edges", fontsize=11)
ax.set_title("Relation Type Distribution", fontsize=13, fontweight="bold")
ax.invert_yaxis()
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(0, max(values) * 1.18)
plt.tight_layout()
out = os.path.join(STATS_DIR, "plot_relation_types.png")
plt.savefig(out, dpi=150)
plt.close()
print(f"Saved: {out}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Top-20 source nodes by out-degree
# ─────────────────────────────────────────────────────────────────────────────
out_deg = sorted(G.out_degree(), key=lambda x: x[1], reverse=True)[:20]
nodes_od, degs_od = zip(*out_deg)

fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(nodes_od))
ax.bar(x, degs_od, color=PALETTE[1], edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(nodes_od, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Out-degree", fontsize=11)
ax.set_title("Top-20 Cancer Types by Out-Degree (# concepts linked)", fontsize=13, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
out = os.path.join(STATS_DIR, "plot_top_out_degree.png")
plt.savefig(out, dpi=150)
plt.close()
print(f"Saved: {out}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — Top-20 concept nodes by in-degree
# ─────────────────────────────────────────────────────────────────────────────
in_deg = sorted(G.in_degree(), key=lambda x: x[1], reverse=True)[:20]
nodes_id, degs_id = zip(*in_deg)

fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(nodes_id))
ax.bar(x, degs_id, color=PALETTE[2], edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(nodes_id, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("In-degree", fontsize=11)
ax.set_title("Top-20 Concept Nodes by In-Degree (# cancer types that reference them)", fontsize=13, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
out = os.path.join(STATS_DIR, "plot_top_in_degree.png")
plt.savefig(out, dpi=150)
plt.close()
print(f"Saved: {out}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 — Degree distribution (histogram, undirected)
# ─────────────────────────────────────────────────────────────────────────────
G_undir = G.to_undirected()
degrees = [d for _, d in G_undir.degree()]

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(degrees, bins=range(1, max(degrees) + 2), color=PALETTE[3],
        edgecolor="white", align="left", rwidth=0.8)
ax.set_xlabel("Degree (undirected)", fontsize=11)
ax.set_ylabel("Number of nodes", fontsize=11)
ax.set_title("Node Degree Distribution", fontsize=13, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
out = os.path.join(STATS_DIR, "plot_degree_distribution.png")
plt.savefig(out, dpi=150)
plt.close()
print(f"Saved: {out}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 5 — Graph layout (spring, colour-coded by relation type)
# ─────────────────────────────────────────────────────────────────────────────
relation_types = list(rel_counts.keys())
rel_color_map  = {r: PALETTE[i % len(PALETTE)] for i, r in enumerate(relation_types)}

pos = nx.spring_layout(G, seed=42, k=0.6)

fig, ax = plt.subplots(figsize=(16, 12))

# Draw edges grouped by relation for the legend
for rel in relation_types:
    edge_list = [(u, v) for u, v, d in G.edges(data=True) if d["relation"] == rel]
    if edge_list:
        nx.draw_networkx_edges(
            G, pos, edgelist=edge_list,
            edge_color=rel_color_map[rel],
            arrows=True, arrowsize=10,
            alpha=0.7, width=1.2, ax=ax,
            connectionstyle="arc3,rad=0.05",
        )

# Colour nodes: sources (out-degree > 0, in-degree 0) vs concepts
source_nodes  = [n for n in G.nodes() if G.in_degree(n) == 0]
concept_nodes = [n for n in G.nodes() if G.in_degree(n) > 0]

nx.draw_networkx_nodes(G, pos, nodelist=source_nodes,
                       node_color="#2563EB", node_size=120, alpha=0.9, ax=ax)
nx.draw_networkx_nodes(G, pos, nodelist=concept_nodes,
                       node_color="#F59E0B", node_size=80, alpha=0.9, ax=ax)

# Labels only for high-degree nodes to avoid clutter
high_deg = {n for n, d in G.degree() if d >= 8}
labels = {n: n for n in high_deg}
nx.draw_networkx_labels(G, pos, labels=labels, font_size=6, ax=ax)

# Legend — relation types
legend_handles = [
    plt.Line2D([0], [0], color=rel_color_map[r], linewidth=2, label=r)
    for r in relation_types
]
legend_handles += [
    plt.scatter([], [], c="#2563EB", s=60, label="Cancer type (source)"),
    plt.scatter([], [], c="#F59E0B", s=60, label="Morphology concept"),
]
ax.legend(handles=legend_handles, loc="upper left", fontsize=8, framealpha=0.85)
ax.set_title("Knowledge Graph — Spring Layout (edge colour = relation type)", fontsize=14, fontweight="bold")
ax.axis("off")
plt.tight_layout()
out = os.path.join(STATS_DIR, "plot_graph_layout.png")
plt.savefig(out, dpi=150)
plt.close()
print(f"Saved: {out}")

print("\nAll plots saved to:", STATS_DIR)
