"""
kg_plot.py  -  Visualise KG statistics saved by kg_stat.py.
"""

import json
import os
import sys

# Import base directories
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from paths_config import DATA_DIR
import networkx as nx
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# ── Font Configuration ───────────────────────────────────────────────────────
# Use CMU Serif / Computer Modern Roman fonts
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "axes.unicode_minus": False,
    "mathtext.fontset": "cm",
})

# ── Paths ────────────────────────────────────────────────────────────────────
STATS_DIR    = os.path.join(DATA_DIR, "wsi_bench", "kg_data", "stats")
CSV_PATH     = os.path.join(DATA_DIR, "wsi_bench", "custom_kg", "relations.csv")

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

# ── Custom Relation-Type Colors ───────────────────────────────────────────────
# Map each relation name to a hex color string.
# Any relation NOT listed here will fall back to the tab10 palette automatically.
RELATION_COLORS: dict[str, str] = {
    "has_cytological_feature":    "#c7c5fe",
    "has_nuclear_feature":         "#d9f2d0",
    "found_in":          "#dceaf7",
    "has_background_feature":    "#f2cfee",
    "has_growth_pattern":      "#f6c6ad",
    "has_invasion_pattern":     "#fbffa9",
}

# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 - Relation type distribution (horizontal bar)
# ─────────────────────────────────────────────────────────────────────────────
rel_counts = basic["relation_type_counts"]
labels = list(rel_counts.keys())
values = list(rel_counts.values())

fig, ax = plt.subplots(figsize=(10, 5))
colors = [
    RELATION_COLORS.get(lbl, PALETTE[i % len(PALETTE)])
    for i, lbl in enumerate(labels)
]

# height=1.0 removes gap, edgecolor="black" adds outline
bars = ax.barh(labels, values, color=colors, edgecolor="black", height=1.0, linewidth=1.5)

# Add values at the end of bars
ax.bar_label(bars, padding=8, fontsize=28, fontweight="bold")

# Add relation type names INSIDE the bars
for i, (label, value) in enumerate(zip(labels, values)):
    ax.text(2, i, label.replace("_", " "), color="black", va="center", ha="left", 
            fontsize=28, fontweight="bold", path_effects=None) # Optional: add shadow if needed

# Aesthetics
ax.invert_yaxis()
ax.set_yticks([]) # Hide y-axis labels (moved inside)
ax.xaxis.set_visible(False) # Hide x-axis
ax.spines[["top", "right", "bottom", "left"]].set_visible(False)

plt.tight_layout()
out = os.path.join(STATS_DIR, "plot_relation_types.pdf")
plt.savefig(out)
plt.close()
print(f"Saved: {out}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 - Top-20 source nodes by out-degree
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
out = os.path.join(STATS_DIR, "plot_top_out_degree.pdf")
plt.savefig(out)
plt.close()
print(f"Saved: {out}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 - Top-20 concept nodes by in-degree
# ─────────────────────────────────────────────────────────────────────────────
in_deg = sorted(G.in_degree(), key=lambda x: x[1], reverse=True)[:15]
nodes_id, degs_id = zip(*in_deg)

# Wider figure to fit long concept names
fig, ax = plt.subplots(figsize=(14, 8))
y_pos = np.arange(len(nodes_id))

# Horizontal bars, touching (height=1.0), black outline
bars = ax.barh(y_pos, degs_id, color="gold", edgecolor="black", height=1.0, linewidth=1.2)

# Add relation type names INSIDE the bars
for i, node in enumerate(nodes_id):
    ax.text(0.3, i, node, color="black", va="center", ha="left", 
            fontsize=22, fontweight="bold")

# Add values at the end of bars
ax.bar_label(bars, padding=8, fontsize=22, fontweight="bold")

# Aesthetics
ax.invert_yaxis()
ax.set_yticks([]) # Hide y-axis labels (moved inside)
ax.xaxis.set_visible(False) # Hide x-axis (numbers at end of bars)
ax.spines[["top", "right", "bottom", "left"]].set_visible(False)

plt.tight_layout()
out = os.path.join(STATS_DIR, "plot_top_in_degree.pdf")
plt.savefig(out)
plt.close()
print(f"Saved: {out}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 - Degree distribution (horizontal bars, undirected)
# ─────────────────────────────────────────────────────────────────────────────
from collections import Counter
G_undir = G.to_undirected()
degrees = [d for _, d in G_undir.degree()]
counts = Counter(degrees)
sorted_counts = sorted(counts.items())
unique_degs, node_freqs = zip(*sorted_counts)

fig, ax = plt.subplots(figsize=(10, 7))
y_pos = np.arange(len(unique_degs))

# Horizontal bars, touching (height=1.0), black outline
bars = ax.barh(y_pos, node_freqs, color="#c7c5fe", edgecolor="black", height=1.0, linewidth=1.2)

# Values at the end of bars
ax.bar_label(bars, padding=8, fontsize=22, fontweight="bold")

# Aesthetics
ax.invert_yaxis()

# Clean Y-axis with discrete degree numbers
ax.set_yticks(y_pos)
ax.set_yticklabels(unique_degs, fontsize=22, fontweight="bold")
ax.set_ylabel("Degree", fontsize=22, fontweight="bold")

# Limit the y-axis spine to just the extent of the bars
ax.set_ylim(y_pos[-1] + 0.5, y_pos[0] - 0.5)

ax.xaxis.set_visible(False) # Hide x-axis
ax.spines[["top", "right", "bottom"]].set_visible(False)
ax.spines["left"].set_linewidth(1.5)

plt.tight_layout()
out = os.path.join(STATS_DIR, "plot_degree_distribution.pdf")
plt.savefig(out)
plt.close()
print(f"Saved: {out}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 5 - Graph layout (spring, colour-coded by relation type)
# ─────────────────────────────────────────────────────────────────────────────
relation_types = list(rel_counts.keys())
rel_color_map  = {
    r: RELATION_COLORS.get(r, PALETTE[i % len(PALETTE)])
    for i, r in enumerate(relation_types)
}

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

# Legend - relation types
legend_handles = [
    plt.Line2D([0], [0], color=rel_color_map[r], linewidth=2, label=r)
    for r in relation_types
]
legend_handles += [
    plt.scatter([], [], c="#2563EB", s=60, label="Cancer type (source)"),
    plt.scatter([], [], c="#F59E0B", s=60, label="Morphology concept"),
]
ax.legend(handles=legend_handles, loc="upper left", fontsize=8, framealpha=0.85)
ax.set_title("Knowledge Graph - Spring Layout (edge colour = relation type)", fontsize=14, fontweight="bold")
ax.axis("off")
plt.tight_layout()
out = os.path.join(STATS_DIR, "plot_graph_layout.pdf")
plt.savefig(out)
plt.close()
print(f"Saved: {out}")

print("\nAll plots saved to:", STATS_DIR)
