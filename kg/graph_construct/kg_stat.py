import pandas as pd
import json
import os
import networkx as nx
from collections import Counter

# ── Paths ────────────────────────────────────────────────────────────────────
graph_incidence_list = "/home/nadun/wd/datasets/wsi_bench/custom_kg/relations.csv"
stats_output_path    = "/home/nadun/wd/datasets/wsi_bench/kg_data/stats"
os.makedirs(stats_output_path, exist_ok=True)

# ── Load CSV ─────────────────────────────────────────────────────────────────
df = pd.read_csv(graph_incidence_list)
df.columns = df.columns.str.strip()
df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

print(f"Loaded {len(df)} edges from: {graph_incidence_list}")
print(f"Columns: {list(df.columns)}\n")

# ── Build directed + undirected graphs ───────────────────────────────────────
G_dir   = nx.DiGraph()
G_undir = nx.Graph()

for _, row in df.iterrows():
    src, dst, rel = row["reference_entity"], row["related_entity"], row["relation_name"]
    G_dir.add_edge(src, dst, relation=rel)
    G_undir.add_edge(src, dst, relation=rel)

# ── Basic graph stats ─────────────────────────────────────────────────────────
num_nodes     = G_dir.number_of_nodes()
num_edges     = G_dir.number_of_edges()
is_dag        = nx.is_directed_acyclic_graph(G_dir)
density       = nx.density(G_dir)
avg_in_deg    = sum(d for _, d in G_dir.in_degree())  / num_nodes
avg_out_deg   = sum(d for _, d in G_dir.out_degree()) / num_nodes

top_out_nodes = sorted(G_dir.out_degree(), key=lambda x: x[1], reverse=True)[:10]
top_in_nodes  = sorted(G_dir.in_degree(),  key=lambda x: x[1], reverse=True)[:10]

relation_counts = Counter(df["relation_name"])

print("=" * 60)
print("  BASIC GRAPH STATISTICS")
print("=" * 60)
print(f"  Nodes            : {num_nodes}")
print(f"  Edges            : {num_edges}")
print(f"  Is DAG           : {is_dag}")
print(f"  Density          : {density:.6f}")
print(f"  Avg out-degree   : {avg_out_deg:.3f}")
print(f"  Avg in-degree    : {avg_in_deg:.3f}")
print()

print("  Relation type counts:")
for rel, cnt in relation_counts.most_common():
    print(f"    {rel:<40} {cnt}")
print()

print("  Top-10 nodes by out-degree (source nodes):")
for node, deg in top_out_nodes:
    print(f"    [{deg:>3}]  {node}")
print()

print("  Top-10 nodes by in-degree (target / concept nodes):")
for node, deg in top_in_nodes:
    print(f"    [{deg:>3}]  {node}")
print()

# ── Connected components (undirected) ─────────────────────────────────────────
components = list(nx.connected_components(G_undir))
components_sorted = sorted(components, key=len, reverse=True)

print("=" * 60)
print("  CONNECTED COMPONENTS (undirected)")
print("=" * 60)
print(f"  Number of components : {len(components_sorted)}")
print()

component_summaries = []
for i, comp in enumerate(components_sorted):
    subg      = G_undir.subgraph(comp)
    n_nodes   = len(comp)
    n_edges   = subg.number_of_edges()
    node_list = sorted(comp)

    print(f"  Component {i+1:>3}  ({n_nodes} nodes, {n_edges} edges)")
    # Print first 10 nodes as preview
    preview = node_list[:10]
    suffix  = f"  ... +{n_nodes-10} more" if n_nodes > 10 else ""
    print(f"    Nodes: {preview}{suffix}")
    print()

    component_summaries.append({
        "component_id" : i + 1,
        "num_nodes"    : n_nodes,
        "num_edges"    : n_edges,
        "nodes"        : node_list,
    })

# ── Weakly connected components (directed) ────────────────────────────────────
wcc = list(nx.weakly_connected_components(G_dir))
print(f"  Weakly connected components (directed) : {len(wcc)}")
scc = list(nx.strongly_connected_components(G_dir))
print(f"  Strongly connected components          : {len(scc)}")
print(f"  Largest SCC size                       : {max(len(c) for c in scc)}")
print()

# ── Save stat files ───────────────────────────────────────────────────────────
# 1. Basic stats JSON
basic_stats = {
    "num_nodes"           : num_nodes,
    "num_edges"           : num_edges,
    "is_dag"              : is_dag,
    "density"             : density,
    "avg_out_degree"      : avg_out_deg,
    "avg_in_degree"       : avg_in_deg,
    "num_weakly_connected_components"    : len(wcc),
    "num_strongly_connected_components" : len(scc),
    "largest_scc_size"    : max(len(c) for c in scc),
    "relation_type_counts": dict(relation_counts.most_common()),
    "top10_out_degree"    : [{"node": n, "degree": d} for n, d in top_out_nodes],
    "top10_in_degree"     : [{"node": n, "degree": d} for n, d in top_in_nodes],
}
basic_stats_path = os.path.join(stats_output_path, "basic_stats.json")
with open(basic_stats_path, "w") as f:
    json.dump(basic_stats, f, indent=2)
print(f"  Saved: {basic_stats_path}")

# 2. Connected components JSON
components_path = os.path.join(stats_output_path, "connected_components.json")
with open(components_path, "w") as f:
    json.dump(component_summaries, f, indent=2)
print(f"  Saved: {components_path}")

# 3. Incidence list as JSON (for downstream use)
incidence_records = df.to_dict(orient="records")
incidence_path = os.path.join(stats_output_path, "incidence_list.json")
with open(incidence_path, "w") as f:
    json.dump(incidence_records, f, indent=2)
print(f"  Saved: {incidence_path}")

print("\nDone.")
