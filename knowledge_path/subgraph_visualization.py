
import pandas as pd
import networkx as nx
import gravis as gv
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Import base directories
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths_config import DATA_DIR

df = pd.read_csv(os.path.join(DATA_DIR, "wsi_bench", "custom_kg", "relations.csv"))

G = nx.Graph()

# -------------------------
# Build graph
# -------------------------
reference_entities = set(df["reference_entity"].unique())

for _, row in df.iterrows():
    src = row["reference_entity"]
    tgt = row["related_entity"]
    rel = row["relation_name"]
    G.add_edge(src, tgt, label=rel)

center_node = ""

green_nodes = {}
red_nodes = {}

# -------------------------
# Node styling defaults
# -------------------------
for n in G.nodes():
    G.nodes[n]["color"] = "#d3d1d1"  # default grey
    G.nodes[n]["size"] = 15

# Highlight ALL reference entities with same color
for n in reference_entities:
    if n in G.nodes():
        G.nodes[n]["color"] = "#1f77b4"  # blue for all reference entities

# -------------------------
# Edge color mapping per relation type
relations = list(df["relation_name"].unique())

# Manually chosen 6 distinct colors (no blue)
palette_6 = [
    "#e41a1c",  # red
    "#4daf4a",  # green
    "#ff7f00",  # orange
    "#984ea3",  # purple
    "#a65628",  # brown
    "#f781bf",  # pink
]

# Safety check (optional)
if len(relations) > 6:
    raise ValueError(f"Expected max 6 relations, found {len(relations)}")

relation_color_map = {
    rel: palette_6[i] for i, rel in enumerate(relations)
}
print(relation_color_map)
breakpoint()
for u, v in G.edges():
    rel = G.edges[u, v].get("label", "")
    G.edges[u, v]["color"] = relation_color_map.get(rel, "#999999")
    G.edges[u, v]["width"] = 2
    # -------------------------
# Optional center node styling
# -------------------------
if center_node in G.nodes():
    G.nodes[center_node]["color"] = "#00008b"
    G.nodes[center_node]["size"] = 15

# -------------------------
# Green / red highlight logic (kept as-is)
# -------------------------
for node in green_nodes:
    if node in G.nodes():
        G.nodes[node]["color"] = "#2ca02c"
        if center_node and G.has_edge(center_node, node):
            G.edges[center_node, node]["color"] = "#2ca02c"
            G.edges[center_node, node]["width"] = 6

for node in red_nodes:
    if node in G.nodes():
        G.nodes[node]["color"] = "#d62728"
        if center_node and G.has_edge(center_node, node):
            G.edges[center_node, node]["color"] = "#d62728"
            G.edges[center_node, node]["width"] = 6

# -------------------------
# Draw graph
# -------------------------
# Gravis d3 supports graph_height (pixels) for canvas sizing.
GRAPH_HEIGHT = 1400

fig = gv.d3(
    G,
    graph_height=GRAPH_HEIGHT,
    edge_label_data_source="label",
    many_body_force_strength=-50,
    links_force_distance=35,
    links_force_strength=0.6,
    collision_force_radius=8
)

fig.display()

# -------------------------
# Save HTML
# -------------------------
output_path = os.path.join(
    DATA_DIR,
    "wsi_bench",
    "custom_kg",
    "subgraph_visualization.html"
)

if os.path.exists(output_path):
    os.remove(output_path)

fig.export_html(output_path)









# import pandas as pd
# import networkx as nx
# import gravis as gv
# import os
# import sys

# # Import base directories
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# from paths_config import DATA_DIR

# df = pd.read_csv(os.path.join(DATA_DIR, "wsi_bench", "custom_kg", "relations.csv"))

# G = nx.Graph()

# # Build Graph
# for _, row in df.iterrows():
#     src = row["reference_entity"]
#     tgt = row["related_entity"]
#     rel = row["relation_name"]
#     G.add_edge(src, tgt, label=rel)

# center_node = "Lymphoid Neoplasm Diffuse Large B-cell Lymphoma"

# # Nodes to highlight
# green_nodes = {"Hyperchromatic nuclei","Prominent nucleoli","Necrosis","Large Cell","Increased Mitotic Figures","Nuclear pleomorphism"}
# red_nodes = {}

# # Default styling (everything grey first)
# for n in G.nodes():
#     G.nodes[n]["color"] = "#d3d1d1"
#     G.nodes[n]["size"] = 15

# for u, v in G.edges():
#     G.edges[u, v]["color"] = "#cccccc"
#     G.edges[u, v]["width"] = 2

# # Highlight center node
# if center_node in G.nodes():
#     G.nodes[center_node]["color"] = "#00008b"  # dark blue
#     G.nodes[center_node]["size"] = 15

# # Highlight green nodes
# for node in green_nodes:
#     if node in G.nodes():
#         G.nodes[node]["color"] = "#2ca02c"  # green
#         G.nodes[node]["size"] = 15
#         # Only highlight edge if it exists
#         if G.has_edge(center_node, node):
#             G.edges[center_node, node]["color"] = "#2ca02c"
#             G.edges[center_node, node]["width"] = 6

# # Highlight red nodes (no edges to center)
# for node in red_nodes:
#     if node in G.nodes():
#         G.nodes[node]["color"] = "#d62728"  # red
#         G.nodes[node]["size"] = 15
#         if G.has_edge(center_node, node):
#             G.edges[center_node, node]["color"] = "#d62728"
#             G.edges[center_node, node]["width"] = 6

# # Draw graph
# fig = gv.d3(
#     G,
#     graph_height=900,
#     edge_label_data_source="label",
#     many_body_force_strength=-50,
#     links_force_distance=35,
#     links_force_strength=0.6,
#     collision_force_radius=8
# )


# fig.display()
# #save the figure as an HTML file
# output_path = os.path.join(
#     DATA_DIR,
#     "wsi_bench",
#     "custom_kg",
#     "subgraph_visualization.html"
# )

# if os.path.exists(output_path):
#     os.remove(output_path)

# fig.export_html(output_path)


# # import pandas as pd
# # import networkx as nx
# # import gravis as gv
# # import os
# # import sys
# # import matplotlib.pyplot as plt
# # import matplotlib.colors as mcolors

# # # Import base directories
# # sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# # from paths_config import DATA_DIR

# # df = pd.read_csv(os.path.join(DATA_DIR, "wsi_bench", "custom_kg", "relations.csv"))

# # G = nx.Graph()

# # # -------------------------
# # # Build graph
# # # -------------------------
# # reference_entities = set(df["reference_entity"].unique())
# # child_entities = set(df["related_entity"].unique())

# # for _, row in df.iterrows():
# #     src = row["reference_entity"]
# #     tgt = row["related_entity"]
# #     rel = row["relation_name"]
# #     G.add_edge(src, tgt, label=rel)

# # center_node = ""

# # green_nodes = {}
# # red_nodes = {}

# # # -------------------------
# # # FONT STYLING (2 types)
# # # -------------------------
# # histology_font = "Georgia, serif"
# # feature_font = "Arial, sans-serif"

# # # -------------------------
# # # Node styling defaults + fonts
# # # -------------------------
# # for n in G.nodes():
# #     G.nodes[n]["color"] = "#d3d1d1"  # default grey
# #     G.nodes[n]["size"] = 15

# #     # Histology (reference entities)
# #     if n in reference_entities:
# #         G.nodes[n]["color"] = "#1f77b4"
# #         G.nodes[n]["label"] = f"<span style='font-family:{histology_font}; font-weight:600'>{n}</span>"

# #     # Feature / child nodes
# #     elif n in child_entities:
# #         G.nodes[n]["label"] = f"<span style='font-family:{feature_font}'>{n}</span>"

# # # -------------------------
# # # Edge color mapping (6 relations, NO BLUE)
# # # -------------------------
# # relations = list(df["relation_name"].unique())

# # palette_6 = [
# #     "#e41a1c",  # red
# #     "#4daf4a",  # green
# #     "#ff7f00",  # orange
# #     "#984ea3",  # purple
# #     "#a65628",  # brown
# #     "#f781bf",  # pink
# # ]

# # if len(relations) > 6:
# #     raise ValueError(f"Expected max 6 relations, found {len(relations)}")

# # relation_color_map = {
# #     rel: palette_6[i] for i, rel in enumerate(relations)
# # }

# # for u, v in G.edges():
# #     rel = G.edges[u, v].get("label", "")
# #     G.edges[u, v]["color"] = relation_color_map.get(rel, "#999999")
# #     G.edges[u, v]["width"] = 2

# # # -------------------------
# # # Optional center node styling
# # # -------------------------
# # if center_node in G.nodes():
# #     G.nodes[center_node]["color"] = "#00008b"
# #     G.nodes[center_node]["size"] = 15

# # # -------------------------
# # # Green / red highlight logic
# # # -------------------------
# # for node in green_nodes:
# #     if node in G.nodes():
# #         G.nodes[node]["color"] = "#2ca02c"
# #         if center_node and G.has_edge(center_node, node):
# #             G.edges[center_node, node]["color"] = "#2ca02c"
# #             G.edges[center_node, node]["width"] = 6

# # for node in red_nodes:
# #     if node in G.nodes():
# #         G.nodes[node]["color"] = "#d62728"
# #         if center_node and G.has_edge(center_node, node):
# #             G.edges[center_node, node]["color"] = "#d62728"
# #             G.edges[center_node, node]["width"] = 6

# # # -------------------------
# # # Draw graph
# # # -------------------------
# # fig = gv.d3(
# #     G,
# #     graph_height=900,
# #     edge_label_data_source="label",
# #     many_body_force_strength=-50,
# #     links_force_distance=35,
# #     links_force_strength=0.6,
# #     collision_force_radius=8
# # )

# # fig.display()

# # # -------------------------
# # # Save HTML
# # # -------------------------
# # output_path = os.path.join(
# #     DATA_DIR,
# #     "wsi_bench",
# #     "custom_kg",
# #     "subgraph_visualization.html"
# # )

# # if os.path.exists(output_path):
# #     os.remove(output_path)

# # fig.export_html(output_path)












############All gray one 
# import pandas as pd
# import networkx as nx
# import gravis as gv
# import os
# import sys

# # Import base directories
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# from paths_config import DATA_DIR

# df = pd.read_csv(os.path.join(DATA_DIR, "wsi_bench", "custom_kg", "relations.csv"))

# G = nx.Graph()

# # Build Graph
# for _, row in df.iterrows():
#     src = row["reference_entity"]
#     tgt = row["related_entity"]
#     rel = row["relation_name"]
#     G.add_edge(src, tgt, label=rel)

# center_node = "Lymphoid Neoplasm Diffuse Large B-cell Lymphoma"

# # Nodes to highlight
# green_nodes = {"Hyperchromatic nuclei","Prominent nucleoli","Necrosis","Large Cell","Increased Mitotic Figures","Nuclear pleomorphism"}
# red_nodes = {}

# # Default styling (everything grey first)
# for n in G.nodes():
#     G.nodes[n]["color"] = "#d3d1d1"
#     G.nodes[n]["size"] = 15

# for u, v in G.edges():
#     G.edges[u, v]["color"] = "#cccccc"
#     G.edges[u, v]["width"] = 2

# # Highlight center node
# if center_node in G.nodes():
#     G.nodes[center_node]["color"] = "#d3d1d1"  # dark blue
#     G.nodes[center_node]["size"] = 15

# # Highlight green nodes
# for node in green_nodes:
#     if node in G.nodes():
#         G.nodes[node]["color"] = "#d3d1d1"  # green
#         G.nodes[node]["size"] = 15
#         # Only highlight edge if it exists
#         if G.has_edge(center_node, node):
#             G.edges[center_node, node]["color"] = "#d3d1d1"
#             G.edges[center_node, node]["width"] = 6

# # Highlight red nodes (no edges to center)
# for node in red_nodes:
#     if node in G.nodes():
#         G.nodes[node]["color"] = "#d3d1d1"  # red
#         G.nodes[node]["size"] = 15
#         if G.has_edge(center_node, node):
#             G.edges[center_node, node]["color"] = "#d3d1d1"
#             G.edges[center_node, node]["width"] = 6

# # Draw graph
# fig = gv.d3(
#     G,
#     graph_height=1900,
#     edge_label_data_source="label",
#     many_body_force_strength=-50,
#     links_force_distance=35,
#     links_force_strength=0.6,
#     collision_force_radius=8
# )