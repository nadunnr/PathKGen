# import pandas as pd
# import networkx as nx
# import gravis as gv


# df = pd.read_csv("/home/nadun/wd/datasets/wsi_bench/custom_kg/relations.csv")

# G = nx.Graph()


# # Build Graph

# for _, row in df.iterrows():
#     src = row["reference_entity"]
#     tgt = row["related_entity"]
#     rel = row["relation_name"]
#     G.add_edge(src, tgt, label=rel)

# print(G)


# # Extract 1-hop graph

# center_node = "Head and Neck Squamous Cell Carcinoma" #add the histology you want to highlight here
# subG = nx.ego_graph(G, center_node, radius=0.5)

# ego_nodes = set(subG.nodes())
# ego_edges = set(subG.edges())

# #styling
# for n in G.nodes():
#     if n in ego_nodes:
#         G.nodes[n]["color"] = "#d62728"  # red
#         G.nodes[n]["size"] = 20
#     else:
#         G.nodes[n]["color"] = "#b5b3b3" # light grey
#         G.nodes[n]["size"] = 20


# if center_node in G.nodes():
#     G.nodes[center_node]["color"] = "#8b0000"
#     G.nodes[center_node]["size"] = 20


# for u, v in G.edges():
#     if (u, v) in ego_edges or (v, u) in ego_edges:
#         G.edges[u, v]["color"] = "#d62728"
#         G.edges[u, v]["width"] = 3.5
#     else:
#         G.edges[u, v]["color"] = "#5a5e5b"
#         G.edges[u, v]["width"] = 0.5

# fig = gv.d3(
#     G,
#     graph_height=900,
#     edge_label_data_source="label",

#     many_body_force_strength = -50,
#     links_force_distance = 35,
#     links_force_strength = 0.6,
#     collision_force_radius = 8
# )



# fig.display()
# # fig.export_html("test.html")


import pandas as pd
import networkx as nx
import gravis as gv

df = pd.read_csv("/home/nadun/wd/datasets/wsi_bench/custom_kg/relations.csv")

G = nx.Graph()

# Build Graph
for _, row in df.iterrows():
    src = row["reference_entity"]
    tgt = row["related_entity"]
    rel = row["relation_name"]
    G.add_edge(src, tgt, label=rel)

center_node = "Head and Neck Squamous Cell Carcinoma"

# Nodes to highlight
green_nodes = {"Keratinization", "intercellular bridge", "Sheet-Like Growth Pattern"}
red_nodes = {"Lymph-vascular invasion", "Perineural Invasion", "Necrosis"}

# Default styling (everything grey first)
for n in G.nodes():
    G.nodes[n]["color"] = "#b5b3b3"
    G.nodes[n]["size"] = 15

for u, v in G.edges():
    G.edges[u, v]["color"] = "#cccccc"
    G.edges[u, v]["width"] = 2

# Highlight center node
if center_node in G.nodes():
    G.nodes[center_node]["color"] = "#00008b"  # dark blue
    G.nodes[center_node]["size"] = 15

# Highlight green nodes
for node in green_nodes:
    if node in G.nodes():
        G.nodes[node]["color"] = "#2ca02c"  # green
        G.nodes[node]["size"] = 15
        # Only highlight edge if it exists
        if G.has_edge(center_node, node):
            G.edges[center_node, node]["color"] = "#2ca02c"
            G.edges[center_node, node]["width"] = 6

# Highlight red nodes (no edges to center)
for node in red_nodes:
    if node in G.nodes():
        G.nodes[node]["color"] = "#d62728"  # red
        G.nodes[node]["size"] = 15
        if G.has_edge(center_node, node):
            G.edges[center_node, node]["color"] = "#d62728"
            G.edges[center_node, node]["width"] = 6

# Draw graph
fig = gv.d3(
    G,
    graph_height=900,
    edge_label_data_source="label",
    many_body_force_strength=-50,
    links_force_distance=35,
    links_force_strength=0.6,
    collision_force_radius=8
)

fig.display()