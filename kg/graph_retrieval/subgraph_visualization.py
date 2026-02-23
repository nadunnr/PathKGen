import pandas as pd
import networkx as nx
import gravis as gv

# Load Data
df = pd.read_csv("/home/nadun/wd/datasets/wsi_bench/custom_kg/relations.csv")

G = nx.Graph()

# Build Graph
for _, row in df.iterrows():
    src = row["reference_entity"]
    tgt = row["related_entity"]
    G.add_edge(src, tgt)

# Generate Gravis Visualization (Cleaned up Physics)
fig = gv.d3(
    G,
    graph_height=900,
    show_node_label=False,
    show_edge_label=False,   

    many_body_force_strength=-180,  
    links_force_distance=100,       
    collision_force_radius=50,      
    node_size_factor=1.5,          
    edge_size_factor=0.7,           
    
    zoom_factor=0.7                
) 


fig.export_html("graph.html")
print("Cleaned-up graph successfully exported to graph.html")