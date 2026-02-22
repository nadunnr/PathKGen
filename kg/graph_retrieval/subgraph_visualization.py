from neo4j import GraphDatabase
from pyvis.network import Network
import argparse

# --- CONFIG ---
URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "11111111"

cuis_to_search = [
      "C0027540",
      "C0521157",
      "C1334022",
      "C1707265",
      "C1335070",
      "C1518437",
      "C1332352",
      "C1707602",
      "C1334428",
      "C0032219"
    ]


# Define the relationships relevant to histology/pathology
HISTOLOGY_RELS = [
    "disease_has_abnormal_cell",
    "disease_has_associated_anatomic_site",
    "disease_has_finding",
    "disease_has_normal_cell_origin",
    "disease_has_normal_tissue_origin",
    "disease_has_primary_anatomic_site",
    "has_associated_morphology",
    "has_finding_site",
    "disease_may_have_finding"
]

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

def get_histology_subgraph(cui, hops):
    rel_filter = "|".join(HISTOLOGY_RELS)
    
    # 1. Cleaner Query: Unwinds relationships and grabs their endpoints
    query = f"""
    MATCH (c {{cui: $cui}})
    MATCH path = (c)-[:{rel_filter}*1..{hops}]-(n)
    UNWIND relationships(path) AS rel
    WITH DISTINCT rel
    RETURN startNode(rel) AS start_n, endNode(rel) AS end_n, type(rel) AS rel_type
    LIMIT 1000
    """

    nodes = {}
    edges = []

    with driver.session() as session:
        result = session.run(query, cui=cui)
        for record in result:
            # 2. Extract the variables we explicitly RETURNed in the query
            start_n = record["start_n"]
            end_n = record["end_n"]
            rel_type = record["rel_type"]
            
            # Store start node data
            if start_n["cui"] not in nodes:
                nodes[start_n["cui"]] = {
                    "label": start_n.get("name", start_n["cui"]),
                    "type": list(start_n.labels)[0] if start_n.labels else "Concept"
                }
                
            # Store end node data
            if end_n["cui"] not in nodes:
                nodes[end_n["cui"]] = {
                    "label": end_n.get("name", end_n["cui"]),
                    "type": list(end_n.labels)[0] if end_n.labels else "Concept"
                }
                
            # Store the relationship connecting them
            edges.append({
                "from": start_n["cui"],
                "to": end_n["cui"],
                "label": rel_type
            })

    # Optional: Final deduplication just to ensure pyvis doesn't double-draw lines
    unique_edges = { (e["from"], e["to"], e["label"]): e for e in edges }.values()

    return nodes, list(unique_edges)

def visualize_styled(nodes, edges, center_cui):
    net = Network(height="800px", width="100%", directed=True, bgcolor="#ffffff", font_color="black")
    
    # Simple color mapping for clarity
    for cui, data in nodes.items():
        color = "#97c2fc" # Default blue
        size = 20
        
        if cui == center_cui:
            color = "#ff4d4d" # Deep red for center
            size = 35
        
        net.add_node(cui, label=data["label"], color=color, size=size, title=f"Type: {data['type']}")

    for edge in edges:
        # Style edges based on the relationship type
        color = "#cccccc"
        if "anatomic_site" in edge["label"]: color = "#2ecc71" # Green for anatomy
        if "abnormal_cell" in edge["label"]: color = "#e67e22" # Orange for cells
        
        net.add_edge(edge["from"], edge["to"], label=edge["label"], color=color, width=2)

    net.force_atlas_2based() # Better layout for medical clusters
    net.show("pathology_focus_graph.html", notebook=False)
    print("Graph generated: pathology_focus_graph.html")

def available_cuis(nodes):
    print("Available CUIs in the graph:")
    for cui, data in nodes.items():
        if cui in cuis_to_search:
            cui_found = cui
            print(f"{cui}: {data['label']} ({data['type']})")

        #highlight and make it thicken the found_cui nodes and edge



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cui", default="C0853879", help="Input CUI")
    parser.add_argument("--hops", type=int, default=1, help="Hops (1 is usually enough for histology attributes)")



    args = parser.parse_args()

    print(f"Generating histology graph for {args.cui}...")
    nodes, edges = get_histology_subgraph(args.cui, args.hops)

    available_cuis(nodes)
    visualize_styled(nodes, edges, args.cui)

