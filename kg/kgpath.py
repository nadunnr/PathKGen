import pandas as pd
import csv
from graphs import Neo4jQuery

# Neo4j connection configuration
NEO4J_URL = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "11111111"  # Change this to your Neo4j password

# CSV file paths (adjust these paths as needed)
NODES_CSV = "/home/nadun/wd/datasets/UMLS_sql/UMLS_DATA/FYP/KGs/nodes.csv"
RELATIONSHIPS_CSV = "/home/nadun/wd/datasets/UMLS_sql/UMLS_DATA/FYP/KGs/relationships.csv"

ALLOWED_RELATIONS = {
    "isa",
    "inverse_isa",
    "has_associated_morphology"
}

REL_TEMPLATES = {
    "isa": "{A} is a type of {B}.",
    "inverse_isa": "{B} is a type of {A}.",
    "has_associated_morphology": "{A} shows morphology of {B}."
}

def extract_local_subgraph(seed_cui, max_neighbors=500):
    """
    Extract local subgraph structure using Neo4j. Returns the neighbours and corresponding edges of the seed CUI
    Returns:
        - nodes: set of CUIs
        - edges: list of tuples (source_cui, target_cui, rel_type)
        - cui_to_name: dictionary mapping CUI to name
    Only queries the graph database, doesn't load full CSV files.
    """
    # Start neo4j query
    query = Neo4jQuery(NEO4J_URL, NEO4J_USER, NEO4J_PASSWORD)
    database = "dataset"
    nodes = set()
    edges = []
    cui_to_name = {}
    
    # Add seed node and get its name
    nodes.add(seed_cui)
    seed_entity = query.get_entity_by_cui(seed_cui, database=database)
    if seed_entity:
        cui_to_name[seed_cui] = seed_entity['name']
    
    # Get neighbors of the seed CUI
    neighbors = query.get_neighbors(seed_cui, direction="incoming", limit=max_neighbors, database=database)

    for neighbor in neighbors:
        # Get neighbor CUI
        neighbor_cui = neighbor['cui'] 

        # Get neighbor name
        neighbor_name = neighbor['name']

        # Get neighbor relation
        neighbor_relation = neighbor['rela']
        
        # Filter by allowed relations
        if neighbor_relation in ALLOWED_RELATIONS and neighbor_cui not in nodes:
            nodes.add(neighbor_cui)
            cui_to_name[neighbor_cui] = neighbor_name
            # Store edge as (source, target, relation_type)
            edges.append((seed_cui, neighbor_cui, neighbor_relation))
    
    query.close()
    print(f"Extracted subgraph: {len(nodes)} nodes, {len(edges)} edges")
    return nodes, edges, cui_to_name

def build_prompt(nodes, edges, cui_to_name):
    """
    Build the knowledge prompt from the local subgraph.
    """
    lines = []
    lines.append("Relevant medical knowledge:\n")
    
    # - Entity summaries -
    for cui in nodes:
        name = cui_to_name.get(cui, cui)
        lines.append(f"{name}")
    
    # - Relations -
    lines.append("\nKey relationships:\n")
    
    for src, dst, rel in edges:
        A = cui_to_name.get(src, src)
        B = cui_to_name.get(dst, dst)
        
        template = REL_TEMPLATES.get(rel)
        if template:
            lines.append(template.format(A=A, B=B))
    
    return "\n".join(lines)

def get_prompt(seed_cui):
    """
    Get the knowledge prompt for a given CUI.
    """
    # Step 1: Extract a local subgraph structure (filtered by allowed relations)
    nodes, edges, cui_to_name = extract_local_subgraph(seed_cui)

    # Step 2: Build and return the prompt
    return build_prompt(nodes, edges, cui_to_name)

# Main execution
if __name__ == "__main__":

    SEED_CUI = "C0678222"
    prompt = get_prompt(SEED_CUI)
    
    print("\n" + "="*80)
    print(prompt)
    print("="*80)

