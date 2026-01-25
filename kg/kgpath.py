import pandas as pd

MRCONSO_PATH = "/home/nadun/wd/datasets/UMLS_sql/2025AB/META/MRCONSO.RRF"
MRSTY_PATH = "/home/nadun/wd/datasets/UMLS_sql/2025AB/META/MRSTY.RRF"
MRDEF_PATH = "/home/nadun/wd/datasets/UMLS_sql/2025AB/META/MRDEF.RRF"
MRREL_PATH = "/home/nadun/wd/datasets/UMLS_sql/2025AB/META/MRREL.RRF"
MAX_NEIGHBORS = 12

ALLOWED_RELATIONS = {
    "ISA",
    "PART_OF",
    "ASSOCIATED_WITH",
    "HAS_LOCATION",
    "HAS_MORPHOLOGY",
    "CAUSES",
    "MANIFESTATION_OF"
}

ALLOWED_SEMTYPES = {
    "Neoplastic Process",
    "Disease or Syndrome",
    "Pathologic Function",
    "Tissue",
    "Cell",
    "Anatomical Structure"
}

def extract_subgraph_structure(seed_cui):
    """
    Extract subgraph structure by reading only the relations file.
    Returns the set of CUIs needed and the edges.
    """
    print(f"Extracting subgraph for seed CUI: {seed_cui}")
    
    neighbors = []
    
    # Read relations file line by line to find neighbors of seed_cui
    with open(MRREL_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) < 5:
                continue
            
            cui1 = parts[0]
            rel = parts[3]
            cui2 = parts[4]
            
            # Only process if this is our seed CUI and relation is allowed
            if cui1 == seed_cui and rel in ALLOWED_RELATIONS:
                neighbors.append((cui2, rel))
                
                # Stop if we have enough neighbors
                if len(neighbors) >= MAX_NEIGHBORS:
                    break
    
    # Build node set and edges
    nodes = {seed_cui}
    edges = []
    
    for dst, rel in neighbors:
        nodes.add(dst)
        edges.append((seed_cui, dst, rel))
    
    print(f"Found {len(nodes)} nodes in subgraph")
    return nodes, edges


def load_concept_names_for_cuis(cui_set):
    """
    Load concept names only for the specified CUIs.
    """
    print(f"Loading concept names for {len(cui_set)} CUIs...")
    cui_to_name = {}
    
    with open(MRCONSO_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) < 15:
                continue
            
            cui = parts[0]
            
            # Only process if this CUI is in our set
            if cui not in cui_set:
                continue
            
            # Check if it's English preferred term
            lat = parts[1]
            ispref = parts[6]
            name = parts[14]
            
            if lat == "ENG" and ispref == "Y":
                # Only store if we haven't seen this CUI yet
                if cui not in cui_to_name:
                    cui_to_name[cui] = name
            
            # Early exit if we've found all CUIs
            if len(cui_to_name) == len(cui_set):
                break
    
    return cui_to_name


def load_semantic_types_for_cuis(cui_set):
    """
    Load semantic types only for the specified CUIs.
    """
    print(f"Loading semantic types for {len(cui_set)} CUIs...")
    cui_to_semtype = {}
    
    with open(MRSTY_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) < 4:
                continue
            
            cui = parts[0]
            sty = parts[3]
            
            # Only process if this CUI is in our set
            if cui in cui_set:
                if cui not in cui_to_semtype:
                    cui_to_semtype[cui] = []
                cui_to_semtype[cui].append(sty)
    
    return cui_to_semtype


def load_definitions_for_cuis(cui_set):
    """
    Load definitions only for the specified CUIs.
    """
    print(f"Loading definitions for {len(cui_set)} CUIs...")
    cui_to_def = {}
    
    with open(MRDEF_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) < 6:
                continue
            
            cui = parts[0]
            definition = parts[5]
            
            # Only process if this CUI is in our set and we haven't seen it yet
            if cui in cui_set and cui not in cui_to_def:
                cui_to_def[cui] = definition
            
            # Early exit if we've found all CUIs
            if len(cui_to_def) == len(cui_set):
                break
    
    return cui_to_def


def filter_nodes(nodes, edges, cui_to_semtype, seed_cui):
    """
    Filter nodes based on semantic types.
    """
    filtered = set()
    
    for cui in nodes:
        semtypes = set(cui_to_semtype.get(cui, []))
        if semtypes & ALLOWED_SEMTYPES or cui == seed_cui:
            filtered.add(cui)
    
    filtered_edges = [
        e for e in edges
        if e[0] in filtered and e[1] in filtered
    ]
    
    print(f"Filtered to {len(filtered)} nodes after semantic type filtering")
    return filtered, filtered_edges


REL_TEMPLATES = {
    "ISA": "{A} is a type of {B}.",
    "PART_OF": "{A} is part of {B}.",
    "ASSOCIATED_WITH": "{A} is associated with {B}.",
    "HAS_LOCATION": "{A} occurs in {B}.",
    "HAS_MORPHOLOGY": "{A} shows morphology of {B}.",
    "CAUSES": "{A} can cause {B}.",
    "MANIFESTATION_OF": "{A} is a manifestation of {B}."
}


def build_prompt(nodes, edges, cui_to_name, cui_to_semtype, cui_to_def):
    """
    Build the knowledge prompt from the subgraph.
    """
    lines = []
    lines.append("Relevant medical knowledge:\n")
    
    # - Entity summaries -
    for cui in nodes:
        name = cui_to_name.get(cui, cui)
        definition = cui_to_def.get(cui, "")
        semtypes = ", ".join(cui_to_semtype.get(cui, []))
        
        if definition:
            lines.append(f"{name}: {definition}")
        else:
            lines.append(f"{name} ({semtypes})")
    
    # - Relations -
    lines.append("\nKey relationships:\n")
    
    for src, dst, rel in edges:
        A = cui_to_name.get(src, src)
        B = cui_to_name.get(dst, dst)
        
        template = REL_TEMPLATES.get(rel)
        if template:
            lines.append(template.format(A=A, B=B))
    
    return "\n".join(lines)


# Main execution
SEED_CUI = "C0206692"   # change to yours

# Step 1: Extract subgraph structure (only reads relations file)
nodes, edges = extract_subgraph_structure(SEED_CUI)

# Step 2: Load metadata only for the CUIs in the subgraph
cui_to_name = load_concept_names_for_cuis(nodes)
cui_to_semtype = load_semantic_types_for_cuis(nodes)
cui_to_def = load_definitions_for_cuis(nodes)

# Step 3: Filter nodes based on semantic types
nodes, edges = filter_nodes(nodes, edges, cui_to_semtype, SEED_CUI)

# Step 4: Build and print the prompt
prompt = build_prompt(nodes, edges, cui_to_name, cui_to_semtype, cui_to_def)

print("\n" + "="*80)
print(prompt)
print("="*80)
