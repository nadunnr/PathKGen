Histopathology Knowledge Graph Construction (E2 Pipeline)
This repository documents the approaches for constructing meaningful subgraphs from Whole Slide Image (WSI) concept extractions. The goal is to transform a "bag of concepts" (UMLS CUIs) extracted by the concept head into structured, medically relevant knowledge graphs to guide the Vision-Language Model (VLM).

We utilize Neo4j as our graph database and the Neo4j Graph Data Science (GDS) library for advanced pathfinding algorithms.

1. Graph Filtering: Semantic Pruning
Before applying any connection logic, we must reduce the noise in the massive UMLS metathesaurus. We filter nodes and relationships based on medical relevance to histopathology.

1.1. Allowed Semantic Types (TUIs)
We only project nodes belonging to these Semantic Types into our working graph.

Anatomy (Primary Sites): T023 (Body Part/Organ), T024 (Tissue), T029 (Body Location).

Pathology (Histology & Variants): T191 (Neoplastic Process), T047 (Disease or Syndrome), T019 (Congenital Abnormality).

Morphology (Features): T033 (Finding), T044 (Molecular Function), T045 (Function).

1.2. Allowed Semantic Relations
We prioritize "definitional" and "locational" edges over generic ones.

Hierarchical: isa, CHD (Child of).

Locational: finding_site_of, has_location.

Associative: associated_morphology_of, manifestation_of.

(Note: Generic relations like associated_with are assigned high cost or filtered out).

1.3. Neo4j GDS Projection
Create an in-memory graph projection for the algorithms to run on.

Cypher
CALL gds.graph.project(
  'histo-graph-filtered',
  {
    Concept: {
      label: 'Concept',
      properties: 'confidence_score' // Optional: if available from concept head
    }
  },
  {
    RELATION: {
      type: 'RELATED_TO',
      orientation: 'UNDIRECTED',
      properties: 'weight' // 0.1 for hierarchical, 1.0 for generic
    }
  },
  {
    nodePredicate: 'n.tui IN ["T191", "T023", "T033", "T024"]'
  }
)
## 2. Approach A: Medical Ruled Graph (The "Anchor" Method)
This heuristic approach forces the graph into a predictable "Star" or "Tree" topology based on biological hierarchy. It guarantees interpretability but may miss latent connections.

### Logic
1. **Identify Anchor**: Select the node with TUI = T191 (Neoplastic Process) and the highest confidence score.
2. **Vertical Expansion**: Query 1-hop up (isa) to get the parent class (e.g., Adenocarcinoma → Carcinoma).
3. **Horizontal Expansion**: Query specifically for location_of (Site) and associated_morphology_of (Features) relative to the Anchor.

### Implementation Pseudocode

```python
def build_medical_ruled_graph(cui_list, grade=None, site=None):
    # Step 1: Identify Anchor
    anchor = None
    for cui in cui_list:
        tui = get_semantic_type(cui)
        if tui == 'T191':  # Neoplastic Process
            if anchor is None or get_confidence(cui) > get_confidence(anchor):
                anchor = cui
    
    if anchor is None:
        # Fallback: select from pathology types (T047)
        anchor = select_highest_confidence(cui_list, allowed_tuis=['T047'])
    
    # Step 2: Vertical Expansion (get parent hierarchy)
    parents = query_relationships(anchor, rel_type='isa', direction='outgoing')
    
    # Step 3: Horizontal Expansion
    sites = []
    features = []
    
    for cui in cui_list:
        tui = get_semantic_type(cui)
        
        # Check if this CUI is connected to anchor as location
        if tui in ['T023', 'T024']:  # Anatomy
            if has_relationship(anchor, cui, rel_types=['location_of', 'finding_site_of']):
                sites.append(cui)
        
        # Check if this CUI is connected to anchor as morphology
        elif tui == 'T033':  # Finding
            if has_relationship(anchor, cui, rel_types=['associated_morphology_of']):
                features.append(cui)
    
    # Build graph structure
    graph = {
        'anchor': anchor,
        'parents': parents,
        'sites': sites,
        'features': features,
        'grade': grade
    }
    
    return graph
```

### Example: Endometrioid Adenocarcinoma

**Input CUIs** (from slide example):
```python
extracted_cuis = [
    "C0206693",  # Endometrioid adenocarcinoma (T191)
    "C0007137",  # Squamous differentiation (T033)
    "C0042149",  # Uterus (T023)
    "C0877041",  # Glandular structure (T033)
    "C0030351",  # Papillary structure (T033)
]
grade = "G1"
```

**Expected Output**:
```
Graph Structure:
  Anchor: C0206693 (Endometrioid adenocarcinoma)
    ├─ isa → C0001418 (Adenocarcinoma)
    │   └─ isa → C0007097 (Carcinoma)
    ├─ location_of → C0042149 (Uterus)
    └─ associated_morphology_of → C0877041 (Glandular structure)
         └─ associated_morphology_of → C0030351 (Papillary structure)

Additional Features:
  - Squamous differentiation (C0007137)
  - Tumor Grade: G1

Text Prompt:
"Endometrioid adenocarcinoma (a subtype of adenocarcinoma and carcinoma) located in the uterus, 
characterized by glandular and papillary structures with squamous differentiation. Tumor grade: G1."
```

### Cypher Implementation (Pattern Matching)

```cypher
// Assume $anchorCui is the high-confidence histology finding
MATCH (anchor:Concept {cui: $anchorCui})

// 1. Get Parent Hierarchy (up to 3 levels)
OPTIONAL MATCH (anchor)-[:isa*1..3]->(parent)

// 2. Get Location (if extracted)
OPTIONAL MATCH (anchor)-[:location_of|:finding_site_of]-(site) 
WHERE site.cui IN $extracted_cuis AND site.tui IN ['T023', 'T024']

// 3. Get Morphology (if extracted)
OPTIONAL MATCH (anchor)-[:associated_morphology_of]-(morph)
WHERE morph.cui IN $extracted_cuis AND morph.tui = 'T033'

RETURN anchor, collect(DISTINCT parent) as parents, 
       collect(DISTINCT site) as sites, 
       collect(DISTINCT morph) as morphology
```

### Strengths & Limitations

**Strengths**:
- **Fully interpretable**: Every edge follows explicit medical relationships
- **Predictable structure**: Always centered on the primary neoplasm
- **Fast execution**: Simple pattern matching, no optimization needed

**Limitations**:
- **Rigid**: May miss indirect relationships (e.g., morphology features not directly linked to anchor)
- **Incomplete**: If a CUI isn't directly connected to anchor, it gets dropped
- **No discovery**: Cannot identify intermediate concepts not in the extracted CUI list
## 3. Approach B: Steiner Tree Graph (Topological Optimization)
This approach treats the subgraph construction as a cost-optimization problem. It finds the minimum spanning tree that connects all detected concepts, potentially discovering **"bridging" nodes** that were not explicitly detected but explain the relationship between two findings.

### Logic
1. **Source Nodes**: The list of CUIs extracted by the concept head.
2. **Algorithm**: Neo4j GDS Steiner Tree (or Directed Steiner Tree).
3. **Heuristic**: "Connectedness" implies "Causality" or "Association."
4. **Discovery**: The algorithm may introduce intermediate concepts (Steiner points) that weren't in the original extraction but are necessary to connect the extracted concepts optimally.

### Implementation Workflow

```python
def build_steiner_tree_graph(cui_list, prune_hubs=True):
    """
    Build subgraph using Steiner tree optimization.
    
    Returns: Graph with nodes, edges, and optional bridging concepts
    """
    # Step 1: Configure edge weights
    # Lower weight = preferred path (more specific relationships)
    edge_weights = {
        'isa': 0.1,
        'CHD': 0.1,
        'associated_morphology_of': 0.2,
        'finding_site_of': 0.2,
        'location_of': 0.2,
        'manifestation_of': 0.3,
        'associated_with': 1.0,  # Discouraged (too generic)
    }
    
    # Step 2: Run Steiner tree algorithm
    # Use existing steiner_tree_retrieval() from Neo4jQuery
    source_cui = cui_list[0]  # Pick highest confidence as source
    target_cuis = cui_list[1:]
    
    steiner_result = neo4j_query.steiner_tree_retrieval(
        cui_list=cui_list,
        source_cui=source_cui,
        relationship_weight_property='weight',
        apply_rerouting=False
    )
    
    # Step 3: Post-process (Hub Pruning)
    if prune_hubs:
        bridging_nodes = steiner_result['steiner_points']
        filtered_bridging = []
        
        for node in bridging_nodes:
            degree = get_node_degree(node)
            tui = get_semantic_type(node)
            
            # Keep specific bridging nodes, remove generic hubs
            if degree < 50 and tui in ALLOWED_TUIS:
                filtered_bridging.append(node)
        
        steiner_result['steiner_points'] = filtered_bridging
    
    return steiner_result
```

### Hub Pruning Criteria

**Why Prune?** The Steiner algorithm may introduce overly-generic "hub" concepts like:
- `C0012634` (Disease) - too broad
- `C0006826` (Neoplasm) - too generic for specific histology
- `C0229960` (Entire body system) - not clinically useful

**Pruning Rules**:
1. **Degree Threshold**: If a bridging node has >50 connections globally, it's likely a generic hub → **Remove**
2. **Semantic Type Check**: Only keep bridging nodes with allowed TUIs (Anatomy, Pathology, Morphology)
3. **Specificity Check**: Prefer nodes with `definition` length >50 characters (more specific concepts have detailed definitions)

**Keep Useful Bridges** like:
- `C0014129` (Endometrial Tissue) - connects endometrioid adenocarcinoma to uterus
- `C0014180` (Endometrium) - anatomically specific
- `C0677578` (Glandular Epithelium) - morphologically meaningful

### Example: Endometrioid Adenocarcinoma

**Input CUIs**:
```python
extracted_cuis = [
    "C0206693",  # Endometrioid adenocarcinoma (T191)
    "C0007137",  # Squamous differentiation (T033)
    "C0042149",  # Uterus (T023)
    "C0877041",  # Glandular structure (T033)
    "C0030351",  # Papillary structure (T033)
]
```

**Steiner Tree Result** (with bridging nodes):
```
Graph Structure:
  Source: C0206693 (Endometrioid adenocarcinoma)
    ├─ isa → C0001418 (Adenocarcinoma)
    │   └─ isa → C0007097 (Carcinoma)
    ├─ finding_site_of → [C0014180 (Endometrium)] ← BRIDGING NODE
    │   └─ part_of → C0042149 (Uterus)
    └─ associated_morphology_of → [C0677578 (Glandular Epithelium)] ← BRIDGING NODE
        ├─ C0877041 (Glandular structure)
        └─ C0030351 (Papillary structure)

Additional Features:
  - C0007137 (Squamous differentiation) - connected via manifestation_of

Bridging Nodes Discovered:
  1. C0014180 (Endometrium) - connects tumor to anatomical site
  2. C0677578 (Glandular Epithelium) - connects morphology features

Text Prompt:
"Endometrioid adenocarcinoma (adenocarcinoma subtype, carcinoma) originates in the endometrium 
of the uterus. It is characterized by glandular epithelium showing glandular and papillary structures, 
with squamous differentiation present."
```

### Neo4j GDS Implementation

```cypher
// Step 1: Project filtered graph (if not already done)
CALL gds.graph.project(
  'histo-graph-filtered',
  {
    Concept: {
      label: 'Concept',
      properties: ['tui', 'name', 'definition']
    }
  },
  {
    RELATED_TO: {
      type: '*',  // All relationship types
      orientation: 'UNDIRECTED',
      properties: 'weight'
    }
  },
  {
    nodeFilter: 'n.tui IN ["T191", "T023", "T033", "T024", "T047", "T029"]'
  }
)

// Step 2: Run Steiner Tree
MATCH (source:Concept)
WHERE source.cui IN $extracted_cuis
WITH collect(source) AS sourceNodes

CALL gds.steinerTree.stream('histo-graph-filtered', {
  sourceNodes: sourceNodes,
  relationshipWeightProperty: 'weight',
  delta: 2.0  // Relaxation parameter for near-optimal solutions
})
YIELD nodeId, parentId, weight
RETURN gds.util.asNode(nodeId) AS concept, 
       gds.util.asNode(parentId) AS connectedTo,
       weight
```

### Key Configuration Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `relationshipWeightProperty` | `'weight'` | Use pre-assigned edge weights (isa=0.1, generic=1.0) |
| `delta` | `2.0` | Allow slightly suboptimal paths to find more interpretable trees |
| `apply_rerouting` | `False` | Keep original paths (rerouting may introduce confusing detours) |
| `sourceNodes` | All extracted CUIs | Connect all concepts, not just pairs |

### Strengths & Limitations

**Strengths**:
- **Discovery**: Finds meaningful intermediate concepts (e.g., "Endometrium" connecting tumor to uterus)
- **Completeness**: All extracted CUIs guaranteed to be in the graph (if connected component exists)
- **Optimality**: Minimizes total "semantic distance" between concepts

**Limitations**:
- **Requires tuning**: Edge weights and hub thresholds need careful configuration
- **Computational cost**: More expensive than pattern matching (GDS projection + algorithm)
- **Interpretability**: Bridging nodes may not always be clinically obvious
- **Failure mode**: If concepts are disconnected in UMLS, returns partial trees

**When It Excels**:
- When extracted concepts are sparse (few CUIs, need context to connect them)
- When relationships aren't all direct (e.g., feature → epithelium → tumor, not feature → tumor)
- When you want to "fill in the gaps" of incomplete concept extraction


4. Approach C: Hybrid Strategy (Recommended)
This combines the safety of semantic filtering with the discovery power of Steiner trees.

Workflow
Strict Semantic Projection: Use the GDS projection from Step 1 to strictly limit the search space to medical concepts (removing administrative nodes).

Prize-Collecting Bias: Use a variable weighting scheme.

Set edge weights for isa and associated_morphology very low (0.1).

Set edge weights for generic relations high (0.9).

Run Steiner Tree: Execute the Steiner algorithm on this biased projection.

Post-Pruning (The "Hub" Check):

The Steiner tree might introduce intermediate nodes (Steiner points) to connect your concepts.

Rule: If a Steiner point is a generic hub (e.g., "Disease", "Body Part") with >50 global connections, collapse it. Only keep specific bridging nodes (e.g., "Glandular Epithelium" connecting "Adenocarcinoma" and "Lumen").

Why this is the "E2" Standard:
Interpretation: It doesn't just list concepts; it explains how they are related via the path.

Robustness: If the VLM misses a specific link, the Steiner path fills it in.

Cleanliness: The semantic filter prevents the "butterfly effect" where the graph wanders into irrelevant medical sub-domains (e.g., billing codes).

---

## 5. Comparison Framework: Approach A vs. Approach B

To determine which approach produces more meaningful subgraphs for VLM guidance, evaluate both on the following criteria:

### 5.1. Evaluation Metrics

| Metric | Approach A (Medical Ruled) | Approach B (Steiner Tree) | How to Measure |
|--------|---------------------------|---------------------------|----------------|
| **Interpretability** | ★★★★★ Always follows explicit medical relationships | ★★★☆☆ Bridging nodes may be non-obvious | Pathologist reviews: "Can I trace the clinical logic?" |
| **Completeness** | ★★☆☆☆ May drop unconnected CUIs | ★★★★★ Guarantees all CUIs in graph (if reachable) | Count: # input CUIs ÷ # CUIs in graph |
| **Discovery** | ★☆☆☆☆ No intermediate concepts | ★★★★★ Finds bridging concepts | Count bridging nodes; assess clinical value |
| **Computational Cost** | ★★★★★ Fast pattern matching | ★★☆☆☆ GDS projection + algorithm | Query execution time (ms) |
| **Robustness** | ★★☆☆☆ Fails if anchor not found | ★★★★☆ Works with any connected concepts | Success rate on edge cases |
| **Clinical Utility** | ★★★★☆ Familiar hierarchy-based reasoning | ★★★★☆ May reveal non-obvious connections | VLM downstream QA performance |

### 5.2. When to Use Each Approach

**Use Approach A (Medical Ruled Graph)** when:
- ✅ You have high-confidence histology extraction (clear T191 anchor)
- ✅ Concepts are directly related (site, morphology linked to tumor)
- ✅ Interpretability is critical (regulatory, clinical deployment)
- ✅ Low computational budget (embedded systems, high throughput)
- ✅ You want a "safe" baseline that never surprises

**Use Approach B (Steiner Tree)** when:
- ✅ Concepts are sparse or indirect (need context to connect them)
- ✅ You want to "fill in gaps" from incomplete extraction
- ✅ Exploring research questions (what connects A to B?)
- ✅ VLM needs rich contextual knowledge beyond extracted CUIs
- ✅ You can afford tuning edge weights and hub thresholds

### 5.3. Recommended Testing Protocol

1. **Prepare Test Set**: Select 50 slides with diverse histology types (adenocarcinoma, squamous, sarcoma, etc.)

2. **Generate Both Graphs**: For each slide:
   ```python
   graph_a = builder.build_medical_ruled_graph(cui_list)
   graph_b = builder.build_steiner_tree_graph(cui_list, prune_hubs=True)
   ```

3. **Compute Metrics**:
   - Node/edge counts
   - Concept coverage (% input CUIs in graph)
   - Bridging node relevance (manual review of Steiner points)
   - Graph diameter (longest shortest path)
   - Query time

4. **Qualitative Review**:
   - Sample 10 slides
   - Show both graphs to pathologist
   - Ask: "Which graph better explains the histology?"
   - Rate on 5-point scale: clinical accuracy, completeness, clarity

5. **VLM Downstream Evaluation** (if applicable):
   - Use graphs as prompts for VLM QA tasks
   - Compare accuracy on histology classification, grading, feature detection
   - Metric: F1 score, answer correctness

### 5.4. Expected Findings

Based on the design of each approach, you may observe:

**Scenario 1: Well-Structured Extraction**
- Input: Clear histology, site, 2-3 features
- **Approach A**: Clean star graph, 100% concept coverage, fast
- **Approach B**: Similar graph + 0-2 bridging nodes (e.g., tissue type)
- **Winner**: Tie or Approach A (simpler is better)

**Scenario 2: Sparse Extraction**
- Input: Only histology + 1 feature (missing site)
- **Approach A**: Small graph, missing connections, low coverage
- **Approach B**: Adds intermediate nodes (e.g., "epithelium", "anatomical site class")
- **Winner**: Approach B (fills in context)

**Scenario 3: Complex Multi-System Tumor**
- Input: Multiple sites, rare variant, many features
- **Approach A**: May not connect all concepts (some dropped)
- **Approach B**: Large tree with many bridging nodes (some spurious)
- **Winner**: Depends on hub pruning quality; potentially hybrid approach

### 5.5. Decision Framework

```
START
  ↓
Is interpretability > discovery?
  ├─ YES → Use Approach A
  └─ NO → Continue
  ↓
Do you have high-quality anchor extraction (T191)?
  ├─ NO → Use Approach B (no reliable anchor)
  └─ YES → Continue
  ↓
Are most concepts directly related to anchor?
  ├─ YES → Use Approach A (simple graph sufficient)
  └─ NO → Use Approach B (need bridging context)
  ↓
[Optional] Try Both → Compare on your specific dataset
```

### 5.6. Hybrid Approach (Future Work)

**Concept**: Combine strengths of both methods:

1. **Phase 1**: Run Approach A to build core anchor-centered graph
2. **Phase 2**: For any unconnected CUIs, run Steiner Tree on `[anchor, unconnected_cui]` pairs
3. **Phase 3**: Add high-value bridging nodes (degree < 50, TUI allowed) to core graph

**Advantages**:
- Guaranteed interpretable core (from Approach A)
- Fills gaps for edge cases (from Approach B)
- Controlled complexity (only add bridges when needed)

**Implementation**:
```python
def build_hybrid_graph(cui_list):
    # Step 1: Anchor-based core
    core_graph = build_medical_ruled_graph(cui_list)
    connected_cuis = get_cuis_in_graph(core_graph)
    
    # Step 2: Find unconnected CUIs
    unconnected = [c for c in cui_list if c not in connected_cuis]
    
    # Step 3: Bridge unconnected via Steiner
    for cui in unconnected:
        bridge_graph = build_steiner_tree_graph([core_graph['anchor'], cui], prune_hubs=True)
        merge_graphs(core_graph, bridge_graph)
    
    return core_graph
```

This hybrid is recommended for production systems where you need both reliability (Approach A) and completeness (Approach B).