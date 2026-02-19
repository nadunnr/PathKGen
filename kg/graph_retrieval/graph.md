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
2. Approach A: Medical Ruled Graph (The "Anchor" Method)
This heuristic approach forces the graph into a predictable "Star" or "Tree" topology based on biological hierarchy. It guarantees interpretability but may miss latent connections.

Logic
Identify Anchor: Select the node with TUI = T191 (Neoplastic Process) and the highest confidence score.

Vertical Expansion: Query 1-hop up (isa) to get the parent class (e.g., Adenocarcinoma -> Carcinoma).

Horizontal Expansion: Query specifically for location_of (Site) and associated_morphology_of (Features) relative to the Anchor.

Cypher Implementation (Pattern Matching)
Cypher
// Assume $anchorCui is the high-confidence histology finding
MATCH path = (anchor:Concept {cui: $anchorCui})
// 1. Get Parent Hierarchy
OPTIONAL MATCH (anchor)-[:isa]->(parent)
// 2. Get Location (if extracted)
OPTIONAL MATCH (anchor)-[:location_of]-(site) 
WHERE site.cui IN $extracted_cuis AND site.tui IN ['T023', 'T024']
// 3. Get Morphology (if extracted)
OPTIONAL MATCH (anchor)-[:associated_morphology_of]-(morph)
WHERE morph.cui IN $extracted_cuis AND morph.tui = 'T033'
RETURN path
3. Approach B: Steiner Tree Graph (Topological Optimization)
This approach treats the subgraph construction as a cost-optimization problem. It finds the minimum spanning tree that connects all detected concepts, potentially discovering "bridging" nodes that were not explicitly detected but explain the relationship between two findings.

Logic
Source Nodes: The list of CUIs extracted by the concept head.

Algorithm: Neo4j GDS Steiner Tree (or Directed Steiner Tree).

Heuristic: "Connectedness" implies "Causality" or "Association."

Neo4j GDS Implementation
We use the gds.steinerTree algorithm. Note that standard Steiner Tree is for undirected graphs; for directed hierarchies, ensure your projection handles directionality or use the Directed Steiner Tree implementation if edges imply strict flow.

Cypher
MATCH (source:Concept)
WHERE source.cui IN $extracted_cuis
WITH collect(source) AS sourceNodes

CALL gds.steinerTree.stream('histo-graph-filtered', {
  sourceNodes: sourceNodes,
  relationshipWeightProperty: 'weight' // Critical: 'isa' should be cheap (0.1), 'associated_with' expensive (1.0)
})
YIELD nodeId, parents
RETURN gds.util.asNode(nodeId).name AS concept, 
       [p IN parents | gds.util.asNode(p).name] AS connected_to
Key Configuration:

Weights: You must invert confidence scores or use preset edge costs (e.g., cost = 1/confidence). The algorithm minimizes cost.

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