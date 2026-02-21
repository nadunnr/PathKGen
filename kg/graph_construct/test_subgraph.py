"""
Test script for comparing UMLS subgraph construction approaches.

This script tests both Medical Ruled Graph and Steiner Tree approaches
with sample histopathology concepts.
"""

from graphs import Neo4jQuery
from subgraph_builder import SubgraphBuilder
import json


def print_separator(title=""):
    """Print a visual separator."""
    print("\n" + "=" * 80)
    if title:
        print(f" {title}")
        print("=" * 80)
    print()


def print_subgraph_summary(subgraph: dict, approach_name: str):
    """Print a summary of a subgraph."""
    print(f"Approach: {approach_name}")
    print(f"Metadata: {json.dumps(subgraph['metadata'], indent=2)}")
    
    print(f"\nNodes ({len(subgraph['nodes'])}):")
    for node in subgraph['nodes'][:10]:  # Show first 10
        print(f"  - {node.get('name', 'Unknown')} ({node['cui']})")
    if len(subgraph['nodes']) > 10:
        print(f"  ... and {len(subgraph['nodes']) - 10} more")
    
    print(f"\nEdges ({len(subgraph['edges'])}):")
    for edge in subgraph['edges'][:10]:  # Show first 10
        print(f"  {edge.get('source_name', '?')} "
              f"--[{edge.get('rel_type', '?')}]--> "
              f"{edge.get('target_name', '?')}")
    if len(subgraph['edges']) > 10:
        print(f"  ... and {len(subgraph['edges']) - 10} more")
    print()


def test_simple_case(builder: SubgraphBuilder, use_gds: bool = False):
    """Test Case 1: Simple endometrioid carcinoma example."""
    print_separator("TEST CASE 1: Simple Endometrioid Carcinoma")
    
    # Search for concepts
    query = builder.query
    
    # Find histology: endometrioid adenocarcinoma
    histology_results = query.search_entity("endometrioid adenocarcinoma")
    if not histology_results:
        print("⚠️  Could not find endometrioid adenocarcinoma")
        return
    histology_cui = histology_results[0]['cui']
    print(f"Histology: {histology_results[0]['name']} ({histology_cui})")
    
    # Find grade: G1 / Grade 1 / well differentiated
    grade_results = query.search_entity("well differentiated")
    if not grade_results:
        grade_results = query.search_entity("grade 1")
    if grade_results:
        grade_cui = grade_results[0]['cui']
        print(f"Grade: {grade_results[0]['name']} ({grade_cui})")
    else:
        print("⚠️  Could not find grade concept")
        grade_cui = None
    
    # Find site: uterus
    site_results = query.search_entity("uterus")
    if not site_results:
        print("⚠️  Could not find uterus")
        return
    site_cui = site_results[0]['cui']
    print(f"Site: {site_results[0]['name']} ({site_cui})")
    
    # Build CUI list
    cui_list = [histology_cui, site_cui]
    if grade_cui:
        cui_list.append(grade_cui)
    
    print(f"\nInput CUIs: {cui_list}")
    
    # Test Approach A: Medical Ruled Graph
    print_separator("Approach A: Medical Ruled Graph")
    medical_graph = builder.build_medical_ruled_graph(cui_list)
    print_subgraph_summary(medical_graph, "Medical Ruled Graph")
    
    # Test Approach B: Steiner Tree
    method_name = "Steiner Tree (GDS)" if use_gds else "Steiner Tree (Cypher)"
    print_separator(f"Approach B: {method_name}")
    steiner_graph = builder.build_steiner_tree_graph(cui_list, use_gds=use_gds)
    print_subgraph_summary(steiner_graph, method_name)
    
    # Compare
    print_separator("Comparison")
    medical_cuis = set(n['cui'] for n in medical_graph['nodes'])
    steiner_cuis = set(n['cui'] for n in steiner_graph['nodes'])
    
    print(f"Medical Ruled Graph: {len(medical_cuis)} nodes, {len(medical_graph['edges'])} edges")
    print(f"Steiner Tree: {len(steiner_cuis)} nodes, {len(steiner_graph['edges'])} edges")
    print(f"Shared nodes: {len(medical_cuis & steiner_cuis)}")
    print(f"Unique to Medical: {len(medical_cuis - steiner_cuis)}")
    print(f"Unique to Steiner: {len(steiner_cuis - medical_cuis)}")
    
    # Show Steiner nodes (discovered by Steiner tree)
    input_cui_set = set(cui_list)
    steiner_discovered = [
        n for n in steiner_graph['nodes']
        if n['cui'] not in input_cui_set
    ]
    
    if steiner_discovered:
        print(f"\nSteiner nodes discovered ({len(steiner_discovered)}):")
        for node in steiner_discovered[:5]:
            print(f"  - {node.get('name', 'Unknown')} ({node['cui']})")
        if len(steiner_discovered) > 5:
            print(f"  ... and {len(steiner_discovered) - 5} more")
    
    # Export to files
    builder.export_to_json(medical_graph, "/tmp/medical_graph.json")
    builder.export_to_json(steiner_graph, "/tmp/steiner_graph.json")
    builder.export_to_graphml(medical_graph, "/tmp/medical_graph.graphml")
    builder.export_to_graphml(steiner_graph, "/tmp/steiner_graph.graphml")
    
    print("\n✅ Exported graphs to /tmp/*.json and /tmp/*.graphml")


def test_complex_case(builder: SubgraphBuilder):
    """Test Case 2: More complex case with multiple concepts."""
    print_separator("TEST CASE 2: Complex Multi-Concept Case")
    
    query = builder.query
    
    # Multiple related concepts
    concepts = {
        "invasive ductal carcinoma": None,
        "breast": None,
        "lymph node": None,
        "metastasis": None,
        "estrogen receptor": None,
    }
    
    cui_list = []
    for concept_name in concepts.keys():
        results = query.search_entity(concept_name)
        if results:
            cui = results[0]['cui']
            concepts[concept_name] = cui
            cui_list.append(cui)
            print(f"{concept_name}: {results[0]['name']} ({cui})")
        else:
            print(f"⚠️  Could not find: {concept_name}")
    
    if len(cui_list) < 3:
        print("⚠️  Not enough concepts found to test")
        return
    
    print(f"\nInput CUIs: {cui_list}")
    
    # Run comparison
    print_separator("Running Both Approaches")
    comparison = builder.compare_approaches(cui_list)
    
    print("Comparison Results:")
    print(json.dumps(comparison['comparison'], indent=2))


def test_minimal_case(builder: SubgraphBuilder):
    """Test Case 3: Minimal case with just 2 concepts."""
    print_separator("TEST CASE 3: Minimal 2-Concept Case")
    
    query = builder.query
    
    # Just histology and site
    carcinoma = query.search_entity("carcinoma")
    lung = query.search_entity("lung")
    
    if not carcinoma or not lung:
        print("⚠️  Could not find required concepts")
        return
    
    cui_list = [carcinoma[0]['cui'], lung[0]['cui']]
    print(f"Testing with: {carcinoma[0]['name']} and {lung[0]['name']}")
    print(f"CUIs: {cui_list}")
    
    # Medical Ruled Graph
    medical_graph = builder.build_medical_ruled_graph(cui_list)
    print(f"\nMedical Ruled Graph: {len(medical_graph['nodes'])} nodes, "
          f"{len(medical_graph['edges'])} edges")
    
    # Steiner Tree
    steiner_graph = builder.build_steiner_tree_graph(cui_list, use_gds=False)
    print(f"Steiner Tree: {len(steiner_graph['nodes'])} nodes, "
          f"{len(steiner_graph['edges'])} edges")


def main():
    """Main test execution."""
    # Initialize Neo4j connection
    NEO4J_URL = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "11111111"  # Change this to your password
    NEO4J_DATABASE = "neo4jv8"  # Change this to your database name
    
    # Set to True to use GDS Steiner Tree, False for Cypher approximation
    USE_GDS = True  # Change to False if GDS is not available
    
    print("Initializing Neo4j connection...")
    query = Neo4jQuery(NEO4J_URL, NEO4J_USER, NEO4J_PASSWORD, database=NEO4J_DATABASE)
    
    print("Creating SubgraphBuilder...")
    builder = SubgraphBuilder(query)
    
    # Run tests
    try:
        print(f"\n🔧 Using {'GDS' if USE_GDS else 'Cypher'} for Steiner Tree\n")
        test_simple_case(builder, use_gds=USE_GDS)
        # test_complex_case(builder)
        # test_minimal_case(builder)
    finally:
        query.close()
    
    print_separator("All Tests Complete")


if __name__ == "__main__":
    main()
