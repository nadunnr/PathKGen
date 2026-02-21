"""
Test subgraph builder with extracted labeled dataset using new dictionary API.

Demonstrates the recommended build_medical_ruled_graph_from_concepts() method.
"""

from graphs import Neo4jQuery
from subgraph_builder import SubgraphBuilder
import csv
from pathlib import Path


def test_with_dataset():
    """Test new dictionary API with extracted dataset samples."""
    
    # Initialize Neo4j
    query = Neo4jQuery(
        "bolt://localhost:7687",
        "neo4j",
        "11111111",
        database="neo4jv8"
    )
    
    builder = SubgraphBuilder(query)
    
    # Load dataset
    labels_file = Path("/home/nadun/wd/vlm_histo/concept-head/labelling/labels_test/merged_labels.csv")
    
    if not labels_file.exists():
        print(f"❌ Dataset not found: {labels_file}")
        print("   Run: cd /home/nadun/wd/vlm_histo/concept-head/labelling && python merge_labels.py")
        return
    
    with open(labels_file, 'r') as f:
        reader = csv.DictReader(f)
        samples = list(reader)
    
    print(f"✅ Loaded {len(samples)} samples\n")
    
    # Test first 3 samples
    for i, sample in enumerate(samples[:3], 1):
        print("=" * 80)
        print(f"SAMPLE {i}: {sample['tcga_id'][:60]}")
        print("=" * 80)
        
        # Build concepts dictionary
        concepts = {
            'histology': sample['histology_cui'] if sample['histology_cui'] else None,
            'variant': sample['variant_cui'] if sample['variant_cui'] else None,
            'site': sample['site_cui'] if sample['site_cui'] else None,
            'grade': sample['grade_cui'] if sample['grade_cui'] else None,
            'tumor_features': []  # Will be added later
        }
        
        # Print input concepts
        print(f"\n📋 Input Concepts:")
        print(f"  Histology: {sample['histology_name']} ({sample['histology_cui']})")
        if sample['variant_cui']:
            print(f"  Variant:   {sample['variant_name']} ({sample['variant_cui']}) ⭐ ANCHOR")
        else:
            print(f"  Variant:   None ⭐ ANCHOR = histology")
        print(f"  Site:      {sample['site_name']} ({sample['site_cui']})")
        print(f"  Grade:     {sample['grade_name']} ({sample['grade_cui']})")
        
        # Build subgraph using new API
        try:
            subgraph = builder.build_medical_ruled_graph_from_concepts(
                concepts=concepts,
                include_hierarchy=True,
                hierarchy_depth=2,
                semantic_expansion=True
            )
            
            # Print results
            print(f"\n📊 Subgraph Results:")
            print(f"  Nodes:      {subgraph['metadata']['node_count']}")
            print(f"  Edges:      {subgraph['metadata']['edge_count']}")
            print(f"  Discovered: {subgraph['metadata']['discovered_node_count']} new nodes")
            print(f"  Anchor:     {subgraph['anchor_cui']}")
            
            # Show discovered nodes
            input_cuis = {concepts['histology'], concepts['variant'], 
                         concepts['site'], concepts['grade']}
            input_cuis.discard(None)
            
            discovered = [n for n in subgraph['nodes'] if n['cui'] not in input_cuis]
            
            if discovered:
                print(f"\n  Sample Discovered Nodes:")
                for node in discovered[:5]:
                    print(f"    • {node['name'][:60]} ({node['cui']})")
                if len(discovered) > 5:
                    print(f"    ... and {len(discovered) - 5} more")
            
            # Export
            builder.export_to_json(subgraph, f"/tmp/sample_{i}_subgraph.json")
            print(f"\n  💾 Exported to: /tmp/sample_{i}_subgraph.json")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
        
        print()
    
    query.close()
    print("=" * 80)
    print("✅ Test Complete!")
    print("=" * 80)


if __name__ == "__main__":
    test_with_dataset()
