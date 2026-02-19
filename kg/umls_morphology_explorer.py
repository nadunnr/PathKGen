"""
UMLS Morphology Feature Explorer

This script explores UMLS graph traversal paths from histology + primary site CUIs
to morphological feature CUIs. It processes the WSI-Bench dataset and queries Neo4j
to find relationships.
"""

import json
import re
from typing import Dict, List, Set, Tuple
from neo4j import GraphDatabase
from tqdm import tqdm


class UMLSMorphologyExplorer:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="11111111"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def extract_morphological_features(self, t_answer: str) -> List[str]:
        """
        Extract morphological features from T-answer text.
        Features include: pleomorphism, nuclear features, mitotic activity, 
        differentiation, invasion patterns, necrosis, etc.
        """
        features = []
        
        # Morphological feature keywords to look for
        feature_patterns = {
            'pleomorphism': r'pleomorph\w*',
            'nuclear_variability': r'nuclear.*variability|variability.*nuclear',
            'nuclear_size': r'nuclear size',
            'mitotic_activity': r'mitotic activity|mitosis|mitotic',
            'differentiation': r'differentiat\w*',
            'necrosis': r'necrosis|necrotic',
            'invasion': r'invasion|invasive|infiltrat\w*',
            'nuclear_grade': r'nuclear grade',
            'cellular_morphology': r'cellular morphology',
            'nuclear_atypia': r'nuclear atypia|atypia',
            'hyperchromatic': r'hyperchromat\w*',
            'nucleoli': r'nucleol\w*',
            'architecture': r'architecture|architectural',
            'glandular': r'glandular',
            'vascular_invasion': r'vascular invasion',
            'perineural_invasion': r'perineural invasion',
            'grade': r'grade \d|grading',
        }
        
        t_answer_lower = t_answer.lower()
        for feature_name, pattern in feature_patterns.items():
            if re.search(pattern, t_answer_lower):
                features.append(feature_name)
        
        return list(set(features))  # Remove duplicates
    
    def extract_histology_and_variant(self, project_name: str, t_answer: str) -> Tuple[str, str]:
        """
        Extract histology type and variant from project_name and T-answer.
        """
        histology = project_name.strip()
        
        # Try to extract variant from T-answer if it's a diagnostic answer
        variant = ""
        # Common patterns for variants in answers
        variant_patterns = [
            r'(ductal|lobular|papillary|follicular|clear cell|squamous|adenocarcinoma)',
            r'(well|moderately|poorly) differentiated',
            r'grade [I1-4IV]+',
        ]
        
        for pattern in variant_patterns:
            match = re.search(pattern, t_answer, re.IGNORECASE)
            if match:
                variant = match.group(0)
                break
        
        return histology, variant
    
    def search_cui_by_text(self, text: str, limit: int = 5):
        """Search UMLS for CUIs matching the given text."""
        with self.driver.session() as session:
            query = """
            MATCH (c:Concept)
            WHERE toLower(c.name) CONTAINS toLower($text)
            RETURN c.cui as cui, c.name as name
            LIMIT $limit
            """
            result = session.run(query, text=text, limit=limit)
            return [(record["cui"], record["name"]) for record in result]
    
    def find_paths_to_morphology(self, histology_cui: str, site_cui: str, 
                                  morphology_features: List[str], max_depth: int = 5):
        """
        Find paths from histology+site CUIs to morphology feature CUIs.
        Returns list of paths with relationship types.
        """
        with self.driver.session() as session:
            # First, let's explore what relationships exist from histology node
            query = """
            MATCH path = (h:Concept {cui: $histology_cui})-[r*1..$max_depth]->(m:Concept)
            WHERE ANY(feature IN $features WHERE toLower(m.name) CONTAINS toLower(feature))
            RETURN path, 
                   [rel in relationships(path) | type(rel)] as rel_types,
                   [node in nodes(path) | node.name] as node_names,
                   length(path) as depth
            ORDER BY depth
            LIMIT 20
            """
            
            result = session.run(
                query, 
                histology_cui=histology_cui, 
                max_depth=max_depth,
                features=morphology_features
            )
            
            paths = []
            for record in result:
                paths.append({
                    'rel_types': record["rel_types"],
                    'node_names': record["node_names"],
                    'depth': record["depth"]
                })
            
            return paths
    
    def explore_instance(self, instance: Dict) -> Dict:
        """
        Explore a single instance from the dataset.
        Returns extracted information and found paths.
        """
        # Extract information
        primary_site = instance.get("primary_site", "")
        project_name = instance.get("project_name", "")
        t_answer = instance.get("T-answer", "")
        task = instance.get("task", "")
        tcga_id = instance.get("tcga_id", "")
        
        # Extract histology and morphological features
        histology, variant = self.extract_histology_and_variant(project_name, t_answer)
        morphological_features = self.extract_morphological_features(t_answer)
        
        # Search for CUIs
        histology_cuis = self.search_cui_by_text(histology, limit=3)
        site_cuis = self.search_cui_by_text(primary_site, limit=3)
        
        result = {
            'tcga_id': tcga_id,
            'task': task,
            'primary_site': primary_site,
            'histology': histology,
            'variant': variant,
            'morphological_features': morphological_features,
            'histology_cuis': histology_cuis,
            'site_cuis': site_cuis,
            't_answer_snippet': t_answer[:200] + '...' if len(t_answer) > 200 else t_answer,
        }
        
        # Find paths if we have CUIs
        if histology_cuis and morphological_features:
            histology_cui = histology_cuis[0][0]  # Use first match
            site_cui = site_cuis[0][0] if site_cuis else None
            
            paths = self.find_paths_to_morphology(
                histology_cui, 
                site_cui,
                morphological_features,
                max_depth=5
            )
            result['paths'] = paths
        else:
            result['paths'] = []
        
        return result


def main():
    # Load dataset
    dataset_path = "/home/nadun/wd/datasets/wsi_bench/dataset/WSI-Bench-train-formatted.json"
    print(f"Loading dataset from {dataset_path}...")
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} instances")
    
    # Filter for morphology-related tasks
    morphology_tasks = ['Morphology', 'Morphology_choice', 'MorDetail']
    morphology_instances = [inst for inst in data if inst.get('task') in morphology_tasks]
    
    print(f"Found {len(morphology_instances)} morphology-related instances")
    
    # Initialize explorer
    explorer = UMLSMorphologyExplorer()
    
    # Process a sample of instances
    sample_size = 100  # Start with first 100 instances
    results = []
    
    print(f"\nProcessing {sample_size} instances...")
    for instance in tqdm(morphology_instances[:sample_size]):
        try:
            result = explorer.explore_instance(instance)
            results.append(result)
        except Exception as e:
            print(f"\nError processing {instance.get('tcga_id')}: {e}")
    
    # Save results
    output_path = "/home/nadun/wd/vlm_histo/kg/morphology_exploration_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved results to {output_path}")
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    instances_with_paths = sum(1 for r in results if r.get('paths'))
    print(f"Instances with found paths: {instances_with_paths}/{len(results)}")
    
    # Most common morphological features
    all_features = []
    for r in results:
        all_features.extend(r.get('morphological_features', []))
    
    from collections import Counter
    feature_counts = Counter(all_features)
    print("\nTop 10 morphological features:")
    for feature, count in feature_counts.most_common(10):
        print(f"  {feature}: {count}")
    
    explorer.close()


if __name__ == "__main__":
    main()
