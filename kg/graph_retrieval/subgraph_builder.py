"""
UMLS Subgraph Construction for Histopathology Slides

This module provides two approaches for building meaningful knowledge subgraphs
from extracted slide concepts (histology, grade, site, etc.):

1. Medical Ruled Graph (Approach A): Anchor-based semantic expansion
2. Steiner Tree Graph (Approach B): Optimization-based spanning tree
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict
import json


class SubgraphBuilder:
    """
    Constructs UMLS knowledge subgraphs from extracted slide concepts.
    
    Supports two approaches:
    - Medical Ruled Graph: Uses semantic rules and hierarchies
    - Steiner Tree: Finds minimum spanning tree connecting concepts
    """
    
    # Allowed semantic types for histopathology
    ALLOWED_TUIS = {
        # Anatomy (Primary Sites)
        'T023',  # Body Part/Organ
        'T024',  # Tissue
        'T029',  # Body Location
        # Pathology (Histology & Variants)
        'T191',  # Neoplastic Process
        'T047',  # Disease or Syndrome
        'T019',  # Congenital Abnormality
        # Morphology (Features)
        'T033',  # Finding
        'T044',  # Molecular Function
        'T045',  # Function
    }
    
    # Hierarchical relations (low cost/high priority)
    HIERARCHICAL_RELS = {'isa', 'CHD', 'PAR'}
    
    # Semantic relations (low cost/high priority)
    SEMANTIC_RELS = {
        'has_finding_site',
        'finding_site_of',
        'location_of',
        'has_location',
        'has_associated_morphology',
        'associated_morphology_of',
        'manifestation_of',
        'has_grade',
        'has_stage',
        'part_of',
    }
    
    # Generic relations (high cost/low priority)
    GENERIC_RELS = {'associated_with', 'related_to'}
    
    def __init__(self, neo4j_query_instance):
        """
        Initialize SubgraphBuilder with a Neo4jQuery instance.
        
        Args:
            neo4j_query_instance: Instance of Neo4jQuery for database access
        """
        self.query = neo4j_query_instance
    
    def build_medical_ruled_graph_from_concepts(
        self,
        concepts: Dict[str, str],
        include_hierarchy: bool = True,
        hierarchy_depth: int = 2,
        semantic_expansion: bool = True
    ) -> Dict[str, Any]:
        """
        Build Medical Ruled Graph from structured concept dictionary.
        
        Recommended API for working with extracted slide concepts.
        Anchor: variant (if present), else histology.
        
        Args:
            concepts: Dict with CUIs - 'histology' (required), 'variant', 'site', 'grade', 'tumor_features'
            include_hierarchy: Expand vertically (parent/child)
            hierarchy_depth: Hierarchy levels to include
            semantic_expansion: Expand horizontally (semantic relations)
            
        Returns:
            Dict with nodes, edges, anchor_cui, metadata
        """
        nodes = {}
        edges = []
        
        # Extract CUIs
        histology_cui = concepts.get('histology')
        variant_cui = concepts.get('variant')
        site_cui = concepts.get('site')
        grade_cui = concepts.get('grade')
        tumor_features = concepts.get('tumor_features', [])
        
        if not histology_cui:
            raise ValueError("histology CUI required")
        
        # Anchor: variant if present, else histology
        anchor_cui = variant_cui if variant_cui else histology_cui
        
        # Build CUI list
        cui_list = [c for c in [histology_cui, variant_cui, site_cui, grade_cui] if c]
        if isinstance(tumor_features, list):
            cui_list.extend([f for f in tumor_features if f])
        elif tumor_features:
            cui_list.append(tumor_features)
        
        # Fetch nodes
        for cui in cui_list:
            node_data = self.query.get_entity_by_cui(cui)
            if node_data:
                nodes[cui] = node_data
        
        # Connect histology-variant if both exist
        if histology_cui and variant_cui and histology_cui != variant_cui:
            path = self.query.find_shortest_path(histology_cui, variant_cui, max_depth=2)
            if path and path['nodes']:
                for node in path['nodes']:
                    if node['cui'] not in nodes:
                        nodes[node['cui']] = node
                
                for i in range(len(path['nodes']) - 1):
                    src, tgt = path['nodes'][i], path['nodes'][i + 1]
                    rel = path['relationships'][i] if i < len(path['relationships']) else {}
                    edges.append({
                        'source_cui': src['cui'],
                        'source_name': src['name'],
                        'target_cui': tgt['cui'],
                        'target_name': tgt['name'],
                        'rel_type': rel.get('type', 'RELATED_TO'),
                        'rela': rel.get('rela'),
                        'weight': 0.1
                    })
        
        # Hierarchical expansion
        if include_hierarchy and anchor_cui:
            self._expand_hierarchy(anchor_cui, nodes, edges, depth=hierarchy_depth)
        
        # Semantic expansion
        if semantic_expansion:
            self._expand_semantic_relations(cui_list, nodes, edges)
        
        return {
            'nodes': list(nodes.values()),
            'edges': edges,
            'anchor_cui': anchor_cui,
            'metadata': {
                'approach': 'medical_ruled_graph',
                'node_count': len(nodes),
                'edge_count': len(edges),
                'input_cui_count': len(cui_list),
                'discovered_node_count': len(nodes) - len(cui_list),
                'has_variant': bool(variant_cui)
            }
        }
        
    def build_medical_ruled_graph(
        self,
        cui_list: List[str],
        include_hierarchy: bool = True,
        hierarchy_depth: int = 2,
        semantic_expansion: bool = True,
        anchor_cui: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Approach A: Build Medical Ruled Graph using anchor-based expansion.
        
        Creates a predictable Star/Tree topology based on biological hierarchy.
        Guarantees interpretability but may miss latent connections.
        
        Args:
            cui_list: List of CUI identifiers from concept extraction
            include_hierarchy: Whether to expand vertically (parent/child)
            hierarchy_depth: How many levels of hierarchy to include
            semantic_expansion: Whether to expand horizontally (semantic relations)
            anchor_cui: Optional explicit anchor CUI (e.g., histology or histology+variant)
                       If not provided, will auto-detect from cui_list
            
        Returns:
            Dictionary containing:
                - nodes: List of all nodes in the graph
                - edges: List of all edges in the graph
                - anchor_cui: The anchor node CUI (if found)
                - metadata: Graph statistics and information
        """
        nodes = {}  # cui -> node_data
        edges = []  # list of edge dicts
        
        # Step 1: Fetch all input nodes
        # Auto-detect anchor if not explicitly provided via parameter
        detected_anchor = None
        
        for cui in cui_list:
            node_data = self.query.get_entity_by_cui(cui)
            if node_data:
                nodes[cui] = node_data
                
                # Auto-detect anchor if not explicitly provided
                if anchor_cui is None and detected_anchor is None:
                    # Prioritize neoplastic processes (histology/variant)
                    if self._is_neoplastic(cui):
                        detected_anchor = cui
        
        # Use explicit anchor if provided, otherwise use detected or first CUI
        if anchor_cui:
            final_anchor = anchor_cui
        elif detected_anchor:
            final_anchor = detected_anchor
        elif cui_list:
            final_anchor = cui_list[0]
        else:
            final_anchor = None
        
        # If final anchor not in nodes (explicit anchor case), fetch it
        if final_anchor and final_anchor not in nodes:
            node_data = self.query.get_entity_by_cui(final_anchor)
            if node_data:
                nodes[final_anchor] = node_data
        
        # Step 2: Hierarchical expansion (vertical)
        if include_hierarchy and final_anchor:
            self._expand_hierarchy(
                final_anchor, nodes, edges, 
                depth=hierarchy_depth
            )
        
        # Step 3: Semantic expansion (horizontal)
        if semantic_expansion:
            self._expand_semantic_relations(
                cui_list, nodes, edges
            )
        
        # Step 4: Build result
        return {
            'nodes': list(nodes.values()),
            'edges': edges,
            'anchor_cui': final_anchor,
            'metadata': {
                'approach': 'medical_ruled_graph',
                'node_count': len(nodes),
                'edge_count': len(edges),
                'input_cui_count': len(cui_list),
                'discovered_node_count': len(nodes) - len(cui_list)
            }
        }
    
    def build_steiner_tree_graph(
        self,
        cui_list: List[str],
        max_depth: int = 4,
        use_gds: bool = False
    ) -> Dict[str, Any]:
        """
        Approach B: Build Steiner Tree Graph using optimization.
        
        Finds minimum spanning tree connecting all concepts, potentially
        discovering bridging nodes that explain relationships.
        
        Args:
            cui_list: List of CUI identifiers from concept extraction
            max_depth: Maximum path length to search
            use_gds: Whether to use Neo4j GDS (requires GDS plugin)
            
        Returns:
            Dictionary containing:
                - nodes: List of all nodes in the graph
                - edges: List of all edges in the graph
                - steiner_nodes: Intermediate nodes discovered
                - metadata: Graph statistics and information
        """
        if use_gds:
            # GDS-based implementation (requires Neo4j GDS plugin)
            return self._build_steiner_tree_gds(cui_list)
        else:
            # Cypher-based fallback approximation
            return self._build_steiner_tree_cypher(cui_list, max_depth)
    
    def _build_steiner_tree_cypher(
        self,
        cui_list: List[str],
        max_depth: int = 4
    ) -> Dict[str, Any]:
        """
        Cypher-based Steiner tree approximation.
        
        Uses shortest paths between all pairs of concepts, then merges
        the paths to create a spanning tree.
        """
        nodes = {}  # cui -> node_data
        edges = []
        edge_set = set()  # (source_cui, target_cui, rel_type) for deduplication
        
        # Fetch all input nodes
        for cui in cui_list:
            node_data = self.query.get_entity_by_cui(cui)
            if node_data:
                nodes[cui] = node_data
        
        # Find shortest paths between all pairs
        for i, source_cui in enumerate(cui_list):
            for target_cui in cui_list[i+1:]:
                path = self.query.find_shortest_path(
                    source_cui, target_cui, 
                    max_depth=max_depth
                )
                
                if path:
                    # Add all nodes and edges from this path
                    path_nodes = path.get('nodes', [])
                    path_rels = path.get('relationships', [])
                    
                    # Add all nodes from the path
                    for node in path_nodes:
                        cui = node['cui']
                        if cui not in nodes:
                            nodes[cui] = node
                    
                    # Build edges with proper source and target CUIs
                    # The relationships array corresponds to edges between consecutive nodes
                    for idx, rel in enumerate(path_rels):
                        if idx < len(path_nodes) - 1:
                            source_node = path_nodes[idx]
                            target_node = path_nodes[idx + 1]
                            
                            edge_key = (
                                source_node['cui'],
                                target_node['cui'],
                                rel.get('type', 'RELATED_TO')
                            )
                            
                            if edge_key not in edge_set:
                                edge_set.add(edge_key)
                                edges.append({
                                    'source_cui': source_node['cui'],
                                    'source_name': source_node.get('name'),
                                    'target_cui': target_node['cui'],
                                    'target_name': target_node.get('name'),
                                    'rel_type': rel.get('type', 'RELATED_TO'),
                                    'rela': rel.get('rela'),
                                    'weight': 1.0
                                })
        
        # Identify Steiner nodes (not in original input)
        steiner_nodes = [
            cui for cui in nodes.keys() 
            if cui not in cui_list
        ]
        
        # Post-process: Remove generic hubs (>50 connections)
        filtered_nodes, filtered_edges = self._filter_generic_hubs(
            nodes, edges, threshold=50
        )
        
        return {
            'nodes': list(filtered_nodes.values()),
            'edges': filtered_edges,
            'steiner_nodes': [
                nodes[cui] for cui in steiner_nodes 
                if cui in filtered_nodes
            ],
            'metadata': {
                'approach': 'steiner_tree_cypher',
                'node_count': len(filtered_nodes),
                'edge_count': len(filtered_edges),
                'input_cui_count': len(cui_list),
                'steiner_node_count': len([
                    cui for cui in steiner_nodes 
                    if cui in filtered_nodes
                ]),
                'removed_hubs': len(nodes) - len(filtered_nodes)
            }
        }
    
    def _build_steiner_tree_gds(self, cui_list: List[str]) -> Dict[str, Any]:
        """
        GDS-based Steiner tree using Neo4j Graph Data Science.
        
        Uses the 'conceptGraph' projection and gds.steinerTree.stream algorithm.
        """
        if len(cui_list) < 2:
            raise ValueError("Need at least 2 CUIs for Steiner tree")
        
        nodes = {}  # cui -> node_data
        edges = []
        
        with self.query.driver.session(database=self.query.database) as session:
            # Run GDS Steiner Tree query
            result = session.run("""
                MATCH (n:Concept)
                WHERE n.cui IN $cui_list
                WITH collect(n) AS allNodes
                
                // Separate one node as source, rest as targets
                WITH allNodes[0] AS source, 
                     [node IN allNodes WHERE node <> allNodes[0]] AS targets
                
                // Run Steiner Tree algorithm
                CALL gds.steinerTree.stream('conceptGraph', {
                  sourceNode: source,
                  targetNodes: targets
                })
                YIELD nodeId, parentId
                
                // Return node and parent information
                RETURN 
                    gds.util.asNode(nodeId).cui AS node_cui, 
                    gds.util.asNode(nodeId).name AS node_name,
                    gds.util.asNode(parentId).cui AS parent_cui,
                    gds.util.asNode(parentId).name AS parent_name,
                    nodeId,
                    parentId
            """, cui_list=cui_list)
            
            # Process results
            records = [dict(record) for record in result]
            
            # Build nodes dict
            for record in records:
                node_cui = record['node_cui']
                parent_cui = record['parent_cui']
                
                if node_cui and node_cui not in nodes:
                    nodes[node_cui] = {
                        'cui': node_cui,
                        'name': record['node_name']
                    }
                
                if parent_cui and parent_cui not in nodes:
                    nodes[parent_cui] = {
                        'cui': parent_cui,
                        'name': record['parent_name']
                    }
                
                # Build edges (skip root node where nodeId == parentId)
                if record['nodeId'] != record['parentId']:
                    # Get relationship type between parent and child
                    rel_result = session.run("""
                        MATCH (parent:Concept {cui: $parent_cui})-[r]->(child:Concept {cui: $child_cui})
                        RETURN type(r) AS rel_type, r.rela AS rela
                        LIMIT 1
                    """, parent_cui=parent_cui, child_cui=node_cui)
                    
                    rel_record = rel_result.single()
                    rel_type = rel_record['rel_type'] if rel_record else 'RELATED_TO'
                    rela = rel_record['rela'] if rel_record else None
                    
                    edges.append({
                        'source_cui': parent_cui,
                        'source_name': record['parent_name'],
                        'target_cui': node_cui,
                        'target_name': record['node_name'],
                        'rel_type': rel_type,
                        'rela': rela,
                        'weight': 1.0
                    })
        
        # Identify Steiner nodes
        steiner_nodes = [
            cui for cui in nodes.keys() 
            if cui not in cui_list
        ]
        
        return {
            'nodes': list(nodes.values()),
            'edges': edges,
            'steiner_nodes': [
                nodes[cui] for cui in steiner_nodes
            ],
            'metadata': {
                'approach': 'steiner_tree_gds',
                'node_count': len(nodes),
                'edge_count': len(edges),
                'input_cui_count': len(cui_list),
                'steiner_node_count': len(steiner_nodes),
                'removed_hubs': 0  # GDS handles this internally
            }
        }
    
    def _expand_hierarchy(
        self,
        anchor_cui: str,
        nodes: Dict[str, Any],
        edges: List[Dict[str, Any]],
        depth: int = 2
    ) -> None:
        """
        Expand vertically through isa/CHD/PAR hierarchy.
        """
        # Get parents
        parents = self.query.get_concept_hierarchy(
            anchor_cui, depth=depth, direction='parents'
        )
        
        for parent in parents:
            cui = parent['cui']
            if cui not in nodes:
                nodes[cui] = self.query.get_entity_by_cui(cui)
            
            edges.append({
                'source_cui': anchor_cui,
                'source_name': nodes.get(anchor_cui, {}).get('name'),
                'target_cui': cui,
                'target_name': parent['name'],
                'rel_type': parent.get('rel_type', 'isa'),
                'weight': 0.1  # Low weight = high priority
            })
        
        # Get children
        children = self.query.get_concept_hierarchy(
            anchor_cui, depth=depth, direction='children'
        )
        
        for child in children:
            cui = child['cui']
            if cui not in nodes:
                nodes[cui] = self.query.get_entity_by_cui(cui)
            
            edges.append({
                'source_cui': cui,
                'source_name': child['name'],
                'target_cui': anchor_cui,
                'target_name': nodes.get(anchor_cui, {}).get('name'),
                'rel_type': child.get('rel_type', 'isa'),
                'weight': 0.1
            })
    
    def _expand_semantic_relations(
        self,
        cui_list: List[str],
        nodes: Dict[str, Any],
        edges: List[Dict[str, Any]]
    ) -> None:
        """
        Expand horizontally through semantic relations.
        """
        for cui in cui_list:
            if cui not in nodes:
                continue
            
            # Get filtered neighbors with semantic relations
            neighbors = self.query.get_neighbors_filtered(
                cui,
                allowed_rels=list(self.SEMANTIC_RELS),
                limit=20
            )
            
            for neighbor in neighbors:
                neighbor_cui = neighbor['cui']
                
                # Only include if it's one of our extracted concepts
                # or if it's a relevant semantic type
                if neighbor_cui in cui_list or self._is_relevant_type(neighbor_cui):
                    if neighbor_cui not in nodes:
                        nodes[neighbor_cui] = self.query.get_entity_by_cui(neighbor_cui)
                    
                    edges.append({
                        'source_cui': cui,
                        'source_name': nodes[cui].get('name'),
                        'target_cui': neighbor_cui,
                        'target_name': neighbor['name'],
                        'rel_type': neighbor.get('rela', 'RELATED_TO'),
                        'weight': 0.1  # Semantic relations are high priority
                    })
    
    def _filter_generic_hubs(
        self,
        nodes: Dict[str, Any],
        edges: List[Dict[str, Any]],
        threshold: int = 50
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Remove nodes that are generic hubs (too many connections).
        
        Generic nodes like "Disease" or "Body Part" don't add value.
        """
        # Count connections for each node in the UMLS database
        node_connection_counts = {}
        
        for cui in nodes.keys():
            # Get total relationship count from database
            rels = self.query.get_entity_relationships(cui, limit=threshold + 1)
            node_connection_counts[cui] = len(rels)
        
        # Filter out hubs
        hub_cuis = {
            cui for cui, count in node_connection_counts.items()
            if count > threshold
        }
        
        filtered_nodes = {
            cui: data for cui, data in nodes.items()
            if cui not in hub_cuis
        }
        
        filtered_edges = [
            edge for edge in edges
            if edge['source_cui'] not in hub_cuis
            and edge['target_cui'] not in hub_cuis
        ]
        
        return filtered_nodes, filtered_edges
    
    def _is_neoplastic(self, cui: str) -> bool:
        """Check if a CUI is a neoplastic process (T191)."""
        # This would require querying the semantic type
        # For now, we'll use a simple heuristic
        entity = self.query.get_entity_by_cui(cui)
        if not entity:
            return False
        
        # Check if name contains cancer/carcinoma/tumor keywords
        name = entity.get('name', '').lower()
        keywords = ['carcinoma', 'cancer', 'tumor', 'neoplasm', 'adenocarcinoma', 
                   'sarcoma', 'lymphoma', 'leukemia', 'melanoma']
        return any(kw in name for kw in keywords)
    
    def _is_relevant_type(self, cui: str) -> bool:
        """Check if a CUI has a relevant semantic type."""
        # This would require querying the semantic type from the database
        # For now, we'll allow all - in production, you'd check TUI
        # against ALLOWED_TUIS
        return True
    
    def export_to_json(self, subgraph: Dict[str, Any], filepath: str) -> None:
        """Export subgraph to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(subgraph, f, indent=2)
    
    def export_to_graphml(self, subgraph: Dict[str, Any], filepath: str) -> None:
        """Export subgraph to GraphML format for visualization."""
        # Build GraphML XML
        lines = ['<?xml version="1.0" encoding="UTF-8"?>']
        lines.append('<graphml xmlns="http://graphml.graphdrawing.org/xmlns">')
        lines.append('  <graph id="G" edgedefault="directed">')
        
        # Add nodes
        for node in subgraph['nodes']:
            cui = node['cui']
            name = node.get('name', 'Unknown')
            lines.append(f'    <node id="{cui}">')
            lines.append(f'      <data key="name">{self._escape_xml(name)}</data>')
            lines.append('    </node>')
        
        # Add edges
        for i, edge in enumerate(subgraph['edges']):
            source = edge.get('source_cui')
            target = edge.get('target_cui')
            rel_type = edge.get('rel_type', 'RELATED_TO')
            lines.append(f'    <edge id="e{i}" source="{source}" target="{target}">')
            lines.append(f'      <data key="type">{rel_type}</data>')
            lines.append('    </edge>')
        
        lines.append('  </graph>')
        lines.append('</graphml>')
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))
    
    def _escape_xml(self, text: str) -> str:
        """Escape XML special characters."""
        return (text.replace('&', '&amp;')
                   .replace('<', '&lt;')
                   .replace('>', '&gt;')
                   .replace('"', '&quot;')
                   .replace("'", '&apos;'))
    
    def compare_approaches(
        self,
        cui_list: List[str]
    ) -> Dict[str, Any]:
        """
        Run both approaches and return comparison metrics.
        
        Args:
            cui_list: List of CUI identifiers
            
        Returns:
            Dictionary with both subgraphs and comparison data
        """
        # Build both graphs
        medical_graph = self.build_medical_ruled_graph(cui_list)
        steiner_graph = self.build_steiner_tree_graph(cui_list, use_gds=False)
        
        # Compute comparison metrics
        comparison = {
            'approach_a_medical_ruled': medical_graph,
            'approach_b_steiner_tree': steiner_graph,
            'comparison': {
                'medical_ruled': {
                    'nodes': len(medical_graph['nodes']),
                    'edges': len(medical_graph['edges']),
                    'discovered': medical_graph['metadata']['discovered_node_count']
                },
                'steiner_tree': {
                    'nodes': len(steiner_graph['nodes']),
                    'edges': len(steiner_graph['edges']),
                    'steiner_nodes': steiner_graph['metadata']['steiner_node_count']
                },
                'shared_nodes': len(
                    set(n['cui'] for n in medical_graph['nodes']) &
                    set(n['cui'] for n in steiner_graph['nodes'])
                ),
                'unique_to_medical': len(
                    set(n['cui'] for n in medical_graph['nodes']) -
                    set(n['cui'] for n in steiner_graph['nodes'])
                ),
                'unique_to_steiner': len(
                    set(n['cui'] for n in steiner_graph['nodes']) -
                    set(n['cui'] for n in medical_graph['nodes'])
                )
            }
        }
        
        return comparison
