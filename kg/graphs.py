from neo4j import GraphDatabase
import csv
import sys
import json


# Increase field size limit
maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)


class Neo4jLoader:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def create_constraints(self, database="neo4j"):
        """Create uniqueness constraint on CUI to improve performance"""
        with self.driver.session(database=database) as session:
            try:
                session.run("CREATE CONSTRAINT concept_cui IF NOT EXISTS FOR (c:Concept) REQUIRE c.cui IS UNIQUE")
            except Exception as e:
                print(f"Constraint may already exist: {e}")
    
    def load_nodes(self, csv_file, batch_size=1000000, database="neo4j"):
        """Load nodes from CSV file in batches"""
        
        with self.driver.session(database=database) as session:
            batch = []
            count = 0
            
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    # Convert header format: 'cui:ID' -> 'cui', ':LABEL' -> 'LABEL'
                    cui = row.get('cui:ID')
                    name = row.get('name')
                    label = row.get(':LABEL', 'Concept')
                    
                    if cui:
                        batch.append({'cui': cui, 'name': name})
                        count += 1
                        
                        if len(batch) >= batch_size:
                            self._create_nodes_batch(session, batch)
                            print(f"  Loaded {count} nodes...")
                            batch = []
                
                # Load remaining nodes
                if batch:
                    self._create_nodes_batch(session, batch)
                
    def _create_nodes_batch(self, session, batch):
        """Create a batch of nodes using UNWIND"""
        session.run("""
            UNWIND $batch AS node
            MERGE (c:Concept {cui: node.cui})
            SET c.name = node.name
        """, batch=batch)
    
    def load_relationships(self, csv_file, batch_size=1000000, database="neo4j"):
        """Load relationships from CSV file in batches"""        
        with self.driver.session(database=database) as session:
            batch = []
            count = 0
            
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    # Convert header format: ':START_ID' -> start_id, ':END_ID' -> end_id, ':TYPE' -> rel_type
                    start_id = row.get(':START_ID')
                    end_id = row.get(':END_ID')
                    rel_type = row.get(':TYPE', 'RELATED_TO')
                    rela = row.get('RELA', '')
                    
                    if start_id and end_id:
                        batch.append({'start_id': start_id, 'end_id': end_id,
                            'rel_type': rel_type, 'rela': rela})
                        count += 1
                        
                        if len(batch) >= batch_size:
                            self._create_relationships_batch(session, batch)
                            print(f"  Loaded {count} relationships...")
                            batch = []
                
                # Load remaining relationships
                if batch:
                    self._create_relationships_batch(session, batch)
                
    def _create_relationships_batch(self, session, batch):
        """Create a batch of relationships using UNWIND (without APOC)"""
        session.run("""
            UNWIND $batch AS rel
            MATCH (a:Concept {cui: rel.start_id})
            MATCH (b:Concept {cui: rel.end_id})
            CREATE (a)-[r:RELATED_TO {type: rel.rel_type, rela: rel.rela}]->(b)
        """, batch=batch)


class Neo4jQuery:
    def __init__(self, uri, user, password):
        from neo4j import NotificationDisabledCategory
        self.driver = GraphDatabase.driver(
            uri, 
            auth=(user, password),
            notifications_disabled_categories=[NotificationDisabledCategory.DEPRECATION]
        )
    
    def close(self):
        self.driver.close()
    
    def search_entity(self, search_term, database="neo4j"):
        """Search for entities by name"""
        with self.driver.session(database=database) as session:
            result = session.run("""
                MATCH (c:Concept)
                WHERE toLower(c.name) CONTAINS toLower($search_term)
                RETURN c.cui as cui, c.name as name
                LIMIT 20
            """, search_term=search_term)
            
            entities = [dict(record) for record in result]
            return entities
    
    def get_entity_by_cui(self, cui, database="neo4j"):
        """Get entity by CUI"""
        with self.driver.session(database=database) as session:
            result = session.run("""
                MATCH (c:Concept {cui: $cui})
                RETURN c.cui as cui, c.name as name
            """, cui=cui)
            
            entity = result.single()
            return dict(entity) if entity else None
    
    def get_entity_relationships(self, cui, limit=50, database="neo4j"):
        """Get all relationships for an entity"""
        with self.driver.session(database=database) as session:
            result = session.run("""
                MATCH (c:Concept {cui: $cui})-[r]-(related:Concept)
                RETURN c.cui as source_cui, c.name as source_name,
                       type(r) as rel_type, r.type as rel_subtype, r.rela as rela,
                       related.cui as target_cui, related.name as target_name
                LIMIT $limit
            """, cui=cui, limit=limit)
            
            relationships = [dict(record) for record in result]
            return relationships
    
    def get_statistics(self, database="neo4j"):
        """Get database statistics"""
        with self.driver.session(database=database) as session:
            # Count nodes
            nodes_result = session.run("MATCH (c:Concept) RETURN count(c) as count")
            nodes_count = nodes_result.single()["count"]
            
            # Count relationships
            rels_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            rels_count = rels_result.single()["count"]
            
            # Get relationship types distribution (sample for speed)
            rel_types_result = session.run("""
                MATCH ()-[r]->()
                WITH r.type as rel_type
                LIMIT 100000
                RETURN rel_type, count(*) as count
                ORDER BY count DESC
                LIMIT 10
            """)
            rel_types = [dict(record) for record in rel_types_result]
            
            return {
                "total_concepts": nodes_count,
                "total_relationships": rels_count,
                "top_relationship_types": rel_types
            }
    
    
    def find_shortest_path(self, source_cui, target_cui, max_depth=5, database="neo4j"):
        """Find shortest path between two entities"""
        with self.driver.session(database=database) as session:
            result = session.run("""
                MATCH path = shortestPath(
                    (source:Concept {cui: $source_cui})-[*..%s]-(target:Concept {cui: $target_cui})
                )
                WITH path, 
                     [n in nodes(path) | {cui: n.cui, name: n.name}] as node_list,
                     [r in relationships(path) | {type: type(r), rela: r.rela}] as rel_list
                RETURN node_list, rel_list, length(path) as path_length
            """ % max_depth, source_cui=source_cui, target_cui=target_cui)
            
            path_data = result.single()
            if path_data:
                return {
                    "nodes": path_data["node_list"],
                    "relationships": path_data["rel_list"],
                    "length": path_data["path_length"]
                }
            return None
    
    def find_all_paths(self, source_cui, target_cui, max_depth=4, limit=10, database="neo4j"):
        """Find all paths between two entities (limited)"""
        with self.driver.session(database=database) as session:
            result = session.run("""
                MATCH path = (source:Concept {cui: $source_cui})-[*..%s]-(target:Concept {cui: $target_cui})
                WITH path, 
                     [n in nodes(path) | {cui: n.cui, name: n.name}] as node_list,
                     [r in relationships(path) | {type: type(r), rela: r.rela}] as rel_list
                RETURN node_list, rel_list, length(path) as path_length
                ORDER BY path_length
                LIMIT $limit
            """ % max_depth, source_cui=source_cui, target_cui=target_cui, limit=limit)
            
            paths = []
            for record in result:
                paths.append({
                    "nodes": record["node_list"],
                    "relationships": record["rel_list"],
                    "length": record["path_length"]
                })
            return paths
    
    def get_neighbors(self, cui, rel_type=None, direction="both", limit=20, database="neo4j"):
        """Get neighbors of a concept with optional filtering"""
        direction_clause = {
            "outgoing": "-[r]->",
            "incoming": "<-[r]-",
            "both": "-[r]-"
        }.get(direction, "-[r]-")
        
        type_filter = f"AND type(r) = '{rel_type}'" if rel_type else ""
        
        with self.driver.session(database=database) as session:
            result = session.run(f"""
                MATCH (c:Concept {{cui: $cui}}){direction_clause}(neighbor:Concept)
                WHERE c <> neighbor {type_filter}
                RETURN neighbor.cui as cui, neighbor.name as name, 
                       type(r) as rel_type, r.rela as rela
                LIMIT $limit
            """, cui=cui, limit=limit)
            
            neighbors = [dict(record) for record in result]
            return neighbors
    
    
    def get_subgraph(self, cui_list, database="neo4j"):
        """Get subgraph connecting a list of concepts"""
        with self.driver.session(database=database) as session:
            result = session.run("""
                MATCH (source:Concept)-[r]->(target:Concept)
                WHERE source.cui IN $cui_list AND target.cui IN $cui_list
                RETURN source.cui as source_cui, source.name as source_name,
                       type(r) as rel_type, r.rela as rela,
                       target.cui as target_cui, target.name as target_name
            """, cui_list=cui_list)
            
            edges = [dict(record) for record in result]
            return edges
    
    def get_concept_hierarchy(self, cui, depth=2, direction="children", database="neo4j"):
        """Get hierarchical relationships (parent/child)"""
        if direction == "parents":
            rel_pattern = "<-[r:CHD|PAR|isa]-"
        else:  # children
            rel_pattern = "-[r:CHD|PAR|isa]->"
        
        with self.driver.session(database=database) as session:
            result = session.run(f"""
                MATCH path = (c:Concept {{cui: $cui}}){rel_pattern}(related:Concept)
                WHERE length(path) <= $depth
                RETURN related.cui as cui, related.name as name,
                       type(r) as rel_type, length(path) as level
                ORDER BY level, related.name
                LIMIT 100
            """, cui=cui, depth=depth)
            
            hierarchy = [dict(record) for record in result]
            return hierarchy
    
    def steiner_tree_retrieval(self, cui_list, source_cui=None, relationship_weight_property=None, 
                              apply_rerouting=False, delta=2.0, database="neo4j"):
        """
        Find the Steiner tree connecting a list of CUI entities using Neo4j GDS.
        
        The Steiner tree is a minimal tree that connects all specified vertices (target nodes)
        through a source node, potentially including intermediate nodes.
        
        Args:
            cui_list: List of CUI identifiers for target concepts to connect
            source_cui: Optional source CUI. If not provided, the first CUI in the list is used
            relationship_weight_property: Optional property name for relationship weights (e.g., 'weight', 'cost')
            apply_rerouting: If True, applies post-processing heuristic to improve the tree
            delta: Bucket width for grouping nodes (See Delta-Stepping algorithm)
            database: Neo4j database name
            
        Returns:
            Dictionary containing:
                - source_cui: The source node CUI
                - source_name: The source node name
                - total_weight: Sum of all relationship weights in the tree
                - tree_nodes: List of all nodes in the tree (with cui and name)
                - tree_edges: List of all edges in the tree (with source, target, weight, and relationship info)
                - effective_node_count: Number of nodes in the resulting tree
                - target_count: Number of target nodes successfully reached
        """
        if not cui_list or len(cui_list) < 2:
            raise ValueError("cui_list must contain at least 2 CUI identifiers")
        
        # Use first CUI as source if not provided, rest as targets
        if source_cui is None:
            source_cui = cui_list[0]
            target_cuis = cui_list[1:]
        else:
            target_cuis = [cui for cui in cui_list if cui != source_cui]
        
        if not target_cuis:
            raise ValueError("At least one target CUI is required (different from source)")
        
        with self.driver.session(database=database) as session:
            # First, get the actual node IDs for source and targets
            source_result = session.run(
                "MATCH (c:Concept {cui: $cui}) RETURN id(c) as nodeId, c.name as name",
                cui=source_cui
            )
            source_record = source_result.single()
            if not source_record:
                raise ValueError(f"Source CUI '{source_cui}' not found in database")
            
            source_node_id = source_record["nodeId"]
            source_name = source_record["name"]
            
            # Get target node IDs
            target_result = session.run(
                "UNWIND $cuis as cui MATCH (c:Concept {cui: cui}) RETURN id(c) as nodeId, cui",
                cuis=target_cuis
            )
            target_records = [dict(record) for record in target_result]
            target_node_ids = [record["nodeId"] for record in target_records]
            
            if len(target_node_ids) != len(target_cuis):
                found_cuis = [record["cui"] for record in target_records]
                missing = set(target_cuis) - set(found_cuis)
                raise ValueError(f"Some target CUIs not found in database: {missing}")
            
            # Build configuration for Steiner tree
            config = {
                "sourceNode": source_node_id,
                "targetNodes": target_node_ids,
                "delta": delta,
                "applyRerouting": apply_rerouting
            }
            
            if relationship_weight_property:
                config["relationshipWeightProperty"] = relationship_weight_property
            
            # Run Steiner tree algorithm using GDS
            # Note: This requires the graph to be projected into GDS graph catalog
            # For direct Cypher queries without GDS projection, we'll use a custom implementation
            method_used = "gds"
            try:
                # Try using GDS if available
                result = session.run("""
                    CALL gds.steinerTree.stream($graph_name, $config)
                    YIELD nodeId, parentId, weight
                    RETURN nodeId, parentId, weight
                """, graph_name="graph", config=config)
                
                # Process results
                tree_records = [dict(record) for record in result]
                
            except Exception as e:
                # Fall back to custom Cypher implementation if GDS is not available
                # This implements a simple approximation using shortest paths
                method_used = "cypher"
                tree_records = self._steiner_tree_cypher_fallback(
                    session, source_node_id, target_node_ids, relationship_weight_property
                )
            
            # Build the tree structure from records
            tree_nodes = []
            tree_edges = []
            total_weight = 0.0
            node_ids_in_tree = set()
            
            for record in tree_records:
                node_id = record["nodeId"]
                parent_id = record["parentId"]
                weight = record.get("weight", 0.0)
                
                node_ids_in_tree.add(node_id)
                
                # Skip if node is its own parent (root node)
                if node_id == parent_id:
                    continue
                
                node_ids_in_tree.add(parent_id)
                total_weight += weight
                
                # Get node and relationship details
                edge_result = session.run("""
                    MATCH (parent)-[r]->(child)
                    WHERE id(parent) = $parent_id AND id(child) = $node_id
                    RETURN parent.cui as source_cui, parent.name as source_name,
                           child.cui as target_cui, child.name as target_name,
                           type(r) as rel_type, r.rela as rela, r.type as rel_subtype
                    LIMIT 1
                """, parent_id=parent_id, node_id=node_id)
                
                edge_record = edge_result.single()
                if edge_record:
                    tree_edges.append({
                        "source_cui": edge_record["source_cui"],
                        "source_name": edge_record["source_name"],
                        "target_cui": edge_record["target_cui"],
                        "target_name": edge_record["target_name"],
                        "rel_type": edge_record["rel_type"] or "RELATED_TO",
                        "rela": edge_record["rela"],
                        "rel_subtype": edge_record["rel_subtype"],
                        "weight": weight
                    })
            
            # Get all nodes in the tree
            if node_ids_in_tree:
                nodes_result = session.run("""
                    MATCH (c:Concept)
                    WHERE id(c) IN $node_ids
                    RETURN c.cui as cui, c.name as name
                """, node_ids=list(node_ids_in_tree))
                
                tree_nodes = [dict(record) for record in nodes_result]
            
            return {
                "source_cui": source_cui,
                "source_name": source_name,
                "total_weight": total_weight,
                "tree_nodes": tree_nodes,
                "tree_edges": tree_edges,
                "effective_node_count": len(tree_nodes),
                "target_count": len(target_cuis),
                "method": method_used
            }
    
    def _steiner_tree_cypher_fallback(self, session, source_node_id, target_node_ids, weight_property=None):
        """
        Fallback Steiner tree implementation using pure Cypher (when GDS is not available).
        Uses a greedy approximation: finds shortest paths from source to each target.
        """
        tree_records = []
        visited_edges = set()
        
        # Add source as root
        tree_records.append({
            "nodeId": source_node_id,
            "parentId": source_node_id,
            "weight": 0.0
        })
        
        for target_id in target_node_ids:
            # Find shortest path from source to this target and get all nodes in the path
            result = session.run("""
                MATCH path = shortestPath((source)-[*..10]->(target))
                WHERE id(source) = $source_id AND id(target) = $target_id
                RETURN [node IN nodes(path) | id(node)] as nodeIds,
                       [rel IN relationships(path) | id(rel)] as relIds
            """, source_id=source_node_id, target_id=target_id)
            
            path_record = result.single()
            if not path_record:
                continue  # No path found to this target
            
            node_ids = path_record["nodeIds"]
            
            # Add edges from this path (skip first node as it's the source)
            for i in range(1, len(node_ids)):
                parent_id = node_ids[i-1]
                node_id = node_ids[i]
                edge_key = (parent_id, node_id)
                
                if edge_key not in visited_edges:
                    visited_edges.add(edge_key)
                    
                    # Get weight for this edge
                    if weight_property:
                        weight_result = session.run(f"""
                            MATCH (parent)-[r]->(child)
                            WHERE id(parent) = $parent_id AND id(child) = $node_id
                            RETURN r.{weight_property} as weight
                            LIMIT 1
                        """, parent_id=parent_id, node_id=node_id)
                        weight_record = weight_result.single()
                        weight = weight_record["weight"] if weight_record else 1.0
                    else:
                        weight = 1.0
                    
                    tree_records.append({
                        "nodeId": node_id,
                        "parentId": parent_id,
                        "weight": weight
                    })
        
        return tree_records
    
if __name__ == '__main__':
    NEO4J_URL = "neo4j://127.0.0.1:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "11111111"  # Change this to your Neo4j password
    
    NODES_CSV = "nodes.csv"
    RELATIONSHIPS_CSV = "relationships.csv"
    
    # ============================================================
    # EXPERIMENT 1: Load data (uncomment to load)
    # ============================================================
    # loader = Neo4jLoader(NEO4J_URL, NEO4J_USER, NEO4J_PASSWORD)    
    # loader.create_constraints()
    # loader.load_nodes(NODES_CSV, batch_size=100000)
    # loader.load_relationships(RELATIONSHIPS_CSV, batch_size=100000)
    # loader.close()

    query = Neo4jQuery(NEO4J_URL, NEO4J_USER, NEO4J_PASSWORD)
    
    # ============================================================
    # EXPERIMENT 2: Database Statistics
    # ============================================================
    print("\n" + "="*60)
    print("EXPERIMENT 2: Database Statistics")
    print("="*60)
    stats = query.get_statistics()
    print(json.dumps(stats, indent=2))
    
    # ============================================================
    # EXPERIMENT 3: Search for entities
    # ============================================================
    print("\n" + "="*60)
    print("EXPERIMENT 3: Search for entities")
    print("="*60)
    search_term = "carcinoma"  # Change this to search for different terms
    print(f"Searching for: '{search_term}'")
    entities = query.search_entity(search_term)
    print(f"Found {len(entities)} entities:")
    for i, entity in enumerate(entities[:5], 1):  # Show first 5
        print(f"  {i}. {entity['cui']}: {entity['name']}")
    
    # ============================================================
    # EXPERIMENT 4: Get entity relationships
    # ============================================================
    if entities:
        print("\n" + "="*60)
        print("EXPERIMENT 4: Entity Relationships")
        print("="*60)
        test_cui = entities[0]['cui']
        print(f"Getting relationships for: {test_cui} ({entities[0]['name']})")
        relationships = query.get_entity_relationships(test_cui, limit=10)
        print(f"Found {len(relationships)} relationships:")
        for i, rel in enumerate(relationships[:5], 1):  # Show first 5
            print(f"  {i}. {rel['source_name']} --[{rel['rel_type']}/{rel['rela']}]--> {rel['target_name']}")
    
    # ============================================================
    # EXPERIMENT 5: Get neighbors
    # ============================================================
    if entities:
        print("\n" + "="*60)
        print("EXPERIMENT 5: Get Neighbors")
        print("="*60)
        test_cui = entities[0]['cui']
        print(f"Getting neighbors for: {test_cui}")
        neighbors = query.get_neighbors(test_cui, direction="both", limit=10)
        print(f"Found {len(neighbors)} neighbors:")
        for i, neighbor in enumerate(neighbors[:5], 1):
            print(f"  {i}. {neighbor['name']} (via {neighbor['rel_type']})")
    
    # ============================================================
    # EXPERIMENT 6: Find shortest path between two concepts
    # ============================================================
    if len(entities) >= 2:
        print("\n" + "="*60)
        print("EXPERIMENT 6: Shortest Path")
        print("="*60)
        source_cui = entities[0]['cui']
        target_cui = entities[1]['cui']
        print(f"Finding path from {entities[0]['name']} to {entities[1]['name']}")
        path = query.find_shortest_path(source_cui, target_cui, max_depth=5)
        if path:
            print(f"Path length: {path['length']}")
            print("Path:")
            for i, node in enumerate(path['nodes']):
                print(f"  {i+1}. {node['name']} ({node['cui']})")
                if i < len(path['relationships']):
                    print(f"      └─[{path['relationships'][i]['type']}]→")
        else:
            print("No path found")
    
    
    # ============================================================
    # EXPERIMENT 9: Get subgraph
    # ============================================================
    if len(entities) >= 3:
        print("\n" + "="*60)
        print("EXPERIMENT 9: Subgraph Extraction")
        print("="*60)
        cui_list = [e['cui'] for e in entities[:3]]
        print(f"Getting subgraph for {len(cui_list)} concepts")
        subgraph = query.get_subgraph(cui_list)
        print(f"Found {len(subgraph)} edges in subgraph:")
        for edge in subgraph[:5]:
            print(f"  {edge['source_name']} --[{edge['rel_type']}]--> {edge['target_name']}")
    
    # ============================================================
    # EXPERIMENT 10: Steiner Tree Extraction
    # ============================================================
    if len(entities) >= 3:
        print("\n" + "="*60)
        print("EXPERIMENT 10: Steiner Tree Extraction")
        print("="*60)
        
        # Include the first 4 carcinoma entities plus 2 distant concepts
        cui_list = [e['cui'] for e in entities[:4]]  # First 4 carcinomas
        
        # Add 2 conceptually distant nodes to test longer paths
        # Search for inflammation (common condition, likely farther from carcinoma)
        distant_concepts = query.search_entity("inflammation")
        if distant_concepts:
            cui_list.append(distant_concepts[0]['cui'])
        
        # Search for diabetes (metabolic disease, very different from carcinoma)
        metabolic_concepts = query.search_entity("diabetes mellitus")
        if metabolic_concepts:
            cui_list.append(metabolic_concepts[0]['cui'])
        
        print(f"Building Steiner tree for {len(cui_list)} concepts:")
        print(f"  (Including {len(cui_list) - 4} distant nodes to test path discovery)\n")
        for i, cui in enumerate(cui_list, 1):
            entity = query.get_entity_by_cui(cui)
            if entity:
                marker = " [DISTANT]" if i > 4 else ""
                print(f"  {i}. {entity['name']} ({cui}){marker}")
        
        try:
            tree = query.steiner_tree_retrieval(cui_list)
            print(f"\nSteiner Tree Results (using {tree['method'].upper()} method):")
            print(f"  - Optimal source: {tree['source_cui']}")
            entity = query.get_entity_by_cui(tree['source_cui'])
            if entity:
                print(f"    ({entity['name']})")
            print(f"  - Total weight: {tree['total_weight']}")
            print(f"  - Nodes in tree: {len(tree['tree_nodes'])}")
            print(f"  - Edges in tree: {len(tree['tree_edges'])}")
            print(f"  - Steiner nodes discovered: {len(tree['tree_nodes']) - len(cui_list)}")
            
            print(f"\nTree structure:")
            for edge in tree['tree_edges']:
                rel_info = f"{edge.get('rel_type', 'RELATED_TO')}"
                if edge.get('rela'):
                    rel_info += f"/{edge['rela']}"
                print(f"  {edge['source_name']} --[{rel_info}, w={edge['weight']}]--> {edge['target_name']}")
            
            print(f"\nAll nodes in tree:")
            original_cuis = set(cui_list)
            for node in tree['tree_nodes']:
                marker = " [STEINER NODE]" if node['cui'] not in original_cuis else ""
                print(f"  - {node['name']} ({node['cui']}){marker}")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    query.close()
    print("\n" + "="*60)
    print("All experiments completed!")
    print("="*60)


