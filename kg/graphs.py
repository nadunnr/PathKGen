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
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
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


if __name__ == '__main__':
    NEO4J_URI = "http://localhost:7474"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "11111111"  # Change this to your Neo4j password
    
    NODES_CSV = "nodes.csv"
    RELATIONSHIPS_CSV = "relationships.csv"
    
    # Load data
    # loader = Neo4jLoader(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)    
    # loader.create_constraints()
    # loader.load_nodes(NODES_CSV, batch_size=100000)
    # loader.load_relationships(RELATIONSHIPS_CSV, batch_size=100000)
    # loader.close()

    query = Neo4jQuery(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    stats = query.get_statistics()
    print("Database Statistics:", json.dumps(stats, indent=2))


