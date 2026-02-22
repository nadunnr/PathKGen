from neo4j import GraphDatabase
import csv
from collections import defaultdict


class Neo4jQuery:
    def __init__(self, uri, user, password, database="jan27"):
        from neo4j import NotificationDisabledCategory
        self.driver = GraphDatabase.driver(
            uri, 
            auth=(user, password),
            notifications_disabled_categories=[NotificationDisabledCategory.DEPRECATION]
        )
        self.database = database
    
    def close(self):
        self.driver.close()
    
    def create_relationship(self, source_cui, target_cui, rel_type, rela=None):
        """Create a relationship between two concepts"""
        with self.driver.session(database=self.database) as session:
            session.run("""
                MATCH (source:Concept {cui: $source_cui})
                MATCH (target:Concept {cui: $target_cui})
                CREATE (source)-[r:RELATED {type: $rel_type, rela: $rela}]->(target)
                RETURN r
            """, source_cui=source_cui, target_cui=target_cui, rel_type=rel_type, rela=rela)
    
    def create_relationships_batch(self, seed_cui, target_cuis, rel_type, rela=None):
        """Create relationships from seed CUI to multiple target CUIs"""
        with self.driver.session(database=self.database) as session:
            for target_cui in target_cuis:
                session.run("""
                    MATCH (source:Concept {cui: $source_cui})
                    MATCH (target:Concept {cui: $target_cui})
                    CREATE (source)-[r:RELATED {type: $rel_type, rela: $rela}]->(target)
                    RETURN r
                """, source_cui=seed_cui, target_cui=target_cui, rel_type=rel_type, rela=rela)
    
    def create_relationships_batch_efficient(self, seed_cui, target_cuis, rel_type, rela=None):
        """Create relationships from seed CUI to multiple target CUIs (single query)"""
        with self.driver.session(database=self.database) as session:
            session.run("""
                MATCH (source:Concept {cui: $source_cui})
                MATCH (targets:Concept) WHERE targets.cui IN $target_cuis
                CREATE (source)-[r:RELATED {type: $rel_type, rela: $rela}]->(targets)
                RETURN count(r) as created
            """, source_cui=seed_cui, target_cuis=target_cuis, rel_type=rel_type, rela=rela)
    
    def find_shortest_path(self, source_cui, target_cui):
        """Find the shortest path between two concepts"""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (source:Concept {cui: $source_cui}), (target:Concept {cui: $target_cui})
                MATCH p = shortestPath((source)-[*]-(target))
                RETURN p
            """, source_cui=source_cui, target_cui=target_cui)
            return result.single()[0] if result.single() else None



class Neo4jLoader:
    def __init__(self, uri, user, password, database="neo4jv8"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
    
    def close(self):
        self.driver.close()
    
    def create_constraints(self):
        """Create uniqueness constraint on CUI to improve performance"""
        with self.driver.session(database=self.database) as session:
            try:
                session.run("CREATE CONSTRAINT concept_cui IF NOT EXISTS FOR (c:Concept) REQUIRE c.cui IS UNIQUE")
            except Exception as e:
                print(f"Constraint may already exist: {e}")
    
    def load_nodes(self, csv_file, batch_size=1000000):
        """Load nodes from CSV file in batches"""
        
        with self.driver.session(database=self.database) as session:
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
    
    def load_relationships(self, csv_file, batch_size=1000000):
        """Load relationships from CSV file in batches"""        
        with self.driver.session(database=self.database) as session:
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
        # Group relationships by type to create them with proper relationship types
        by_type = defaultdict(list)
        for rel in batch:
            # Use rela if available, otherwise use rel_type, or default to RELATED_TO
            # rel_key = rel['rela'] if rel['rela'] else (rel.get('rel_type') or 'RELATED_TO')
            # Use rela if available, otherwise 'RELATED_TO'
            rel_key = rel['rela'] if rel['rela'] else (rel.get('rel_type') or 'RELATED_TO')
            by_type[rel_key].append(rel)
        
        # Create relationships for each type separately
        for rel_type, rels in by_type.items():
            # Sanitize relationship type to ensure it's a valid Cypher identifier
            safe_rel_type = rel_type.replace('-', '_').replace(' ', '_').replace('/', '_')
            # Skip if empty after sanitization
            if not safe_rel_type:
                safe_rel_type = 'RELATED_TO'
            session.run(f"""
                UNWIND $batch AS rel
                MATCH (a:Concept {{cui: rel.start_id}})
                MATCH (b:Concept {{cui: rel.end_id}})
                CREATE (a)-[r:{safe_rel_type} {{rela: rel.rela}}]->(b)
            """, batch=rels)
            
    def update_node_definitions(self, json_file):
        """Update existing nodes with definitions from JSON file"""
        import json
        
        with self.driver.session(database=self.database) as session:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                updated = 0
                
                for item in data:
                    cui = item.get('cui')
                    definition = item.get('description', '')
                    
                    if cui and definition:
                        result = session.run("""
                            MATCH (c:Concept {cui: $cui})
                            SET c.definition = $definition
                            RETURN count(c) as updated
                        """, cui=cui, definition=definition)
                        updated += result.single()['updated']
                
                print(f"Updated {updated} nodes with definitions")
    
    def update_node_semantic_types(self, csv_file, batch_size=1000):
        """Update existing nodes with semantic types from CSV file"""
        # Read TUI and TUI names for each CUI
        cui_tui_map = {}
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                cui = row.get('cui:ID')
                tui = row.get('tui', '')
                tui_name = row.get('semantic_type', '')
                
                if cui and tui:
                    cui_tui_map[cui] = {'tui': tui, 'tui_name': tui_name}
        
        # Now update nodes in batches
        with self.driver.session(database=self.database) as session:
            batch = []
            count = 0
            updated = 0
            
            for cui, tui_data in cui_tui_map.items():
                batch.append({
                    'cui': cui,
                    'tui': tui_data['tui'],
                    'tui_name': tui_data['tui_name']
                })
                count += 1
                
                if len(batch) >= batch_size:
                    result = self._update_semantic_types_batch(session, batch)
                    updated += result
                    print(f"  Processed {count} CUIs, updated {updated} nodes...")
                    batch = []
            
            # Update remaining nodes
            if batch:
                result = self._update_semantic_types_batch(session, batch)
                updated += result
                
        print(f"Total processed: {count} CUIs, updated: {updated} nodes")
    
    def _update_semantic_types_batch(self, session, batch):
        """Update a batch of nodes with TUI codes and semantic type names"""
        result = session.run("""
            UNWIND $batch AS item
            MATCH (c:Concept {cui: item.cui})
            SET c.tui = item.tui, c.tui_name = item.tui_name
            RETURN count(c) as updated
        """, batch=batch)
        return result.single()['updated']



if __name__ == '__main__':
    NEO4J_URL = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "11111111"

    loader = Neo4jLoader(NEO4J_URL, NEO4J_USER, NEO4J_PASSWORD)
    loader.create_constraints()
    loader.update_node_semantic_types('/home/nadun/wd/datasets/UMLS_sql/neo4jdb/v8/semantic_types.csv')
    # loader.update_node_definitions('/home/nadun/wd/vlm_histo/kg/graph_construct/descriptions.json')
    loader.close()
