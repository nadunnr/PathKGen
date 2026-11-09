#!/bin/bash
# Load the UMLS-derived knowledge graph into Neo4j and select the top CUIs per histology.
#
# Usage:  bash pre_processing/build_knowledge_graph.sh
# Needs:  a running Neo4j instance; set NEO4J_URL / NEO4J_USER / NEO4J_PASSWORD in
#         knowledge_path/graphs.py and knowledge_path/graph_construct/cui_selecter.py
#         $DATA_DIR/UMLS_sql/neo4jdb/v8/{nodes,relationships,definitions}.csv
#         $DATA_DIR/wsi_bench/kg_data/cui_jsons/*.json
# Output: $DATA_DIR/wsi_bench/kg_data/top_cuis/
set -euo pipefail

cd "$(dirname "$0")/../knowledge_path"

python graphs.py
python graph_construct/cui_selecter.py
