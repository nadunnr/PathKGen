
import sys
from pathlib import Path
from collections import Counter
import csv
import os
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graphs import Neo4jQuery

TOP_N = 150

def get_top_n_cuis(all_cuis, n):
    cui_counter = Counter(all_cuis)
    top_n_cuis = cui_counter.most_common(n)
    top_n_cuis = [cui[0] for cui in top_n_cuis]

    return top_n_cuis 

if __name__ == "__main__":  
    NEO4J_URL = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "11111111"

    neo4j_query = Neo4jQuery(NEO4J_URL, NEO4J_USER, NEO4J_PASSWORD)

    histology_jsons = "/home/nadun/wd/datasets/wsi_bench/kg_data/cui_jsons"
    top_cuis_json = f"/home/nadun/wd/datasets/wsi_bench/kg_data/top_cuis/top_{TOP_N}_cuis.json"

    # Load existing data or create new dict
    if os.path.exists(top_cuis_json):
        with open(top_cuis_json, "r") as f:
            top_cuis_data = json.load(f)
    else:
        top_cuis_data = {}

    for json_file in os.listdir(histology_jsons):
        all_cuis = []
        
        if json_file.endswith(".json"):
            histology_name = json_file.replace(".json", "")
            print(histology_name)

            with open(os.path.join(histology_jsons, json_file), "r") as f:
                data = json.load(f)
            for item in data:
                all_cuis.extend(item["cuis"])

        top_n_cuis = get_top_n_cuis(all_cuis, TOP_N)

        top_cuis_names = []
        for cui in top_n_cuis:
            top_cuis_names.append(neo4j_query.get_entity_by_cui(cui)['name'])
        # print(top_n_cuis)
        # print(top_cuis_names)

        
        # Add entry to dictionary
        top_cuis_data[histology_name] = {"cuis": top_n_cuis, "names": top_cuis_names}

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(top_cuis_json), exist_ok=True)
    # Save the complete dictionary as valid JSON
    with open(top_cuis_json, "w") as f:
        json.dump(top_cuis_data, f, indent=2)

    print(f"Saved top {TOP_N} CUIs to {top_cuis_json}")
            


    