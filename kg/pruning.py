from utils import entity_linking, extract_unique_entities
import json, csv
import pandas as pd


def collect_entities(dataset_path = "/home/nadun/wd/datasets/tcga_report_histgen/primary_site_splits/breast_dataset.json"):
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entities = set()
    nlp, linker = entity_linking()
    for tvt in data:
        for report in data[tvt]:
            entities.update(extract_unique_entities(nlp(report['report'])))
    with open("/home/nadun/wd/vlm_histo/kg/docs/who-breast/auto/who-breast.md", "r", encoding="utf-8") as f:
        entities.update(extract_unique_entities(nlp(f.read())))


    with open("tmp/filtered.csv", "w", newline="") as f:
        csv.writer(f).writerows([[x] for x in entities])
    return entities


def clean_nodes(input_path = "/home/nadun/wd/datasets/UMLS_sql/neo4jdb/v3/nodes.csv", output_path = "tmp/nodes.csv"):
    umls_nodes = pd.read_csv(input_path)
    dataset_nodes = set(pd.read_csv("tmp/filtered.csv").iloc[:, 0])
    
    # Filter UMLS nodes: keep only rows where first column is in dataset_nodes
    filtered_nodes = umls_nodes[umls_nodes.iloc[:, 0].isin(dataset_nodes)]
    
    filtered_nodes.to_csv(output_path, index=False, header=False)


def clean_edges(input_path = "/home/nadun/wd/datasets/UMLS_sql/neo4jdb/v3/relationships.csv", output_path = "tmp/edges.csv"):
    umls_nodes = set(pd.read_csv("tmp/nodes.csv").iloc[:, 0])
    edges = pd.read_csv(input_path, header=None)

    # filter: keep edge only if both CUIs are in node list
    cleaned_edges = edges[edges[0].isin(umls_nodes) & edges[1].isin(umls_nodes)]

    cleaned_edges.to_csv(output_path, index=False, header=False)

    print(f"Cleaned edges saved to: {output_path}")
    print(f"Before: {len(edges)} edges | After: {len(cleaned_edges)} edges")


if __name__ == "__main__":
    collect_entities()
    clean_nodes()
    clean_edges()