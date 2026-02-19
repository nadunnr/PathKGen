import os
import json
import torch
import faiss
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel


MODEL_PATH = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
CONCEPT_FILE = "/home/nadun/wd/datasets/UMLS_sql/neo4jdb/v8/nodes.csv"
INDEX_FILE = "/home/nadun/wd/datasets/wsi_bench/extracted_cuis/umls_index.faiss"

JSON_FOLDER = "/home/nadun/wd/datasets/wsi_bench/extracted_features/outputs"
OUTPUT_FOLDER = "/home/nadun/wd/vlm_histo/data_pre-processing/extracted_cuis/cui_csvs"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("Loading SapBERT...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModel.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()

print("Loading FAISS index...")
index = faiss.read_index(INDEX_FILE)

print("Loading concept metadata...")
meta = pd.read_csv(CONCEPT_FILE)

cui_list = meta["cui:ID"].values
name_list = meta["name"].values

def encode_text(text):
    tokens = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        emb = model(**tokens)[0][:, 0, :].cpu().numpy()

    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10
    return emb.astype("float32")

def get_top_concept(phrase):
    emb = encode_text(phrase)
    scores, indices = index.search(emb, 1)
    idx = indices[0][0]

    return cui_list[idx], name_list[idx]

def process_json(json_path, output_csv):
    print(f"\nProcessing: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    rows = []

    for item in data:
        tcga_id = item["tcga_id"]
        features = [f.strip() for f in item["tumor_features"].split(",")]

        cui_set = set()
        name_set = set()

        for feature in features:
            if feature:
                cui, name = get_top_concept(feature)
                cui_set.add(cui)
                name_set.add(name)

        rows.append({
            "tcga_id": tcga_id,
            "cuis": ";".join(sorted(cui_set)),
            "concept_names": ";".join(sorted(name_set))
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")


json_files = [f for f in os.listdir(JSON_FOLDER) if f.endswith(".json")]

print(f"\nFound {len(json_files)} JSON files")

for jf in json_files:
    json_path = os.path.join(JSON_FOLDER, jf)
    csv_name = jf.replace(".json", ".csv")
    output_csv = os.path.join(OUTPUT_FOLDER, csv_name)

    process_json(json_path, output_csv)

print("\nAll files processed.")
