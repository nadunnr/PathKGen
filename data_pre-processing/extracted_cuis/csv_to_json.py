import csv
import json
import os
from glob import glob

INPUT_FOLDER = "/home/nadun/wd/vlm_histo/data_pre-processing/extracted_cuis/cui_csvs/"
OUTPUT_FOLDER = "/home/nadun/wd/vlm_histo/data_pre-processing/extracted_cuis/cui_jsons/"


os.makedirs(OUTPUT_FOLDER, exist_ok=True)

csv_files = glob(os.path.join(INPUT_FOLDER, "*.csv"))

for csv_file in csv_files:
    data = []
    
    with open(csv_file, "r", newline='', encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        
        for row in reader:
            tcga_id = row["tcga_id"]
            cuis = [c.strip() for c in row["cuis"].split(";") if c.strip()]
            concepts = [c.strip() for c in row["concept_names"].split(";") if c.strip()]
            
            data.append({
                "tcga_id": tcga_id,
                "cuis": cuis,
                "concept_names": concepts
            })
    
    base_name = os.path.splitext(os.path.basename(csv_file))[0]
    output_path = os.path.join(OUTPUT_FOLDER, base_name + ".json")
    
    # Save JSON
    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(data, outfile, indent=4)
    
    print(f"Saved: {output_path}")

print("All files converted.")
