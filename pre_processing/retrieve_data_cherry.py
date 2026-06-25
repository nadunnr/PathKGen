import json
import os

# Define paths
CHERRY_PICKED_PATH = '/home/nadun/wd/datasets/wsi_bench/results/qwen_7b_bestresults/cherry_picked.json'
REPORT_KNOWLEDGE_PATH = '/home/nadun/wd/datasets/wsi_bench/dataset/wsi_bench_Report_knowledge/WSI-Bench-test_Report_knowledge.json'
OUTPUT_PATH = '/home/nadun/wd/datasets/wsi_bench/results/qwen_7b_bestresults/cherry_picked_updated.json'

def update_cherry_picked():
    # 1. Load the knowledge report
    print("Loading report knowledge...")
    with open(REPORT_KNOWLEDGE_PATH, 'r') as f:
        knowledge_data = json.load(f)

    # 2. Create a lookup dictionary for fast searching by tcga_id
    # We map tcga_id -> {primary_site, histology, pred_histology}
    knowledge_lookup = {}
    for entry in knowledge_data:
        tid = entry.get("tcga_id")
        if tid:
            knowledge_lookup[tid] = {
                "primary_site": entry.get("primary_site"),
                "histology": entry.get("histology"),
                "pred_histology": entry.get("pred_histology")
            }

    # 3. Load the cherry_picked results
    print("Loading cherry picked results...")
    with open(CHERRY_PICKED_PATH, 'r') as f:
        cherry_picked = json.load(f)

    # 4. Update cherry_picked with data from lookup
    updated_count = 0
    missing_count = 0
    
    for item in cherry_picked:
        tcga_id = item.get("tcga_id")
        if tcga_id in knowledge_lookup:
            info = knowledge_lookup[tcga_id]
            item["primary_site"] = info["primary_site"]
            item["histology"] = info["histology"]
            item["pred_histology"] = info["pred_histology"]
            updated_count += 1
        else:
            # Optional: handle cases where ID isn't found
            item["primary_site"] = None
            item["histology"] = None
            item["pred_histology"] = None
            missing_count += 1

    # 5. Save the updated file
    print(f"Saving updated file to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(cherry_picked, f, indent=4)

    print(f"Done! Updated: {updated_count} records. Missing in knowledge: {missing_count} records.")

if __name__ == "__main__":
    update_cherry_picked()