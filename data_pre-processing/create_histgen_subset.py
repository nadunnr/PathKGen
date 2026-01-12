"""
This script creates a subset of the HistGen dataset by randomly sampling a specified ratio of items from each split (train, val, test).

Usage:
python create_histgen_subset.py

Configuration:
- INPUT_JSON_FILE: Path to the original HistGen dataset JSON file
- OUTPUT_JSON_FILE: Path to save the subset JSON file
- SAMPLE_RATIO: Ratio of items to sample from each split (e.g., 0.1 for 10%)
"""

import json
import random
import os

# --- Configuration ---
INPUT_JSON_FILE = "/home/nadun/wd/datasets/histgen_data/annotation712_update.json" 
OUTPUT_JSON_FILE = "/home/nadun/wd/datasets/tcga_report_histgen/histgen_1.0.json"
SAMPLE_RATIO = 1

def create_subset(input_path, output_path, ratio):
    if not os.path.exists(input_path):
        print(f"Error: File '{input_path}' not found.")
        return

    print(f"Loading original dataset: {input_path}")
    with open(input_path, 'r') as f:
        data = json.load(f)

    new_data = {}
    
    # Process each split (train, val, test) found in the JSON
    for split_key, items in data.items():
        if not isinstance(items, list):
            continue
            
        total_items = len(items)
        sample_size = int(total_items * ratio)
        
        print(f"Processing '{split_key}': Sampling {sample_size} from {total_items} items...")
        
        # 1. Randomly sample the requested portion
        sampled_items = random.sample(items, sample_size)
        
        # 2. Clean the data (Remove 'image_path')
        cleaned_items = []
        for item in sampled_items:
            new_entry = {
                "id": item.get("id"),
                "report": item.get("report")
            }
            # Only add keys that actually existed
            cleaned_items.append(new_entry)
            
        new_data[split_key] = cleaned_items

    # Save the new JSON file
    print(f"Saving subset to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(new_data, f, indent=4)
        
    print("Done! Subset creation complete.")

if __name__ == "__main__":
    # Set a fixed seed for reproducibility (optional)
    random.seed(42) 
    create_subset(INPUT_JSON_FILE, OUTPUT_JSON_FILE, SAMPLE_RATIO)