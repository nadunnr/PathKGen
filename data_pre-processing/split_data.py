"""
This script splits the HistGen dataset into primary site-specific splits based on the 'organ' label.

Usage:
python split_data.py

Configuration:
- INPUT_DIR: Directory containing the input JSON file
- INPUT_FILENAME: Name of the input JSON file
- OUTPUT_DIR: Directory to save the split JSON files
- TARGET_ORGANS: Mapping of user names to GDC 'organ' labels
- TRAIN_RATIO: Ratio of items to include in the train split
- VAL_RATIO: Ratio of items to include in the validation split
"""

import json
import os
import random
import csv
import math

# --- CONFIGURATION ---
# Path to the file created in the previous step
INPUT_DIR = "/home/nadun/wd/datasets/tcga_report_histgen/"
INPUT_FILENAME = "histgen_with_organs.json"

# Output directory (same as input)
OUTPUT_DIR = os.path.join(INPUT_DIR, "primary_site_splits")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Organ Mapping: User Name -> GDC 'organ' Label
# Based on your previous logs, Lung is labeled as "bronchus and lung"
TARGET_ORGANS = {
    "breast": "breast",
    "kidney": "kidney",
    "bronchus and lung":   "bronchus and lung",
    "brain":    "brain",
}

# Split Ratios
TRAIN_RATIO = 0.7
VAL_RATIO   = 0.1
# Test is remaining (0.2)

def process_organ_splits():
    input_path = os.path.join(INPUT_DIR, INPUT_FILENAME)
    
    print(f"Loading {input_path}...")
    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        return

    # 1. Pool data by organ (ignore original split for now)
    # structure: pools['breast'] = [item1, item2...]
    pools = {k: [] for k in TARGET_ORGANS.keys()}
    
    print("Filtering and pooling data...")
    total_items = 0
    
    # Iterate through existing 'train', 'val', 'test' in the input file
    for original_split in data:
        for item in data[original_split]:
            organ_label = item.get('organ', '').lower()
            
            # Check if this item belongs to our target organs
            for user_name, gdc_label in TARGET_ORGANS.items():
                if organ_label == gdc_label:
                    pools[user_name].append(item)
                    total_items += 1
                    break

    print(f"Total targeted items found: {total_items}")
    print("-" * 30)

    # 2. Process each organ
    random.seed(42) # Fixed seed for reproducibility

    for organ_name, items in pools.items():
        count = len(items)
        print(f"Processing {organ_name.upper()} ({count} items)...")
        
        if count == 0:
            print(f"  [!] Warning: No items found for {organ_name}")
            continue

        # --- A. Save IDs to CSV ---
        csv_filename = f"{organ_name}_ids.csv"
        csv_path = os.path.join(OUTPUT_DIR, csv_filename)
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id']) # Header
            for item in items:
                writer.writerow([item['id']])
        print(f"  -> Saved CSV: {csv_filename}")

        # --- B. Shuffle and Re-Split ---
        random.shuffle(items)
        
        n_train = math.floor(count * TRAIN_RATIO)
        n_val   = math.floor(count * VAL_RATIO)
        # n_test takes the remainder to ensure no data loss due to rounding
        
        new_train = items[:n_train]
        new_val   = items[n_train : n_train + n_val]
        new_test  = items[n_train + n_val :]
        
        print(f"  -> Split Counts: Train: {len(new_train)}, Val: {len(new_val)}, Test: {len(new_test)}")

        # --- C. Save to new JSON ---
        new_dataset = {
            "train": new_train,
            "val":   new_val,
            "test":  new_test
        }
        
        json_filename = f"{organ_name}_dataset.json"
        json_path = os.path.join(OUTPUT_DIR, json_filename)
        
        with open(json_path, 'w') as f:
            json.dump(new_dataset, f, indent=4)
        print(f"  -> Saved JSON: {json_filename}")
        print("-" * 30)

    print("Processing Complete.")

if __name__ == "__main__":
    process_organ_splits()