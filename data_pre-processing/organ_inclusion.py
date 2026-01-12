"""This script processes a HistGen dataset JSON file to include organ information for each patient.

It queries the GDC API to fetch the primary_site for each TCGA barcode and updates the dataset with this information.

Usage:
python organ_inclusion.py

Configuration:
- INPUT_FILE: Path to the original HistGen dataset JSON file
- OUTPUT_DIR: Directory where both the JSON and the LOG will be saved
- OUTPUT_JSON_NAME: Name of the output JSON file
- LOG_FILE_NAME: Name of the log file"""

import json
import requests
import collections
import os

# --- CONFIGURATION ---
INPUT_FILE = "/home/nadun/wd/datasets/histgen_data/annotation712_update.json"
# Directory where both the JSON and the LOG will be saved
OUTPUT_DIR = "/home/nadun/wd/datasets/tcga_report_histgen/" 
OUTPUT_JSON_NAME = "histgen_with_organs.json"
LOG_FILE_NAME = "class_counts.log"

def get_gdc_data(submitter_ids):
    """
    Queries the GDC API for a list of TCGA barcodes to get the primary_site.
    """
    url = "https://api.gdc.cancer.gov/cases"
    
    payload = {
        "filters": {
            "op": "in",
            "content": {
                "field": "submitter_id",
                "value": submitter_ids
            }
        },
        "fields": "submitter_id,primary_site",
        "format": "json",
        "size": len(submitter_ids) + 100
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()['data']['hits']
    except Exception as e:
        print(f"  [!] API Error: {e}")
        return []

def process_dataset():
    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    output_json_path = os.path.join(OUTPUT_DIR, OUTPUT_JSON_NAME)
    log_file_path = os.path.join(OUTPUT_DIR, LOG_FILE_NAME)

    print(f"Loading {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: Input file not found.")
        return

    # 1. Structure for the new clean dataset
    cleaned_data = {}
    
    # 2. Maps 'TCGA-XX-XXXX' -> list of (split_name, index) to update later
    barcode_map = {} 
    all_barcodes = set()

    for split_name in data:
        print(f"Scanning split: {split_name}...")
        cleaned_data[split_name] = []
        
        for item in data[split_name]:
            # Create a clean item dict: Keep only ID and Report
            clean_item = {
                "id": item.get("id"),
                "report": item.get("report")
                # 'organ' will be added here later
            }
            
            cleaned_data[split_name].append(clean_item)
            
            # Prepare for API Query
            original_id = item.get('id', '')
            if original_id.startswith("TCGA"):
                # Extract TCGA-XX-XXXX (Patient ID)
                short_barcode = "-".join(original_id.split("-")[:3])
                all_barcodes.add(short_barcode)
                
                # Track where this patient is located in our new list
                current_index = len(cleaned_data[split_name]) - 1
                
                if short_barcode not in barcode_map:
                    barcode_map[short_barcode] = []
                barcode_map[short_barcode].append((split_name, current_index))

    print(f"Found {len(all_barcodes)} unique patients. Querying GDC API...")

    # 3. Query GDC API in chunks
    chunk_size = 1000
    barcodes_list = list(all_barcodes)
    site_lookup = {}

    for i in range(0, len(barcodes_list), chunk_size):
        chunk = barcodes_list[i:i+chunk_size]
        print(f"Fetching batch {i} to {i+len(chunk)}...")
        hits = get_gdc_data(chunk)
        
        for hit in hits:
            # Map barcode -> organ (primary_site)
            site_lookup[hit['submitter_id']] = hit['primary_site'].lower()

    # 4. Update the cleaned data and count
    print("Updating dataset with organ labels...")
    organ_counter = collections.Counter()
    split_counters = {split: collections.Counter() for split in cleaned_data}
    
    # We iterate over the barcode map to update every instance
    for barcode, locations in barcode_map.items():
        organ = site_lookup.get(barcode, "unknown")
        
        for split, index in locations:
            cleaned_data[split][index]['organ'] = organ
            organ_counter[organ] += 1
            split_counters[split][organ] += 1

    # 5. Save to JSON file (Overwrites existing file)
    with open(output_json_path, 'w') as f:
        json.dump(cleaned_data, f, indent=4)

    # 6. Save Counts to Log File
    # Note: 'log_content.append' builds the string in memory.
    # The 'w' mode in open() below ensures the file on disk is OVERWRITTEN, not appended to.
    log_content = []
    log_content.append("=" * 40)
    log_content.append(f"Processing Complete.")
    log_content.append(f"Output JSON: {output_json_path}")
    log_content.append("=" * 40)
    log_content.append("\nTOTAL CLASS (ORGAN) COUNTS:")
    log_content.append("-" * 40)
    for organ, count in organ_counter.most_common():
        log_content.append(f"{organ}: {count}")
    
    log_content.append("\n" + "=" * 40)
    log_content.append("COUNTS PER SPLIT")
    log_content.append("=" * 40)
    
    for split, counter in split_counters.items():
        log_content.append(f"\nSPLIT: {split.upper()}")
        log_content.append("-" * 20)
        if not counter:
            log_content.append("  (No data)")
        for organ, count in counter.most_common():
             log_content.append(f"  {organ}: {count}")

    # Mode 'w' = Overwrite. Mode 'a' would be Append.
    with open(log_file_path, 'w') as f:
        f.write("\n".join(log_content))

    # Print to console as well
    print("\n".join(log_content))
    print(f"\nLog file saved to: {log_file_path}")

if __name__ == "__main__":
    process_dataset()