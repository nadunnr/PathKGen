import csv
import json
import requests
import time

# --- Configuration ---
CSV_FILE = "bronchus and lung_ids.csv"
MANIFEST_FILE = "manifest_bronchus_and_lung.txt"
GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"

def extract_barcodes(csv_file):
    """
    Extracts only the Barcode part (e.g., TCGA-97-7553-01Z-00-DX1)
    ignoring the trailing UUID which causes the lookup failure.
    """
    barcodes = []
    try:
        with open(csv_file, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                full_id = row["id"].strip()
                # Split at the first dot to get 'TCGA-XX-XXXX-XXX-XX-XXX'
                if "." in full_id:
                    barcode = full_id.split(".", 1)[0]
                    barcodes.append(barcode)
                else:
                    barcodes.append(full_id)
        return barcodes
    except FileNotFoundError:
        print(f"[!] Error: File {csv_file} not found.")
        return []

def map_barcodes_to_files(barcodes, chunk_size=50):
    """
    Queries GDC to find the File UUID associated with the Slide Barcode.
    """
    manifest_data = []
    
    # Process in chunks
    for i in range(0, len(barcodes), chunk_size):
        chunk = barcodes[i:i + chunk_size]
        print(f"[*] Processing chunk {i//chunk_size + 1} ({len(chunk)} slides)...")
        
        # KEY FIX: Filter by 'cases.samples.portions.slides.submitter_id'
        # This links the physical slide barcode to the digital file.
        filters = {
            "op": "and",
            "content": [
                {
                    "op": "in",
                    "content": {
                        "field": "cases.samples.portions.slides.submitter_id",
                        "value": chunk
                    }
                },
                {
                    "op": "in",
                    "content": {
                        "field": "data_format",
                        "value": ["SVS"] 
                    }
                }
            ]
        }

        # We request the specific fields needed for the manifest
        params = {
            "filters": json.dumps(filters),
            "fields": "file_id,file_name,md5sum,file_size,state,cases.samples.portions.slides.submitter_id",
            "format": "JSON",
            "size": 2000 
        }

        try:
            r = requests.post(GDC_FILES_ENDPOINT, json=params)
            r.raise_for_status()
            hits = r.json().get("data", {}).get("hits", [])
            
            for hit in hits:
                manifest_data.append({
                    "id": hit["file_id"],
                    "filename": hit["file_name"],
                    "md5": hit["md5sum"],
                    "size": hit["file_size"],
                    "state": hit["state"]
                })
                
            print(f"    -> Found {len(hits)} matching SVS files.")
            time.sleep(0.1)

        except Exception as e:
            print(f"    [!] Error in chunk: {e}")

    return manifest_data

def save_manifest(data, output_file):
    if not data:
        print("[!] No data to save.")
        return

    # Standard GDC Manifest Headers
    fieldnames = ["id", "filename", "md5", "size", "state"]
    
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(data)
        
    print(f"\n[✔] Manifest saved to: {output_file}")
    print(f"[✔] Total files found: {len(data)}")

def main():
    print("[*] Reading barcodes from CSV...")
    barcodes = extract_barcodes(CSV_FILE)
    if not barcodes:
        return
    print(f"    -> Loaded {len(barcodes)} barcodes to query.")

    print("[*] Querying GDC for current File UUIDs...")
    manifest_rows = map_barcodes_to_files(barcodes)

    if not manifest_rows:
        print("\n[!] Failed to find any matching files.")
        print("    Ensure your IDs start with 'TCGA-' and look like 'TCGA-XX-XXXX-01Z-00-DX1'.")
        return

    save_manifest(manifest_rows, MANIFEST_FILE)
    print(f"\n[Next Step] Download using:")
    print(f"gdc-client download -m {MANIFEST_FILE}")

if __name__ == "__main__":
    main()