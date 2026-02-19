# import pandas as pd
# import requests
# import os

# # --- Configuration ---
# CSV_FILE = "/home/nadun/wd/datasets/wsi_bench/slide_features/missing_wsi_train.csv"
# MANIFEST_FILE = "/home/nadun/wd/datasets/wsi_bench/slide_features/missing_wsis_train.txt"
# GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
# CSV_ID_COLUMN = "id"

# def query_gdc(filters):
#     """Helper to send a query to GDC"""
#     params = {
#         "filters": filters,
#         "fields": "file_id,file_name,md5sum,file_size,cases.samples.portions.slides.submitter_id,cases.samples.portions.slides.slide_id",
#         "format": "JSON",
#         "size": 1000
#     }
#     try:
#         response = requests.post(GDC_FILES_ENDPOINT, json=params)
#         response.raise_for_status()
#         return response.json().get("data", {}).get("hits", [])
#     except Exception:
#         return []

# def main():
#     # 1. Load Data
#     df = pd.read_csv(CSV_FILE)
#     raw_ids = df[CSV_ID_COLUMN].dropna().unique().tolist()
    
#     print(f"[*] Processing {len(raw_ids)} input IDs...")

#     found_files = {}  # Key: Original Input ID -> Value: File Metadata
#     missing_ids = []

#     # 2. Iterate through each ID individually to ensure we track exactly what is missing
#     for original_id in raw_ids:
#         original_id = str(original_id).strip()
        
#         # Parse the ID
#         # Format: BARCODE.UUID (e.g., TCGA-XX-XXXX.1234-5678...)
#         parts = original_id.split('.')
#         barcode = parts[0]
#         uuid_part = parts[-1] if len(parts) > 1 else None

#         hits = []

#         # --- STRATEGY 1: Search by Slide UUID (Most Precise) ---
#         if uuid_part and len(uuid_part) > 10: 
#             filters = {
#                 "op": "and",
#                 "content": [
#                     {"op": "=","content": {"field": "cases.samples.portions.slides.slide_id", "value": uuid_part}},
#                     {"op": "=","content": {"field": "files.data_format", "value": "SVS"}}
#                 ]
#             }
#             hits = query_gdc(filters)

#         # --- STRATEGY 2: Search by Barcode (Fallback) ---
#         if not hits:
#             # print(f"    -> UUID lookup failed for {barcode}, trying Barcode...")
#             filters = {
#                 "op": "and",
#                 "content": [
#                     {"op": "=","content": {"field": "cases.samples.portions.slides.submitter_id", "value": barcode}},
#                     {"op": "=","content": {"field": "files.data_format", "value": "SVS"}}
#                 ]
#             }
#             hits = query_gdc(filters)

#         # --- Process Results ---
#         if hits:
#             # If multiple files found (e.g. slight version diffs), pick the largest one
#             best_hit = max(hits, key=lambda x: x.get('file_size', 0))
#             found_files[original_id] = {
#                 "id": best_hit.get("file_id"),
#                 "filename": best_hit.get("file_name"),
#                 "md5": best_hit.get("md5sum"),
#                 "size": best_hit.get("file_size"),
#                 "state": "validated"
#             }
#         else:
#             missing_ids.append(original_id)

#     # 3. Save Manifest
#     if found_files:
#         if os.path.isdir(MANIFEST_FILE): os.rmdir(MANIFEST_FILE)
        
#         manifest_df = pd.DataFrame(list(found_files.values()))
#         manifest_df.to_csv(MANIFEST_FILE, sep='\t', index=False)
#         print(f"\n[✔] Success! Generated manifest for {len(found_files)} files.")
#         print(f"[✔] Saved to: {MANIFEST_FILE}")
#     else:
#         print("\n[!] Failed to find ANY files.")

#     # 4. Report Missing
#     if missing_ids:
#         print(f"\n[!] WARNING: Could not find SVS files for {len(missing_ids)} IDs:")
#         for mid in missing_ids:
#             print(f"    - {mid}")
#         print("\nPossible reasons: These slides might be 'Not Available' or retracted in GDC.")

# if __name__ == "__main__":
#     main()

import pandas as pd
import requests
import os

# --- Configuration ---
CSV_FILE = "/home/nadun/wd/datasets/wsi_bench/slide_features/missing_wsi_train.csv"
MANIFEST_FILE = "/home/nadun/wd/datasets/wsi_bench/slide_features/missing_wsis_train.txt"
GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
CSV_ID_COLUMN = "id"
BATCH_SIZE = 50 # Smaller batches for deep patient-level queries

def fetch_patient_files(case_ids):
    """Fetch every SVS file for a list of patients (Cases)"""
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.submitter_id", "value": case_ids}},
            {"op": "in", "content": {"field": "files.data_format", "value": ["SVS", "svs"]}}
        ]
    }
    params = {
        "filters": filters,
        # We request multiple nested submitter_id fields to find the slide link
        "fields": ("file_id,file_name,md5sum,file_size,state,"
                   "cases.submitter_id,"
                   "cases.samples.submitter_id,"
                   "cases.samples.portions.submitter_id,"
                   "cases.samples.portions.slides.submitter_id"),
        "format": "JSON",
        "size": 5000 
    }
    try:
        r = requests.post(GDC_FILES_ENDPOINT, json=params)
        r.raise_for_status()
        return r.json().get("data", {}).get("hits", [])
    except Exception as e:
        print(f"[!] API Request failed: {e}")
        return []

def main():
    if not os.path.exists(CSV_FILE):
        print(f"[!] File not found: {CSV_FILE}")
        return

    # 1. Prepare IDs
    df = pd.read_csv(CSV_FILE)
    raw_ids = df[CSV_ID_COLUMN].dropna().unique().tolist()
    
    # Target barcodes (e.g., TCGA-2G-AAFV-01Z-00-DX1)
    target_barcodes = [rid.split('.')[0] for rid in raw_ids]
    # Patient IDs (e.g., TCGA-2G-AAFV)
    case_ids = list(set([b[:12] for b in target_barcodes]))

    print(f"[*] Input: {len(target_barcodes)} slides for {len(case_ids)} patients.")
    
    final_manifest = {}

    # 2. Batch Query by Patient (Case)
    for i in range(0, len(case_ids), BATCH_SIZE):
        batch = case_ids[i : i + BATCH_SIZE]
        hits = fetch_patient_files(batch)
        
        for hit in hits:
            # GDC nesting is deep. We extract ALL submitter_ids linked to this file.
            linked_ids = []
            for c in hit.get("cases", []):
                linked_ids.append(c.get("submitter_id"))
                for s in c.get("samples", []):
                    linked_ids.append(s.get("submitter_id"))
                    for p in s.get("portions", []):
                        linked_ids.append(p.get("submitter_id"))
                        for slide in p.get("slides", []):
                            linked_ids.append(slide.get("submitter_id"))
            
            # Remove None values and check if any match our target slide barcodes
            linked_ids = [lid for lid in linked_ids if lid is not None]
            
            for barcode in target_barcodes:
                if barcode in linked_ids:
                    # If we find a match, keep the largest file (the WSI)
                    if barcode not in final_manifest or hit.get("file_size", 0) > final_manifest[barcode].get("size", 0):
                        final_manifest[barcode] = {
                            "id": hit.get("file_id"),
                            "filename": hit.get("file_name"),
                            "md5": hit.get("md5sum"),
                            "size": hit.get("file_size"),
                            "state": hit.get("state")
                        }

    # 3. Finalize
    manifest_rows = list(final_manifest.values())
    if manifest_rows:
        out_df = pd.DataFrame(manifest_rows)
        out_df.to_csv(MANIFEST_FILE, sep='\t', index=False)
        print(f"\n[✔] SUCCESS: Found {len(out_df)} out of {len(target_barcodes)} slides.")
    else:
        print("\n[!] No files found.")

    # 4. Critical Missing Check
    missing = set(target_barcodes) - set(final_manifest.keys())
    if missing:
        print(f"[!] Still missing {len(missing)} IDs. See 'still_missing.log'.")
        with open("still_missing.log", "w") as f:
            for m in sorted(list(missing)): f.write(f"{m}\n")

if __name__ == "__main__":
    main()