import json
import csv
import ollama
from tqdm import tqdm

# Input and output file paths
input_file = "/home/nadun/wd/datasets/wsi_bench/dataset/report_dataset_test.csv"
output_file = "/home/nadun/wd/vlm_histo/kg/wsi-kg/extracted_test.json"

print(f"Processing CSV file: {input_file}")
print(f"Output will be saved to: {output_file}")

# Count total lines for progress bar
print("Counting total entries...")
with open(input_file, 'r', encoding='utf-8') as f:
    total_lines = sum(1 for _ in f) - 1  # Subtract 1 for header

print(f"Total entries to process: {total_lines}")

# Process each entry line by line
processed_count = 0
error_count = 0
results = []

with open(input_file, 'r', encoding='utf-8') as csv_file:
    
    csv_reader = csv.DictReader(csv_file)
    
    for row in tqdm(csv_reader, total=total_lines, desc="Processing WSI entries"):
        # Extract required fields from CSV
        tcga_id = row.get("tcga_id", "").strip()
        project_name = row.get("project_name", "").strip()
        primary_site = row.get("primary_site", "").strip()
        disease_type = row.get("disease_type", "").strip()
        text = row.get("report_content", "").strip()
        
        # Skip if report_content is empty
        if not text or not tcga_id:
            continue
        
        # Create prompt for extraction
        prompt = f"""You are analyzing a pathology slide with the following metadata and description. Use ALL the provided information to extract the 5 required fields accurately.

**Metadata:**
- Project Name: {project_name}
- Primary Site: {primary_site}
- Disease Type: {disease_type}

**Pathology Description:**
{text}

**Task:** Extract ONLY the following 5 fields based on the metadata AND pathology description. Respond ONLY with a valid JSON object containing short values (no explanations):

- histology: The main cancer type (e.g., "Squamous cell carcinoma", "Adenocarcinoma") - use metadata as reference.
- variant: The histological variant of the main cancer type (e.g.,"keratinizing", "invasive", "clear cell", "papillary") - use metadata as reference, if not mentioned, return "".
- grade: The differentiation grade (strictly follow the example format e.g., "G1", "G2", "G3", "G4") - if not mentioned, return "".
- primary_site: The organ/tissue location (e.g., "Lung", "Breast") - use the metadata as reference, give a single organ name and use most prominent.
- tumor_feature: One key tumor feature (e.g., "Tumor necrosis", "Lymphatic invasion") - give one feature only.

Respond with ONLY a JSON object in this exact format:
{{
  "histology": "...",
  "variant": "...",
  "grade": "...",
  "primary_site": "...",
  "tumor_feature": "..."
}}"""
        
        try:
            # Call Ollama using the Python library
            response = ollama.generate(
                model='qwen2.5:7b',
                prompt=prompt
            )
            
            # Parse response
            model_output = response['response'].strip()
            
            # Extract JSON from response (in case there's extra text)
            start_idx = model_output.find('{')
            end_idx = model_output.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = model_output[start_idx:end_idx]
                extracted_data = json.loads(json_str)
            else:
                extracted_data = json.loads(model_output)
            
            # Create output entry with tcga_id and extracted features
            output_entry = {
                "tcga_id": tcga_id,
                **extracted_data
            }
            
            # Add to results list
            results.append(output_entry)
            processed_count += 1
            
            # Save periodically (every 100 entries) to avoid data loss
            if processed_count % 100 == 0:
                with open(output_file, 'w', encoding='utf-8') as out_f:
                    json.dump(results, out_f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            # Log errors but continue processing
            error_entry = {
                "tcga_id": tcga_id,
                "error": str(e)
            }
            results.append(error_entry)
            error_count += 1
            print(f"\nError processing {tcga_id}: {e}")

# Final save
print("\nSaving final results...")
with open(output_file, 'w', encoding='utf-8') as out_f:
    json.dump(results, out_f, indent=2, ensure_ascii=False)



print(f"\nProcessing complete!")
print(f"Successfully processed: {processed_count}")
print(f"Errors encountered: {error_count}")
print(f"Results saved to: {output_file}")

