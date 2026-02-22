import json
import csv
import ollama
from tqdm import tqdm


import json
import ollama

def get_descriptors(project_name: str, primary_site: str, text: str):
    system_prompt = (
        "You are a pathology information extraction engine.\n"
        "Extract structured labels from pathology reports.\n"
        "Use ONLY explicitly stated information.\n"
        "Do NOT infer or guess.\n"
        "Return empty strings \"\" for missing fields.\n"
        "Output ONLY valid JSON."
    )

    user_prompt = f"""
        Extract structured tumor descriptors from the pathology report below.
        Pathology report:
        {text}

        Extraction rules:
        - histology: main cancer type explicitly diagnosed (eg: Renal Cell Carcinoma)
        - variant: histologic variant or subtype; (eg: Renal Cell Carcinoma, Clear Cell Type)
        - grade: select either from G1, G2, G3, G4 or from poorly/moderately/well differentiated. Do not use both.
        - site: single organ name, do not use multiple organs. use most prominent organ inferred from the report.
        - tumor_features: meaningful summarized tumor features, at most five, seperated by commas

        Output JSON only:
        {{
        "histology": "",
        "variant": "",
        "grade": "",
        "site": "",
        "tumor_features": ""
        }}
        """

    response = ollama.generate(
        model="qwen2.5:7b",
        system=system_prompt,
        prompt=user_prompt,
        options={
            "temperature": 0.0,
            "top_p": 1.0
        }
    )

    return response

if __name__ == "__main__":
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
            
            try:
                # Get model output
                response = get_descriptors(project_name, primary_site, text)
                
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

                if processed_count % 10 == 0:
                    print(f"Original text: {text} + {primary_site}\n")
                    print(f"Extracted data: {extracted_data}\n")
                
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

