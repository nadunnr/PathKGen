import json
import csv
import sys


def filter_and_save_report_tasks(json_file, output_csv):
    """
    Reads JSON file, filters by task == "report", and saves to CSV.
    
    Args:
        json_file: Path to JSON file containing the data
        output_csv: Path to output CSV file
    """
    try:
        # Read JSON file
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Ensure data is a list
        if isinstance(data, dict):
            data = [data]
        
        # Filter for "report" task
        report_tasks = [item for item in data if item.get("task") == "Report"]
        
        if not report_tasks:
            print("[!] No tasks with task='report' found in the JSON file.")
            return
        
        # Write to CSV
        fieldnames = ["tcga_id", "report_content", "project_name", "primary_site", "disease_type"]
        
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for item in report_tasks:
                # Use T-answer as report content
                report_content = item.get('T-answer', '')
                
                row = {
                    "tcga_id": item.get("tcga_id", ""),
                    "report_content": report_content,
                    "project_name": item.get("project_name", ""),
                    "primary_site": item.get("primary_site", ""),
                    "disease_type": item.get("disease_type", "")
                }
                writer.writerow(row)
        
        print(f"[✔] Successfully saved {len(report_tasks)} report tasks to {output_csv}")
    
    except FileNotFoundError:
        print(f"[!] Error: File '{json_file}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"[!] Error: Invalid JSON format in '{json_file}'.")
        sys.exit(1)
    except Exception as e:
        print(f"[!] Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_dataset_csv.py <input_json_file> <output_csv_file>")
        print("Example: python generate_dataset_csv.py data.json report_dataset.csv")
        sys.exit(1)
    
    json_file = sys.argv[1]
    output_csv = sys.argv[2]
    filter_and_save_report_tasks(json_file, output_csv)
