import csv
import re
from lookup import SPLIT_RULES, FINAL_CLASS_REMAP, HIST_TO_CUI

INPUT_CSV = "/home/nadun/wd/datasets/wsi_bench/dataset/report_dataset_test.csv"
OUTPUT_CSV = "labels_test/variant_extracted_test.csv"

# Helper
def split_compound_class(tcga_id: str, project_name: str, report_text: str) -> str:
    text = report_text.lower()

    for final_class, patterns in SPLIT_RULES[project_name]:
        for pat in patterns:
            if re.search(pat, text):
                return final_class
    
    print(f"Exact class not decided for {tcga_id}")

    return None
 
# Main processing
with open(INPUT_CSV, newline="", encoding="utf-8") as infile, \
     open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as outfile:

    reader = csv.DictReader(infile)
    writer = csv.DictWriter(outfile, fieldnames=["tcga_id", "final_class", "cui"])
    writer.writeheader()

    for row in reader:
        tcga_id = row["tcga_id"]
        project = row["project_name"]
        report = row["report_content"]

        # Step 1: split compound projects if needed
        if project in SPLIT_RULES:
            final_class = split_compound_class(tcga_id, project, report)
            if final_class is None:
                print(f"Skipping {tcga_id} as exact class not decided")
                continue
        else:
            final_class = project

        # Step 2: remap to desired name
        final_class = FINAL_CLASS_REMAP.get(final_class, final_class)

        cui = HIST_TO_CUI.get(final_class, "")

        writer.writerow({
            "tcga_id": tcga_id,
            "final_class": final_class,
            "cui": cui
        })

print("Class remapping complete. Output written to:", OUTPUT_CSV)
