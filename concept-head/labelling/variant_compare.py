import csv
import re

INPUT_CSV = "/home/nadun/wd/datasets/wsi_bench/dataset/report_dataset_train.csv"
OUTPUT_CSV = "diagnoses_with_project.csv"

# Regex to capture everything after "diagnosis is"
DIAGNOSIS_REGEX = re.compile(
    r"diagnosis is\s*(.+)$",
    re.IGNORECASE
)

def normalize(text):
    """Basic normalization to avoid trivial duplicates"""
    text = text.strip()
    text = re.sub(r"\.$", "", text)      # remove trailing period
    text = re.sub(r"\s+", " ", text)     # normalize spaces
    return text.lower()

def extract_diagnoses():
    rows = []

    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            report = row.get("report_content")
            project_name = row.get("project_name", "")
            if not report:
                continue

            match = DIAGNOSIS_REGEX.search(report)
            if match:
                diagnosis = normalize(match.group(1))
                rows.append({"diagnosis": diagnosis, "project_name": project_name})

    # Write CSV sorted by project_name for easy scanning
    rows.sort(key=lambda r: (r["project_name"], r["diagnosis"]))

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["diagnosis", "project_name"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Extracted {len(rows)} diagnosis entries.")
    print(f"Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    extract_diagnoses()