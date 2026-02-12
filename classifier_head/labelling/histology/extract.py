import csv
import re
from pathlib import Path
from tqdm import tqdm
from histology import *
from organ import *
from grade import *

def extract_descriptor(row, descriptor):
    text_fields = []

    for key in [
        "report_content",
        "project_name",
        "primary_site",
        "disease_type"
    ]:
        if key in row and row[key]:
            text_fields.append(row[key].lower())

    full_text = " ".join(text_fields)

    if descriptor == "histology":
        for hist in HISTOLOGY_PRIORITY:
            for pattern in HISTOLOGY_PATTERNS[hist]:
                if re.search(pattern, full_text):
                    return hist
    elif descriptor == "site":
        for site in SITE_PRIORITY:
            for pattern in SITE_PATTERNS[site]:
                if re.search(pattern, full_text):
                    return site
    elif descriptor == "grade":
        for grad in GRADE_PRIORITY:
            for pattern in GRADE_PATTERNS[grad]:
                if re.search(pattern, full_text):
                    return grad

    return None


def process_csv(input_csv, output_csv, descriptor="histology"):
    if descriptor not in ["histology", "site", "grade"]:
        raise ValueError("Descriptor must be either 'histology' or 'site' or 'grade'")

    output_csv = Path(output_csv)
    write_header = not output_csv.exists()

    with open(output_csv, "a", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=["tcga_id", descriptor, "cui"])

        if write_header:
            writer.writeheader()

        with open(input_csv, "r", encoding="utf-8") as in_f:
            reader = csv.DictReader(in_f)
            rows = list(reader)

            for row in tqdm(rows, desc="Processing slides", unit="slide"):
                tcga_id = row.get("tcga_id")
                description = extract_descriptor(row, descriptor)

                if tcga_id and description:
                    if descriptor == "histology":
                        cui = HISTOLOGY_TO_CUI.get(description)
                    elif descriptor == "site":
                        cui = SITE_TO_CUI.get(description)
                    elif descriptor == "grade":
                        cui = GRADE_TO_CUI.get(description)

                    writer.writerow({
                        "tcga_id": tcga_id,
                        f"{descriptor}": description,
                        "cui": cui
                    })
                    out_f.flush()
                else:
                    print(f"Could not extract {descriptor} for {tcga_id}")

if __name__ == "__main__":
    input_csv = "/home/nadun/wd/datasets/wsi_bench/dataset/report_dataset_test.csv"
    output_csv = "grade_extracted.csv"
    process_csv(input_csv, output_csv, descriptor="grade")
