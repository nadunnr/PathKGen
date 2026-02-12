import csv
import re
from pathlib import Path
from tqdm import tqdm
from histology_dictionary import *

def extract_histology(row):

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

    for hist in HISTOLOGY_PRIORITY:
        for pattern in HISTOLOGY_PATTERNS[hist]:
            if re.search(pattern, full_text):
                return hist

    return None


def process_csv(input_csv, output_csv):
    output_csv = Path(output_csv)
    write_header = not output_csv.exists()

    with open(output_csv, "a", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=["tcga_id", "histology", "cui"])

        if write_header:
            writer.writeheader()

        with open(input_csv, "r", encoding="utf-8") as in_f:
            reader = csv.DictReader(in_f)
            rows = list(reader)

            for row in tqdm(rows, desc="Processing slides", unit="slide"):
                tcga_id = row.get("tcga_id")
                histology = extract_histology(row)

                if tcga_id and histology:
                    cui = HISTOLOGY_TO_CUI.get(histology)

                    writer.writerow({
                        "tcga_id": tcga_id,
                        "histology": histology,
                        "cui": cui
                    })
                    out_f.flush()
                else:
                    print(f"Could not extract histology for {tcga_id}")


if __name__ == "__main__":
    input_csv = "/home/nadun/wd/datasets/wsi_bench/dataset/report_dataset_train.csv"
    output_csv = "histology_extracted_train.csv"
    process_csv(input_csv, output_csv)
