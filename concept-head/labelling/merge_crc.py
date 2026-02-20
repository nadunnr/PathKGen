"""
merge_crc.py

Reads variant_extracted_train.csv, merges 'Colon Adenocarcinoma' and
'Rectum Adenocarcinoma' into a single 'Colorectal Adenocarcinoma' class,
and writes the result to a new CSV file.
"""

import csv
import os

INPUT_CSV = os.path.join(os.path.dirname(__file__), "labels_train", "variant_extracted_train.csv")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "labels_train", "merged_crc_train.csv")

# Classes to merge and the resulting merged class name / placeholder CUI
CLASSES_TO_MERGE = {"Colon Adenocarcinoma", "Rectum Adenocarcinoma"}
MERGED_CLASS_NAME = "Colorectal Adenocarcinoma"
MERGED_CLASS_CUI = "XXXXXX"  # Placeholder CUI — replace with the real UMLS CUI when available


def merge_crc(input_path: str, output_path: str) -> None:
    rows_read = 0
    rows_merged = 0

    with open(input_path, newline="", encoding="utf-8") as infile, \
         open(output_path, "w", newline="", encoding="utf-8") as outfile:

        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames  # ['tcga_id', 'final_class', 'cui']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            rows_read += 1
            if row["final_class"] in CLASSES_TO_MERGE:
                row["final_class"] = MERGED_CLASS_NAME
                row["cui"] = MERGED_CLASS_CUI
                rows_merged += 1
            writer.writerow(row)

    print(f"Done.")
    print(f"  Rows read         : {rows_read}")
    print(f"  Rows merged (CRC) : {rows_merged}")
    print(f"  Output written to : {output_path}")


if __name__ == "__main__":
    merge_crc(INPUT_CSV, OUTPUT_CSV)
