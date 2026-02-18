import csv
import re

INPUT_CSV = "/home/nadun/wd/datasets/wsi_bench/dataset/report_dataset_train.csv"
OUTPUT_CSV = "tcga_final_classes_train.csv"

SPLIT_RULES = {

    "Cervical Squamous Cell Carcinoma and Endocervical Adenocarcinoma": [
        ("Cervical Squamous Cell Carcinoma", [
            r"squamous",
            r"scc",
            r"keratinizing",
            r"non[- ]keratinizing"
        ]),
        ("Endocervical Adenocarcinoma", [
            r"adenocarcinoma",
            r"endocervical",
            r"glandular"
        ])
    ],

    "Pheochromocytoma and Paraganglioma": [
        ("Pheochromocytoma", [
            r"pheochromocytoma",
            r"adrenal medulla"
        ]),
        ("Paraganglioma", [
            r"paraganglioma",
            r"extra[- ]adrenal"
        ])
    ]
}

def split_compound_class(project_name: str, report_text: str) -> str:
    """
    Resolve compound TCGA project into atomic class.
    """
    text = report_text.lower()

    for final_class, patterns in SPLIT_RULES[project_name]:
        for pat in patterns:
            if re.search(pat, text):
                return final_class

    # deterministic fallback (should be rare)
    return project_name


# =========================
# Main processing
# =========================

with open(INPUT_CSV, newline="", encoding="utf-8") as infile, \
     open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as outfile:

    reader = csv.DictReader(infile)
    writer = csv.DictWriter(outfile, fieldnames=["tcga_id", "final_class"])
    writer.writeheader()

    for row in reader:
        tcga_id = row["tcga_id"]
        project = row["project_name"]
        report = row["report_content"]

        if project in SPLIT_RULES:
            final_class = split_compound_class(project, report)
        else:
            final_class = project

        writer.writerow({
            "tcga_id": tcga_id,
            "final_class": final_class
        })

print("✅ Class mapping complete. Output written to:", OUTPUT_CSV)
