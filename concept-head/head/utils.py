import csv

# Load labels
def load_labels(csv_path):
    ids, labels, sites = [], [], []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(row["tcga_id"])
            labels.append(row["final_class"])
            sites.append(row["primary_site"])   # or organ column

    return ids, labels, sites