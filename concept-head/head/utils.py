import json

# Load labels from JSON file
def load_labels(json_path):
    ids, labels, sites, feature_paths = [], [], [], []

    with open(json_path, "r") as f:
        data = json.load(f)

    for entry in data:
        ids.append(entry["tcga_id"])
        labels.append(entry["histology"])
        sites.append(entry["primary_site"])
        feature_paths.append(entry.get("feature_path", None))

    return ids, labels, sites, feature_paths