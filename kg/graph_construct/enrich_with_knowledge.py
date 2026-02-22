import json
import os
import copy
import pandas as pd
from collections import Counter

# ── Input paths ──────────────────────────────────────────────────────────────
WSI_TRAIN_SET = "/home/nadun/wd/datasets/wsi_bench/dataset/wsi_bench_Report_with_features/wsi_bench_Report_train.json"
WSI_TEST_SET = "/home/nadun/wd/datasets/wsi_bench/dataset/wsi_bench_Report_with_features/wsi_bench_Report_test.json"

site_pd = '/home/nadun/wd/datasets/wsi_bench/dataset/extracted_sites/test/with_site_variant_inference_test_pred.csv'

knowledge_path = '/home/nadun/wd/datasets/wsi_bench/kg_data/histology_descriptions_with_features.json'

# ── Output paths ─────────────────────────────────────────────────────────────
OUT_DIR = "/home/nadun/wd/datasets/wsi_bench/dataset/wsi_bench_Report_knowledge"

updated_knowledge_train_set = os.path.join(OUT_DIR, "WSI-Bench-train_Report_knowledge.json")
updated_knowledge_test_set  = os.path.join(OUT_DIR, "WSI-Bench-test_Report_knowledge.json")

train_log_path = os.path.join(OUT_DIR, "WSI-Bench-train_Report_knowledge.log")
test_log_path  = os.path.join(OUT_DIR, "WSI-Bench-test_Report_knowledge.log")


# ── Helpers ───────────────────────────────────────────────────────────────────

def format_table(rows, col_headers, col_widths):
    """Return a simple fixed-width table string."""
    sep = "-" * (sum(col_widths) + len(col_widths))
    header = "".join(h.ljust(w) for h, w in zip(col_headers, col_widths))
    lines = [header, sep]
    for row in rows:
        lines.append("".join(str(v).ljust(w) for v, w in zip(row, col_widths)))
    return "\n".join(lines)


def compute_stats(dataset: dict):
    """Compute per-field statistics for logging."""
    stats = {}
    for field in ("project", "project_name", "primary_site", "disease_type", "histology"):
        counter = Counter(entry.get(field, "N/A") for entry in dataset.values())
        total = sum(counter.values())
        stats[field] = (counter, total)
    return stats


def write_log(log_path: str, source_path: str, dataset: dict,
              dropped_ids: list = None, knowledge_miss: list = None):
    """Write a statistics log file similar to the existing .log format."""
    stats = compute_stats(dataset)

    lines = []
    lines.append(f"Enrichment Log for {source_path}")
    lines.append("=" * 40)
    lines.append("")

    lines.append("--- Post-Enrichment Statistics ---")
    lines.append(f"Total Data Points: {len(dataset)}")
    lines.append("")

    field_labels = {
        "project":      "Project Statistics:",
        "project_name": "Project Name Statistics:",
        "primary_site": "Primary Site Statistics:",
        "disease_type": "Disease Type Statistics:",
        "histology":    "Histology Statistics:",
    }

    for field, label in field_labels.items():
        counter, total = stats[field]
        lines.append(label)
        rows = []
        for name, cnt in sorted(counter.items(), key=lambda x: -x[1]):
            pct = cnt / total * 100 if total > 0 else 0
            rows.append((name, cnt, f"{pct:.2f}    %"))
        col_widths = [55, 11, 12]
        lines.append(format_table(rows, ["Name", "Count", "Percentage"], col_widths))
        lines.append(f"Total entries: {len(counter)}")
        lines.append("")

    # Knowledge-miss summary
    if knowledge_miss:
        lines.append(f"--- Knowledge Lookup Misses ({len(knowledge_miss)} entries) ---")
        for item in knowledge_miss:
            lines.append(f"  {item}")
        lines.append("")

    # Dropped IDs summary
    if dropped_ids:
        lines.append(f"--- Dropped IDs (no prediction found, {len(dropped_ids)} entries) ---")
        for tid in dropped_ids:
            lines.append(f"  {tid}")
        lines.append("")

    with open(log_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Log written → {log_path}")


# ── Core enrichment functions ─────────────────────────────────────────────────

def enrich_train_set(train_path: str, output_path: str, knowledge_dict: dict, log_path: str):
    """
    For each entry in the train set:
      - read the 'histology' field
      - look up its knowledge text in knowledge_dict
      - replace the 'knowledge' field with the looked-up text
    """
    with open(train_path, "r") as f:
        reports = json.load(f)

    print(f"  Loaded {len(reports)} train reports from {train_path}")

    updated = {}
    dropped = []

    for key, entry in reports.items():
        histology = entry.get("histology", "").strip()
        if not histology or histology not in knowledge_dict:
            dropped.append(f"{key}: histology='{histology}'")
            continue
        new_entry = copy.deepcopy(entry)
        new_entry["knowledge"] = knowledge_dict[histology]
        updated[key] = new_entry

    print(f"  Knowledge updated for {len(updated)} / {len(reports)} entries")
    if dropped:
        print(f"  [DROPPED] {len(dropped)} entries (missing or unmatched histology)")

    with open(output_path, "w") as f:
        json.dump(updated, f, indent=2)
    print(f"  Saved → {output_path}")

    write_log(log_path, train_path, updated, knowledge_miss=dropped)
    return updated


def enrich_test_set(test_path: str, output_path: str, knowledge_dict: dict,
                    pred_csv_path: str, log_path: str):
    """
    For each entry in the test set:
      - look up the predicted histology from site_pd CSV via tcga_id
      - replace 'knowledge' field with knowledge_dict[predicted_histology]
      - drop entries whose tcga_id is missing in the predictions CSV
    """
    with open(test_path, "r") as f:
        reports = json.load(f)

    print(f"  Loaded {len(reports)} test reports from {test_path}")

    # Build tcga_id → predicted_label mapping
    pred_df = pd.read_csv(pred_csv_path)
    pred_map = dict(zip(pred_df["tcga_id"].str.strip(), pred_df["predicted_label"].str.strip()))
    print(f"  Loaded {len(pred_map)} predictions from {pred_csv_path}")

    updated = {}
    dropped_ids = []
    knowledge_miss = []

    for key, entry in reports.items():
        tcga_id = entry.get("tcga_id", "").strip()

        if tcga_id not in pred_map:
            dropped_ids.append(tcga_id if tcga_id else key)
            continue

        predicted_histology = pred_map[tcga_id].strip()

        if not predicted_histology or predicted_histology not in knowledge_dict:
            knowledge_miss.append(f"{tcga_id}: predicted_histology='{predicted_histology}'")
            dropped_ids.append(tcga_id)   # drop — no usable knowledge
            continue

        new_entry = copy.deepcopy(entry)
        new_entry["histology"] = predicted_histology
        new_entry["knowledge"] = knowledge_dict[predicted_histology]
        updated[key] = new_entry

    print(f"  Kept {len(updated)} entries  |  Dropped {len(dropped_ids)} entries (no prediction or unmatched histology)")
    if knowledge_miss:
        print(f"  [DROPPED] {len(knowledge_miss)} entries had no knowledge match")

    with open(output_path, "w") as f:
        json.dump(updated, f, indent=2)
    print(f"  Saved → {output_path}")

    write_log(log_path, test_path, updated,
              dropped_ids=dropped_ids, knowledge_miss=knowledge_miss)
    return updated


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load knowledge descriptions
    with open(knowledge_path, "r") as f:
        knowledge_dict = json.load(f)
    print(f"Loaded {len(knowledge_dict)} knowledge entries from {knowledge_path}")
    print("\n" + "=" * 80)

    # ── Training set ──
    print("\nProcessing Training Set...")
    print(f"  Input : {WSI_TRAIN_SET}")
    print(f"  Output: {updated_knowledge_train_set}")
    enrich_train_set(WSI_TRAIN_SET, updated_knowledge_train_set, knowledge_dict, train_log_path)

    print("\n" + "=" * 80)

    # ── Test set ──
    print("\nProcessing Test Set...")
    print(f"  Input    : {WSI_TEST_SET}")
    print(f"  Pred CSV : {site_pd}")
    print(f"  Output   : {updated_knowledge_test_set}")
    enrich_test_set(WSI_TEST_SET, updated_knowledge_test_set, knowledge_dict, site_pd, test_log_path)

    print("\n" + "=" * 80)
    print("\nAll datasets enriched successfully!")


if __name__ == "__main__":
    main()
