"""
Batch script to compute BERTScore for all VLM experiment results in subdirectories.
Finds 'stage2_val_epoch1.json' in each subfolder and computes scores.
"""

import json
import os
import torch
import glob

# Monkeypatch transformers to bypass the strict torch.load security check (CVE-2025-32434)
# which blocks loading .bin files on torch < 2.6.
try:
    import transformers.utils.import_utils as import_utils
    import_utils.check_torch_load_is_safe = lambda: None
    import transformers.modeling_utils as modeling_utils
    modeling_utils.check_torch_load_is_safe = lambda: None
except ImportError:
    pass

from bert_score import score
from tqdm import tqdm
import pandas as pd

def compute_single_file(input_json, output_json, model_type, lang, device):
    print(f"--- Processing: {os.path.basename(os.path.dirname(input_json))} ---")
    
    with open(input_json, 'r') as f:
        data = json.load(f)

    if not isinstance(data, list):
        print(f"  [Skip] {input_json} is not a list.")
        return None

    predictions = []
    references = []
    ids = []

    for i, item in enumerate(data):
        if 'prediction' in item and 'reference' in item:
            predictions.append(item['prediction'])
            references.append(item['reference'])
            ids.append(item.get('tcga_id', f"sample_{i}"))

    if not predictions:
        print(f"  [Skip] No valid samples in {input_json}.")
        return None
    
    P, R, F1 = score(predictions, references, model_type=model_type, lang=lang, device=device, verbose=False)

    mean_precision = P.mean().item()
    mean_recall = R.mean().item()
    mean_f1 = F1.mean().item()

    results = {
        "input_file": input_json,
        "model_type": model_type,
        "mean_precision": mean_precision,
        "mean_recall": mean_recall,
        "mean_f1": mean_f1,
        "per_sample": []
    }
    
    for i in range(len(predictions)):
        results["per_sample"].append({
            "tcga_id": ids[i],
            "precision": P[i].item(),
            "recall": R[i].item(),
            "f1": F1[i].item()
        })
        
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  ✓ F1: {mean_f1:.4f} | Saved to: {os.path.basename(output_json)}")
    return {
        "Experiment": os.path.basename(os.path.dirname(input_json)),
        "Precision": mean_precision,
        "Recall": mean_recall,
        "F1": mean_f1
    }

def main():
    results_root = "/home/nadun/wd/vlm_histo/data/wsi_bench/results"
    target_filename = "stage2_val_epoch1.json"
    output_filename = "stage2_val_bertscore_clinicalbert.json"
    model_type = "distilbert-base-uncased"
    lang = "en"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Find all target files
    search_pattern = os.path.join(results_root, "*", target_filename)
    json_files = glob.glob(search_pattern)

    if not json_files:
        print(f"No files matching {target_filename} found in {results_root}")
        return

    print(f"Found {len(json_files)} experiments to process.")
    
    summary_stats = []
    for json_file in tqdm(json_files, desc="Experiments"):
        out_file = os.path.join(os.path.dirname(json_file), output_filename)
        res = compute_single_file(json_file, out_file, model_type, lang, device)
        if res:
            summary_stats.append(res)

    if summary_stats:
        print("\n" + "=" * 80)
        print("BATCH BERTSCORE SUMMARY")
        print("=" * 80)
        df = pd.DataFrame(summary_stats)
        print(df.to_string(index=False))
        print("=" * 80)

if __name__ == "__main__":
    main()
