"""
Standalone script to compute BERTScore for VLM experiment results.
Reads a JSON file with 'prediction' and 'reference' fields.
"""

import json
import os
import torch
from bert_score import score
from tqdm import tqdm

def main():
    input_json = "/home/nadun/wd/datasets/wsi_bench/results/qwen7b_label_smoothing/stage2_val_epoch1.json"
    output_json = "/home/nadun/wd/datasets/wsi_bench/results/qwen7b_label_smoothing/stage2_val_bertscore.json"
    model_type = "distilbert-base-uncased"
    lang = "en"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading results from {input_json}...")
    if not os.path.exists(input_json):
        print(f"Error: Input file {input_json} not found.")
        return

    with open(input_json, 'r') as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("Error: Input file must be a list of dictionaries.")
        return

    predictions = []
    references = []
    ids = []

    for i, item in enumerate(data):
        if 'prediction' in item and 'reference' in item:
            predictions.append(item['prediction'])
            references.append(item['reference'])
            ids.append(item.get('tcga_id', f"sample_{i}"))

    if not predictions:
        print("Error: No samples with 'prediction' and 'reference' found.")
        return

    print(f"Computing BERTScore for {len(predictions)} samples using {model_type}...")
    
    # Compute BERTScore
    P, R, F1 = score(predictions, references, model_type=model_type, lang=lang, device=device, verbose=True)

    mean_precision = P.mean().item()
    mean_recall = R.mean().item()
    mean_f1 = F1.mean().item()

    print("\n" + "=" * 60)
    print("BERTScore results:")
    print("=" * 60)
    print(f"  Precision: {mean_precision:.4f}")
    print(f"  Recall:    {mean_recall:.4f}")
    print(f"  F1:        {mean_f1:.4f}")
    print("=" * 60)

    if output_json:
        results = {
            "input_file": input_json,
            "model_type": model_type,
            "mean_precision": mean_precision,
            "mean_recall": mean_recall,
            "mean_f1": mean_f1,
            "bert-score": mean_f1,
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
        print(f"Saved results to {output_json}")

if __name__ == "__main__":
    main()
