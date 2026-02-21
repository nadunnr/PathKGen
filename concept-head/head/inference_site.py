import os
import csv
import pandas as pd
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Set-up
LABEL_CSV = f"/home/nadun/wd/vlm_histo/concept-head/labelling/without_tgc/variant_test_with_site.csv"
TRAIN_CSV = f"/home/nadun/wd/vlm_histo/concept-head/labelling/without_tgc/variant_train_with_site.csv"
EMBEDDING_DIR = f"/home/nadun/wd/datasets/wsi_bench/slide_features/test"  # or train for validation
MODEL_PATH = f"/home/nadun/wd/vlm_histo/concept-head/head/with_site/classifier_with_site_best.pt"
OUTPUT_PREFIX = f"with_site/with_site_variant_inference_test"

# Hyper-parameters
BATCH_SIZE = 128
SITE_EMB_DIM = 16

device = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset
class WSIDataset(Dataset):
    def __init__(self, ids, labels, sites, label_to_idx, site_to_idx):
        self.ids = ids
        self.labels = labels
        self.sites = sites
        self.label_to_idx = label_to_idx
        self.site_to_idx = site_to_idx

    def __getitem__(self, idx):
        tcga_id = self.ids[idx]
        emb_path = os.path.join(EMBEDDING_DIR, f"{tcga_id}.h5")

        with h5py.File(emb_path, 'r') as f:
            embedding = f['features'][:]
            embedding = (embedding - embedding.mean()) / (embedding.std() + 1e-6)

        label = self.label_to_idx[self.labels[idx]]
        site_id = self.site_to_idx[self.sites[idx]]

        return (
            torch.tensor(embedding, dtype=torch.float32),
            torch.tensor(site_id, dtype=torch.long),
            label
        )
    
    def __len__(self):
        return len(self.ids)

# Model
class LinearClassifier(nn.Module):
    def __init__(self, emb_dim, num_sites, site_emb_dim, num_classes):
        super().__init__()

        self.site_emb = nn.Embedding(num_sites, site_emb_dim)

        self.fc = nn.Sequential(
            nn.Linear(emb_dim + site_emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, site_id):
        site_vec = self.site_emb(site_id)
        x = torch.cat([x, site_vec], dim=1)
        return self.fc(x)

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

# Inference
def inference():
    print(f"\n{'='*60}")
    print(f"Running inference on {LABEL_CSV}")
    print(f"{'='*60}\n")
    
    # Load checkpoint
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    num_classes = ckpt["fc.3.weight"].shape[0]
    num_sites    = ckpt["site_emb.weight"].shape[0]

    # Build label vocab from train CSV (must match the order used during training)
    train_ids, train_labels, train_sites = load_labels(TRAIN_CSV)
    unique_labels = sorted(set(train_labels))
    label_to_idx  = {l: i for i, l in enumerate(unique_labels)}
    idx_to_label  = {i: l for i, l in enumerate(unique_labels)}

    # Build site vocab from train CSV (must match the order used during training)
    unique_sites = sorted(set(train_sites))
    site_to_idx  = {s: i for i, s in enumerate(unique_sites)}

    print(f"Classes (from checkpoint): {num_classes}")
    print(f"Sites   (from checkpoint): {num_sites}")

    # Load ids, GT labels, and sites from test CSV
    ids, gt_labels, sites = load_labels(LABEL_CSV)
    
    # Filter out samples without embedding files
    print("Checking for missing embedding files...")
    valid_entries = []
    missing_count = 0

    for i, id_ in enumerate(ids):
        emb_path = os.path.join(EMBEDDING_DIR, f"{id_}.h5")
        if os.path.exists(emb_path):
            valid_entries.append(i)
        else:
            missing_count += 1

    if missing_count > 0:
        print(f"Removed {missing_count} samples without embedding files")

    ids       = [ids[i]       for i in valid_entries]
    gt_labels = [gt_labels[i] for i in valid_entries]
    sites     = [sites[i]     for i in valid_entries]
    
    print(f"Total samples: {len(ids)}")
    print(f"Device: {device}\n")
    
    # Create dataset
    dataset    = WSIDataset(ids, gt_labels, sites, label_to_idx, site_to_idx)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Load model
    with h5py.File(os.path.join(EMBEDDING_DIR, f"{ids[0]}.h5"), 'r') as f:
        sample_emb = f['features'][:]
    
    model = LinearClassifier(sample_emb.shape[0], num_sites, SITE_EMB_DIM, num_classes).to(device)
    model.load_state_dict(ckpt)
    model.eval()
    
    print(f"Loaded model from {MODEL_PATH}\n")
    
    # Inference
    all_preds = []

    with torch.no_grad():
        for x, site_id, _ in tqdm(dataloader, desc="Inference"):
            x       = x.to(device)
            site_id = site_id.to(device)
            logits  = model(x, site_id)
            preds   = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
    
    # Save predictions in ground truth format
    output_csv  = f"{OUTPUT_PREFIX}_pred.csv"
    metrics_txt = f"{OUTPUT_PREFIX}_metrics.txt"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    pred_labels = [idx_to_label[p] for p in all_preds]

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tcga_id", "predicted_label"])
        for tcga_id, pred_label in zip(ids, pred_labels):
            writer.writerow([tcga_id, pred_label])

    print(f"\nInference complete.")
    print(f"Saved predictions (GT format) to {output_csv}")

    # Metrics
    if gt_labels is not None:
        acc      = accuracy_score(gt_labels, pred_labels)
        macro_f1 = f1_score(gt_labels, pred_labels, average="macro", zero_division=0)
        report   = classification_report(gt_labels, pred_labels, zero_division=0)

        with open(metrics_txt, "w") as mf:
            mf.write(f"{'='*60}\n")
            mf.write(f"Metrics\n")
            mf.write(f"{'='*60}\n\n")
            mf.write(f"  Accuracy  : {acc * 100:.2f}%\n")
            mf.write(f"  Macro-F1  : {macro_f1:.4f}\n\n")
            mf.write("Per-class report:\n")
            mf.write(report)
            mf.write("\n")

        print(f"Saved metrics to {metrics_txt}")

        # Confusion matrix
        present_labels = sorted(set(gt_labels) | set(pred_labels))
        cm = confusion_matrix(gt_labels, pred_labels, labels=present_labels)

        fig_w = max(12, len(present_labels) * 0.6)
        fig_h = max(10, len(present_labels) * 0.5)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax)

        # Annotate each cell
        thresh = cm.max() / 2.0
        for row in range(cm.shape[0]):
            for col in range(cm.shape[1]):
                ax.text(
                    col, row, str(cm[row, col]),
                    ha="center", va="center", fontsize=7,
                    color="white" if cm[row, col] > thresh else "black",
                )

        ax.set_xticks(range(len(present_labels)))
        ax.set_xticklabels(present_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(present_labels)))
        ax.set_yticklabels(present_labels, rotation=0, fontsize=8)
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Ground Truth", fontsize=12)
        ax.set_title(f"Confusion Matrix (Macro-F1: {macro_f1:.4f})", fontsize=13)
        plt.tight_layout()

        cm_path = f"{OUTPUT_PREFIX}_confusion_matrix.png"
        plt.savefig(cm_path, dpi=150)
        plt.close(fig)
        print(f"Saved confusion matrix to {cm_path}")
    else:
        print("\n[Note] No 'label' column found in CSV — skipping metrics and confusion matrix.")

if __name__ == "__main__":
    inference()