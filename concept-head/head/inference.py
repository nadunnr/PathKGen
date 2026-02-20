import os
import csv
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
TYPE = "variant"
LABEL_CSV = f"/home/nadun/wd/vlm_histo/concept-head/labelling/labels_test/{TYPE}_extracted_test.csv"
EMBEDDING_DIR = "/home/nadun/wd/datasets/wsi_bench/slide_features/test"  # or train for validation
MODEL_PATH = f"/home/nadun/wd/vlm_histo/concept-head/head/variant/variant_classifier_best.pt"
OUTPUT_PREFIX = f"{TYPE}/{TYPE}_inference_test"

# Hyper-parameters
BATCH_SIZE = 128

device = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset
class WSIDataset(Dataset):
    def __init__(self, ids):
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        tcga_id = self.ids[idx]
        emb_path = os.path.join(EMBEDDING_DIR, f"{tcga_id}.h5")
        
        with h5py.File(emb_path, 'r') as f:
            embedding = f['features'][:]
            embedding = (embedding - embedding.mean()) / (embedding.std() + 1e-6)

        return torch.tensor(embedding, dtype=torch.float32), tcga_id

# Model
class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

# Load labels
def load_ids(csv_path):
    """Return (ids, gt_labels). gt_labels is None if the column is absent."""
    ids = []
    gt_labels = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        has_label = "label" in (reader.fieldnames or [])
        for row in reader:
            ids.append(row["tcga_id"])
            if has_label:
                gt_labels.append(row["label"])

    return ids, (gt_labels if gt_labels else None)

# Inference
def inference():
    print(f"\n{'='*60}")
    print(f"Running Inference on Training Set")
    print(f"{'='*60}\n")
    
    # Load the trained model's class list
    trained_classes = ['Adrenocortical Carcinoma', 'Bladder Urothelial Carcinoma', 'Brain Lower Grade Glioma', 'Breast Invasive Carcinoma', 'Cervical Squamous Cell Carcinoma', 'Cholangiocarcinoma', 'Colon Adenocarcinoma', 'Endocervical Adenocarcinoma', 'Esophageal Carcinoma', 'Head and Neck Squamous Cell Carcinoma', 'Kidney Chromophobe', 'Kidney Renal Clear Cell Carcinoma', 'Kidney Renal Papillary Cell Carcinoma', 'Liver Hepatocellular Carcinoma', 'Lung Adenocarcinoma', 'Lung Squamous Cell Carcinoma', 'Lymphoid Neoplasm Diffuse Large B-cell Lymphoma', 'Mesothelioma', 'Ovarian Serous Cystadenocarcinoma', 'Paraganglioma', 'Pheochromocytoma', 'Prostate Adenocarcinoma', 'Rectum Adenocarcinoma', 'Sarcoma', 'Skin Cutaneous Melanoma', 'Stomach Adenocarcinoma', 'Thymoma', 'Thyroid Carcinoma', 'Uterine Carcinosarcoma', 'Uterine Corpus Endometrial Carcinoma', 'Uveal Melanoma']
    
    unique_labels = sorted(trained_classes)
    idx_to_label = {i: l for i, l in enumerate(unique_labels)}
    
    # Load ids and GT labels from CSV
    ids, gt_labels = load_ids(LABEL_CSV)
    print(f"Classes (from trained model): {len(unique_labels)}")
    
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

    ids = [ids[i] for i in valid_entries]
    if gt_labels is not None:
        gt_labels = [gt_labels[i] for i in valid_entries]
    
    print(f"Total samples: {len(ids)}")
    print(f"Device: {device}\n")
    
    # Create dataset
    dataset = WSIDataset(ids)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Load model
    with h5py.File(os.path.join(EMBEDDING_DIR, f"{ids[0]}.h5"), 'r') as f:
        sample_emb = f['features'][:]
    
    model = LinearClassifier(sample_emb.shape[0], len(unique_labels)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    print(f"Loaded model from {MODEL_PATH}\n")
    
    # Inference
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for x, tcga_ids in tqdm(dataloader, desc="Inference"):
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_ids.extend(tcga_ids)
    
    # Save predictions in ground truth format
    output_csv = f"{OUTPUT_PREFIX}_predictions_gt_format.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    pred_labels = [idx_to_label[p] for p in all_preds]

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tcga_id", "predicted_label"])
        for tcga_id, pred_label in zip(all_ids, pred_labels):
            writer.writerow([tcga_id, pred_label])

    print(f"\nInference complete.")
    print(f"Saved predictions (GT format) to {output_csv}")

    # ── Metrics ──────────────────────────────────────────────────────────────
    if gt_labels is not None:
        print(f"\n{'='*60}")
        print("Metrics")
        print(f"{'='*60}")

        acc = accuracy_score(gt_labels, pred_labels)
        macro_f1 = f1_score(gt_labels, pred_labels, average="macro", zero_division=0)

        print(f"  Accuracy  : {acc * 100:.2f}%")
        print(f"  Macro-F1  : {macro_f1:.4f}")
        print(f"\nPer-class report:")
        print(classification_report(gt_labels, pred_labels, zero_division=0))

        # ── Confusion matrix image ────────────────────────────────────────────
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
        ax.set_title(f"{TYPE.capitalize()} — Confusion Matrix (Macro-F1: {macro_f1:.4f})", fontsize=13)
        plt.tight_layout()

        cm_path = f"{OUTPUT_PREFIX}_confusion_matrix.png"
        plt.savefig(cm_path, dpi=150)
        plt.close(fig)
        print(f"Saved confusion matrix to {cm_path}")
    else:
        print("\n[Note] No 'label' column found in CSV — skipping metrics and confusion matrix.")

if __name__ == "__main__":
    inference()