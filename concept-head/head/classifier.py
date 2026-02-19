import os
import csv
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set-up
TYPE = "variant"
LABEL_CSV = f"/home/nadun/wd/vlm_histo/concept-head/labelling/labels_train/{TYPE}_extracted_train.csv"
EMBEDDING_DIR = "/home/nadun/wd/datasets/wsi_bench/slide_features/train"
os.makedirs(f"{TYPE}", exist_ok=True)
OUTPUT_PREFIX = f"{TYPE}/{TYPE}_classifier"

# Hyper-parameters
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
VAL_SPLIT = 0.2
RANDOM_SEED = 42

device = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset
class WSIDataset(Dataset):
    def __init__(self, ids, labels, label_to_idx):
        self.ids = ids
        self.labels = labels
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        tcga_id = self.ids[idx]
        emb_path = os.path.join(EMBEDDING_DIR, f"{tcga_id}.h5")
        
        with h5py.File(emb_path, 'r') as f:
            embedding = f['features'][:]

        label = self.label_to_idx[self.labels[idx]]
        return torch.tensor(embedding, dtype=torch.float32), label

# Model
class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


# Load labels
def load_labels(csv_path):
    ids, labels = [], []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        label_name = reader.fieldnames[1]  # histology / grade / site

        for row in reader:
            ids.append(row["tcga_id"])
            labels.append(row[label_name])

    return ids, labels

# Training 
def train():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    ids, labels = load_labels(LABEL_CSV)
    
    # Filter out samples without embedding files
    print("Checking for missing embedding files...")
    valid_ids = []
    valid_labels = []
    missing_count = 0
    
    for id_, label in zip(ids, labels):
        emb_path = os.path.join(EMBEDDING_DIR, f"{id_}.h5")
        if os.path.exists(emb_path):
            valid_ids.append(id_)
            valid_labels.append(label)
        else:
            missing_count += 1
    
    if missing_count > 0:
        print(f"Removed {missing_count} samples without embedding files")
    
    ids = valid_ids
    labels = valid_labels
    
    # Filter out classes with too few samples (need at least 2 for stratified split)
    from collections import Counter
    label_counts = Counter(labels)
    min_samples_per_class = 2
    
    # Keep only samples from classes with enough samples
    filtered_ids = []
    filtered_labels = []
    for id_, label in zip(ids, labels):
        if label_counts[label] >= min_samples_per_class:
            filtered_ids.append(id_)
            filtered_labels.append(label)
    
    removed_count = len(ids) - len(filtered_ids)
    if removed_count > 0:
        print(f"Removed {removed_count} samples from classes with < {min_samples_per_class} samples\n")
    
    ids = filtered_ids
    labels = filtered_labels
    
    unique_labels = sorted(set(labels))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    
    # Print dataset statistics
    print(f"\n{'='*60}")
    print(f"Dataset Statistics")
    print(f"{'='*60}")
    print(f"Total samples: {len(ids)}")
    print(f"Number of classes: {len(unique_labels)}")
    print(f"Classes: {unique_labels}")
    print(f"Class distribution:")
    label_counts = Counter(labels)
    for label in unique_labels:
        count = label_counts[label]
        print(f"  {label}: {count} ({count/len(labels)*100:.1f}%)")
    print(f"{'='*60}\n")

    X_train, X_val, y_train, y_val = train_test_split(
        ids, labels, test_size=VAL_SPLIT, random_state=RANDOM_SEED, stratify=labels
    )
    
    print(f"Train samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Device: {device}\n")

    train_ds = WSIDataset(X_train, y_train, label_to_idx)
    val_ds = WSIDataset(X_val, y_val, label_to_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # Infer embedding dim
    with h5py.File(os.path.join(EMBEDDING_DIR, f"{ids[0]}.h5"), 'r') as f:
        sample_emb = f['features'][:]
    model = LinearClassifier(sample_emb.shape[0], len(unique_labels)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []

    best_val_acc = 0.0
    best_epoch = 0
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))

        model.eval()
        val_loss = 0.0
        preds, gts = [], []

        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]  ", leave=False):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item()

                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                gts.extend(y.cpu().numpy())

        val_losses.append(val_loss / len(val_loader))
        val_acc = accuracy_score(gts, preds)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), f"{OUTPUT_PREFIX}_best.pt")
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f} | Val Acc: {val_acc:.4f}")

    # Metrics
    acc = accuracy_score(gts, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        gts, preds, average="weighted"
    )

    # Save outputs
    torch.save(model.state_dict(), f"{OUTPUT_PREFIX}_final.pt")
    
    print(f"\n{'='*60}")
    print(f"Final Validation Metrics")
    print(f"{'='*60}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"Best Val Accuracy: {best_val_acc:.4f} (Epoch {best_epoch})")
    print(f"{'='*60}\n")

    with open(f"{OUTPUT_PREFIX}_metrics.txt", "w") as f:
        f.write(f"Classes: {unique_labels}\n")
        f.write(f"Number of classes: {len(unique_labels)}\n")
        f.write(f"Total samples: {len(ids)}\n")
        f.write(f"Train samples: {len(X_train)}\n")
        f.write(f"Val samples: {len(X_val)}\n\n")
        f.write(f"Final Validation Metrics:\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n\n")
        f.write(f"Best Val Accuracy: {best_val_acc:.4f} (Epoch {best_epoch})\n")

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Curve")
    plt.savefig(f"{OUTPUT_PREFIX}_loss.png", dpi=300)
    plt.close()

    print("Training complete.")
    print(f"Saved final model to {OUTPUT_PREFIX}_final.pt")
    print(f"Saved best model to {OUTPUT_PREFIX}_best.pt")
    print(f"Saved metrics to {OUTPUT_PREFIX}_metrics.txt")
    print(f"Saved loss plot to {OUTPUT_PREFIX}_loss.png")


# Run script
if __name__ == "__main__":
    train()
