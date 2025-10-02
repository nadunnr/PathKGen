from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import SlideEmbeddingDataset
from model import BaseClassifier

# Hyper-parameters
BATCH_SIZE = 8
LR = 1e-3
EPOCHS = 50
CLASS_NAMES = ['Cancer', 'Diseased', 'Healthy', 'Treated', 'Tumor']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Load dataset
data_path = Path("/home/nadun/wd/TRIDENT/trident_processed/40x_512px_0px_overlap/slide_features_titan")
dataset = SlideEmbeddingDataset(data_path)

# train/val split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Dataset loaded with Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

# 2. Linear classifier 
model = BaseClassifier(input_dim=768, num_classes=len(CLASS_NAMES)).to(device)
print("Model loaded")

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# 3. Train model
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

print("Training classifier")
for epoch in range(1, EPOCHS + 1):
    # --- Training ---
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for features, labels in train_loader:
        features = features.to(device)
        if features.ndim == 3 and features.shape[-1] == 1:
            features = features.squeeze(-1)

        labels = torch.tensor([CLASS_TO_IDX[l] for l in labels], dtype=torch.long, device=device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * features.size(0)

        # accuracy
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_train_loss = running_loss / len(train_dataset)
    epoch_train_acc = correct / total
    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_acc)

    # --- Validation ---
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            if features.ndim == 3 and features.shape[-1] == 1:
                features = features.squeeze(-1)

            labels = torch.tensor([CLASS_TO_IDX[l] for l in labels], dtype=torch.long, device=device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * features.size(0)

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_val_loss = running_loss / len(val_dataset)
    epoch_val_acc = correct / total
    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_acc)

    print(f"Epoch [{epoch}/{EPOCHS}] "
          f"Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f} | "
          f"Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.4f}")

# ----------------------------
# Save model and losses
# ----------------------------
torch.save(model.state_dict(), "classifier_model.pt")
np.save("train_losses.npy", np.array(train_losses))
np.save("val_losses.npy", np.array(val_losses))
np.save("train_accuracies.npy", np.array(train_accuracies))
np.save("val_accuracies.npy", np.array(val_accuracies))

print("Model and metrics saved.")

# ----------------------------
# Final evaluation
# ----------------------------
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for features, labels in val_loader:
        features = features.to(device)
        if features.ndim == 3 and features.shape[-1] == 1:
            features = features.squeeze(-1)

        labels = torch.tensor([CLASS_TO_IDX[l] for l in labels], dtype=torch.long, device=device)

        outputs = model(features)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

final_acc = correct / total
print(f"Final Validation Accuracy: {final_acc:.4f}")

# ----------------------------
# Plot losses and accuracies
# ----------------------------
epochs = range(1, EPOCHS + 1)

plt.figure(figsize=(12, 5))

# Loss subplot
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curves")
plt.legend()
plt.grid(True)

# Accuracy subplot
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label="Train Acc")
plt.plot(epochs, val_accuracies, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curves")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()