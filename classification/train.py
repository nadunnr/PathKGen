from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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

# train/val split (simple example)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Dataset loaded with Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

# 2. Linear classifier 
model = BaseClassifier(input_dim=768).to(device)
print("Model loaded")

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()


# 3. Train model
train_losses = []
val_losses = []

print("Training classifier")
for epoch in range(1, EPOCHS + 1):
    # --- Training ---
    model.train()
    running_loss = 0.0

    for features, labels in train_loader:
        features = features.to(device)
        # flatten features if shape is (B, D, 1)
        if features.ndim == 3 and features.shape[-1] == 1:
            features = features.squeeze(-1)

        labels = [CLASS_TO_IDX[l] for l in labels]  # convert to int indices
        labels = torch.tensor(labels, dtype=torch.long, device=device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * features.size(0)

    epoch_train_loss = running_loss / len(train_dataset)
    train_losses.append(epoch_train_loss)

    # --- Validation ---
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            if features.ndim == 3 and features.shape[-1] == 1:
                features = features.squeeze(-1)

            labels = [CLASS_TO_IDX[l] for l in labels]
            labels = torch.tensor(labels, dtype=torch.long, device=device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * features.size(0)

    epoch_val_loss = running_loss / len(val_dataset)
    val_losses.append(epoch_val_loss)

    print(f"Epoch [{epoch}/{EPOCHS}] "
          f"Train Loss: {epoch_train_loss:.4f} | "
          f"Val Loss: {epoch_val_loss:.4f}")

# 4. Plot losses
plt.figure(figsize=(8, 6))
plt.plot(range(1, EPOCHS + 1), train_losses, label="Train Loss")
plt.plot(range(1, EPOCHS + 1), val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=150)
plt.show()