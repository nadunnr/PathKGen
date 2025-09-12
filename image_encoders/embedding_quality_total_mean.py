import torch
import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# Paths
embeddings_path = Path("/home/nadun/hdd/CLAM/data/conch_features/pt_files")
hest_path = Path("/home/nadun/hdd/HEST/hest_data/metadata")

all_embeddings = []
all_labels = []

# Loop through all .pt files in folder
for embedding_file in embeddings_path.glob("*.pt"):
    file_name = embedding_file.name

    # Load embeddings [patches, 512]
    embeddings = torch.load(embedding_file)

    # Take mean across patches → [512]
    mean_embedding = embeddings.mean(dim=0).numpy()
    all_embeddings.append(mean_embedding)

    # Load corresponding JSON metadata
    metadata_file = hest_path / file_name.replace(".pt", ".json")
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        disease_state = metadata.get("disease_state", "Unknown")
    else:
        disease_state = "Unknown"

    all_labels.append(disease_state)

    print(f"Processed {file_name}: mean_embedding {mean_embedding.shape}, disease_state={disease_state}")

# Convert to numpy for t-SNE
X = np.vstack(all_embeddings)

# Run t-SNE on WSI-level embeddings
tsne = TSNE(n_components=2, random_state=42, init="pca", perplexity=10)
X_2d = tsne.fit_transform(X)

# Plot
plt.figure(figsize=(10, 8))

# Map disease_state to color
color_map = {"Cancer": "tab:red", "Healthy": "tab:green", "Unknown": "tab:gray"}

for label in set(all_labels):
    idx = [i for i, l in enumerate(all_labels) if l == label]
    plt.scatter(X_2d[idx, 0], X_2d[idx, 1], s=60, alpha=0.8,
                c=color_map.get(label, "tab:blue"), label=label)

plt.title("t-SNE of WSIs (mean patch embeddings per WSI)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend(title="Disease State")
plt.tight_layout()

# Save plot
output_file = "results/tsne_all_wsi_mean.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
print(f"t-SNE plot saved to {output_file}")