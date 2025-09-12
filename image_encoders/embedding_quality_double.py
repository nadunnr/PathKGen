import torch
import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# Experiment settings
encoder = "uni"
dataset = "hest1k"

# Paths
embeddings_path = Path(f"/home/nadun/hdd/CLAM/data/{encoder}_features/pt_files")
hest_path = Path("/home/nadun/hdd/HEST/hest_data/metadata")

# Hardcode two files
files = ["TENX27.pt", "TENX30.pt"]

all_embeddings = []
labels = []

for file_name in files:
    # Load embeddings
    embedding_file = embeddings_path / file_name
    embeddings = torch.load(embedding_file)  # shape [patches, 512]

    # Load metadata
    metadata_file = hest_path / file_name.replace(".pt", ".json")
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    disease_state = metadata.get("disease_state", None)
    label = f"{file_name.replace('.pt','')} ({disease_state})"

    print(f"{file_name}: {embeddings.shape}, disease_state={disease_state}")

    # Store embeddings and labels
    all_embeddings.append(embeddings.numpy())
    labels.extend([label] * embeddings.shape[0])

# Concatenate embeddings
X = np.vstack(all_embeddings)

# Run t-SNE on combined embeddings
tsne = TSNE(n_components=2, random_state=42, init="pca", perplexity=30)
X_2d = tsne.fit_transform(X)

# Plot
plt.figure(figsize=(10, 8))

unique_labels = list(set(labels))
colors = ["tab:blue", "tab:red"]  # extend if more files

for i, ulab in enumerate(unique_labels):
    idx = [j for j, lab in enumerate(labels) if lab == ulab]
    plt.scatter(X_2d[idx, 0], X_2d[idx, 1], s=10, alpha=0.7, c=colors[i], label=ulab)

plt.title("t-SNE Visualization of Two WSIs (patch embeddings)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend()

# Save plot
output_file = f"results/{encoder}_{dataset}/tsne_two_wsi.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
print(f"t-SNE plot saved to {output_file}")