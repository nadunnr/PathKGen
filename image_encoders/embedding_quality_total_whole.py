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

all_embeddings = []
all_labels = []

# Max patches to sample per WSI (adjust to control memory/compute)
MAX_PATCHES = 500

for embedding_file in embeddings_path.glob("*.pt"):
    file_name = embedding_file.name

    # Load embeddings [patches, 512]
    embeddings = torch.load(embedding_file)

    # Sample patches if too many
    if embeddings.shape[0] > MAX_PATCHES:
        idx = np.random.choice(embeddings.shape[0], MAX_PATCHES, replace=False)
        embeddings = embeddings[idx]

    # Convert to numpy
    embeddings_np = embeddings.numpy()
    all_embeddings.append(embeddings_np)

    # Load metadata
    metadata_file = hest_path / file_name.replace(".pt", ".json")
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        disease_state = metadata.get("disease_state", "Unknown")
    else:
        disease_state = "Unknown"

    # Add labels (one per patch)
    all_labels.extend([disease_state] * embeddings_np.shape[0])

    print(f"Processed {file_name}: using {embeddings_np.shape[0]} patches, disease_state={disease_state}")

# Stack all sampled embeddings
X = np.vstack(all_embeddings)

# Run t-SNE on all patch embeddings
tsne = TSNE(n_components=2, random_state=42, init="pca", perplexity=30)
X_2d = tsne.fit_transform(X)

# Plot
plt.figure(figsize=(12, 10))

color_map = {"Cancer": "tab:red", "Healthy": "tab:green", "Unknown": "tab:gray"}

for label in set(all_labels):
    idx = [i for i, l in enumerate(all_labels) if l == label]
    plt.scatter(X_2d[idx, 0], X_2d[idx, 1], s=5, alpha=0.6,
                c=color_map.get(label, "tab:blue"), label=label)

plt.title("t-SNE of All Patch Embeddings (sampled per WSI)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend(title="Disease State")
plt.tight_layout()

# Save plot
output_file = f"results/{encoder}_{dataset}/tsne_all_wsi_whole.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
print(f"t-SNE plot saved to {output_file}")