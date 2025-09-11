import torch
import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Experiment settings
encoder = "uni"
dataset = "hest1k"

# Path to embeddings
file_name = "MEND63.pt"
embeddings_path = Path(f"/home/nadun/hdd/CLAM/data/{encoder}_features/pt_files")
hest_path = Path("/home/nadun/hdd/HEST/hest_data/metadata")
embedding_file = embeddings_path / file_name

# Load embedding tensor
embeddings = torch.load(embedding_file)  # shape: [patches, 512]
print("Embedding shape:", embeddings.shape)

# Load JSON file
metadata_file = hest_path / file_name.replace('.pt', '.json')
with open(metadata_file, "r") as f:
    metadata = json.load(f)

# Get class
disease_state = metadata.get("disease_state", None)
print(f"Disease State for {metadata.get('id', 'Unknown')}: {disease_state}")

# Convert to numpy
X = embeddings.numpy()

# Run t-SNE (reduce to 2D)
tsne = TSNE(n_components=2, random_state=42, init="pca", perplexity=30)
X_2d = tsne.fit_transform(X)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], s=10, alpha=0.7)
plt.title(f"t-SNE Visualization of {embedding_file.name}  with class {disease_state}")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")

# Save plot
output_file = f"results/{encoder}_{dataset}/tsne_{file_name.replace('.pt', '.png')}"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
print(f"t-SNE plot saved to {output_file}")