import torch
import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import warnings

from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize

# -------------------
# Paths
# -------------------
proj_root = Path("/home/nadun/hdd/literature-guided-digital-histopathology")
prompts_file = proj_root / "text_encoders/hest_metadata_prompts.jsonl"
metadata_path = Path("/home/nadun/hdd/HEST/hest_data/metadata")
checkpoint_path = proj_root / "checkpoints/conch/pytorch_model.bin"

# -------------------
# Setup
# -------------------
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.models.layers")
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Load CONCH model
model_cfg = "conch_ViT-B-16"
model, preprocess = create_model_from_pretrained(str(model_cfg), str(checkpoint_path))
model = model.to(device).eval()

# Tokenizer
tokenizer = get_tokenizer()

# -------------------
# Load prompts + disease state
# -------------------
texts = []
labels = []
ids = []

with open(prompts_file, "r") as f:
    for line in f:
        data = json.loads(line)
        wsi_id = data["id"]
        prompt = data["response"]

        # Match metadata JSON
        metadata_file = metadata_path / f"{wsi_id}.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as mf:
                metadata = json.load(mf)
            disease_state = metadata.get("disease_state", "Unknown")
        else:
            disease_state = "Unknown"

        ids.append(wsi_id)
        texts.append(prompt)
        labels.append(disease_state)

print(f"Loaded {len(texts)} prompts.")

# -------------------
# Encode text embeddings
# -------------------
tokenized_prompts = tokenize(texts=texts, tokenizer=tokenizer).to(device)

with torch.inference_mode():
    text_embeddings = model.encode_text(tokenized_prompts)  # shape [N, 512]

embeddings_np = text_embeddings.cpu().numpy()

# -------------------
# Run t-SNE
# -------------------
tsne = TSNE(n_components=2, random_state=42, init="pca", perplexity=5)
X_2d = tsne.fit_transform(embeddings_np)

# -------------------
# Plot
# -------------------
plt.figure(figsize=(10, 8))

color_map = {"Cancer": "tab:red", "Healthy": "tab:green", "Unknown": "tab:gray"}

for label in set(labels):
    idx = [i for i, l in enumerate(labels) if l == label]
    plt.scatter(X_2d[idx, 0], X_2d[idx, 1], s=60, alpha=0.8,
                c=color_map.get(label, "tab:blue"), label=label)

plt.title("t-SNE of Text Prompt Embeddings (by disease state)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend(title="Disease State")
plt.tight_layout()

# Save plot
output_file = "results/tsne_text_prompts.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
print(f"✅ t-SNE plot saved to {output_file}")