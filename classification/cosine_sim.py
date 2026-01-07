from pathlib import Path
import h5py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

# --- Paths ---
vision_data_path = Path("/home/nadun/wd/TRIDENT/trident_processed/40x_256px_0px_overlap/features_conch_v1")
text_data_path = Path("/home/nadun/hdd/literature-guided-digital-histopathology/text_encoders/text_descriptions_embeds")

# --- Collect all similarities ---
all_similarities = []
slide_names = []

for vision_file in sorted(vision_data_path.glob("*.h5")):
    slide_id = vision_file.stem  # e.g., 'ZEN49'
    text_file = text_data_path / f"{slide_id}.h5"

    if not text_file.exists():
        print(f"Skipping {slide_id} — text embedding not found.")
        continue

    with h5py.File(vision_file, "r") as vf:
        image_embs = vf["features"][:]
      
    with h5py.File(text_file, "r") as tf:
        text_emb = tf["text_embedding"][:]
  

    # Ensure proper shape
    if image_embs.ndim == 1:
        image_embs = image_embs[np.newaxis, :]
     
    if text_emb.ndim == 1:
        text_emb = text_emb[np.newaxis, :]
      

    # # Normalize
    image_embs = image_embs / np.linalg.norm(image_embs, axis=1, keepdims=True)
    text_emb = text_emb / np.linalg.norm(text_emb, axis=1, keepdims=True)

    # Compute cosine similarity
    sim_matrix = cosine_similarity(image_embs, text_emb)  # (N_patches, 1)
    mean_sim = np.mean(sim_matrix)                         # average across patches
    all_similarities.append(mean_sim)
    slide_names.append(slide_id)   


# --- Plot as a bar chart ---
plt.figure(figsize=(10, 5))
plt.bar(slide_names, all_similarities, color='mediumseagreen')

plt.title("Mean Pairwise Cosine Similarity — Image vs Text Embeddings")
plt.xlabel("Slide ID")
plt.ylabel("Mean Cosine Similarity")
plt.xticks(rotation=45, ha="right")
plt.ylim(0, 1)
plt.tight_layout()

# --- Save the plot ---
output_path = Path("pairwise_similarity_plot.png")
plt.savefig(output_path, dpi=300)
plt.show()

print(f"\n Saved plot to: {output_path.resolve()}")