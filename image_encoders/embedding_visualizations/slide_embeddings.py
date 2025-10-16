import argparse
import json
from pathlib import Path
import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# -----------------------
# Helpers
# -----------------------
def read_label_from_metadata(json_path: Path, key: str, default="disease_state"):
    if json_path.exists():
        with open(json_path, "r") as f:
            md = json.load(f)
        return str(md.get(key, default))
    return default

def get_distinguishable_colors(n: int):
    """Return n strong, distinct colors cycling through a hand-picked palette."""
    base_colors = [
        "red", "blue", "black", "green", "orange",
        "purple", "brown", "pink", "cyan", "magenta",
    ]
    if n <= len(base_colors):
        return base_colors[:n]
    return [base_colors[i % len(base_colors)] for i in range(n)]

def load_embeddings_and_labels(embedding_base_path: Path, zoom_level: str, patch_size: int, slide_encoder: str, metadata_key: str = "disease_state"):
    """Load embeddings and labels for a given zoom level."""
    embeddings_path = embedding_base_path / f"{zoom_level}_{patch_size}px_0px_overlap" / f"slide_features_{slide_encoder}"
    hest_path = Path(f"/home/nadun/wd/HEST/hest_data_{zoom_level.replace('x', '')}/metadata")
    
    all_embeddings = []
    all_labels = []
    
    for embedding_file in sorted(embeddings_path.glob("*.h5")):
        with h5py.File(embedding_file, "r") as f:
            feats = f["features"][:]
        all_embeddings.append(feats)
        
        meta_file = hest_path / embedding_file.with_suffix(".json").name
        label = read_label_from_metadata(meta_file, metadata_key, default="Unknown")
        all_labels.append(label)
    
    return all_embeddings, all_labels

def plot_tsne(embedding_base_path: Path, encoder_name: str, patch_size: int, metadata_key: str = "disease_state", 
              zoom_levels: list = ["20x", "40x"], output_dir: str = "results"):
    """
    Generate t-SNE plot for multiple zoom levels.
    
    Args:
        embedding_base_path: Base path containing zoom level folders
        encoder_name: Name of the slide encoder (used in output filename)
        metadata_key: Metadata field to use for coloring
        zoom_levels: List of zoom levels to process (e.g., ["20x", "40x"])
        output_dir: Directory to save the plot
    """
    
    # -----------------------
    # Load embeddings + labels for all zoom levels
    # -----------------------
    all_embeddings = []
    all_labels = []
    
    for zoom in zoom_levels:
        embeddings, labels = load_embeddings_and_labels(embedding_base_path, zoom, patch_size, encoder_name, metadata_key)
        all_embeddings.extend(embeddings)
        all_labels.extend(labels)
    
    X = np.vstack(all_embeddings)
    
    unique_labels = sorted(set(all_labels))
    num_classes = len(unique_labels)
    colors = get_distinguishable_colors(num_classes)
    label_to_color = {lbl: colors[i] for i, lbl in enumerate(unique_labels)}
    
    print(f"Classes: {unique_labels}")
    print(f"Total samples: {len(all_labels)} from {len(all_embeddings)} slides")
    
    # -----------------------
    # Run t-SNE
    # -----------------------
    tsne = TSNE(n_components=2, perplexity=30, init="pca", 
                random_state=42, learning_rate="auto")
    X_2d = tsne.fit_transform(X)
    
    # -----------------------
    # Plot
    # -----------------------
    fig, ax = plt.subplots(figsize=(12, 8))
    labels_arr = np.array(all_labels)
    
    for lbl in unique_labels:
        idx = np.where(labels_arr == lbl)[0]
        if len(idx) > 0:
            ax.scatter(X_2d[idx, 0], X_2d[idx, 1],
                      s=5, alpha=0.6,
                      c=label_to_color[lbl],
                      label=lbl)
    
    ax.set_title(f"t-SNE colored by '{metadata_key}' - {encoder_name}")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(title=metadata_key, frameon=True, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    out_file = output_path / f"tsne_{encoder_name}.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"t-SNE plot saved to {out_file}")
    
    return str(out_file)

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate t-SNE plots from slide embeddings")
    parser.add_argument("--embedding_path", type=str, default = "/home/nadun/wd/TRIDENT/trident_processed",
                       help="Base path containing embedding folders (e.g., /home/nadun/wd/TRIDENT/trident_processed)")
    parser.add_argument("--encoder_name", type=str, required=True,
                       help="Name of the slide encoder (e.g., 'titan')")
    parser.add_argument("--metadata_key", type=str, default="disease_state",
                       help="Metadata key for coloring (default: disease_state)")
    parser.add_argument("--zoom_levels", type=str, nargs="+", default=["20x", "40x"],
                       help="Zoom levels to process (default: 20x 40x)")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Output directory for plots (default: results)")
    parser.add_argument("--patch_size", type=int, default=512,
                       help="Patch size used in embeddings (default: 512)")
    args = parser.parse_args()
    
    plot_tsne(
        embedding_base_path=Path(args.embedding_path),
        encoder_name=args.encoder_name,
        patch_size=args.patch_size,
        metadata_key=args.metadata_key,
        zoom_levels=args.zoom_levels,
        output_dir=args.output_dir
    )
