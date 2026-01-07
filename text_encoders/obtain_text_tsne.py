import argparse
import h5py
from pathlib import Path
import numpy as np
from sklearn.manifold import TSNE
import json

import matplotlib.pyplot as plt


def get_distinguishable_colors(n: int):
    """Return n strong, distinct colors cycling through a hand-picked palette."""
    base_colors = [
        "red", "blue", "black", "green", "orange",
        "purple", "brown", "pink", "cyan", "magenta",
    ]
    if n <= len(base_colors):
        return base_colors[:n]
    return [base_colors[i % len(base_colors)] for i in range(n)]


def load_text_embeddings(embedding_file: Path):
    """Load text embeddings from HDF5 file."""
    with h5py.File(embedding_file, 'r') as f:
        embeddings = f['text_embedding'][:]
        # Get slide ID from filename (e.g., "TENX95_slide.h5" -> "TENX95")
        slide_id = embedding_file.stem.replace('_slide', '')
        
        # Find corresponding metadata file
        # Try both metadata paths
        meta_paths = [
            Path("/home/nadun/wd/HEST/hest_data_20/metadata"),
            Path("/home/nadun/wd/HEST/hest_data_40/metadata")
        ]
        
        # Read labels from metadata
        label = 'Unknown'
        for hest_path in meta_paths:
            meta_file = hest_path / f"{slide_id}.json"
            if meta_file.exists():
                with open(meta_file, 'r') as mf:
                    metadata = json.load(mf)
                    label = metadata.get('disease_state', 'Unknown')
                    break  # Stop after finding the first matching metadata file
        
        # Assign the same label to all embeddings from this slide
        labels = [label] * len(embeddings)
        
        return embeddings, labels


def plot_text_tsne(embedding_folders: list, output_file: str = "tsne_prompts.png"):
    """
    Generate t-SNE plot for text embeddings from all .h5 files in specified folders.
    
    Args:
        embedding_folders: List of folder paths containing text embedding .h5 files
        output_file: Output filename for the plot
    """
    
    all_embeddings = []
    all_labels = []
    
    # Collect all .h5 files from all folders
    all_h5_files = []
    for folder in embedding_folders:
        folder_path = Path(folder)
        h5_files = sorted(folder_path.glob("*.h5"))
        all_h5_files.extend(h5_files)
        print(f"Found {len(h5_files)} .h5 files in {folder}")
    
    print(f"Total .h5 files to process: {len(all_h5_files)}")
    
    # Load embeddings from all files
    for h5_file in all_h5_files:
        print(f"Loading {h5_file}")
        embeddings, labels = load_text_embeddings(h5_file)
        all_embeddings.append(embeddings)
        all_labels.extend(labels)
    
    # Stack all embeddings
    X = np.vstack(all_embeddings)
    
    # Get unique labels and assign colors
    unique_labels = sorted(set(all_labels))
    num_classes = len(unique_labels)
    colors = get_distinguishable_colors(num_classes)
    label_to_color = {lbl: colors[i] for i, lbl in enumerate(unique_labels)}
    
    print(f"Classes: {unique_labels}")
    print(f"Total samples: {len(all_labels)}")
    print(f"Embedding shape: {X.shape}")
    
    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=min(30, len(all_labels) - 1), 
                init="pca", random_state=42, learning_rate="auto")
    X_2d = tsne.fit_transform(X)
    
    # Plot
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    labels_arr = np.array(all_labels)
    
    for lbl in unique_labels:
        idx = np.where(labels_arr == lbl)[0]
        if len(idx) > 0:
            ax.scatter(X_2d[idx, 0], X_2d[idx, 1],
                      s=100, alpha=0.7,
                      c=label_to_color[lbl],
                      label=lbl,
                      edgecolors='black',
                      linewidth=0.5)
    
    ax.set_title("t-SNE of Text Embeddings (Disease Prompts)", fontsize=14, fontweight='bold')
    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)
    ax.legend(title="Disease State", frameon=True, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"t-SNE plot saved to {output_path}")
    
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate t-SNE plots from text embeddings")
    parser.add_argument("--embedding_folders", type=str, nargs="+",
                       default=[
                           "/home/nadun/wd/literature-guided-digital-histopathology/text_encoders/text_embeds_20",
                           "/home/nadun/wd/literature-guided-digital-histopathology/text_encoders/text_embeds_40"
                       ],
                       help="Folders containing text embedding .h5 files")
    parser.add_argument("--output_file", type=str, default="tsne_prompts.png",
                       help="Output filename for the plot")
    
    args = parser.parse_args()
    
    plot_text_tsne(
        embedding_folders=args.embedding_folders,
        output_file=args.output_file
    )