"""
Visualize training progress from saved history.

Usage:
    python pipeline/plot_training.py \
        --stage1_history /path/to/stage1/training_history.json \
        --stage2_history /path/to/stage2/training_history.json \
        --output_dir /path/to/plots
"""

import argparse
import json
import matplotlib.pyplot as plt
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Plot training history")
    parser.add_argument("--stage1_history", type=str, default=None,
                       help="Path to Stage 1 training_history.json")
    parser.add_argument("--stage2_history", type=str, default=None,
                       help="Path to Stage 2 training_history.json")
    parser.add_argument("--output_dir", type=str, 
                       default="/home/nadun/wd/vlm_histo/plots",
                       help="Output directory for plots")
    return parser.parse_args()


def plot_losses(history, stage_name, output_dir):
    """Plot train and validation losses."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    ax.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-s', label='Val Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'{stage_name} Training Progress', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add best validation loss marker
    best_epoch = history['val_loss'].index(min(history['val_loss'])) + 1
    best_loss = min(history['val_loss'])
    ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best Val (Epoch {best_epoch})')
    ax.text(best_epoch, best_loss, f' {best_loss:.4f}', fontsize=10, color='green')
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / f'{stage_name.lower().replace(" ", "_")}_losses.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to: {output_path}")
    
    plt.close()


def plot_combined_stages(history1, history2, output_dir):
    """Plot both stages side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Stage 1
    epochs1 = range(1, len(history1['train_loss']) + 1)
    axes[0].plot(epochs1, history1['train_loss'], 'b-o', label='Train Loss', linewidth=2)
    axes[0].plot(epochs1, history1['val_loss'], 'r-s', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Stage 1: Projection Only', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Stage 2
    epochs2 = range(1, len(history2['train_loss']) + 1)
    axes[1].plot(epochs2, history2['train_loss'], 'b-o', label='Train Loss', linewidth=2)
    axes[1].plot(epochs2, history2['val_loss'], 'r-s', label='Val Loss', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Stage 2: Projection + QLoRA', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'combined_training_progress.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved combined plot to: {output_path}")
    
    plt.close()


def print_summary(history, stage_name):
    """Print training summary statistics."""
    print(f"\n{'='*60}")
    print(f"{stage_name} Summary")
    print(f"{'='*60}")
    print(f"Total epochs: {len(history['train_loss'])}")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")
    print(f"Best val loss: {min(history['val_loss']):.4f} (Epoch {history['val_loss'].index(min(history['val_loss'])) + 1})")
    print(f"{'='*60}")


def main():
    args = parse_args()
    
    if args.stage1_history is None and args.stage2_history is None:
        print("Error: Must specify at least one history file")
        return
    
    # Load histories
    history1 = None
    history2 = None
    
    if args.stage1_history:
        with open(args.stage1_history, 'r') as f:
            history1 = json.load(f)
        plot_losses(history1, "Stage 1", args.output_dir)
        print_summary(history1, "Stage 1")
    
    if args.stage2_history:
        with open(args.stage2_history, 'r') as f:
            history2 = json.load(f)
        plot_losses(history2, "Stage 2", args.output_dir)
        print_summary(history2, "Stage 2")
    
    # Combined plot if both are available
    if history1 and history2:
        plot_combined_stages(history1, history2, args.output_dir)
    
    print("\n✓ All plots generated successfully!")


if __name__ == "__main__":
    main()
