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
    parser.add_argument("--stage1_history", type=str, default="/home/nadun/wd/vlm_histo/pipeline/checkpoints/stage1/23.01.2026_epoch 50/training_history.json",
                       help="Path to Stage 1 training_history.json")
    parser.add_argument("--stage2_history", type=str, default=None,
                       help="Path to Stage 2 training_history.json")
    return parser.parse_args()

def plot_losses(history, stage_name, output_dir):
    """Plot train and validation losses with comparison table."""
    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.3)
    
    # Left subplot: Training curves
    ax = fig.add_subplot(gs[0])
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    ax.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-s', label='Val Loss', linewidth=2)
    ax.plot(epochs, history['bleu1'], 'g-o', label='BLEU-1', linewidth=2)
    ax.plot(epochs, history['bleu2'], 'c-o', label='BLEU-2', linewidth=2)
    ax.plot(epochs, history['bleu3'], 'm-o', label='BLEU-3', linewidth=2)
    ax.plot(epochs, history['bleu4'], 'y-o', label='BLEU-4', linewidth=2)
    ax.plot(epochs, history['meteor'], 'orange', marker='o', label='METEOR', linewidth=2)
    ax.plot(epochs, history['rouge_l'], 'purple', marker='o', label='ROUGE-L', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Score / Loss', fontsize=12)
    ax.set_title(f'{stage_name} Training Progress', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Mark best train loss
    best_train_epoch = history['train_loss'].index(min(history['train_loss'])) + 1
    best_train_loss = min(history['train_loss'])
    ax.plot(best_train_epoch, best_train_loss, 'b*', markersize=15, 
            markeredgecolor='darkblue', markeredgewidth=2)
    ax.annotate(f'Best Train\n{best_train_loss:.4f}', 
                xy=(best_train_epoch, best_train_loss),
                xytext=(10, 10), textcoords='offset points',
                fontsize=8, color='blue', weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7),
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
    
    # Mark best validation loss
    best_val_epoch = history['val_loss'].index(min(history['val_loss'])) + 1
    best_val_loss = min(history['val_loss'])
    ax.plot(best_val_epoch, best_val_loss, 'r*', markersize=15,
            markeredgecolor='darkred', markeredgewidth=2)
    ax.annotate(f'Best Val\n{best_val_loss:.4f}', 
                xy=(best_val_epoch, best_val_loss),
                xytext=(10, -15), textcoords='offset points',
                fontsize=8, color='red', weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    
    # Mark highest NLP scores
    metrics_to_mark = [
        ('bleu1', 'BLEU-1', 'green', 'lightgreen'),
        ('bleu2', 'BLEU-2', 'blue', 'lightblue'),
        ('bleu3', 'BLEU-3', 'cyan', 'lightcyan'),
        ('bleu4', 'BLEU-4', 'gold', 'yellow'),
        ('meteor', 'METEOR', 'darkorange', 'lightyellow'),
        ('rouge_l', 'ROUGE-L', 'purple', 'plum')
    ]
    
    offset_y = 20
    for metric_key, metric_name, color, bg_color in metrics_to_mark:
        if metric_key in history and len(history[metric_key]) > 0:
            best_epoch = history[metric_key].index(max(history[metric_key])) + 1
            best_score = max(history[metric_key])
            ax.plot(best_epoch, best_score, '*', color=color, markersize=15,
                    markeredgecolor='black', markeredgewidth=1.5)
            ax.annotate(f'Best {metric_name}\n{best_score:.4f}', 
                        xy=(best_epoch, best_score),
                        xytext=(-50, offset_y), textcoords='offset points',
                        fontsize=8, color=color, weight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=bg_color, alpha=0.7),
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
            offset_y += 15
    
    # Right subplot: Comparison table
    ax_table = fig.add_subplot(gs[1])
    ax_table.axis('off')
    
    # Build comparison data
    nlp_metrics = ['bleu1', 'bleu2', 'bleu3', 'bleu4', 'meteor', 'rouge_l']
    metric_names = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'METEOR', 'ROUGE-L']
    
    table_data = []
    cell_colors = []
    
    # Header row
    table_data.append(['Metric', 'Best Score\n(Epoch)', 'Score @ Best\nVal Loss'])
    cell_colors.append(['#E0E0E0', '#E0E0E0', '#E0E0E0'])
    
    for metric_key, metric_name in zip(nlp_metrics, metric_names):
        if metric_key in history and len(history[metric_key]) > 0:
            # Best score
            best_score = max(history[metric_key])
            best_score_epoch = history[metric_key].index(best_score) + 1
            
            # Score at best val loss epoch
            score_at_best_val = history[metric_key][best_val_epoch - 1]
            
            # Determine if best score matches score at best val loss
            if best_score_epoch == best_val_epoch:
                row_color = ['#D4EDDA', '#D4EDDA', '#D4EDDA']  # Light green - aligned
            else:
                row_color = ['#FFF3CD', '#FFF3CD', '#FFF3CD']  # Light yellow - different
            
            table_data.append([
                metric_name,
                f'{best_score:.4f}\n(E{best_score_epoch})',
                f'{score_at_best_val:.4f}'
            ])
            cell_colors.append(row_color)
    
    # Create table
    table = ax_table.table(cellText=table_data, cellColours=cell_colors,
                           cellLoc='center', loc='center',
                           bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(3):
        table[(0, i)].set_facecolor('#4A90E2')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Add title above table
    ax_table.text(0.5, 1.1, f'NLP Metrics Comparison\n(Best Val Loss @ Epoch {best_val_epoch})',
                  ha='center', va='top', fontsize=12, fontweight='bold',
                  transform=ax_table.transAxes)
    
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
    print(f"\n📉 Losses:")
    print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final val loss: {history['val_loss'][-1]:.4f}")
    print(f"  Best train loss: {min(history['train_loss']):.4f} (Epoch {history['train_loss'].index(min(history['train_loss'])) + 1})")
    print(f"  Best val loss: {min(history['val_loss']):.4f} (Epoch {history['val_loss'].index(min(history['val_loss'])) + 1})")
    
    print(f"\n📊 Best NLP Metrics:")
    if 'bleu1' in history and len(history['bleu1']) > 0:
        print(f"  BLEU-1: {max(history['bleu1']):.4f} (Epoch {history['bleu1'].index(max(history['bleu1'])) + 1})")
    if 'bleu2' in history and len(history['bleu2']) > 0:
        print(f"  BLEU-2: {max(history['bleu2']):.4f} (Epoch {history['bleu2'].index(max(history['bleu2'])) + 1})")
    if 'bleu3' in history and len(history['bleu3']) > 0:
        print(f"  BLEU-3: {max(history['bleu3']):.4f} (Epoch {history['bleu3'].index(max(history['bleu3'])) + 1})")
    if 'bleu4' in history and len(history['bleu4']) > 0:
        print(f"  BLEU-4: {max(history['bleu4']):.4f} (Epoch {history['bleu4'].index(max(history['bleu4'])) + 1})")
    if 'meteor' in history and len(history['meteor']) > 0:
        print(f"  METEOR: {max(history['meteor']):.4f} (Epoch {history['meteor'].index(max(history['meteor'])) + 1})")
    if 'rouge_l' in history and len(history['rouge_l']) > 0:
        print(f"  ROUGE-L: {max(history['rouge_l']):.4f} (Epoch {history['rouge_l'].index(max(history['rouge_l'])) + 1})")
    print(f"{'='*60}")


def main():
    args = parse_args()
    
    if args.stage1_history is None and args.stage2_history is None:
        print("Error: Must specify at least one history file (--stage1_history or --stage2_history)")
        return
    
    # Load histories
    history1 = None
    history2 = None
    
    if args.stage1_history:
        print(f"Loading Stage 1 history from: {args.stage1_history}")
        with open(args.stage1_history, 'r') as f:
            history1 = json.load(f)
        
        # Save plots in the same directory as the history file
        stage1_dir = str(Path(args.stage1_history).parent)
        print(f"Saving Stage 1 plots to: {stage1_dir}")
        plot_losses(history1, "Stage 1", stage1_dir)
        print_summary(history1, "Stage 1")
    
    if args.stage2_history:
        print(f"\nLoading Stage 2 history from: {args.stage2_history}")
        with open(args.stage2_history, 'r') as f:
            history2 = json.load(f)
        
        # Save plots in the same directory as the history file
        stage2_dir = str(Path(args.stage2_history).parent)
        print(f"Saving Stage 2 plots to: {stage2_dir}")
        plot_losses(history2, "Stage 2", stage2_dir)
        print_summary(history2, "Stage 2")
    
    # Combined plot if both are available (save to Stage 2 directory, or Stage 1 if only that exists)
    if history1 and history2:
        combined_dir = str(Path(args.stage2_history).parent)
        print(f"\nSaving combined plot to: {combined_dir}")
        plot_combined_stages(history1, history2, combined_dir)
    
    print("\n✓ All plots generated successfully!")


if __name__ == "__main__":
    main()
