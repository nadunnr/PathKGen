#!/usr/bin/env python3
"""
Inspect actual predictions vs references to understand why NLU scores are low.
This will show sample outputs to diagnose the issue.
"""
import json
import sys

checkpoint_dir = '/home/nadun/wd/vlm_histo/checkpoints/stage2/2026-02-01_13-09_vt64_ml512_e5_bs2'

# Load training history
history_path = f'{checkpoint_dir}/training_history.json'
try:
    with open(history_path) as f:
        history = json.load(f)
    
    print("=" * 80)
    print("Training History Analysis")
    print("=" * 80)
    
    print("\nMetrics available:")
    for key in sorted(history.keys()):
        if isinstance(history[key], list):
            print(f"  - {key}: {len(history[key])} values")
        else:
            print(f"  - {key}: {type(history[key]).__name__}")
    
    # Check if final_evaluation exists
    if 'final_evaluation' in history:
        print("\n" + "=" * 80)
        print("Final Evaluation Results")
        print("=" * 80)
        
        final = history['final_evaluation']
        print(f"\nBest epoch: {final.get('best_epoch', 'N/A')}")
        print(f"Val loss: {final.get('val_loss', 'N/A'):.4f}")
        
        if 'metrics' in final:
            metrics = final['metrics']
            print("\nNLU Metrics:")
            print(f"  BLEU-1: {metrics.get('bleu1', 0):.2f}")
            print(f"  BLEU-2: {metrics.get('bleu2', 0):.2f}")
            print(f"  BLEU-3: {metrics.get('bleu3', 0):.2f}")
            print(f"  BLEU-4: {metrics.get('bleu4', 0):.2f}")
            print(f"  METEOR: {metrics.get('meteor', 0):.2f}")
            print(f"  ROUGE-L: {metrics.get('rouge_l', 0):.2f}")
        
        # Check for sample predictions
        if 'sample_outputs' in final:
            print("\n" + "=" * 80)
            print("Sample Predictions")
            print("=" * 80)
            
            samples = final['sample_outputs']
            for i, sample in enumerate(samples[:3], 1):
                print(f"\n--- Sample {i} ---")
                print(f"Prediction: {sample.get('prediction', 'N/A')[:200]}...")
                print(f"Reference:  {sample.get('reference', 'N/A')[:200]}...")
    
    # Analyze training progression
    if 'train_loss' in history and 'val_loss' in history:
        print("\n" + "=" * 80)
        print("Training Progression")
        print("=" * 80)
        
        train_losses = history['train_loss']
        val_losses = history['val_loss']
        
        print(f"\nEpochs trained: {len(train_losses)}")
        print(f"\nEpoch | Train Loss | Val Loss | Improvement")
        print("-" * 55)
        
        prev_val = float('inf')
        for epoch, (train, val) in enumerate(zip(train_losses, val_losses), 1):
            improvement = "↓" if val < prev_val else "↑" if val > prev_val else "="
            print(f"  {epoch:2d}  |   {train:6.4f}   | {val:6.4f}  | {improvement}")
            prev_val = val
        
        best_epoch = val_losses.index(min(val_losses)) + 1
        best_val = min(val_losses)
        print(f"\nBest: Epoch {best_epoch}, Val Loss {best_val:.4f}")
        
        # Check for overfitting
        final_gap = train_losses[-1] - val_losses[-1]
        if final_gap < -0.1:
            print(f"\n⚠️  Possible underfitting (train > val by {abs(final_gap):.3f})")
        elif final_gap > 0.3:
            print(f"\n⚠️  Possible overfitting (val > train by {final_gap:.3f})")
        else:
            print(f"\n✓ Good generalization gap: {final_gap:.3f}")

except FileNotFoundError:
    print(f"❌ Training history not found: {history_path}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("Analysis complete")
print("=" * 80)
