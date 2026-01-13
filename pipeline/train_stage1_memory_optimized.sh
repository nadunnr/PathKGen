#!/bin/bash
# Memory-optimized training script for Stage 1

echo "Starting Stage 1 Training with Memory Optimizations"
echo "===================================================="
echo ""
echo "Optimizations enabled:"
echo "  - Gradient checkpointing: Saves ~40% activation memory"
echo "  - Reduced batch size: 4 (from 8)"
echo "  - Gradient accumulation: 2 steps (effective batch size = 8)"
echo "  - Reduced max_length: 512 tokens (from 512)"
echo ""

python train_stage1.py \
    --epochs 10 \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --max_length 512 \
    --gradient_checkpointing \
    --lr 1e-4 \
    --output_dir checkpoints/stage1 \
    --save_every 2

echo ""
echo "Training complete!"
