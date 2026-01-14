#!/bin/bash
# Memory-optimized training script for Stage 2 with QLoRA

echo "Starting Stage 2 Training"
echo "===================================================="

python train_stage2.py \
    --stage1_checkpoint checkpoints/stage1/best_model.pt \
    --epochs 5 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --output_dir checkpoints/stage2

echo ""
echo "Training complete!"
