#!/bin/bash
# Memory-optimized training script for Stage 2 with QLoRA

echo "Starting Stage 2 Training"
echo "===================================================="

python train_stage2.py \
    --stage1_checkpoint /home/nadun/wd/vlm_histo/pipeline/checkpoints/stage1/2026-01-24_11-15_vt64_ml512_e12_bs2/best_model.pt \
    --epochs 5 \
    --train_split train,val \
    --val_split test \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr 5e-6 \
    --projection_lr 5e-5 \
    --weight_decay 0.1 \
    --gradient_checkpointing \
    --val_batch_size 1 \
    --output_dir checkpoints/stage2

echo ""
echo "Training complete!"
