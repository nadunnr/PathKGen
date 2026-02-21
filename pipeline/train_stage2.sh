#!/bin/bash
# Memory-optimized training script for Stage 2 with QLoRA
# Optimized configuration based on WSI-LLaVA analysis

# Set PyTorch memory allocation to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting Stage 2 Training (Optimized)"
echo "===================================================="

parent_dir="nadun/histo_vlm/checkpoints/stage2/$(date +%Y-%m-%d_%H-%M)_optimized wd/vlm_histo/pipeline"

python train_stage2.py \
    --stage1_checkpoint /home/sandun/nadun/histo_vlm/pipeline/checkpoints/stage1/2026-01-31_08-18_vt64_ml512_e20_bs2/best_model.pt \
    --epochs 5 \
    --train_split train,val \
    --val_split test \
    --batch_size 1 \
    --gradient_accumulation_steps 32 \
    --lr 2e-4 \
    --projection_lr 2e-5 \
    --weight_decay 0.0 \
    --lora_rank 128 \
    --lora_alpha 256 \
    --feature_key tokens \
    --gradient_checkpointing \
    --val_batch_size 1 \
    --output_dir "$parent_dir/checkpoints/stage2/"

echo ""
echo "Training complete!"
