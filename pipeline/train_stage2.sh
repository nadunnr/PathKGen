#!/bin/bash
# Memory-optimized training script for Stage 2 with QLoRA

# Set PyTorch memory allocation to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting Stage 2 Training"
echo "===================================================="

parent_dir="wd/vlm_histo/checkpoints/stage2/2026-02-01_13-09_vt64_ml512_e5_bs2 wd/vlm_histo/pipeline"

python train_stage2.py \
    --stage1_checkpoint /home/nadun/wd/vlm_histo/pipeline/checkpoints/stage1/2026-01-31_08-18_vt64_ml512_e20_bs2/best_model.pt \
    --epochs 5\
    --train_split train,val \
    --val_split test \
    --batch_size 2 \
    --gradient_accumulation_steps 16 \
    --lr 1e-5 \
    --projection_lr 5e-5 \
    --weight_decay 0.01 \
    --gradient_checkpointing \
    # --use_knowledge_guidance \
    --val_batch_size 1 \
    --output_dir "$parent_dir/checkpoints/stage2/"

echo ""
echo "Training complete!"
