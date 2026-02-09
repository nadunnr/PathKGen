#!/bin/bash
# Training script for WSI-Bench dataset
# Optimized configuration based on WSI-LLaVA analysis

# Set PyTorch memory allocation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting Stage 2 Training with WSI-Bench Dataset (Optimized)"
echo "=============================================================="

# WSI-Bench dataset paths
TRAIN_JSON="/home/nadun/wd/datasets/wsi_bench/dataset/WSI-Bench-train-temp_Report_split0.1_train.json"
TEST_JSON="/home/nadun/wd/datasets/wsi_bench/dataset/WSI-Bench-train-temp_Report_split0.1_test.json"

# Configuration
STAGE1_CHECKPOINT="/home/nadun/wd/vlm_histo/checkpoints/stage1_wsi_bench/2026-02-08_23-34_vt128_ml512_e20_bs8/best_model.pt"
OUTPUT_DIR="/home/nadun/wd/vlm_histo/checkpoints/stage2_wsi_bench"

python train_stage2.py \
    --train_json "$TRAIN_JSON" \
    --test_json "$TEST_JSON" \
    --dataset_format wsi-bench \
    --stage1_checkpoint "$STAGE1_CHECKPOINT" \
    --epochs 10\
    --batch_size 8 \
    --gradient_accumulation_steps 8 \
    --lr 2e-4 \
    --projection_lr 2e-5 \
    --weight_decay 0.0 \
    --early_stopping \
    --patience 2 \
    --lora_rank 128 \
    --lora_alpha 256 \
    --feature_key tokens \
    --gradient_checkpointing \
    --val_batch_size 1 \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "Training complete! Checkpoints saved to: $OUTPUT_DIR"
