#!/bin/bash
# Stage 1 Training script for WSI-Bench dataset
# Trains only the projection layer with optimized configuration

# Set PyTorch memory allocation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting Stage 1 Training with WSI-Bench Dataset"
echo "================================================="

# WSI-Bench dataset paths
TRAIN_JSON="/home/nadun/wd/datasets/wsi_bench/dataset/WSI-Bench-train_Report.json"
TEST_JSON="/home/nadun/wd/datasets/wsi_bench/dataset/WSI-Bench-test_Report.json"

# Configuration
OUTPUT_DIR="/home/nadun/wd/vlm_histo/checkpoints/stage1_wsi_bench"

python train_stage1.py \
    --train_json "$TRAIN_JSON" \
    --test_json "$TEST_JSON" \
    --dataset_format wsi-bench \
    --epochs 20 \
    --batch_size 8 \
    --gradient_accumulation_steps 8 \
    --lr 5e-4 \
    --early_stopping \
    --patience 3 \
    --weight_decay 0.01 \
    --feature_key tokens \
    --gradient_checkpointing \
    --val_batch_size 4 \
    --output_dir "$OUTPUT_DIR"
q
echo ""
echo "Stage 1 training complete! Checkpoints saved to: $OUTPUT_DIR"
