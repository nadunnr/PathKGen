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

KNOWLEDGE_TRAIN_JSON="/home/nadun/wd/datasets/wsi_bench/dataset/WSI-Bench-train_Report_knowledge.json"
KNOWLEDGE_TEST_JSON="/home/nadun/wd/datasets/wsi_bench/dataset/WSI-Bench-test_Report_knowledge.json"

# Configuration
OUTPUT_DIR="/home/nadun/wd/vlm_histo/checkpoints/stage1_wsi_bench_knowledge"

python train_stage1.py \
    --train_json "$KNOWLEDGE_TRAIN_JSON" \
    --test_json "$KNOWLEDGE_TEST_JSON" \
    --dataset_format wsi-bench \
    --epochs 10 \
    --batch_size 4 \
    --gradient_accumulation_steps 16 \
    --lr 5e-4 \
    --early_stopping \
    --patience 3 \
    --weight_decay 0.01 \
    --feature_key tokens \
    --gradient_checkpointing \
    --val_batch_size 4 \
    --output_dir "$OUTPUT_DIR" \
    --use_knowledge_guidance

echo ""
echo "Stage 1 training complete! Checkpoints saved to: $OUTPUT_DIR"
