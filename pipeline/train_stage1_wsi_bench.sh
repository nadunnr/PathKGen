#!/bin/bash
# Stage 1 Training script for WSI-Bench dataset
# Trains only the projection layer with optimized configuration

# Set PyTorch memory allocation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting Stage 1 Training with WSI-Bench Dataset"
echo "================================================="

# WSI-Bench dataset paths
WSI_BENCH_TRAIN="/home/sandun/nadun/data/dataset/wsi_bench_Report_with_features/wsi_bench_Report_train.json"
WSI_BENCH_TEST="/home/sandun/nadun/data/dataset/wsi_bench_Report_with_features/wsi_bench_Report_test.json"

KNOWLEDGE_TRAIN_JSON="/home/sandun/nadun/data/dataset/wsi_bench_Report_knowledge/WSI-Bench-train_Report_knowledge.json"
KNOWLEDGE_TEST_JSON="/home/sandun/nadun/data/dataset/wsi_bench_Report_knowledge/WSI-Bench-test_Report_knowledge.json"

OUTPUT_DIR="/home/sandun/nadun/vlm_histo/checkpoints/stage1_wsi_bench_knowledge"

python train_stage1.py \
    --train_json "$KNOWLEDGE_TRAIN_JSON" \
    --test_json "$KNOWLEDGE_TEST_JSON" \
    --dataset_format wsi-bench \
    --epochs 10 \
    --batch_size 1 \
    --gradient_accumulation_steps 64 \
    --lr 5e-4 \
    --early_stopping \
    --patience 3 \
    --weight_decay 0.01 \
    --feature_key tokens \
    --gradient_checkpointing \
    --val_batch_size 1 \
    --output_dir "$OUTPUT_DIR" \

echo ""
echo "Stage 1 training complete! Checkpoints saved to: $OUTPUT_DIR"
