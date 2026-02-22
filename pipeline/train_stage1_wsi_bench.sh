#!/bin/bash
# Stage 1 Training script for WSI-Bench dataset
# Trains only the projection layer with optimized configuration

# Set PyTorch memory allocation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting Stage 1 Training with WSI-Bench Dataset"
echo "================================================="

# WSI-Bench dataset paths
TRAIN_JSON="/home/nadun/wd/datasets/wsi_bench/dataset/wsi_bench_Report_with_features/wsi_bench_Report_train.json"
TEST_JSON="/home/nadun/wd/datasets/wsi_bench/dataset/wsi_bench_Report_with_features/wsi_bench_Report_test.json"

KNOWLEDGE_TRAIN_JSON="/home/nadun/wd/datasets/wsi_bench/dataset/wsi_bench_Report_knowledge/WSI-Bench-train_Report_knowledge.json"
KNOWLEDGE_TEST_JSON="/home/nadun/wd/datasets/wsi_bench/dataset/wsi_bench_Report_knowledge/WSI-Bench-test_Report_knowledge.json"

DEFAULT_LLM_PATH="/home/nadun/wd/vlm_histo/checkpoints/Qwen2.5-7B-Instruct"
UMLS_LLM_PATH="/home/nadun/wd/vlm_histo/checkpoints/Qwen-UMLS-7B-Instruct"

# Configuration
OUTPUT_DIR="/home/nadun/wd/vlm_histo/checkpoints/stage1_wsi_bench_knowledge"

python train_stage1.py \
    --train_json "$KNOWLEDGE_TRAIN_JSON" \
    --test_json "$KNOWLEDGE_TEST_JSON" \
    --llm_path "$UMLS_LLM_PATH"\
    --dataset_format wsi-bench \
    --epochs 10 \
    --batch_size 4 \
    --gradient_accumulation_steps 32 \
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
