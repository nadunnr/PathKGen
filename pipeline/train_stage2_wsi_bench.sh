#!/bin/bash
# Training script for WSI-Bench dataset
# Optimized configuration based on WSI-LLaVA analysis

# Set PyTorch memory allocation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting Stage 2 Training with WSI-Bench Dataset (Optimized)"
echo "=============================================================="

# WSI-Bench dataset paths
TRAIN_JSON="/home/nadun/wd/datasets/wsi_bench/dataset/wsi_bench_Report_with_features/wsi_bench_Report_train.json"
TEST_JSON="/home/nadun/wd/datasets/wsi_bench/dataset/wsi_bench_Report_with_features/wsi_bench_Report_test.json"

KNOWLEDGE_TRAIN_JSON="/home/nadun/wd/datasets/wsi_bench/dataset/wsi_bench_Report_knowledge/WSI-Bench-train_Report_knowledge.json"
KNOWLEDGE_TEST_JSON="/home/nadun/wd/datasets/wsi_bench/dataset/wsi_bench_Report_knowledge/WSI-Bench-test_Report_knowledge.json"

DEFAULT_LLM_PATH="/home/nadun/wd/vlm_histo/checkpoints/Qwen2.5-7B-Instruct"
UMLS_LLM_PATH="/home/nadun/wd/vlm_histo/checkpoints/Qwen-UMLS-7B-Instruct"

# Configuration
OUTPUT_DIR="/home/nadun/wd/vlm_histo/checkpoints/stage2_wsi_bench_knowledge"
OUTPUT_RESULTS_DIR="/home/nadun/wd/vlm_histo/results/stage2_wsi_bench_knowledge"

# Configuration
STAGE1_CHECKPOINT="/home/nadun/wd/vlm_histo/checkpoints/stage1_wsi_bench_knowledge/2026-02-20_04-09_vt128_ml768_e10_bs4/best_model.pt"

python train_stage2.py \
    --train_json "$KNOWLEDGE_TRAIN_JSON" \
    --test_json "$KNOWLEDGE_TEST_JSON" \
    --llm_path "$UMLS_LLM_PATH" \
    --dataset_format wsi-bench \
    --stage1_checkpoint "$STAGE1_CHECKPOINT" \
    --output_results_dir "$OUTPUT_RESULTS_DIR" \
    --compute_metrics_during_training \
    --epochs 2\
    --batch_size 4 \
    --gradient_accumulation_steps 32 \
    --num_workers 8 \
    --lr 1e-4 \
    --projection_lr 1e-5 \
    --weight_decay 0.0 \
    --early_stopping \
    --patience 1 \
    --lora_rank 128 \
    --lora_alpha 256 \
    --feature_key tokens \
    --gradient_checkpointing \
    --val_batch_size 4 \
    --output_dir "$OUTPUT_DIR"\
    --use_knowledge_guidance

echo ""
echo "Training complete! Checkpoints saved to: $OUTPUT_DIR"
