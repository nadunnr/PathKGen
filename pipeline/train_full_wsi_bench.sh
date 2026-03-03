#!/bin/bash
# Full pipeline: Stage 1 (projection) → Stage 2 (QLoRA) on WSI-Bench + Knowledge
# Stage 1 best_model.pt is auto-detected and passed to Stage 2.

set -e  # Exit immediately on error

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Paths ──────────────────────────────────────────────────────────────────────
KNOWLEDGE_TRAIN_JSON="/home/nadun/wd/datasets/wsi_bench/dataset/wsi_bench_Report_knowledge/WSI-Bench-train_Report_knowledge.json"
KNOWLEDGE_TEST_JSON="/home/nadun/wd/datasets/wsi_bench/dataset/wsi_bench_Report_knowledge/WSI-Bench-test_Report_knowledge.json"

UMLS_LLM_PATH="/home/nadun/wd/vlm_histo/checkpoints/Qwen-UMLS-7B-Instruct"
DEFAULT_LLM_PATH="/home/nadun/wd/vlm_histo/checkpoints/Qwen2.5-7B-Instruct"
QWEN_15B_PATH="/home/nadun/wd/vlm_histo/checkpoints/Qwen2.5-1.5B-Instruct"

STAGE1_OUTPUT_DIR="/home/nadun/wd/vlm_histo/checkpoints/stage1_wsi_bench_knowledge"
STAGE2_OUTPUT_DIR="/home/nadun/wd/vlm_histo/checkpoints/stage2_wsi_bench_knowledge"

STAGE1_RESULTS_DIR="/home/nadun/wd/vlm_histo/results/stage1_wsi_bench_knowledge"
STAGE2_RESULTS_DIR="/home/nadun/wd/vlm_histo/results/stage2_wsi_bench_knowledge"

# ── Stage 1 Training ───────────────────────────────────────────────────────────
echo ""
echo "================================================="
echo "  STAGE 1: Projection Layer Training"
echo "================================================="

STAGE1_ARGS=(
    --train_json "$KNOWLEDGE_TRAIN_JSON"
    --test_json  "$KNOWLEDGE_TEST_JSON"
    --llm_path   "$QWEN_15B_PATH"
    --dataset_format wsi-bench
    --epochs 8
    --batch_size 8
    --gradient_accumulation_steps 16
    --lr 5e-4
    --early_stopping
    --patience 2
    --weight_decay 0.01
    --feature_key tokens
    --gradient_checkpointing
    --val_batch_size 8
    --output_dir "$STAGE1_OUTPUT_DIR"
)

python train_stage1.py "${STAGE1_ARGS[@]}"

# ── Auto-detect Stage 1 best_model.pt ─────────────────────────────────────────
# The training script creates a timestamped subdir; find the newest one.
STAGE1_EXP_DIR=$(ls -td "$STAGE1_OUTPUT_DIR"/*/  2>/dev/null | head -1)
STAGE1_CHECKPOINT="${STAGE1_EXP_DIR}best_model.pt"

if [ ! -f "$STAGE1_CHECKPOINT" ]; then
    echo "ERROR: Could not find Stage 1 best_model.pt at: $STAGE1_CHECKPOINT"
    exit 1
fi

echo ""
echo "Stage 1 complete. Best model: $STAGE1_CHECKPOINT"
echo "Running with Stage 1 checkpoint: $STAGE1_CHECKPOINT"

# ── Stage 2 Training ───────────────────────────────────────────────────────────
echo ""
echo "================================================="
echo "  STAGE 2: QLoRA Fine-tuning"
echo "================================================="

python train_stage2.py \
    --train_json "$KNOWLEDGE_TRAIN_JSON" \
    --test_json  "$KNOWLEDGE_TEST_JSON" \
    --llm_path   "$QWEN_15B_PATH" \
    --dataset_format wsi-bench \
    --stage1_checkpoint "$STAGE1_CHECKPOINT" \
    --output_results_dir "$STAGE2_RESULTS_DIR" \
    --compute_metrics_during_training \
    --label_smoothing 0.1 \
    --warmup_ratio 0.1 \
    --eval_num_beams 4 \
    --eval_length_penalty 1.0 \
    --no_repeat_ngram_size 3 \
    --epochs 1 \
    --batch_size 8 \
    --gradient_accumulation_steps 16 \
    --num_workers 8 \
    --lr 1e-4 \
    --projection_lr 1e-5 \
    --weight_decay 0.01 \
    --early_stopping \
    --patience 1 \
    --lora_rank 92 \
    --lora_alpha 196 \
    --feature_key tokens \
    --gradient_checkpointing \
    --val_batch_size 8 \
    --output_dir "$STAGE2_OUTPUT_DIR"

echo ""
echo "================================================="
echo "  Pipeline complete!"
echo "  Stage 1 checkpoints: $STAGE1_OUTPUT_DIR"
echo "  Stage 2 checkpoints: $STAGE2_OUTPUT_DIR"
echo "================================================="
