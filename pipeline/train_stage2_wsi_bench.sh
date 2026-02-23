#!/bin/bash
# Training script for WSI-Bench dataset
# Optimized configuration based on WSI-LLaVA analysis
# Enhanced with NLU score improvements

# Set PyTorch memory allocation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting Stage 2 Training with WSI-Bench Dataset (NLU Optimized)"
echo "=================================================================="

# WSI-Bench dataset paths
WSI_BENCH_TRAIN="/home/sandun/nadun/data/dataset/wsi_bench_Report_with_features/wsi_bench_Report_train.json"
WSI_BENCH_TEST="/home/sandun/nadun/data/dataset/wsi_bench_Report_with_features/wsi_bench_Report_test.json"

KNOWLEDGE_TRAIN_JSON="/home/nadun/wd/datasets/wsi_bench/dataset/WSI-Bench-train_Report_knowledge.json"
KNOWLEDGE_TEST_JSON="/home/nadun/wd/datasets/wsi_bench/dataset/WSI-Bench-test_Report_knowledge.json"

# Configuration
OUTPUT_DIR="/home/sandun/nadun/histo_vlm/checkpoints/stage2_wsi_bench"

# Configuration
STAGE1_CHECKPOINT="/home/sandun/nadun/vlm_histo_old/checkpoints/stage1_wsi_bench/best_model.pt"
# Retry logic: repeat if error occurs
python train_stage2.py \
    --use_bleu_penalty \
    --bleu_metric bleu4 \
    --train_json "$WSI_BENCH_TRAIN" \
    --test_json "$WSI_BENCH_TEST" \
    --dataset_format wsi-bench \
    --stage1_checkpoint "$STAGE1_CHECKPOINT" \
    --compute_metrics_during_training \
    --epochs 3 \
    --batch_size 1 \
    --gradient_accumulation_steps 32 \
    --num_workers 8 \
    --lr 1e-4 \
    --projection_lr 1e-5 \
    --weight_decay 0.01 \
    --early_stopping \
    --patience 1 \
    --lora_rank 128 \
    --lora_alpha 256 \
    --feature_key tokens \
    --gradient_checkpointing \
    --val_batch_size 1 \
    --output_dir "$OUTPUT_DIR" \
    \
    `# ============ NLU Score Improvements ============` \
    --label_smoothing 0.1 \
    --warmup_ratio 0.1 \
    --eval_num_beams 4 \
    --eval_length_penalty 1.0 \
    --no_repeat_ngram_size 3 \
    --use_entity_f1



if [ $? -eq 0 ]; then
    echo ""
    echo "Training complete! Checkpoints saved to: $OUTPUT_DIR"
else
    echo ""
    echo "Error occurred during training."
fi
