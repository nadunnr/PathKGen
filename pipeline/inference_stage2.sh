#!/bin/bash
# Inference using Stage 2 checkpoint (if Stage 2 training completed)

echo "============================================================"
echo "Running Inference with Stage 2 (QLoRA) Model"
echo "============================================================"
echo ""

# Configuration
CHECKPOINT="checkpoints/stage2/best_model.pt"
DATA_JSON="/home/nadun/wd/vlm_histo/data/curated_breast_dataset.json"
OUTPUT_DIR="./results/stage2_inference"

# Check if Stage 2 checkpoint exists
if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: Stage 2 checkpoint not found at ${CHECKPOINT}"
    echo "Please complete Stage 2 training first, or use inference_batch.sh for Stage 1"
    exit 1
fi

echo "Checkpoint: ${CHECKPOINT}"
echo "Dataset: ${DATA_JSON}"
echo "Output: ${OUTPUT_DIR}"
echo ""

python inference.py \
    --checkpoint ${CHECKPOINT} \
    --data_json ${DATA_JSON} \
    --split test \
    --output_dir ${OUTPUT_DIR} \
    --skip_slidechat_init \
    --save_json \
    --max_new_tokens 512 \
    --temperature 0.7 \
    --top_p 0.9 \
    --repetition_penalty 1.2

echo ""
echo "============================================================"
echo "Stage 2 inference complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "============================================================"
