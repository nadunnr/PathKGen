#!/bin/bash
# Batch inference on test split using trained checkpoint

echo "============================================================"
echo "Running Batch Inference on Test Split"
echo "============================================================"
echo ""

# Configuration
CHECKPOINT="checkpoints/stage1/best_model.pt"
DATA_JSON="/home/nadun/wd/vlm_histo/data/curated_breast_dataset.json"
OUTPUT_DIR="results/stage1_inference"

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
echo "Batch inference complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "  - Generated reports: ${OUTPUT_DIR}/generated_reports.json"
echo "  - Individual files: ${OUTPUT_DIR}/*.txt"
echo "============================================================"
