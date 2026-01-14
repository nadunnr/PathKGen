#!/bin/bash
# Inference script for single slide using Stage 1 checkpoint

echo "============================================================"
echo "Running Inference on Single Slide"
echo "============================================================"
echo ""

# Configuration
CHECKPOINT="checkpoints/stage1/best_model.pt"
SLIDE_ID="TCGA-EW-A6SC-01Z-00-DX1.C2D50E7C-3AD0-4038-9A1C-CAE54A3FBE2F"
SLIDE_FEATURES="/home/nadun/wd/datasets/trident_processed/20x_512px_0px_overlap/slide_features_titan/${SLIDE_ID}.h5"

echo "Checkpoint: ${CHECKPOINT}"
echo "Slide ID: ${SLIDE_ID}"
echo "Features: ${SLIDE_FEATURES}"
echo ""

python inference.py \
    --checkpoint ${CHECKPOINT} \
    --slide_features ${SLIDE_FEATURES} \
    --slide_id ${SLIDE_ID} \
    --skip_slidechat_init \
    --max_new_tokens 512 \
    --temperature 0.7 \
    --top_p 0.9 \
    --repetition_penalty 1.2

echo ""
echo "Inference complete!"
