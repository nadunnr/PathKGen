#!/bin/bash
# Convert Stage 2 QLoRA checkpoint to inference-ready format

echo "============================================================"
echo "Converting Stage 2 QLoRA Checkpoint"
echo "============================================================"
echo ""

INPUT="checkpoints/stage2/best_model.pt"
OUTPUT="checkpoints/stage2/best_model_merged.pt"

if [ ! -f "${INPUT}" ]; then
    echo "ERROR: Stage 2 checkpoint not found at ${INPUT}"
    echo "Please train Stage 2 first or specify correct path"
    exit 1
fi

echo "Input checkpoint:  ${INPUT}"
echo "Output checkpoint: ${OUTPUT}"
echo ""
echo "This will:"
echo "  1. Load your Stage 2 QLoRA checkpoint"
echo "  2. Merge LoRA weights into base model"
echo "  3. Save as regular checkpoint (no QLoRA)"
echo "  4. Result works perfectly for inference!"
echo ""
echo "Starting conversion..."
echo ""

python convert_stage2_checkpoint.py \
    --input ${INPUT} \
    --output ${OUTPUT}

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✓ Conversion Successful!"
    echo "============================================================"
    echo ""
    echo "Merged checkpoint: ${OUTPUT}"
    echo ""
    echo "Run inference with:"
    echo "  python inference.py --checkpoint ${OUTPUT} ..."
    echo ""
else
    echo ""
    echo "ERROR: Conversion failed!"
    echo "Check the error messages above"
fi