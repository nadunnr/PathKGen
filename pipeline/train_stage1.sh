#!/bin/bash
# Memory-optimized training script for Stage 1

epochs=20
batch_size=2
gradient_accumulation_steps=8
max_length=512
train_split=train,val
val_split=test
gradient_checkpointing=True
lr=1e-4
output_dir=checkpoints/stage1
save_every=2

parent_dir="wd/vlm_histo/checkpoints/stage2/2026-02-01_13-09_vt64_ml512_e5_bs2 wd/vlm_histo/pipeline"

echo "Starting Stage 1 Training with Memory Optimizations"
echo "===================================================="
echo ""
echo "Optimizations enabled:"
echo "  - Gradient checkpointing: $gradient_checkpointing"
echo "  - Batch size: $batch_size"
echo "  - Gradient accumulation: $gradient_accumulation_steps steps (effective batch size = $batch_size * $gradient_accumulation_steps)"
echo "  - Max length: $max_length"
echo "  - Train split: $train_split"
echo "  - Val split: $val_split"
echo "  - Learning rate: $lr"
echo "  - Output directory: $output_dir"
echo "  - Save every: $save_every epochs"
echo ""

python train_stage1.py \
    --epochs $epochs \
    --batch_size $batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps\
    --max_length $max_length \
    --train_split $train_split \
    --val_split $val_split \
    --gradient_checkpointing \
    --lr $lr \
    --output_dir "$parent_dir/$output_dir" \
    --save_every $save_every

echo ""
echo "Training complete!"
