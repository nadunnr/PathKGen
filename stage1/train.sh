#!/bin/bash
# Stage 1: WSI-Text Contrastive Alignment Training

cd "$(dirname "$0")"

python train.py \
    --batch_size 64 \
    --lr 1e-3 \
    --epochs 50 \
    --temperature 0.02 \
    --hidden_dim 3584 \
    --alignment_dim 3584 \
    --checkpoint_dir /home/nadun/wd/vlm_histo/checkpoints/stage1 \
    "$@"
