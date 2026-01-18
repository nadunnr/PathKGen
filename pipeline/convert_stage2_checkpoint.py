#!/usr/bin/env python
"""
Convert Stage 2 QLoRA checkpoint to inference-ready checkpoint.

This script:
1. Loads a Stage 2 QLoRA checkpoint
2. Merges LoRA weights into the base model
3. Saves as a regular full-precision checkpoint
4. Result can be used for inference without QLoRA complications

Usage:
    python convert_stage2_checkpoint.py \
        --input checkpoints/stage2/best_model.pt \
        --output checkpoints/stage2/best_model_merged.pt
"""

import argparse
import torch
from model import VisionLanguageModel


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Stage 2 QLoRA checkpoint for inference")
    
    parser.add_argument("--input", type=str, required=True,
                       help="Path to Stage 2 QLoRA checkpoint")
    parser.add_argument("--output", type=str, required=True,
                       help="Path to save merged checkpoint")
    parser.add_argument("--llm_path", type=str,
                       default="/home/nadun/wd/vlm_histo/checkpoints/Qwen2.5-7B-Instruct",
                       help="Path to base LLM")
    parser.add_argument("--visual_dim", type=int, default=768,
                       help="Dimension of slide features")
    parser.add_argument("--num_visual_tokens", type=int, default=32,
                       help="Number of visual tokens")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\n" + "="*60)
    print("Converting Stage 2 QLoRA Checkpoint")
    print("="*60)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print()
    
    # Step 1: Load checkpoint metadata to verify it's QLoRA
    print("Step 1: Checking checkpoint type...")
    checkpoint_cpu = torch.load(args.input, map_location='cpu')
    
    if 'lora_state_dict' not in checkpoint_cpu:
        print("ERROR: This doesn't appear to be a Stage 2 QLoRA checkpoint!")
        print("Stage 2 checkpoints should contain 'lora_state_dict'")
        return
    
    print("✓ Confirmed Stage 2 QLoRA checkpoint")
    del checkpoint_cpu
    torch.cuda.empty_cache()
    
    # Step 2: Initialize model with QLoRA
    print("\nStep 2: Loading model with QLoRA...")
    model = VisionLanguageModel(
        llm_path=args.llm_path,
        slidechat_checkpoint=None,  # Skip SlideChat (weights in checkpoint)
        visual_dim=args.visual_dim,
        num_visual_tokens=args.num_visual_tokens,
        use_qlora=True,
        freeze_llm=False,
        device=device
    )
    
    # Step 3: Load checkpoint
    print("\nStep 3: Loading checkpoint weights...")
    model.load_checkpoint(args.input, load_optimizer=False)
    
    # Step 4: Merge LoRA weights
    print("\nStep 4: Merging LoRA weights into base model...")
    print("This may take a few minutes...")
    model.merge_lora_weights()
    
    # Step 5: Save merged checkpoint
    print("\nStep 5: Saving merged checkpoint...")
    model.save_checkpoint(args.output, epoch=0, merge_lora=False)  # Already merged
    
    print("\n" + "="*60)
    print("✓ Conversion Complete!")
    print("="*60)
    print(f"\nMerged checkpoint saved to: {args.output}")
    print("\nYou can now use this checkpoint for inference:")
    print(f"  python inference.py --checkpoint {args.output} ...")
    print("\nNo QLoRA complications, clean and simple!")


if __name__ == "__main__":
    main()
