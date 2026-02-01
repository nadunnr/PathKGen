#!/usr/bin/env python3
"""
Inspect SlideChat checkpoint structure to see what weights it contains.
"""
import torch

print("=" * 80)
print("SlideChat Checkpoint Inspector")
print("=" * 80)

checkpoint_path = "/home/nadun/wd/vlm_histo/checkpoints/SlideChat_Weight/stage2_pth/mp_rank_00_model_states.pt"

print(f"\nLoading checkpoint from: {checkpoint_path}")
print("(This may take a moment...)\n")

ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

print(f"Total keys in checkpoint: {len(ckpt)}")
print("\n" + "=" * 80)
print("ALL KEYS IN CHECKPOINT:")
print("=" * 80)

for i, key in enumerate(ckpt.keys(), 1):
    # Get shape if it's a tensor
    if hasattr(ckpt[key], 'shape'):
        shape_str = f" {list(ckpt[key].shape)}"
    else:
        shape_str = f" (type: {type(ckpt[key]).__name__})"
    print(f"{i:3d}. {key}{shape_str}")

print("\n" + "=" * 80)
print("KEY ANALYSIS:")
print("=" * 80)

# Categorize keys
llm_keys = []
vision_keys = []
other_keys = []

for key in ckpt.keys():
    if key.startswith("model."):
        clean_key = key.replace("model.", "")
        if "mm_projector" in clean_key or "slide_encoder" in clean_key:
            vision_keys.append(key)
        else:
            llm_keys.append(key)
    else:
        other_keys.append(key)

print(f"\nLLM weights (model.* excluding vision): {len(llm_keys)}")
if llm_keys:
    print("  Examples:")
    for key in llm_keys[:5]:
        print(f"    - {key}")

print(f"\nVision weights (mm_projector/slide_encoder): {len(vision_keys)}")
if vision_keys:
    print("  Examples:")
    for key in vision_keys[:5]:
        print(f"    - {key}")

print(f"\nOther keys: {len(other_keys)}")
if other_keys:
    print("  All:")
    for key in other_keys:
        print(f"    - {key}")

print("\n" + "=" * 80)
print("CONCLUSION:")
print("=" * 80)

if len(llm_keys) == 0:
    print("❌ NO LLM weights found in this checkpoint!")
    print("   This checkpoint only contains vision components.")
    print("   SlideChat Stage 2 might store LLM as LoRA adapters elsewhere,")
    print("   or this is just the projection layer checkpoint.")
else:
    print(f"✓ Found {len(llm_keys)} LLM weight tensors")
    print("  These can be loaded into the base model.")
