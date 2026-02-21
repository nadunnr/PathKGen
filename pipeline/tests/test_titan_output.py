import torch
import sys

# Test if TITAN is returning tokens
print("Testing TITAN output format...")

# Load TITAN model
from transformers import AutoModel
model = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
model.eval()

# Create dummy inputs
batch_size = 1
num_patches = 100
patch_dim = 768

patch_features = torch.randn(batch_size, num_patches, patch_dim)
patch_coords = torch.randint(0, 10000, (batch_size, num_patches, 2))
patch_size_lv0 = 512

print(f"\nInput shapes:")
print(f"  patch_features: {patch_features.shape}")
print(f"  patch_coords: {patch_coords.shape}")
print(f"  patch_size_lv0: {patch_size_lv0}")

# Check vision encoder settings
print(f"\nVision encoder settings:")
print(f"  return_all_tokens: {model.vision_encoder.return_all_tokens}")
print(f"  has attn_pool: {model.vision_encoder.attn_pool is not None}")

# Test encode_slide_from_patch_features
with torch.no_grad():
    output = model.encode_slide_from_patch_features(patch_features, patch_coords, patch_size_lv0)

print(f"\nOutput from encode_slide_from_patch_features:")
if isinstance(output, tuple):
    print(f"  ✓ Returns TUPLE with {len(output)} elements")
    print(f"  - Element 0 (pooled) shape: {output[0].shape}")
    print(f"  - Element 1 (tokens) shape: {output[1].shape}")
else:
    print(f"  ✗ Returns SINGLE TENSOR")
    print(f"  - Shape: {output.shape}")
    print(f"  - Dimensions: {output.ndim}D")

print("\n" + "="*60)
if isinstance(output, tuple):
    print("SUCCESS: Model returns both pooled and tokens!")
else:
    print("FAILED: Model only returns single output")
    print("\nPossible issues:")
    print("  1. Python cached old module - try restarting Python")
    print("  2. Changes to modeling_titan.py didn't save")
    print("  3. return_all_tokens is False")
