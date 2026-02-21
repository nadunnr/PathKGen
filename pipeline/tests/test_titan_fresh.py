import torch
import sys
import importlib

# Force reload of all TITAN modules
model_path = '/home/nadun/hdd/.cache/huggingfaceexport/hub/models--MahmoodLab--TITAN/snapshots/dac6773d9961cfc75503440676ff157a2c6e8d2e'

# Remove from sys.modules if already loaded
modules_to_reload = []
for module_name in list(sys.modules.keys()):
    if 'titan' in module_name.lower() or 'vision_transformer' in module_name:
        modules_to_reload.append(module_name)

print(f"Removing {len(modules_to_reload)} cached modules...")
for module_name in modules_to_reload:
    del sys.modules[module_name]

# Add model path to sys.path if not already there
if model_path not in sys.path:
    sys.path.insert(0, model_path)

print("Loading fresh TITAN model...")

# Load TITAN model
from transformers import AutoModel
model = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
model.eval()

print(f"DEBUG: Vision Encoder Module: {model.vision_encoder.__module__}")
import sys
module_name = model.vision_encoder.__module__
if module_name in sys.modules:
    print(f"DEBUG: Module File: {sys.modules[module_name].__file__}")
else:
    # It might be in a parent package
    base_module = module_name.split('.')[0]
    if base_module in sys.modules:
        print(f"DEBUG: Base Module File: {sys.modules[base_module].__file__}")

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

# Check config
print(f"\nConfig settings:")
print(f"  config.vision_config.return_all_tokens: {model.config.vision_config.return_all_tokens}")

# Test encode_slide_from_patch_features
with torch.no_grad():
    output = model.encode_slide_from_patch_features(patch_features, patch_coords, patch_size_lv0)

print(f"\nOutput from encode_slide_from_patch_features:")
if isinstance(output, tuple):
    print(f"  ✓ Returns TUPLE with {len(output)} elements")
    print(f"  - Element 0 (pooled) shape: {output[0].shape}")
    print(f"  - Element 1 (tokens) shape: {output[1].shape}")
    print(f"  - Number of tokens: {output[1].shape[1]}")
    print(f"  - Token dimension: {output[1].shape[2]}")
else:
    print(f"  ✗ Returns SINGLE TENSOR")
    print(f"  - Shape: {output.shape}")
    print(f"  - Dimensions: {output.ndim}D")

print("\n" + "="*60)
if isinstance(output, tuple):
    print("✓ SUCCESS: Model returns both pooled and tokens!")
else:
    print("✗ FAILED: Model only returns single output")
