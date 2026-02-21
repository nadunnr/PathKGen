import torch
from transformers import AutoModel

# Load model
model = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)

# Debug config loading
print("=" * 60)
print("CONFIGURATION DEBUGGING")
print("=" * 60)

# Check what was in config.json for vision_config
import json
config_path = '/home/nadun/hdd/.cache/huggingfaceexport/hub/models--MahmoodLab--TITAN/snapshots/dac6773d9961cfc75503440676ff157a2c6e8d2e/config.json'
with open(config_path, 'r') as f:
    config_json = json.load(f)

print("\n1. From config.json:")
print(f"   vision_config.return_all_tokens: {config_json['vision_config'].get('return_all_tokens', 'NOT FOUND')}")

# Check loaded config object
print("\n2. From loaded config object:")
print(f"   config.vision_config.__dict__: ")
for key in sorted(model.config.vision_config.__dict__.keys()):
    if not key.startswith('_'):
        val = getattr(model.config.vision_config, key)
        if key == 'return_all_tokens':
            print(f"   >>> {key}: {val}")
        else:
            print(f"       {key}: {val}")

# Check the actual vision encoder instance
print("\n3. From vision_encoder instance:")
print(f"   vision_encoder.return_all_tokens: {model.vision_encoder.return_all_tokens}")

# Check if build_vision_tower is correctly passing the parameter
print("\n4. Checking build_vision_tower signature...")
from vision_transformer import build_vision_tower
import inspect
sig = inspect.signature(build_vision_tower)
print(f"   Signature: {sig}")

# Manually test build_vision_tower
print("\n5. Manually calling build_vision_tower...")
test_vision = build_vision_tower(model.config.vision_config)
print(f"   Manually built vision_encoder.return_all_tokens: {test_vision.return_all_tokens}")

print("\n" + "=" * 60)
