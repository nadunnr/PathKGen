import torch
from transformers import AutoModel

#Load model
model = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)

# FORCE set return_all_tokens to True
model.vision_encoder.return_all_tokens = True

# Test it
patch_features = torch.randn(1, 100, 768)
patch_coords = torch.randint(0, 10000, (1, 100, 2))
output = model.encode_slide_from_patch_features(patch_features, patch_coords, 512)

if isinstance(output, tuple):
    print("✓ SUCCESS: Returns tuple!")
    print(f"  Pooled: {output[0].shape}")
    print(f"  Tokens: {output[1].shape}")
else:
    print("✗ Still returns single tensor")