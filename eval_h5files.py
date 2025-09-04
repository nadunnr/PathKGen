import torch
import h5py
from PIL import Image
import os
from pathlib import Path
import warnings
from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize

# Miscellaneous settings
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.models.layers")
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Make paths absolute to avoid CWD confusion
proj_root = Path("/mnt/usbhdd1/literature-guided-digital-histopathology")  # Project path
checkpoint_path = proj_root / "checkpoints/conch/pytorch_model.bin"
image_path = proj_root / "data/MEND63.h5"

assert checkpoint_path.is_file(), f"Missing checkpoint: {checkpoint_path}"
assert image_path.is_file(), f"Missing image: {image_path}"

# Load model
model_cfg = "conch_ViT-B-16"
model, preprocess = create_model_from_pretrained(str(model_cfg), str(checkpoint_path))
# model, preprocess = create_model_from_pretrained(str(model_cfg), str(checkpoint_path), force_img_size=224, device='cuda')
model = model.to(device).eval()

# Load image
with h5py.File(image_path, "r") as f:
    imgs_np = f["img"][:]   # shape (1422, 224, 224, 3), dtype uint8
print("Loaded WSI patches:", imgs_np.shape, imgs_np.dtype)

# # Pre-process and batch all patches
# pil_list = [Image.fromarray(p).convert("RGB") for p in imgs_np]
# batch_tensors = torch.stack([preprocess(p) for p in pil_list], dim=0).to(device)
# print("Batch tensor shape:", batch_tensors.shape)  # (1422, 3, 448, 448)

# with torch.inference_mode():
#     image_embeddings = model.encode_image(batch_tensors)
print("Encoding patches one-by-one")
embeds = []  # will hold (1, D) tensors on CPU

model.eval()
with torch.inference_mode():
    for patch_np in imgs_np:
        pil_img = Image.fromarray(patch_np).convert("RGB")
        x = preprocess(pil_img).unsqueeze(0).to(device)   # (1, 3, 448, 448)
        e = model.encode_image(x).cpu()                   # (1, D) -> move to CPU
        embeds.append(e)

# Stack to (N, D)
image_embeddings = torch.cat(embeds, dim=0)
print("Embeddings:", image_embeddings.shape, image_embeddings.dtype)

# Tokenize and encode text
tokenizer = get_tokenizer()
classes = ["cancer", "healthy"]
prompts = [
    "The image is showing the brain tissue from the brain organ, preserved with fresh frozen. The image shows a pediatric brain tumor.",
    "The image shows healthy cerebellum tissue preserved with fresh frozen in the Brain organ of Homo sapiens.",
]

tokenized_prompts = tokenize(texts=prompts, tokenizer=tokenizer).to(device)
print("Tokenized prompts shape:", tokenized_prompts.shape)

with torch.inference_mode():
    text_embeddings = model.encode_text(tokenized_prompts)

print("Text embedding shape: ", text_embeddings.shape)

# Compute similarity
with torch.inference_mode():
    img_emb = image_embeddings.to(device)
    txt_emb = text_embeddings.to(device)

    # (N, D) @ (D, T) -> (N, T)
    logits = img_emb @ txt_emb.T * model.logit_scale.exp()

    # average across patches (dim=0)
    mean_logits = logits.mean(dim=0, keepdim=True)  # shape (1, T)

    # softmax over text prompts
    sim_scores = mean_logits.softmax(dim=-1).cpu().numpy()  # (1, T)

# Classification
pred = classes[sim_scores[0].argmax()]
print("Predicted class:", pred)
print("Final similarity scores:",
      [f"{cls}: {score:.3f}" for cls, score in zip(classes, sim_scores[0])])