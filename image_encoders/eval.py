import torch
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
image_path = proj_root / "data/roi1.jpg"

assert checkpoint_path.is_file(), f"Missing checkpoint: {checkpoint_path}"
assert image_path.is_file(), f"Missing image: {image_path}"

# Load model
model_cfg = "conch_ViT-B-16"
model, preprocess = create_model_from_pretrained(str(model_cfg), str(checkpoint_path))
model = model.to(device).eval()

# Encode image 
image = Image.open(str(image_path)).convert("RGB")
image = preprocess(image).unsqueeze(0).to(device)
print("Image shape:", image.shape)

with torch.inference_mode():
    image_embeddings = model.encode_image(image)

print("Image embedding shape:", image_embeddings.shape)

# Tokenize and encode text
tokenizer = get_tokenizer()
classes = ["invasive ductal carcinoma", "invasive lobular carcinoma"]
prompts = [
    "an H&E image is characterized by malignant epithelial cells infiltrating the surrounding stroma, often arranged in irregular nests, cords, or sheets rather than normal ductal structures. The tumor cells typically show nuclear pleomorphism, hyperchromasia, and prominent nucleoli, with a high nuclear-to-cytoplasmic ratio. A dense desmoplastic stromal response is commonly seen around the invasive clusters, along with increased mitotic figures. The absence of a myoepithelial cell layer and occasional lymphovascular invasion further distinguish  in situ lesions.",
    "an H&E image is characterized by small, uniform, discohesive tumor cells infiltrating the stroma in linear “single-file” patterns or as targetoid arrangements around normal ducts. Gland or tubule formation is typically absent. The cells often have round to oval nuclei with mild pleomorphism and scant cytoplasm, and loss of E-cadherin expression underlies their reduced cohesion. A fibrous stromal reaction may be present, but necrosis and calcifications are uncommon .",
]

tokenized_prompts = tokenize(texts=prompts, tokenizer=tokenizer).to(device)
print("Tokenized prompts shape:", tokenized_prompts.shape)

with torch.inference_mode():
    text_embeddings = model.encode_text(tokenized_prompts)

print("Text embedding shape: ", text_embeddings.shape)

# Compute similarity
with torch.inference_mode():
    logits = (image_embeddings @ text_embeddings.T) * model.logit_scale.exp()
    sim_scores = logits.softmax(dim=-1).detach().cpu().numpy()

pred = classes[sim_scores[0].argmax()]
print("Predicted class:", pred)
print("Normalized similarity scores:",
      [f"{cls}: {score:.3f}" for cls, score in zip(classes, sim_scores[0])])