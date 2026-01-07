import os
import json
import h5py
from tqdm import tqdm
from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize
import torch
from PIL import Image

model_cfg = 'conch_ViT-B-16'
# checkpoint_path = '/checkpoints/CONCH/pytorch_model.bin'
checkpoint_path = "/home/nadun/hdd/literature-guided-digital-histopathology/checkpoints/conch/pytorch_model.bin"

# === CONFIG ===
INPUT_JSONL = "./text_encoders/hest_responses_20.jsonl"        
OUTPUT_DIR = "./text_encoders/text_embeds_20"            
# ===============

def conch_encode_text(checkpoint_path, texts, device, model_cfg='conch_ViT-B-16'):

    #For image-only tasks, it is common to directly use the representation before the projection head and l2-normalization.
    # This is done by setting **proj_contrast=False** and **normalize=False**.
    model, preprocess = create_model_from_pretrained(model_cfg, checkpoint_path, device=device)

    # model, preprocess = create_model_from_pretrained(model_cfg, checkpoint_path, force_img_size=224, device='cuda:2')
    _ = model.eval()

    ### Embed texts
    #The **.encode_text()** method encodes a batch of texts into a batch of l2-normalized text embeddings used for computing the similarity scores such as between images and texts. 

    tokenizer = get_tokenizer() # load tokenizer
    text_tokens = tokenize(texts=texts, tokenizer=tokenizer).to(device) # tokenize the text
    text_embs = model.encode_text(text_tokens)

    #print(f"text embeddings: text_embs.shape", text_embs.shape)

    return text_embs

def preprocess_text(text):
    # This creates a 1 dim for each sentence
    if isinstance(text, list) or isinstance(text, tuple):
        return text
    elif isinstance(text, str):
        return [text]
    
def main():
    torch.cuda.empty_cache()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    #device = "cpu"

    with open(INPUT_JSONL, "r") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Processing responses"):
        entry = json.loads(line.strip())
        id = entry["id"]
        text = preprocess_text(entry["response"])

        embedding = conch_encode_text(checkpoint_path, text, device)

        # Save as id.h5
        out_path = os.path.join(OUTPUT_DIR, f"{id}.h5")
        with h5py.File(out_path, "w") as hf:
            hf.create_dataset("text_embedding", data=embedding.detach().cpu().numpy())

    print(f"\nDone! Embeddings saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
