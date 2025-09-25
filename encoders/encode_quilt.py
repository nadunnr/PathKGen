import pandas as pd
from pathlib import Path
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize

model_cfg = 'conch_ViT-B-16'
#4checkpoint_path = '/checkpoints/CONCH/pytorch_model.bin'
checkpoint_path = "/home/nadun/hdd/literature-guided-digital-histopathology/checkpoints/conch/pytorch_model.bin"

model, preprocess = create_model_from_pretrained(model_cfg, checkpoint_path)
_ = model.eval()

def conch_encode_img(image_path, image_only, model_cfg='conch_ViT-B-16'):
    
    image = Image.open(image_path)
    image = preprocess(image).unsqueeze(0)

    with torch.inference_mode():
        if image_only:
            #For image-only tasks, it is common to directly use the representation before the projection head and l2-normalization.
            # This is done by setting **proj_contrast=False** and **normalize=False**.
            image_embs = model.encode_image(image, proj_contrast=False, normalize=False)
        else:
            image_embs = model.encode_image(image)

    print(f"image embeddings: {image_embs.shape}")

    return image_embs
 

def conch_encode_text(texts, model_cfg='conch_ViT-B-16'):

    tokenizer = get_tokenizer() # load tokenizer
    text_tokens = tokenize(texts=texts, tokenizer=tokenizer) # tokenize the text
    text_embs = model.encode_text(text_tokens)

    print(f"text embeddings: text_embs.shape", text_embs.shape)

    return text_embs

class HistopathTextImageDataset(Dataset):
    def __init__(self, csv_path, images_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        # self.image_encoder = image_encoder
        # self.text_encoder = text_encoder
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_rel_path = row["image_path"]
        caption = row["caption"]
        img_full_path = self.images_dir / img_rel_path
        img_embedding = conch_encode_img(img_full_path, image_only=False).detach()
        text_embeds = conch_encode_text([caption]).detach()

        return img_embedding, text_embeds

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Hardcode your paths for now
datapath = Path("/home/nadun/hdd/QUILT1M")
csv_path = datapath / "quilt_1M_lookup.csv"
img_path = datapath / "quilt_1m"

dataset = HistopathTextImageDataset(csv_path=csv_path, images_dir=img_path)
# dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

# def safe_collate(batch):
#     imgs, captions = zip(*batch)
#     imgs = torch.stack(imgs, dim=0)  # default, no out= usage
#     return imgs, list(captions)

def safe_collate(batch):
    imgs, texts = zip(*batch)
    # imgs is tuple of tensors (each [1,512] or [C, H, W] etc.)
    imgs = torch.stack(imgs, dim=0)  # simple stacking
    texts = torch.stack(texts, dim=0)  # if they are tensors of same shape
    return imgs, texts

dataloader = DataLoader(dataset, batch_size=16, shuffle=False,
                        num_workers=0, collate_fn=safe_collate)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

count = 0
n = 10

for imgs, captions in dataloader:

    if count == n: break
    print("Iteration", count)

    imgs = imgs.to(device)
    # captions is a list of strings
    print("Batch imgs:", imgs.shape)
    print("Captions:", captions[:3])
    print("Caption length", captions.shape)
    # Here you would tokenize captions and pass them through the text encoder
    # tokenized = tokenizer(captions, ...)
    # text_embeds = text_encoder(**tokenized)
    # img_embeds = image_encoder(imgs)

    count += 1
