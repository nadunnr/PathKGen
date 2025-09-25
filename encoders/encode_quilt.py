import pandas as pd
from pathlib import Path
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

class HistopathTextImageDataset(Dataset):
    def __init__(self, csv_path, images_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_rel_path = row["image_path"]
        caption = row["caption"]
        img_full_path = self.images_dir / img_rel_path
        img = Image.open(img_full_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, caption

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Hardcode your paths for now
datapath = Path("/home/nadun/hdd/QUILT1M")
csv_path = datapath / "quilt_1M_lookup.csv"
img_path = datapath / "quilt_1m"

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

dataset = HistopathTextImageDataset(csv_path=csv_path, images_dir=img_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

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
    print("Caption length", len(captions))
    # Here you would tokenize captions and pass them through the text encoder
    # tokenized = tokenizer(captions, ...)
    # text_embeds = text_encoder(**tokenized)
    # img_embeds = image_encoder(imgs)

    count += 1
