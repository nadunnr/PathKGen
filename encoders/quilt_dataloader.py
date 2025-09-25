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