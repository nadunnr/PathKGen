# dataset.py
import os
from pathlib import Path
import json
import glob
import h5py
import torch
from torch.utils.data import Dataset, DataLoader

class SlideEmbeddingDataset(Dataset):
    def __init__(self, root_dir):
        self.files = sorted(glob.glob(os.path.join(root_dir, "*.h5")))
        self.metadata_dir = Path("/home/nadun/wd/HEST/hest_data_40/metadata")

        if not self.files:
            raise FileNotFoundError(f"No .h5 files found in {root_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        # Load slide embedding
        with h5py.File(file_path, "r") as f:
            features = f["features"][:]   # Features size (768, 1)
        features = torch.tensor(features, dtype=torch.float32)

        # Load class
        slide_id = Path(file_path).stem   # e.g., "INT1"
        json_path = self.metadata_dir / f"{slide_id}.json"

        if not json_path.exists():
            raise FileNotFoundError(f"Metadata file not found for {slide_id}: {json_path}")

        with open(json_path, "r", encoding="utf-8") as jf:
            metadata = json.load(jf)

        # Extract disease_state (class name)
        disease_state = metadata.get("disease_state", "Unknown")

        return features, disease_state






    