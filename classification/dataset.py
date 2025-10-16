# dataset.py
import os
from pathlib import Path
import json
import glob
import h5py
import torch
from torch.utils.data import Dataset, DataLoader

class SlideEmbeddingDataset(Dataset):
    def __init__(self, visual_root_dir, text_root_dir, vision_only:bool=True):

        self.visual_files = sorted(glob.glob(os.path.join(visual_root_dir, "*.h5")))
        self.text_files = sorted(glob.glob(os.path.join(text_root_dir, "*.h5")))
        self.metadata_dir = Path("/home/nadun/wd/HEST/hest_data_40/metadata")
        self.vision_only = vision_only

        # Map files by their ID (stem name)
        visual_map = {Path(f).stem: f for f in self.visual_files}
        text_map = {Path(f).stem: f for f in self.text_files}

        # Find common IDs
        common_ids = sorted(set(visual_map.keys()) & set(text_map.keys()))
        if not common_ids:
            raise ValueError("No matching IDs found between visual and text embeddings!")

        if not self.visual_files:
            raise FileNotFoundError(f"No .h5 files found in {visual_root_dir}")
        if not self.text_files:
            raise FileNotFoundError(f"No .h5 files found in {text_root_dir}")
        
        # Keep only matched pairs
        self.visual_files = [visual_map[i] for i in common_ids]
        self.text_files = [text_map[i] for i in common_ids]
        self.ids = common_ids  # keep track of slide IDs

    def __len__(self):
        return len(self.visual_files)

    def __getitem__(self, idx):
        visual_file_path = self.visual_files[idx]
        text_file_path = self.text_files[idx]

        # Load slide embedding
        with h5py.File(visual_file_path, "r") as f:
            features = f["features"][:]   # Features size (768, 1)
            # print(f"Slide embedding shape from {visual_file_path}: {features.shape}")
        slide_embed = torch.tensor(features, dtype=torch.float32)

        # Load text embedding
        with h5py.File(text_file_path, "r") as f:
            text_embed = f["text_embedding"][:]
            #print(f"Text embedding shape from {text_file_path}: {text_embed.shape}")
        text_embed = torch.tensor(text_embed, dtype=torch.float32)

        features = self.concatenate_pooled_features(slide_embed, text_embed)  # (1536,)

        # Load class
        slide_id = Path(visual_file_path).stem   # e.g., "INT1"
        json_path = self.metadata_dir / f"{slide_id}.json"

        if not json_path.exists():
            raise FileNotFoundError(f"Metadata file not found for {slide_id}: {json_path}")

        with open(json_path, "r", encoding="utf-8") as jf:
            metadata = json.load(jf)

        # Extract disease_state (class name)
        disease_state = metadata.get("disease_state", "Unknown")

        return features, disease_state, 
    
    def concatenate_pooled_features(self, visual_features, text_features):
        # Visual features: (768, 1) for conch
        # Text features: (768, 1) for conch

        # No pooling for now
        visual_pooled = visual_features.squeeze(0)  # (512)
        text_pooled = text_features.squeeze(0)      # (512)

        if self.vision_only:
            return visual_pooled
        else:
            concatenated = torch.cat((visual_pooled, text_pooled), dim=0)  # (1024)
            #print("Shape concatenated:", concatenated.shape)
        return concatenated






    