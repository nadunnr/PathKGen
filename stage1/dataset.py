import os
import h5py
import torch
from torch.utils.data import Dataset
from typing import Dict, List
import logging
import csv
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class AlignmentDataset(Dataset):
    """Dataset yielding (tokens, text_embedding) for contrastive alignment."""
    
    def __init__(
        self, samples: List[Dict],
        slide_features_dir: str, text_embeddings_dir: str):
        """
        Args:
            samples: List of dicts with 'tcga_id' 
        """
        self.samples = samples
        self.slide_features_dir = slide_features_dir
        self.text_embeddings_dir = text_embeddings_dir
        
        # Filter out samples with missing files
        valid_samples = []
        for s in samples:
            slide_path = os.path.join(slide_features_dir, f"{s['tcga_id']}.h5")
            text_path = os.path.join(text_embeddings_dir, f"{s['tcga_id']}.h5")
            
            if os.path.exists(slide_path) and os.path.exists(text_path):
                valid_samples.append(s)
        
        if len(valid_samples) < len(samples):
            logger.warning(f"Dropped {len(samples) - len(valid_samples)} samples with missing h5 files")
        
        self.samples = valid_samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        tcga_id = sample['tcga_id']
        
        # Load slide features (TITAN tokens)
        slide_path = os.path.join(self.slide_features_dir, f"{tcga_id}.h5")
        with h5py.File(slide_path, 'r') as f:
            tokens = torch.from_numpy(f['tokens'][:]).float()   #In titan embeddings token key had 2d tokens

        # Load text embeddings
        text_path = os.path.join(self.text_embeddings_dir, f"{tcga_id}.h5")
        with h5py.File(text_path, 'r') as f:
            text_emb = torch.from_numpy(f['features'][:]).float()     #only key used in text embedding h5 files is 'features' 
        
        return {
            'tokens': tokens,
            'text_embeddings': text_emb,
            'sample_id': tcga_id,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate batch into tensors."""
    return {
        'tokens': torch.stack([b['tokens'] for b in batch]),
        'text_embeddings': torch.stack([b['text_embeddings'] for b in batch]),
        'sample_ids': [b['sample_id'] for b in batch],
    }


def load_dataset_csv(
    csv_path: str,
    split: str = "train",
) -> List[Dict]:
    """
    Load dataset from CSV file and return samples.
    Args:
        csv_path: Path to CSV file (train or test)
        split: 'train' or 'test' (for logging)
        max_samples: Max number of samples to load (for debugging)
    
    Returns:
        List of dicts with 'tcga_id', 'report'
    """

    
    samples = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            samples.append({'tcga_id': row['tcga_id']})
    
    logger.info(f"Loaded {len(samples)} samples for '{split}' split")
    return samples


if __name__ == '__main__':
    # Simple test to load dataset
    samples = load_dataset_csv(
        csv_path="/home/sandun/nadun/data/dataset/report_dataset_train.csv",
        split="train",
        max_samples=5,
    )
    print(f"Loaded {len(samples)} samples")
    print("First sample:", samples[0])
    dataset = AlignmentDataset(
        samples=samples,
        slide_features_dir="/home/sandun/nadun/data/slide_features/train",
        text_embeddings_dir="/home/sandun/nadun/data/text_embeddings/train",
    )


    dataloader = DataLoader(
        dataset,
        batch_size=5,
        shuffle=False,
        collate_fn=collate_fn,
    )

    for batch in dataloader:
        print("Batch keys:", batch.keys())
        print("Tokens shape:", batch['tokens'].shape)
        print("Text embeddings shape:", batch['text_embeddings'].shape)
        print("Sample IDs:", batch['sample_ids'])
        break
