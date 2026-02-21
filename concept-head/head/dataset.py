import os
import h5py
import torch
import torch.nn as nn

class ConceptDatasetwithSite(Dataset):
    def __init__(self, ids, labels, sites, label_to_idx, site_to_idx):
        self.ids = ids
        self.labels = labels
        self.sites = sites
        self.label_to_idx = label_to_idx
        self.site_to_idx = site_to_idx

    def __getitem__(self, idx):
        tcga_id = self.ids[idx]
        emb_path = os.path.join(EMBEDDING_DIR, f"{tcga_id}.h5")

        with h5py.File(emb_path, 'r') as f:
            embedding = f['features'][:]
            embedding = (embedding - embedding.mean()) / (embedding.std() + 1e-6)

        label = self.label_to_idx[self.labels[idx]]
        site_id = self.site_to_idx[self.sites[idx]]

        return (
            torch.tensor(embedding, dtype=torch.float32),
            torch.tensor(site_id, dtype=torch.long),
            label
        )
    
    def __len__(self):
        return len(self.ids)