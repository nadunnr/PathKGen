import os
import json
import h5py
import torch
from torch.utils.data import Dataset, DataLoader

class WSI_Report_Dataset(Dataset):
    def __init__(self, json_path, split = 'train', feature_key='features'):
        """
        Custom Dataset for loading Whole Slide Image (WSI) features and corresponding reports.
        
        Args:
            json_path (str): Path to the JSON file containing dataset information.
            split (str): Dataset split to use ('train', 'val', 'test').
            feature_key (str): Key in the H5 file where slide features are stored.
        """
        self.feature_key = feature_key
        self.split = split

        # Load dataset information from JSON
        with open(json_path, 'r') as info_file:
            full_dataset = json.load(info_file)
        
        #Select the split
        if isinstance(split, str):
            self.dataset = full_dataset[split]
        elif isinstance(split, (list, tuple, set)):
            self.dataset = []
            for s in split:
                self.dataset.extend(full_dataset[s])
        else:
            raise ValueError("split must be a string or a list/tuple/set of strings.")
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Retrieve a single data point from the dataset.
        """
        item = self.dataset[idx]
        feature_path = item['feature_path']
        report_text = item['report']

        # Load slide features from H5 file
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Feature file not found: {feature_path}")

        with h5py.File(feature_path, 'r') as h5_file:
            slide_features = h5_file[self.feature_key]
            slide_embedding = torch.tensor(slide_features[:], dtype=torch.float32)

        slide_id = item.get('id', str(idx))
        organ = item.get('organ', 'unknown')

        return {
            'id': slide_id,
            'organ': organ,
            'slide_embedding': slide_embedding,
            'report': report_text
        }
            


if __name__ == "__main__":
    
    # --- Configuration ---
    json_file = "/home/nadun/wd/vlm_histo/data/curated_breast_dataset.json"


    try:
        split = ['train','val']
        dataset = WSI_Report_Dataset(json_file, split=split, feature_key='features')

        loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

        print(f"Dataset length: {len(dataset)}")

        for batch in loader:
            slide_embeddings = batch['slide_embedding']
            reports = batch['report']
            ids = batch['id']

            print(f"Batch IDs: {ids}")
            print(f"Slide Embeddings Shape: {slide_embeddings.shape}")
            print(f"Number of Reports in Batch: {len(reports)}")
            print(f"Reports: {reports[0][:50]}...")  # Print first 50 chars of first report
            break 

    except Exception as e:
        print(f"Error loading dataset: {e}")

            
        