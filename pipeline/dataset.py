import os
import json
import h5py
import torch
from torch.utils.data import Dataset, DataLoader

class WSI_Report_Dataset(Dataset):
    def __init__(self, json_path, split = 'train', feature_key='features', use_knowledge_guidance=False):
        """
        Custom Dataset for loading Whole Slide Image (WSI) features and corresponding reports.
        
        Args:
            json_path (str): Path to the JSON file containing dataset information.
            split (str): Dataset split to use ('train', 'val', 'test').
            feature_key (str): Key in the H5 file where slide features are stored.
        """
        self.feature_key = feature_key
        self.split = split
        self.use_knowledge_guidance = use_knowledge_guidance

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
        
        # Filter out samples without feature_path (some val samples may not have features extracted)
        original_len = len(self.dataset)
        self.dataset = [item for item in self.dataset if 'feature_path' in item]
        filtered_len = len(self.dataset)
        
        if filtered_len < original_len:
            print(f"Warning: Filtered out {original_len - filtered_len} samples without feature_path")
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Retrieve a single data point from the dataset.
        """
        item = self.dataset[idx]
        feature_path = item['feature_path']
        report_text = item['report']

        # Get pre-computed knowledge text if using knowledge guidance
        if self.use_knowledge_guidance:
            knowledge_text = item.get("knowledge", "")
        else:
            knowledge_text = ""

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
            'report': report_text,
            'knowledge_text': knowledge_text
        }
            


if __name__ == "__main__":
    
    # --- Configuration ---
    json_file = "/home/nadun/wd/datasets/tcga_report_histgen/primary_site_splits/breast_dataset_split_0.7_0.1_0.2_knowledge.json"

    print("="*80)
    print("TESTING DATASET WITHOUT KNOWLEDGE GUIDANCE")
    print("="*80)
    try:
        split = ['train','val']
        dataset = WSI_Report_Dataset(json_file, split=split, feature_key='features', use_knowledge_guidance=False)

        loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

        print(f"Dataset length: {len(dataset)}")

        for batch in loader:
            slide_embeddings = batch['slide_embedding']
            reports = batch['report']
            knowledge_texts = batch['knowledge_text']
            ids = batch['id']

            print(f"Batch IDs: {ids}")
            print(f"Slide Embeddings Shape: {slide_embeddings.shape}")
            print(f"Number of Reports in Batch: {len(reports)}")
            print(f"Reports: {reports[0][:50]}...")
            print(f"Knowledge texts (should be empty): {knowledge_texts}")
            break 

    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)
    print("TESTING DATASET WITH KNOWLEDGE GUIDANCE")
    print("="*80)
    try:
        split = 'train'
        dataset_kg = WSI_Report_Dataset(json_file, split=split, feature_key='features', use_knowledge_guidance=True)

        # Use num_workers=0 to avoid multiprocessing issues during testing
        loader_kg = DataLoader(dataset_kg, batch_size=2, shuffle=True, num_workers=0)

        print(f"Dataset length: {len(dataset_kg)}")

        for batch in loader_kg:
            slide_embeddings = batch['slide_embedding']
            reports = batch['report']
            knowledge_texts = batch['knowledge_text']
            ids = batch['id']

            print(f"\nBatch IDs: {ids}")
            print(f"Slide Embeddings Shape: {slide_embeddings.shape}")
            print(f"Number of Reports in Batch: {len(reports)}")
            print(f"\nReport sample:\n{reports[0][:200]}...")
            print(f"\nKnowledge text sample (first 500 chars):\n{knowledge_texts[0][:500] if knowledge_texts[0] else '(empty)'}...")
            break 

        print("\n✓ Knowledge extraction working correctly!")

    except Exception as e:
        print(f"Error loading dataset with knowledge guidance: {e}")
        import traceback
        traceback.print_exc()


            
        