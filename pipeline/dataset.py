import os
import json
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class WSI_Report_Dataset(Dataset):
    def __init__(self, json_path, dataset_format='wsi-bench', split='train', feature_key='tokens', use_knowledge_guidance=False):
        """
        Custom Dataset for loading Whole Slide Image (WSI) features and corresponding reports.
        
        Args:
            json_path (str): Path to the JSON file containing dataset information.
            dataset_format (str): Format of the dataset - 'wsi-bench' (flat list) or 'histgen' (nested dict).
            split (str or None): Dataset split to use ('train', 'val', 'test'). 
                                 Set to None for WSI-Bench (already pre-split files).
            feature_key (str): Key in the H5 file where slide features are stored.
            use_knowledge_guidance (bool): Whether to use knowledge-guided prompts.
        """
        self.feature_key = feature_key
        self.split = split
        self.use_knowledge_guidance = use_knowledge_guidance
        self.dataset_format = dataset_format

        # Load dataset information from JSON
        with open(json_path, 'r') as info_file:
            data = json.load(info_file)
        
        # Handle different dataset formats
        if dataset_format == 'histgen':
            # HistGen format: nested dict with splits {"train": [...], "val": [...], "test": [...]}
            if split is None:
                raise ValueError("split must be specified for HistGen format")
            
            if isinstance(split, str):
                self.dataset = data[split]
            elif isinstance(split, (list, tuple, set)):
                self.dataset = []
                for s in split:
                    self.dataset.extend(data[s])
            else:
                raise ValueError("split must be a string or a list/tuple/set of strings.")
            
            # Field mappings for HistGen format
            self.answer_key = 'report'
            self.has_questions = False
            
        elif dataset_format == 'wsi-bench':
            # WSI-Bench format: flat list OR dict (already split into separate files)
            if isinstance(data, dict):
                self.dataset = list(data.values())
            elif isinstance(data, list):
                self.dataset = data
            else:
                raise ValueError(
                    f"WSI-Bench format expects a flat list or dict, got {type(data)}. "
                    f"Are you using the correct dataset_format?"
                )
            
            # Field mappings for WSI-Bench format
            self.answer_key = 'T-answer'
            self.has_questions = True
            
        else:
            raise ValueError(f"Unknown dataset_format: {dataset_format}. Use 'histgen' or 'wsi-bench'.")
        
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
        
        # Extract question and answer based on dataset format
        if self.has_questions:
            # WSI-Bench format: has question field
            question = item['question']
            answer = item[self.answer_key]  # 'T-answer'
        else:
            # HistGen format: no question field, will use fixed prompt
            question = None
            answer = item[self.answer_key]  # 'report'

        # Get pre-computed knowledge text if using knowledge guidance
        if self.use_knowledge_guidance:
            knowledge_text = item.get("knowledge", "")
        else:
            knowledge_text = ""

        # Load slide features from H5 file
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Feature file not found: {feature_path}")

        with h5py.File(feature_path, 'r') as h5_file:
            slide_features = h5_file[self.feature_key][:]
            
            # Validate shape: must be 2D spatial features [num_tokens, feature_dim]
            if slide_features.ndim != 2:
                raise ValueError(
                    f"Feature file {feature_path} has shape {slide_features.shape}. "
                    f"Expected 2D spatial features [num_tokens, feature_dim]."
                )
            
            slide_embedding = torch.tensor(slide_features, dtype=torch.float32)
            # Shape: [128, 768] or [N, 768] depending on encoder

        # Get sample metadata
        slide_id = item.get('tcga_id', item.get('id', str(idx)))
        organ = item.get('primary_site', item.get('organ', 'unknown'))

        return {
            'id': slide_id,
            'organ': organ,
            'slide_embedding': slide_embedding,
            'question': question,  # None for HistGen, actual question for WSI-Bench
            'answer': answer,      # 'report' for HistGen, 'T-answer' for WSI-Bench
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
        dataset = WSI_Report_Dataset(json_file, split=split, feature_key='tokens', use_knowledge_guidance=False)

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


            
        