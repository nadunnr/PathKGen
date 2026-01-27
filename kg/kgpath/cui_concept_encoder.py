import h5py
import torch
import pandas as pd
import numpy as np
from typing import List, Union, Tuple
from titan_encoder import TitanEncoder


class CUIConceptEncoder(TitanEncoder):
    """Class for encoding CUI concepts and finding top-k similar concepts."""
    
    def __init__(self, 
                model: str = 'MahmoodLab/TITAN', 
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(model, device)
        self.cui_embeddings = None
        self.cui_ids = None
        self.cui_names = None
    
    def encode_cui_concepts(self, 
                           csv_path: str, 
                           output_path: str,
                           normalize: bool = True) -> None:
        """
        Encode CUI concepts from a CSV file with cui:ID and name columns.
        
        Args:
            csv_path: Path to the CSV file containing CUI concepts
                     Expected columns: cui:ID, name, description, :LABEL
            output_path: Path to save the embeddings (as .h5 file)
            normalize: Whether to normalize embeddings
        """
        # Load CSV with header
        df = pd.read_csv(csv_path)
        
        # Extract CUI IDs and names
        cui_ids = df['cui:ID'].tolist()
        cui_names = df['name'].tolist()
        cui_description = df['description'].tolist()
        
        # Encode all concept names
        embeddings_list = []
        for name, description in zip(cui_names, cui_description):
            caption = f"Whole slide image of a {name}, which is {cui_description}"
            embedding = self.encode_text(caption, normalize=normalize)
            embeddings_list.append(embedding.cpu().numpy())
        
        # Stack all embeddings
        embeddings_array = np.vstack(embeddings_list)
        
        # Save to H5 file
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('embeddings', data=embeddings_array)
            # Save CUI IDs and names as strings
            dt = h5py.string_dtype(encoding='utf-8')
            f.create_dataset('cui_ids', data=cui_ids, dtype=dt)
            f.create_dataset('cui_names', data=cui_names, dtype=dt)
        
        print(f"Encoded {len(cui_ids)} CUI concepts and saved to {output_path}")
    
    def load_cui_embeddings(self, embedding_path: str) -> Tuple[torch.Tensor, List[str], List[str]]:
        """
        Load CUI embeddings from file.
        
        Args:
            embedding_path: Path to the H5 file containing embeddings
            
        Returns:
            Tuple of (embeddings tensor, list of CUI IDs, list of CUI names)
        """
        with h5py.File(embedding_path, 'r') as f:
            embeddings = torch.tensor(f['embeddings'][:], dtype=torch.float32).to(self.device)
            cui_ids = [c.decode('utf-8') if isinstance(c, bytes) else c for c in f['cui_ids'][:]]
            cui_names = [c.decode('utf-8') if isinstance(c, bytes) else c for c in f['cui_names'][:]]
        
        self.cui_embeddings = embeddings
        self.cui_ids = cui_ids
        self.cui_names = cui_names
        
        return embeddings, cui_ids, cui_names
    
    def get_top_k_similar_concepts(self, 
                                   slide_h5_path: str, 
                                   embedding_path: str,
                                   k: int = 10) -> List[Tuple[str, str, float]]:
        """
        Given a slide embedding, find top-k most similar CUI concepts.
        
        Args:
            slide_h5_path: Path to the slide embedding H5 file
            embedding_path: Path to the CUI embeddings H5 file
            k: Number of top similar concepts to return
            
        Returns:
            List of tuples (cui_id, cui_name, similarity_score) ordered by similarity
        """
        # Load slide embedding
        slide_embedding = self.load_slide_embedding(slide_h5_path)
        
        # Load CUI embeddings if not already loaded
        if self.cui_embeddings is None or self.cui_ids is None or self.cui_names is None:
            self.load_cui_embeddings(embedding_path)
        
        # Calculate similarities
        similarities = torch.nn.functional.cosine_similarity(
            slide_embedding.unsqueeze(0), 
            self.cui_embeddings, 
            dim=1
        )
        
        # Get top k indices
        top_k_values, top_k_indices = torch.topk(similarities, k)
        
        # Create result list with both CUI ID and name
        results = [
            (self.cui_ids[idx.item()], self.cui_names[idx.item()], score.item()) 
            for idx, score in zip(top_k_indices, top_k_values)
        ]
        
        return results
