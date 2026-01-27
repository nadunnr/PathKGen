import torch
import numpy as np
from typing import List, Tuple
from sklearn.cluster import KMeans
from titan_encoder import TitanEncoder


class SentenceClusterer(TitanEncoder):
    """Class for clustering sentences and finding closest sentences to cluster means."""
    
    def __init__(self, 
                model: str = 'MahmoodLab/TITAN', 
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(model, device)
    
    def cluster_sentences(self, 
                         sentences: List[str], 
                         k: int,
                         normalize: bool = True) -> Tuple[List[str], List[int], np.ndarray]:
        """
        Cluster sentences into k classes and return the closest sentence to each cluster mean.
        
        Args:
            sentences: List of sentence strings
            k: Number of clusters
            normalize: Whether to normalize embeddings
            
        Returns:
            Tuple of (closest_sentences, cluster_labels, cluster_centers)
            - closest_sentences: List of sentences closest to each cluster mean
            - cluster_labels: Cluster assignment for each sentence
            - cluster_centers: Cluster center coordinates
        """
        # Encode all sentences
        embeddings_list = []
        for sentence in sentences:
            embedding = self.encode_text(sentence, normalize=normalize)
            embeddings_list.append(embedding.cpu().numpy())
        
        # Stack embeddings
        embeddings_array = np.vstack(embeddings_list)
        
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_array)
        cluster_centers = kmeans.cluster_centers_
        
        # Find closest sentence to each cluster center
        closest_sentences = []
        for i in range(k):
            # Get indices of sentences in this cluster
            cluster_indices = np.where(cluster_labels == i)[0]
            
            if len(cluster_indices) == 0:
                closest_sentences.append(None)
                continue
            
            # Get embeddings for sentences in this cluster
            cluster_embeddings = embeddings_array[cluster_indices]
            
            # Calculate distances to cluster center
            distances = np.linalg.norm(cluster_embeddings - cluster_centers[i], axis=1)
            
            # Find index of closest sentence
            closest_idx_in_cluster = np.argmin(distances)
            closest_idx = cluster_indices[closest_idx_in_cluster]
            
            closest_sentences.append(sentences[closest_idx])
        
        # return closest_sentences, cluster_labels.tolist(), cluster_centers
        return closest_sentences
