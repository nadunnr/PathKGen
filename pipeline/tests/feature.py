import h5py
import torch

file_path = "/home/nadun/wd/datasets/wsi_bench/processed/20x_512px_0px_overlap/slide_features_titan/TCGA-2G-AAGG-01Z-00-DX1.546E8AD0-FEE4-44C3-9874-F9BBBD75E6BF.h5"

print(f"Reading: {file_path}\n")

with h5py.File(file_path, 'r') as h5_file:
    print("Available keys:", list(h5_file.keys()))
    print()
    
    # Read pooled features if available
    if 'features' in h5_file:
        pooled_features = h5_file["features"][:]
        pooled_embedding = torch.tensor(pooled_features, dtype=torch.float32)
        print(f"Pooled features shape: {pooled_embedding.shape}")
        print(f"Pooled features (1D embedding): {pooled_embedding.shape}")
    else:
        print("No pooled features found")
    
    print()
    
    # Read 2D tokens if available
    if 'tokens' in h5_file:
        token_features = h5_file["tokens"][:]
        token_embedding = torch.tensor(token_features, dtype=torch.float32)
        print(f"Token features shape: {token_embedding.shape}")
        print(f"Token features (2D tokens): {token_embedding.shape}")
        print(f"Number of tokens: {token_embedding.shape[0]}")
        print(f"Token dimension: {token_embedding.shape[1]}")
    else:
        print("No token features found")
