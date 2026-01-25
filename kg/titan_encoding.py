import h5py
import torch

from transformers import AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load titan
titan = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
titan = titan.to(device)

# load data
from huggingface_hub import hf_hub_download
demo_h5_path = hf_hub_download(
    "MahmoodLab/TITAN", 
    filename="TCGA_demo_features/TCGA-PC-A5DK-01Z-00-DX1.C2D3BC09-411F-46CF-811B-FDBA7C2A295B.h5",
)
file = h5py.File(demo_h5_path, 'r')
features = torch.from_numpy(file['features'][:])
coords = torch.from_numpy(file['coords'][:])
patch_size_lv0 = file['coords'].attrs['patch_size_level0']

# Define pathology text
pathology_text = "The pathology report describes the specimens labeled A, B, and C, which include wide local excisions and an additional margin as part of the surgical treatment of a female patient with bilateral masses and a preoperative diagnosis of bilateral invasive ductal carcinoma. The report reveals that the right breast excision (Part A) measures 2.4 cm and contains invasive ductal carcinoma, SBR grade 2, mixed with ductal carcinoma in situ, lobular carcinoma in situ, and biopsy site changes. The report also indicates that invasive tumor is present at the superior margins and within 1-mm from medial in parts A and B, respectively. The left breast excision (Part C) measures 4 x 4 x 3.5 cm and contains invasive ductal carcinoma, SBR grade 2, and lobular carcinoma in situ, but with negative surgical resection margins. The report further provides additional information, such as specimen processing, margin status, tumor characteristics, lymph node status, and ER/PR/HER2 results."

# Encode pathology text
with torch.autocast(device_type='cuda', dtype=torch.bfloat16), torch.inference_mode():
    # Tokenize the text using TITAN's text encoder tokenizer
    input_ids = titan.text_encoder.tokenizer(pathology_text).to(device)
    
    # Get text embedding using TITAN's encode_text method
    text_embedding = titan.encode_text(input_ids, normalize=True)
    
    print(f"Text embedding shape: {text_embedding.shape}")

# Extract slide embedding
with torch.autocast(device_type='cuda', dtype=torch.bfloat16), torch.inference_mode():
    features = features.to(device)
    coords = coords.to(device)
    slide_embedding = titan.encode_slide_from_patch_features(features, coords, patch_size_lv0)

    print(f"Slide embedding shape: {slide_embedding.shape}")

