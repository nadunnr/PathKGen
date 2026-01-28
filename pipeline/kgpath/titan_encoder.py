import h5py
import torch
from transformers import AutoModel

class TitanEncoder:
    def __init__(self, 
                model: str = 'MahmoodLab/TITAN', 
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
                ):
        self.device = device
        self.model = AutoModel.from_pretrained(model, trust_remote_code=True).to(self.device)

    def encode_text(self, text, normalize: bool = True):
        with torch.inference_mode():
            input_ids = self.model.text_encoder.tokenizer(text).to(self.device)
            
            # Use autocast only for CUDA to avoid dtype issues on CPU
            if self.device == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    text_embedding = self.model.encode_text(input_ids, normalize=normalize)
            else:
                text_embedding = self.model.encode_text(input_ids, normalize=normalize)
            
            return text_embedding
    
    def encode_slide(self, features, coords, patch_size_lv0, normalize: bool = True):
        with torch.inference_mode():
            features = features.to(self.device)
            coords = coords.to(self.device)
            
            # Use autocast only for CUDA to avoid dtype issues on CPU
            if self.device == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    slide_embedding = self.model.encode_slide_from_patch_features(features, coords, patch_size_lv0)
            else:
                slide_embedding = self.model.encode_slide_from_patch_features(features, coords, patch_size_lv0)
            
            return slide_embedding
        
    def load_slide_embedding(self, h5_path):
        file = h5py.File(h5_path, 'r')
        slide_embedding = torch.tensor(file['features'][:], dtype = torch.float32).to(self.device)
        return slide_embedding

    def calculate_similarity(self, embedding1, embedding2):
        return torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=0)