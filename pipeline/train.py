import torch
from model import SlideQwenModel

# Configuration
BASE_MODEL = "/home/nadun/wd/vlm_histo/checkpoints/Qwen2.5-7B-Instruct"
CHECKPOINT = "/home/nadun/wd/vlm_histo/checkpoints/SlideChat_Weight/stage2_pth/mp_rank_00_model_states.pt"

def load_slidechat_weights(model, checkpoint_path):
    print(f"Loading SlideChat weights from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            clean_key = key.replace("model.", "")
            # We specifically want the LLM weights, not the old vision projector
            if "mm_projector" not in clean_key and "slide_encoder" not in clean_key:
                new_state_dict[clean_key] = value
                
    # Inject into the internal LLM backbone
    model.llm.load_state_dict(new_state_dict, strict=False)
    print("SlideChat weights successfully injected into Qwen backbone.")

# 1. Initialize Model
model = SlideQwenModel(BASE_MODEL)

# 2. Load SlideChat Weights for Warm Start
load_slidechat_weights(model, CHECKPOINT)

# 3. Training Setup: Freeze LLM, Train Projector
model.llm.requires_grad_(False)
model.projector.requires_grad_(True)

optimizer = torch.optim.AdamW(model.projector.parameters(), lr=1e-4)

# --- MOCK TRAINING STEP ---
# In reality, these would come from your PyTorch Dataloader
mock_knowledge = torch.randn(1, 10, 3584).to("cuda").to(torch.bfloat16)
mock_slides = torch.randn(1, 256, 1024).to("cuda").to(torch.bfloat16)
mock_input_ids = torch.randint(0, 151643, (1, 20)).to("cuda") # Target report IDs

print("Starting training step...")
outputs = model(mock_knowledge, mock_slides, mock_input_ids, labels=mock_input_ids)
loss = outputs.loss
loss.backward()
optimizer.step()
optimizer.zero_grad()

print(f"Step Complete. Loss: {loss.item():.4f}")