import torch
import torch.nn as nn

class SlideChatProjector(nn.Module):
    def __init__(self, visual_dim=1024, llm_dim=3584):
        super().__init__()
        # Standard LLaVA-style MLP projector
        self.model = nn.Sequential(
            nn.Linear(visual_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )

    def forward(self, x):
        return self.model(x)