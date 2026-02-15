"""
Stage 1 Models: Projection heads for WSI-Text contrastive alignment.

Following WSI-LLaVA:
- Two separate MLP projectors (image + text) → shared alignment space
- TITAN encoder frozen, CLIP vision tower frozen; only projectors are trained
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPVisionConfig


class CLIPVisionTower(nn.Module):
    """
    Frozen CLIP Vision Tower that processes TITAN patch tokens.
    
    Zero-pads TITAN tokens (768-dim) → CLIP dimension (1024-dim),
    then processes through frozen CLIP encoder.
    """
    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-large-patch14-336",
        titan_dim: int = 768,
        freeze: bool = True,
    ):
        super().__init__()
        
        # Load CLIP vision model
        self.vision_model = CLIPVisionModel.from_pretrained(clip_model_name)
        self.clip_hidden_size = self.vision_model.config.hidden_size  # 1024 for ViT-L
        self.titan_dim = titan_dim
        
        # Freeze CLIP weights
        if freeze:
            self._freeze_clip()
    
    def _freeze_clip(self):
        """Freeze all CLIP vision model parameters."""
        for param in self.vision_model.parameters():
            param.requires_grad = False
        self.vision_model.eval()
    
    @property
    def output_dim(self) -> int:
        """Output dimension per patch token."""
        return self.clip_hidden_size
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Process TITAN patch tokens through CLIP vision encoder.
        
        Args:
            tokens: [batch, num_tokens, titan_dim] e.g., [B, 128, 768]
        Returns:
            patch_features: [batch, num_tokens, clip_hidden_size] e.g., [B, 128, 1024]
        """
        # Zero-pad TITAN tokens to CLIP dimension
        B, N, D = tokens.shape
        padding_size = self.clip_hidden_size - D
        hidden_states = F.pad(tokens, (0, padding_size), value=0)  # [B, 128, 768] → [B, 128, 1024]

        # Pass through CLIP encoder
        encoder_outputs = self.vision_model.vision_model.encoder(
            hidden_states,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        
        # Apply final layer norm
        patch_features = self.vision_model.vision_model.post_layernorm(
            encoder_outputs.last_hidden_state
        )
        return patch_features


class MLPProjector(nn.Module):
    """
    Token-wise MLP projection head.
    Projects each token independently: [batch, num_tokens, input_dim] → [batch, num_tokens, output_dim]
    
    Architecture: Linear → GELU → Linear → L2-normalize
    """
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 3584, output_dim: int = 3584):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, num_tokens, input_dim] or [batch, input_dim]
        Returns:
            [batch, num_tokens, output_dim] or [batch, output_dim] - L2-normalized
        """
        # Project to alignment space (nn.Linear handles any dim automatically)
        projected = self.net(x)
        
        # L2 normalize along last dimension
        projected = F.normalize(projected, p=2, dim=-1)
        
        return projected


class AttentionPooling(nn.Module):
    """
    Attention pooling to aggregate TITAN spatial tokens → single vector.
    Learns a query vector that attends over the token sequence.
    """
    def __init__(self, dim: int = 768):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [batch, num_tokens, dim]  e.g. [B, 128, 768]
        Returns:
            pooled: [batch, dim]  e.g. [B, 768]
        """
        B = tokens.shape[0]
        query = self.query.expand(B, -1, -1)  # [B, 1, 768]
        pooled, _ = self.attn(query, tokens, tokens)  # [B, 1, 768]
        pooled = self.norm(pooled.squeeze(1))  # [B, 768]
        return pooled


class Stage1AlignmentModel(nn.Module):
    """
    Complete Stage 1 model: CLIP vision tower + token-wise projection for contrastive alignment.
    
    WSI path: TITAN tokens [128,768] → zero-pad → CLIP [128,1024] → token-wise MLP → [128,3584]
    Text path: text_embedding [768] → MLP → [3584]
    
    InfoNCE loss between pooled image tokens and text (pooling done in loss).
    """
    def __init__(
        self,
        titan_dim: int = 768,
        text_dim: int = 768,
        hidden_dim: int = 3584,
        alignment_dim: int = 3584,
        clip_model_name: str = "openai/clip-vit-large-patch14-336",
        freeze_clip: bool = True,
    ):
        super().__init__()
        
        self.freeze_clip = freeze_clip
        
        # Image path: TITAN tokens → CLIP tower → token-wise projection
        self.clip_tower = CLIPVisionTower(
            clip_model_name=clip_model_name,
            titan_dim=titan_dim,
            freeze=freeze_clip,
        )
        image_proj_input_dim = self.clip_tower.output_dim  # 1024
        
        # Token-wise image projector: [B, 128, 1024] → [B, 128, 3584]
        self.image_projector = MLPProjector(image_proj_input_dim, hidden_dim, alignment_dim)
        # Text projector: [B, 768] → [B, 3584]
        self.text_projector = MLPProjector(text_dim, hidden_dim, alignment_dim)
    
    def encode_image(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Process WSI tokens through CLIP and project token-wise.
        
        Args:
            tokens: [batch, num_tokens, titan_dim] e.g., [B, 128, 768]
        Returns:
            [batch, num_tokens, alignment_dim] e.g., [B, 128, 3584]
        """
        # TITAN tokens → zero-pad → CLIP → token-wise projection
        patch_features = self.clip_tower(tokens)  # [B, 128, 768] → [B, 128, 1024]
        projected_tokens = self.image_projector(patch_features)  # [B, 128, 1024] → [B, 128, 3584]
        return projected_tokens
    
    def encode_text(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Project text embeddings to alignment space.
        text_embeddings: [batch, 768] -> [batch, alignment_dim(3584)]
        """
        return self.text_projector(text_embeddings)
    
    def forward(self, tokens, text_embeddings):
        """
        Forward pass returning projected embeddings.
        
        Args:
            tokens: [batch, num_tokens, titan_dim] e.g., [B, 128, 768]
            text_embeddings: [batch, text_dim] e.g., [B, 768]
        Returns:
            image_tokens: [batch, num_tokens, alignment_dim] e.g., [B, 128, 3584]
            text_embeds: [batch, alignment_dim] e.g., [B, 3584]
        """
        image_tokens = self.encode_image(tokens)
        text_embeds = self.encode_text(text_embeddings)
        return image_tokens, text_embeds
    
    def trainable_parameters(self):
        """
        Return trainable parameters: image and text projectors only.
        CLIP vision tower is frozen.
        """
        return list(self.image_projector.parameters()) + list(self.text_projector.parameters())
    
    def print_trainable_params(self):
        """Print summary of trainable vs frozen parameters."""
        trainable = sum(p.numel() for p in self.trainable_parameters())
        total = sum(p.numel() for p in self.parameters())
        frozen = total - trainable
        print(f"Trainable params: {trainable:,} | Frozen params: {frozen:,} | Total: {total:,}")
        print(f"Trainable ratio: {100 * trainable / total:.2f}%")


class InfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE contrastive loss (CLIP-style).
    
    For image token sequences, mean-pools tokens first, then computes
    contrastive loss with text embeddings.
    
    L = 0.5 * (CE(sim_i2t, labels) + CE(sim_t2i, labels))
    where sim = (img_proj @ txt_proj^T) / temperature
    """
    def __init__(self, temperature: float = 0.02):
        super().__init__()
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, image_embeds: torch.Tensor, text_embeds: torch.Tensor):
        """
        Args:
            image_embeds: [B, N, D] token sequence OR [B, D] pooled - L2-normalized
            text_embeds: [B, D] L2-normalized
        Returns:
            loss: scalar symmetric InfoNCE loss
            logits: [B, B] similarity matrix (for accuracy computation)
        """
        # If image_embeds is 3D (token sequence), mean-pool to 2D
        if image_embeds.dim() == 3:
            image_embeds = image_embeds.mean(dim=1)  # [B, N, D] -> [B, D]
            # Re-normalize after pooling
            image_embeds = F.normalize(image_embeds, p=2, dim=-1)
        
        # Cosine similarity scaled by temperature
        logits = torch.matmul(image_embeds, text_embeds.T) / self.temperature  # [B, B]
        
        # Positive pairs are on the diagonal
        labels = torch.arange(logits.shape[0], device=logits.device)
        
        loss_i2t = self.ce(logits, labels)
        loss_t2i = self.ce(logits.T, labels)
        
        loss = (loss_i2t + loss_t2i) / 2.0
        return loss, logits
