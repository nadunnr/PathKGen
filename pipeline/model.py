import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import Optional, Dict, Tuple
import warnings


class ProjectionLayer(nn.Module):
    """
    Projects global WSI slide features (768-dim) into multiple visual tokens 
    in LLM embedding space (3584-dim for Qwen 2.5 7B).
    
    Design: 
        - Input: [batch, 768] global slide embedding
        - Expand to multiple learnable visual tokens
        - Output: [batch, num_visual_tokens, 3584]
    """
    def __init__(self, visual_dim: int = 768, llm_hidden_dim: int = 3584, num_visual_tokens: int = 32, hidden_dim: int = 2048):
        super().__init__()
        self.num_visual_tokens = num_visual_tokens

        self.projector = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, llm_hidden_dim * num_visual_tokens)
        )

        # Learnable visual tokens
        self.visual_token_template = nn.Parameter(torch.randn(1,num_visual_tokens, llm_hidden_dim)*0.02)

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:

        batch_size = visual_features.shape[0]
        
        # Project visual features
        projected_features = self.projector(visual_features)
        
        # Expand visual tokens
        visual_tokens = projected_features.reshape(batch_size, self.num_visual_tokens, -1)

        # Add learnable visual tokens
        visual_tokens = visual_tokens + self.visual_token_template

        return visual_tokens

class VisionLanguageModel(nn.Module):
    """
    Complete VLM for histopathology report generation.
    
    Components:
        - Projection layer: WSI features → visual tokens
        - Language model: Qwen 2.5 7B with SlideChat weights
        - QLoRA support for efficient fine-tuning
    """
    def __init__(
        self,
        llm_path: str = "/home/nadun/wd/vlm_histo/checkpoints/Qwen2.5-7B-Instruct",
        slidechat_checkpoint: str = "/home/nadun/wd/vlm_histo/checkpoints/SlideChat_Weight/stage2_pth/mp_rank_00_model_states.pt",
        visual_dim: int = 768,
        llm_hidden_dim: int = 3584,
        num_visual_tokens: int = 32,
        use_qlora: bool = False,
        freeze_llm: bool = True,
        gradient_checkpointing: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        
        self.device = device
        self.use_qlora = use_qlora
        self.freeze_llm = freeze_llm
        
        # 1. Initialize tokenizer
        print(f"Loading tokenizer from {llm_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # 2. Initialize projection layer
        llm_hidden_dim = 3584  # Qwen 2.5 7B
        self.projection = ProjectionLayer(
            visual_dim=visual_dim,
            llm_hidden_dim=llm_hidden_dim,
            num_visual_tokens=num_visual_tokens
        )
        
        # 3. Load language model
        if use_qlora:
            print("Loading LLM with 4-bit quantization for QLoRA...")
            self.llm = self._load_llm_with_qlora(llm_path)
        else:
            print(f"Loading LLM from {llm_path}")
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
        
        # 4. Load SlideChat weights if provided
        if slidechat_checkpoint is not None:
            print(f"Loading SlideChat pathology weights from {slidechat_checkpoint}")
            self._load_slidechat_weights(slidechat_checkpoint)
        
        # 5. Enable gradient checkpointing if requested (saves ~40% activation memory)
        if gradient_checkpointing:
            print("Enabling gradient checkpointing for memory efficiency...")
            self.llm.gradient_checkpointing_enable()
        
        # 6. Freeze LLM if specified
        if freeze_llm and not use_qlora:
            print("Freezing LLM parameters...")
            for param in self.llm.parameters():
                param.requires_grad = False
                
        # 7. Move projection to device and match LLM dtype
        self.projection = self.projection.to(device)
        # Convert projection to bfloat16 to match LLM
        self.projection = self.projection.to(torch.bfloat16)
        
        # Store the LLM's dtype for consistency
        self.llm_dtype = torch.bfloat16
        
        print(f"VLM initialized successfully!")
        print(f"  - Projection: {visual_dim} → {num_visual_tokens} tokens × {llm_hidden_dim}")
        print(f"  - LLM frozen: {freeze_llm}")
        print(f"  - Using QLoRA: {use_qlora}")
        print(f"  - Gradient checkpointing: {gradient_checkpointing}")
        
    def _load_llm_with_qlora(self, llm_path: str) -> nn.Module:
        """Load LLM with 4-bit quantization and LoRA adapters."""
        # BitsAndBytes config for 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load base model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            llm_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Prepare for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        # LoRA configuration
        lora_config = LoraConfig(
            r=64,  # Rank
            lora_alpha=128,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        return model
    
    def _load_slidechat_weights(self, checkpoint_path: str):
        """Load SlideChat pathology-specific weights into the LLM."""
        try:
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            
            # Clean state dict (remove 'model.' prefix and vision components)
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("model."):
                    clean_key = key.replace("model.", "")
                    # Skip vision-only parameters
                    if "mm_projector" not in clean_key and "slide_encoder" not in clean_key:
                        new_state_dict[clean_key] = value
            
            # Load into LLM (strict=False to allow missing vision keys)
            if self.use_qlora:
                # For PEFT models, load into base_model
                msg = self.llm.base_model.model.load_state_dict(new_state_dict, strict=False)
            else:
                msg = self.llm.load_state_dict(new_state_dict, strict=False)
            
            print(f"✓ SlideChat weights loaded successfully")
            print(f"  - Missing keys: {len(msg.missing_keys)} (expected for vision components)")
            print(f"  - Unexpected keys: {len(msg.unexpected_keys)}")
            
        except Exception as e:
            warnings.warn(f"Failed to load SlideChat weights: {e}")
            print("Continuing with base Qwen 2.5 weights...")
    
    def get_trainable_params(self) -> Dict[str, torch.nn.Parameter]:
        """Get dictionary of trainable parameters for optimizer."""
        trainable = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable[name] = param
        return trainable
    
    def count_parameters(self) -> Tuple[int, int]:
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
    
    def forward(
        self,
        visual_features: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            visual_features: [batch, 768] slide embeddings
            input_ids: [batch, seq_len] tokenized text
            attention_mask: [batch, seq_len] attention mask
            labels: [batch, seq_len] labels for language modeling loss
            
        Returns:
            Dictionary with 'loss' and 'logits'
        """
        batch_size = visual_features.shape[0]
        
        # 1. Project visual features to tokens (ensure dtype matches LLM)
        visual_features = visual_features.to(self.llm_dtype)
        visual_tokens = self.projection(visual_features)  # [batch, num_visual_tokens, hidden_dim]
        
        # 2. Get text embeddings
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)  # [batch, seq_len, hidden_dim]
        
        # 3. Concatenate visual tokens before text
        combined_embeds = torch.cat([visual_tokens, inputs_embeds], dim=1)  # [batch, num_visual_tokens + seq_len, hidden_dim]
        
        # 4. Adjust attention mask to include visual tokens
        if attention_mask is not None:
            visual_attention = torch.ones(
                batch_size, visual_tokens.shape[1],
                dtype=attention_mask.dtype,
                device=attention_mask.device  
            )
            combined_attention_mask = torch.cat([visual_attention, attention_mask], dim=1)
        else:
            combined_attention_mask = None
        
        # 5. Adjust labels to include visual tokens (set to -100 for ignore)
        if labels is not None:
            visual_labels = torch.full(
                (batch_size, visual_tokens.shape[1]),
                -100,
                dtype=labels.dtype,
                device=labels.device
            )
            combined_labels = torch.cat([visual_labels, labels], dim=1)
        else:
            combined_labels = None
        
        # 6. Forward through LLM
        outputs = self.llm(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask,
            labels=combined_labels,
            return_dict=True
        )
        
        return {
            "loss": outputs.loss,
            "logits": outputs.logits
        }
    
    @torch.no_grad()
    def generate(
        self,
        visual_features: torch.Tensor,
        prompt: str = "Provide a concise single-paragraph pathology diagnosis:",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2
    ) -> str:
        """
        Generate report for a single WSI slide.
        
        Args:
            visual_features: [1, 768] or [768] slide embedding
            prompt: Text prompt to prepend
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            repetition_penalty: Penalty for repetition
            
        Returns:
            Generated report text
        """
        self.eval()
        
        # Ensure batch dimension
        if visual_features.dim() == 1:
            visual_features = visual_features.unsqueeze(0)
        
        # Project visual features (ensure dtype matches)
        visual_features = visual_features.to(self.device).to(self.llm_dtype)
        visual_tokens = self.projection(visual_features)
        
        # Tokenize prompt
        prompt_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = prompt_ids.input_ids.to(self.device)
        
        # Get prompt embeddings
        prompt_embeds = self.llm.get_input_embeddings()(input_ids)
        
        # Combine visual tokens + prompt
        combined_embeds = torch.cat([visual_tokens, prompt_embeds], dim=1)
        
        # Generate
        outputs = self.llm.generate(
            inputs_embeds=combined_embeds,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        # Decode (skip visual tokens in output)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove prompt from output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
    
    def merge_lora_weights(self):
        """
        Merge LoRA adapter weights into the base model.
        
        Converts a QLoRA model into a regular full-precision model by merging 
        the low-rank adaptation weights back into the base weights.
        After merging, the model can be saved/loaded without PEFT/QLoRA.
        
        Returns:
            VisionLanguageModel: Self (for chaining)
        """
        if not self.use_qlora:
            print("Model is not using QLoRA, nothing to merge")
            return self
        
        print("Merging LoRA weights into base model...")
        
        # Merge and unload LoRA weights
        self.llm = self.llm.merge_and_unload()
        
        # Update flag
        self.use_qlora = False
        
        print("✓ LoRA weights merged successfully")
        print("  Model is now a regular full-precision model")
        
        return self
    
    def save_checkpoint(self, save_path: str, epoch: int, optimizer_state: Optional[dict] = None):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'projection_state_dict': self.projection.state_dict(),
            'model_config': {
                'visual_dim': self.projection.projector[0].in_features,
                'num_visual_tokens': self.projection.num_visual_tokens,
                'use_qlora': self.use_qlora
            }
        }
        
        # Save LLM weights (for QLoRA, only save adapters)
        if self.use_qlora:
            checkpoint['lora_state_dict'] = self.llm.state_dict()
        elif not self.freeze_llm:
            checkpoint['llm_state_dict'] = self.llm.state_dict()
        
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
        
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")
    
    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = False):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load projection
        self.projection.load_state_dict(checkpoint['projection_state_dict'])
        print(f"✓ Loaded projection weights from {checkpoint_path}")
        
        # Load LLM/LoRA if present
        if 'lora_state_dict' in checkpoint and self.use_qlora:
            # Use strict=False to handle quantization metadata keys
            missing_keys, unexpected_keys = self.llm.load_state_dict(
                checkpoint['lora_state_dict'], strict=False
            )
            print(f"✓ Loaded LoRA weights")
            if len(unexpected_keys) > 0:
                print(f"  Note: {len(unexpected_keys)} unexpected keys (quantization metadata - this is normal)")
        elif 'llm_state_dict' in checkpoint and not self.freeze_llm:
            self.llm.load_state_dict(checkpoint['llm_state_dict'], strict=False)
            print(f"✓ Loaded LLM weights")
        
        optimizer_state = checkpoint.get('optimizer_state_dict') if load_optimizer else None
        
        return checkpoint['epoch'], optimizer_state


if __name__ == "__main__":
    # Test model initialization
    print("="*60)
    print("Testing VLM Model")
    print("="*60)
    
    # Stage 1: Projection only
    print("\n[Stage 1 Test] Projection layer training")
    model_stage1 = VisionLanguageModel(
        visual_dim=768,
        num_visual_tokens=32,
        use_qlora=False,
        freeze_llm=True
    )
    
    total, trainable = model_stage1.count_parameters()
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Trainable ratio: {100*trainable/total:.2f}%")
    
    # Test forward pass
    dummy_visual = torch.randn(2, 768).to(model_stage1.device)
    dummy_input_ids = torch.randint(0, 1000, (2, 50)).to(model_stage1.device)
    dummy_labels = dummy_input_ids.clone()
    
    outputs = model_stage1(
        visual_features=dummy_visual,
        input_ids=dummy_input_ids,
        labels=dummy_labels
    )
    print(f"\nForward pass successful!")
    print(f"Loss: {outputs['loss'].item():.4f}")
    print(f"Logits shape: {outputs['logits'].shape}")
    
    # Test generation
    print("\n[Testing Generation]")
    report = model_stage1.generate(
        visual_features=dummy_visual[0],
        prompt="Provide a concise single-paragraph pathology diagnosis:",
        max_new_tokens=50
    )
    print(f"Generated report: {report[:200]}...")
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)
