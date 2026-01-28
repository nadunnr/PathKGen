"""
Data collator for Vision-Language Model training.

Handles batching of slide features and text reports with proper padding.
"""

import torch
from typing import Dict, List, Any
from transformers import PreTrainedTokenizer
from config import TRAINING_PROMPT_TEMPLATE


class VLMDataCollator:
    """
    Collates batches for VLM training.
    
    Handles:
    - Variable-length text sequences
    - Proper padding and attention masks
    - Label preparation for causal LM loss
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        padding: str = "longest",
        prompt_template: str = None,  # Will use config default if None
        use_knowledge_guidance: bool = False,
        max_knowledge_tokens: int = 64
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            padding: Padding strategy ('longest' or 'max_length')
            prompt_template: Template for formatting reports (uses config default if None)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.prompt_template = prompt_template if prompt_template is not None else TRAINING_PROMPT_TEMPLATE
        self.use_knowledge_guidance = use_knowledge_guidance
        self.max_knowledge_tokens = max_knowledge_tokens
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples.
        
        Args:
            batch: List of dicts from dataset with keys:
                - 'slide_embedding': [768] tensor
                - 'report': str
                - 'id': str
                - 'organ': str
                
        Returns:
            Dictionary with:
                - 'visual_features': [batch, 768]
                - 'input_ids': [batch, seq_len]
                - 'attention_mask': [batch, seq_len]
                - 'labels': [batch, seq_len]
                - 'ids': List[str]
        """
        # Extract slide features
        visual_features = torch.stack([item['slide_embedding'] for item in batch])
        
        # Format reports with prompt template
        reports = [
            self.prompt_template.format(report=item['report'])
            for item in batch
        ]
        
        # Tokenize reports
        tokenized = self.tokenizer(
            reports,
            padding=self.padding,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Prepare labels (shift will be handled by the model)
        # Set pad tokens to -100 so they're ignored in loss
        labels = tokenized['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        # Collect metadata
        ids = [item['id'] for item in batch]
        
        # Prepare result dictionary
        result = {
            'visual_features': visual_features,
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': labels,
            'ids': ids
        }
        
        # Add knowledge tokens if guidance is enabled
        if self.use_knowledge_guidance:
            # Extract knowledge texts (without prompt template formatting)
            knowledge_texts = [item['knowledge_text'] for item in batch]
            knowledge_tokens = self.tokenizer(
                knowledge_texts,
                padding=self.padding,
                truncation=True,
                max_length=self.max_knowledge_tokens,
                return_tensors="pt"
            )
            result['knowledge_input_ids'] = knowledge_tokens['input_ids']
            result['knowledge_attention_mask'] = knowledge_tokens['attention_mask']
        
        return result


class VLMInferenceCollator:
    """
    Collator for inference (no labels needed).
    """
    
    def __init__(
        self,
        prompt: str = "Generate a detailed pathology report for this whole slide image:\n\n"
    ):
        self.prompt = prompt
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate batch for inference.
        
        Returns:
            - 'visual_features': [batch, 768]
            - 'ids': List[str]
            - 'organs': List[str]
            - 'prompt': str
        """
        visual_features = torch.stack([item['slide_embedding'] for item in batch])
        ids = [item['id'] for item in batch]
        organs = [item.get('organ', 'unknown') for item in batch]
        
        return {
            'visual_features': visual_features,
            'ids': ids,
            'organs': organs,
            'prompt': self.prompt
        }


if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    # Test collator
    print("Testing VLMDataCollator...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        "/home/nadun/wd/vlm_histo/checkpoints/Qwen2.5-7B-Instruct",
        trust_remote_code=True
    )
    
    collator = VLMDataCollator(tokenizer=tokenizer, max_length=256)
    
    # Create dummy batch
    dummy_batch = [
        {
            'id': 'slide_001',
            'organ': 'breast',
            'slide_embedding': torch.randn(768),
            'report': 'This is a test pathology report number one.'
        },
        {
            'id': 'slide_002',
            'organ': 'breast',
            'slide_embedding': torch.randn(768),
            'report': 'Another pathology report with different length to test padding.'
        }
    ]
    
    # Collate
    batch = collator(dummy_batch)
    
    print(f"\nCollated batch:")
    print(f"  Visual features shape: {batch['visual_features'].shape}")
    print(f"  Input IDs shape: {batch['input_ids'].shape}")
    print(f"  Attention mask shape: {batch['attention_mask'].shape}")
    print(f"  Labels shape: {batch['labels'].shape}")
    print(f"  IDs: {batch['ids']}")
    
    # Decode to verify
    print(f"\nSample decoded text:")
    print(tokenizer.decode(batch['input_ids'][0], skip_special_tokens=False))
    
    print("\n✓ Collator test passed!")
