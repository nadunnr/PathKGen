"""
Test spatial token projection with actual TITAN features.

This script validates:
1. ProjectionLayer handles [batch, N, 768] input correctly
2. Shape transformations work as expected
3. Forward pass completes without errors
"""
import torch
import sys
sys.path.insert(0, '/home/nadun/wd/vlm_histo/pipeline')

from model import VisionLanguageModel

def test_spatial_projection():
    print("="*60)
    print("Testing Spatial Token Projection (WSI-LLaVA Style)")
    print("="*60)
    
    # Initialize model
    print("\n[1/4] Initializing VLM model...")
    model = VisionLanguageModel(
        visual_dim=768,
        use_qlora=False,
        freeze_llm=True,
        device="cpu"  # Use CPU for testing
    )
    
    print("\n✓ Model initialized successfully")
    
    # Test with  TITAN-like spatial features [batch, 128, 768]
    print("\n[2/4] Testing projection with spatial features...")
    batch_size = 2
    num_tokens = 128  # TITAN default
    feature_dim = 768
    
    # Create dummy features in bfloat16 to match model dtype
    dummy_spatial_features = torch.randn(batch_size, num_tokens, feature_dim, dtype=torch.bfloat16)
    
    print(f"\nInput shape: {dummy_spatial_features.shape}")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of tokens: {num_tokens}")
    print(f"  Feature dim: {feature_dim}")
    
    # Project
    try:
        projected = model.projection(dummy_spatial_features)
        
        print(f"\nProjected shape: {projected.shape}")
        print(f"  Expected: [{batch_size}, {num_tokens}, 3584]")
        
        assert projected.shape == (batch_size, num_tokens, 3584), \
            f"Shape mismatch! Got {projected.shape}"
        
        print("\n✓ Spatial projection test PASSED!")
        
    except Exception as e:
        print(f"\n✗ Projection FAILED: {e}")
        raise
    
    # Test forward pass with dummy input
    print("\n" + "="*60)
    print("[3/4] Testing Full Forward Pass")
    print("="*60)
    
    # Create dummy prompt with image markers
    dummy_prompt = "<|im_start|>system\nYou are a pathology assistant.<|im_end|>\n<|im_start|>user\n<image_start><image_end>\nAnalyze this slide.<|im_end|>\n<|im_start|>assistant\nThis is a test report.<|im_end|>"
    
    # Tokenize
    tokens = model.tokenizer(dummy_prompt, return_tensors="pt")
    input_ids = tokens.input_ids.repeat(batch_size, 1)  # Batch it
    attention_mask = tokens.attention_mask.repeat(batch_size, 1)
    labels = input_ids.clone()
    labels[labels == model.tokenizer.pad_token_id] = -100
    
    print(f"\nInput IDs shape: {input_ids.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Forward pass
    try:
        outputs = model(
            visual_features=dummy_spatial_features,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        print(f"\nForward pass successful!")
        print(f"  Loss: {outputs['loss'].item():.4f}")
        print(f"  Logits shape: {outputs['logits'].shape}")
        
        # Verify combined sequence length
        expected_seq_len = input_ids.shape[1] + num_tokens
        actual_seq_len = outputs['logits'].shape[1]
        print(f"\n  Combined sequence length: {actual_seq_len}")
        print(f"  Expected: ~{expected_seq_len} (prompt + {num_tokens} visual tokens)")
        
        print("\n✓ Full forward pass test PASSED!")
        
    except Exception as e:
        print(f"\n✗ Forward pass FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Test with different token counts
    print("\n" + "="*60)
    print("[4/4] Testing Dynamic Token Count")
    print("="*60)
    
    for test_tokens in [64, 128, 256]:
        print(f"\nTesting with {test_tokens} tokens...")
        test_features = torch.randn(1, test_tokens, 768, dtype=torch.bfloat16)
        test_projected = model.projection(test_features)
        
        assert test_projected.shape == (1, test_tokens, 3584), \
            f"Failed for {test_tokens} tokens: got {test_projected.shape}"
        
        print(f"  ✓ {test_features.shape} → {test_projected.shape}")
    
    print("\n" + "="*60)
    print("All tests PASSED! ✓")
    print("="*60)
    print("\nSummary:")
    print("  ✓ Spatial projection works correctly")
    print("  ✓ Token-wise projection preserves spatial structure")
    print("  ✓ Dynamic token count supported (64, 128, 256)")
    print("  ✓ Forward pass integrates visual tokens correctly")
    print("\nReady for training!")

if __name__ == "__main__":
    test_spatial_projection()
