#!/usr/bin/env python3
"""
Comprehensive test suite for VLM pathology diagnosis pipeline.
Tests all critical functionality before training.
"""

import torch
from model import VisionLanguageModel
from config import (
    TRAINING_PROMPT_NO_KNOWLEDGE,
    TRAINING_PROMPT_WITH_KNOWLEDGE, 
    GENERATION_PROMPT_NO_KNOWLEDGE,
    GENERATION_PROMPT_WITH_KNOWLEDGE
)

def test_model():
    print("="*80)
    print("VLM PIPELINE COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    # ========================================================================
    # TEST 1: Model Initialization
    # ========================================================================
    print("\n[TEST 1: Model Initialization]")
    print("-" * 80)
    
    model = VisionLanguageModel(
        visual_dim=768,
        num_visual_tokens=64,
        use_qlora=False,
        freeze_llm=True
    )
    
    total, trainable = model.count_parameters()
    print(f"✓ Model initialized successfully")
    print(f"  Total parameters: {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  Trainable ratio: {100*trainable/total:.2f}%")
    print(f"  Device: {model.device}")
    
    # Verify special tokens
    assert model.image_start_token_id is not None, "Image start token not set"
    assert model.image_end_token_id is not None, "Image end token not set"
    print(f"  Image start token ID: {model.image_start_token_id}")
    print(f"  Image end token ID: {model.image_end_token_id}")
    
    # ========================================================================
    # TEST 2: Forward Pass with Image Markers
    # ========================================================================
    print("\n[TEST 2: Forward Pass with Image Marker Injection]")
    print("-" * 80)
    
    sample_report = "The specimen shows invasive ductal carcinoma with moderate nuclear atypia."
    prompt = TRAINING_PROMPT_NO_KNOWLEDGE.format(report=sample_report)
    
    # Tokenize
    tokenized = model.tokenizer(
        prompt,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)
    
    input_ids = tokenized.input_ids
    
    print(f"  Input sequence length: {input_ids.shape[1]} tokens")
    
    # Find markers
    start_positions = (input_ids[0] == model.image_start_token_id).nonzero(as_tuple=True)[0]
    end_positions = (input_ids[0] == model.image_end_token_id).nonzero(as_tuple=True)[0]
    
    assert len(start_positions) > 0, "Image start marker not found"
    assert len(end_positions) > 0, "Image end marker not found"
    
    start_idx = start_positions[0].item()
    end_idx = end_positions[0].item()
    
    print(f"  ✓ Markers found: start={start_idx}, end={end_idx}")
    assert end_idx == start_idx + 1, "Markers not adjacent!"
    print(f"  ✓ Markers adjacent (single tokens confirmed)")
    
    # Forward pass
    visual_features = torch.randn(1, 768).to(model.device)
    labels = input_ids.clone()
    
    outputs = model(
        visual_features=visual_features,
        input_ids=input_ids,
        labels=labels
    )
    
    expected_length = input_ids.shape[1] + 64  # Original + visual tokens
    actual_length = outputs['logits'].shape[1]
    
    print(f"  Input length: {input_ids.shape[1]}")
    print(f"  Expected output: {expected_length}")
    print(f"  Actual output: {actual_length}")
    assert actual_length == expected_length, f"Length mismatch!"
    print(f"  ✓ Sequence length correct")
    print(f"  ✓ Loss: {outputs['loss'].item():.4f}")
    print(f"  ✓ Logits shape: {outputs['logits'].shape}")
    
    # ========================================================================
    # TEST 3: Batch Forward Pass
    # ========================================================================
    print("\n[TEST 3: Batch Processing (2 samples)]")
    print("-" * 80)
    
    reports = [
        "Invasive ductal carcinoma detected.",
        "Benign fibrocystic changes noted throughout the tissue."
    ]
    
    prompts = [TRAINING_PROMPT_NO_KNOWLEDGE.format(report=r) for r in reports]
    
    batch_tokenized = model.tokenizer(
        prompts,
        padding="longest",
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)
    
    batch_visual = torch.randn(2, 768).to(model.device)
    batch_labels = batch_tokenized.input_ids.clone()
    
    batch_outputs = model(
        visual_features=batch_visual,
        input_ids=batch_tokenized.input_ids,
        labels=batch_labels
    )
    
    print(f"  Batch size: 2")
    print(f"  Input shape: {batch_tokenized.input_ids.shape}")
    print(f"  Output shape: {batch_outputs['logits'].shape}")
    assert batch_outputs['logits'].shape[0] == 2, "Batch dimension mismatch"
    print(f"  ✓ Batch processing successful")
    print(f"  ✓ Loss: {batch_outputs['loss'].item():.4f}")
    
    # ========================================================================
    # TEST 4: Generation (No Knowledge)
    # ========================================================================
    print("\n[TEST 4: Generation without Knowledge]")
    print("-" * 80)
    
    gen_visual = torch.randn(1, 768).to(model.device)
    
    generated_report = model.generate(
        visual_features=gen_visual,
        prompt=None,
        knowledge_text=None,
        max_new_tokens=50,
        temperature=0.7
    )
    
    print(f"  Generated length: {len(generated_report)} chars")
    print(f"  Generated (first 120 chars):")
    print(f"    \"{generated_report[:120]}...\"")
    assert len(generated_report) > 0, "Empty generation"
    print(f"  ✓ Generation successful")
    
    # ========================================================================
    # TEST 5: Generation (With Knowledge)
    # ========================================================================
    print("\n[TEST 5: Generation with Knowledge]")
    print("-" * 80)
    
    knowledge = "Invasive ductal carcinoma is the most common breast cancer type."
    
    generated_with_kg = model.generate(
        visual_features=gen_visual,
        knowledge_text=knowledge,
        max_new_tokens=50,
        temperature=0.7
    )
    
    print(f"  Knowledge: \"{knowledge}\"")
    print(f"  Generated length: {len(generated_with_kg)} chars")
    print(f"  Generated (first 120 chars):")
    print(f"    \"{generated_with_kg[:120]}...\"")
    assert len(generated_with_kg) > 0, "Empty generation"
    print(f"  ✓ Generation with knowledge successful")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("✅ ALL 5 TESTS PASSED!")
    print("="*80)
    print("\nPipeline verified and ready for training:")
    print("  ✓ Model initialization with special tokens")
    print("  ✓ Image marker token injection (vectorized)")
    print("  ✓ Batch processing")
    print("  ✓ Generation without knowledge")
    print("  ✓ Generation with knowledge")
    print("\n🚀 You can now proceed to training!")
    print("="*80)

if __name__ == "__main__":
    test_model()
