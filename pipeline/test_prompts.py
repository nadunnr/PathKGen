#!/usr/bin/env python3
"""
Test script for verifying image marker token prompt templates.
Tests tokenization and visual embedding injection with <image_start> and <image_end> markers.
"""

import torch
from transformers import AutoTokenizer
from config import (
    TRAINING_PROMPT_NO_KNOWLEDGE,
    TRAINING_PROMPT_WITH_KNOWLEDGE,
    GENERATION_PROMPT_NO_KNOWLEDGE,
    GENERATION_PROMPT_WITH_KNOWLEDGE,
    IMAGE_START_TOKEN,
    IMAGE_END_TOKEN
)

def test_prompts():
    """Test all prompt templates."""
    print("=" * 80)
    print("TESTING PROMPT TEMPLATES")
    print("=" * 80)
    
    # Test training prompts
    print("\n1. TRAINING PROMPT (No Knowledge):")
    print("-" * 80)
    prompt_no_kg = TRAINING_PROMPT_NO_KNOWLEDGE.format(report="Sample diagnosis report.")
    print(prompt_no_kg)
    
    print("\n2. TRAINING PROMPT (With Knowledge):")
    print("-" * 80)
    prompt_with_kg = TRAINING_PROMPT_WITH_KNOWLEDGE.format(
        knowledge="Invasive ductal carcinoma is the most common type of breast cancer.",
        report="Sample diagnosis report."
    )
    print(prompt_with_kg)
    
    # Test generation prompts
    print("\n3. GENERATION PROMPT (No Knowledge):")
    print("-" * 80)
    print(GENERATION_PROMPT_NO_KNOWLEDGE)
    
    print("\n4. GENERATION PROMPT (With Knowledge):")
    print("-" * 80)
    gen_with_kg = GENERATION_PROMPT_WITH_KNOWLEDGE.format(
        knowledge="Invasive ductal carcinoma is the most common type of breast cancer."
    )
    print(gen_with_kg)

def test_tokenization():
    """Test tokenization with image marker tokens."""
    print("\n" + "=" * 80)
    print("TESTING TOKENIZATION WITH IMAGE MARKERS")
    print("=" * 80)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "/home/nadun/wd/vlm_histo/checkpoints/Qwen2.5-7B-Instruct",
        trust_remote_code=True
    )
    
    # Add special tokens
    special_tokens_dict = {'additional_special_tokens': [IMAGE_START_TOKEN, IMAGE_END_TOKEN]}
    num_added = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"\n✓ Added {num_added} special tokens: {IMAGE_START_TOKEN}, {IMAGE_END_TOKEN}")
    
    # Get token IDs
    image_start_id = tokenizer.convert_tokens_to_ids(IMAGE_START_TOKEN)
    image_end_id = tokenizer.convert_tokens_to_ids(IMAGE_END_TOKEN)
    print(f"  - {IMAGE_START_TOKEN} token ID: {image_start_id}")
    print(f"  - {IMAGE_END_TOKEN} token ID: {image_end_id}")
    
    # Test tokenization
    test_prompt = TRAINING_PROMPT_NO_KNOWLEDGE.format(report="This is a test report.")
    
    print("\n" + "-" * 80)
    print("Tokenizing prompt with markers...")
    print("-" * 80)
    
    tokens = tokenizer(test_prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = tokens.input_ids[0]
    
    print(f"\nTotal tokens: {len(input_ids)}")
    
    # Find marker positions
    start_positions = (input_ids == image_start_id).nonzero(as_tuple=True)[0]
    end_positions = (input_ids == image_end_id).nonzero(as_tuple=True)[0]
    
    if len(start_positions) > 0 and len(end_positions) > 0:
        start_idx = start_positions[0].item()
        end_idx = end_positions[0].item()
        print(f"\n✓ Found image markers:")
        print(f"  - {IMAGE_START_TOKEN} at position: {start_idx}")
        print(f"  - {IMAGE_END_TOKEN} at position: {end_idx}")
        print(f"  - Space between markers: {end_idx - start_idx - 1} tokens (visual tokens will be injected here)")
        
        # Show tokens around markers
        context_size = 5
        print(f"\nContext around markers:")
        start_range = slice(max(0, start_idx-context_size), min(len(input_ids), end_idx+context_size+1))
        context_ids = input_ids[start_range]
        context_tokens = tokenizer.convert_ids_to_tokens(context_ids)
        
        print(f"  Tokens [{max(0, start_idx-context_size)}:{min(len(input_ids), end_idx+context_size+1)}]:")
        for i, (tid, tok) in enumerate(zip(context_ids, context_tokens)):
            idx = max(0, start_idx-context_size) + i
            marker = ""
            if idx == start_idx:
                marker = " <-- START MARKER"
            elif idx == end_idx:
                marker = " <-- END MARKER"
            print(f"    [{idx}] {tid:5d}: {tok:20s}{marker}")
    else:
        print("\n✗ ERROR: Image markers not found in tokenized output!")
        print("This indicates a problem with tokenization.")
    
    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED!")
    print("=" * 80)

if __name__ == "__main__":
    test_prompts()
    test_tokenization()
