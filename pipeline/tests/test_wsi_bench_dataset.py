"""
Test script to verify WSI-Bench dataset loading and formatting.

This script tests:
1. Dataset loading from WSI-Bench JSON
2. Sample format (question, answer fields)
3. Spatial feature validation
4. Data collator prompt formatting
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import WSI_Report_Dataset
from data_collator import VLMDataCollator
from transformers import AutoTokenizer

def test_dataset_loading():
    """Test WSI-Bench dataset loading."""
    print("="*80)
    print("TEST 1: WSI-Bench Dataset Loading")
    print("="*80)
    
    # Load test dataset (smaller split)
    dataset_path = "/home/nadun/wd/datasets/wsi_bench/dataset/WSI-Bench-train-temp_Report_split0.1_test.json"
    
    print(f"\nLoading dataset from: {dataset_path}")
    dataset = WSI_Report_Dataset(
        json_path=dataset_path,
        dataset_format='wsi-bench',
        split=None,  # WSI-Bench files are already split
        use_knowledge_guidance=False
    )
    
    print(f"✓ Dataset loaded successfully!")
    print(f"  - Total samples: {len(dataset)}")
    
    # Check first sample
    print("\n" + "-"*80)
    print("First sample:")
    print("-"*80)
    sample = dataset[0]
    
    print(f"ID: {sample['id']}")
    print(f"Organ: {sample['organ']}")
    print(f"Slide embedding shape: {sample['slide_embedding'].shape}")
    print(f"\nQuestion: {sample['question'][:200]}...")
    print(f"\nAnswer: {sample['answer'][:200]}...")
    
    # Validate fields
    assert 'question' in sample, "Missing 'question' field"
    assert 'answer' in sample, "Missing 'answer' field"
    assert 'slide_embedding' in sample, "Missing 'slide_embedding' field"
    assert sample['question'] is not None, "Question should not be None for WSI-Bench"
    
    # Validate spatial features
    assert sample['slide_embedding'].dim() == 2, f"Expected 2D features, got {sample['slide_embedding'].dim()}D"
    assert sample['slide_embedding'].shape[1] == 768, f"Expected 768-dim features, got {sample['slide_embedding'].shape[1]}"
    
    print(f"\n✓ All field validations passed!")
    return dataset


def test_collator_formatting():
    """Test data collator prompt formatting."""
    print("\n" + "="*80)
    print("TEST 2: Data Collator Prompt Formatting")
    print("="*80)
    
    # Load tokenizer
    tokenizer_path = "/home/nadun/wd/vlm_histo/checkpoints/Qwen2.5-7B-Instruct"
    print(f"\nLoading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Add special tokens
    tokenizer.add_tokens(['<image_start>', '<image_end>'], special_tokens=True)
    print("✓ Tokenizer loaded and special tokens added")
    
    # Create collator
    print("\nCreating VLMDataCollator with dataset_format='wsi-bench'...")
    collator = VLMDataCollator(
        tokenizer=tokenizer,
        dataset_format='wsi-bench',
        use_knowledge_guidance=False
    )
    print("✓ Data collator created")
    
    # Load dataset
    dataset = WSI_Report_Dataset(
        json_path="/home/nadun/wd/datasets/wsi_bench/dataset/WSI-Bench-train-temp_Report_split0.1_test.json",
        dataset_format='wsi-bench',
        split=None
    )
    
    # Create a small batch
    print("\nCreating test batch with 2 samples...")
    batch = [dataset[0], dataset[1]]
    collated = collator(batch)
    
    print(f"✓ Batch collated successfully")
    print(f"\nBatch structure:")
    print(f"  - visual_features shape: {collated['visual_features'].shape}")
    print(f"  - input_ids shape: {collated['input_ids'].shape}")
    print(f"  - attention_mask shape: {collated['attention_mask'].shape}")
    print(f"  - labels shape: {collated['labels'].shape}")
    print(f"  - Number of IDs: {len(collated['ids'])}")
    
    # Decode first prompt to verify format
    print("\n" + "-"*80)
    print("Decoded prompt (first sample):")
    print("-"*80)
    decoded = tokenizer.decode(collated['input_ids'][0], skip_special_tokens=False)
    print(decoded[:800] + "...")
    
    # Validate prompt structure
    assert '<|im_start|>' in decoded, "Missing ChatML markers"
    assert '<image_start>' in decoded, "Missing <image_start> token"
    assert '<image_end>' in decoded, "Missing <image_end> token"
    assert batch[0]['question'] in decoded, "Question not found in prompt"
    
    print("\n✓ All prompt validations passed!")
    print(f"✓ Confirmed: Dataset questions are being used (not fixed prompts)")
    

def test_histgen_compatibility():
    """Test backward compatibility with HistGen format."""
    print("\n" + "="*80)
    print("TEST 3: HistGen Format Compatibility (Backward Compatibility)")
    print("="*80)
    
    print("\nNote: This test requires a HistGen format dataset.")
    print("Skipping for now (WSI-Bench only workflow)")
    print("✓ WSI-Bench format is primary target")


if __name__ == "__main__":
    try:
        # Run tests
        dataset = test_dataset_loading()
        test_collator_formatting()
        test_histgen_compatibility()
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80)
        print("\nWSI-Bench dataset integration is working correctly!")
        print(f"Ready to use dataset_format='wsi-bench' in training scripts.")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
