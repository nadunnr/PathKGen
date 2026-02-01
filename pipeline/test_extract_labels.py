#!/usr/bin/env python3
"""
Test script to verify extract_report_from_labels function works correctly
with ChatML formatted prompts.
"""

import torch
from transformers import AutoTokenizer
from config import (
    TRAINING_PROMPT_NO_KNOWLEDGE,
    TRAINING_PROMPT_WITH_KNOWLEDGE,
    extract_report_from_labels
)

def test_extraction():
    print("="*80)
    print("Testing extract_report_from_labels Function")
    print("="*80)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "/home/nadun/wd/vlm_histo/checkpoints/Qwen2.5-7B-Instruct",
        trust_remote_code=True
    )
    
    # Test case 1: No knowledge
    print("\n[TEST 1: No Knowledge Template]")
    print("-" * 80)
    
    ground_truth_report = "The specimen shows invasive ductal carcinoma with moderate nuclear atypia and scattered mitotic figures."
    prompt_no_kg = TRAINING_PROMPT_NO_KNOWLEDGE.format(report=ground_truth_report)
    
    # Tokenize and simulate label creation (as done in data collator)
    tokens = tokenizer(prompt_no_kg, add_special_tokens=False, return_tensors="pt")
    input_ids = tokens.input_ids[0]
    
    # Simulate filtering -100 (in real scenario, some positions would be -100)
    # For testing, we keep all tokens
    clean_input_ids = input_ids[input_ids != -100]
    
    # Extract report
    extracted_report = extract_report_from_labels(tokenizer, clean_input_ids)
    
    print(f"Ground truth: {ground_truth_report}")
    print(f"Extracted:    {extracted_report}")
    
    # Check if they match
    match = extracted_report.strip() == ground_truth_report.strip()
    print(f"\n✓ Match: {match}")
    
    if not match:
        print("WARNING: Extracted report doesn't match!")
        print(f"Length GT: {len(ground_truth_report)}, Extracted: {len(extracted_report)}")
    
    # Test case 2: With knowledge
    print("\n[TEST 2: With Knowledge Template]")
    print("-" * 80)
    
    ground_truth_report2 = "This tissue sample demonstrates benign fibrocystic changes."
    knowledge_text = "Fibrocystic changes are benign breast alterations."
    prompt_with_kg = TRAINING_PROMPT_WITH_KNOWLEDGE.format(
        knowledge=knowledge_text,
        report=ground_truth_report2
    )
    
    # Tokenize
    tokens2 = tokenizer(prompt_with_kg, add_special_tokens=False, return_tensors="pt")
    input_ids2 = tokens2.input_ids[0]
    clean_input_ids2 = input_ids2[input_ids2 != -100]
    
    # Extract report
    extracted_report2 = extract_report_from_labels(tokenizer, clean_input_ids2)
    
    print(f"Ground truth: {ground_truth_report2}")
    print(f"Extracted:    {extracted_report2}")
    
    # Check if they match
    match2 = extracted_report2.strip() == ground_truth_report2.strip()
    print(f"\n✓ Match: {match2}")
    
    if not match2:
        print("WARNING: Extracted report doesn't match!")
        print(f"Length GT: {len(ground_truth_report2)}, Extracted: {len(extracted_report2)}")
    
    # Test case 3: Long report
    print("\n[TEST 3: Long Report]")
    print("-" * 80)
    
    long_report = (
        "The specimen demonstrates invasive ductal carcinoma, grade 2/3, with extensive "
        "lymphovascular invasion. Tumor cells show moderate nuclear pleomorphism and "
        "increased mitotic activity. Margins are negative for malignancy with the closest "
        "margin measuring 0.5 cm. Three out of five sentinel lymph nodes show metastatic "
        "carcinoma with extracapsular extension."
    )
    
    prompt_long = TRAINING_PROMPT_NO_KNOWLEDGE.format(report=long_report)
    tokens3 = tokenizer(prompt_long, add_special_tokens=False, return_tensors="pt")
    input_ids3 = tokens3.input_ids[0]
    clean_input_ids3 = input_ids3[input_ids3 != -100]
    
    extracted_long = extract_report_from_labels(tokenizer, clean_input_ids3)
    
    print(f"Ground truth length: {len(long_report)} chars")
    print(f"Extracted length:    {len(extracted_long)} chars")
    print(f"Ground truth (first 100): {long_report[:100]}...")
    print(f"Extracted (first 100):    {extracted_long[:100]}...")
    
    match3 = extracted_long.strip() == long_report.strip()
    print(f"\n✓ Match: {match3}")
    
    # Summary
    print("\n" + "="*80)
    if match and match2 and match3:
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        print("\nThe extract_report_from_labels function works correctly.")
        print("NLU evaluation should now properly extract ground truth reports.")
    else:
        print("⚠️  SOME TESTS FAILED!")
        print("="*80)
        failed = []
        if not match: failed.append("Test 1 (No Knowledge)")
        if not match2: failed.append("Test 2 (With Knowledge)")
        if not match3: failed.append("Test 3 (Long Report)")
        print(f"Failed tests: {', '.join(failed)}")

if __name__ == "__main__":
    test_extraction()
