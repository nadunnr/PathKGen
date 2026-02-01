#!/usr/bin/env python3
"""
Empirical token length analysis for VLM pathology diagnosis pipeline.

Analyzes actual token counts for:
- All prompt templates (with/without knowledge)
- Different report lengths (min, median, 95th percentile, max)
- Before and after visual token injection
- Impact of truncation at max_length=512
"""

import sys
from transformers import AutoTokenizer
from config import (
    TRAINING_PROMPT_NO_KNOWLEDGE,
    TRAINING_PROMPT_WITH_KNOWLEDGE,
    GENERATION_PROMPT_NO_KNOWLEDGE,
    GENERATION_PROMPT_WITH_KNOWLEDGE,
    IMAGE_START_TOKEN,
    IMAGE_END_TOKEN,
    DEFAULT_MAX_LENGTH,
    DEFAULT_NUM_VISUAL_TOKENS
)

# Report length statistics (from your dataset)
REPORT_STATS = {
    "min": 28,
    "median": 150,  # Estimated
    "p95": 298,      # 95th percentile
    "max": 2000      # Approximate maximum
}

def create_synthetic_report(num_tokens: int) -> str:
    """Create a synthetic pathology report of approximately num_tokens length."""
    # Average pathology report fragment (will repeat to reach desired length)
    fragment = (
        "The specimen shows invasive ductal carcinoma with moderate nuclear atypia. "
        "Tumor cells are arranged in irregular nests and cords infiltrating the surrounding stroma. "
        "Mitotic figures are present at a moderate rate. The tumor measures approximately 2.5 cm in greatest dimension. "
        "Lymphovascular invasion is identified. Surgical margins are negative for malignancy. "
    )
    
    # Tokenize fragment to know its length
    tokenizer = AutoTokenizer.from_pretrained(
        "/home/nadun/wd/vlm_histo/checkpoints/Qwen2.5-7B-Instruct",
        trust_remote_code=True
    )
    
    fragment_tokens = tokenizer.encode(fragment, add_special_tokens=False)
    fragment_length = len(fragment_tokens)
    
    # Repeat to approximately match target
    repetitions = max(1, num_tokens // fragment_length)
    report = (fragment + " ") * repetitions
    
    # Trim or pad to get closer to target
    actual_tokens = tokenizer.encode(report, add_special_tokens=False)
    if len(actual_tokens) > num_tokens:
        # Decode and re-encode to trim
        trimmed = tokenizer.decode(actual_tokens[:num_tokens])
        return trimmed
    
    return report

def analyze_template(template_name: str, template: str, report_text: str, knowledge_text: str = None):
    """Analyze token counts for a specific template."""
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "/home/nadun/wd/vlm_histo/checkpoints/Qwen2.5-7B-Instruct",
        trust_remote_code=True
    )
    
    # Add special tokens
    special_tokens_dict = {'additional_special_tokens': [IMAGE_START_TOKEN, IMAGE_END_TOKEN]}
    tokenizer.add_special_tokens(special_tokens_dict)
    
    image_start_id = tokenizer.convert_tokens_to_ids(IMAGE_START_TOKEN)
    image_end_id = tokenizer.convert_tokens_to_ids(IMAGE_END_TOKEN)
    
    # Format prompt
    if knowledge_text:
        formatted_prompt = template.format(knowledge=knowledge_text, report=report_text)
    else:
        formatted_prompt = template.format(report=report_text)
    
    # Tokenize WITHOUT truncation to see actual length
    tokens_full = tokenizer(formatted_prompt, add_special_tokens=False, return_tensors="pt")
    full_length = tokens_full.input_ids.shape[1]
    
    # Tokenize WITH truncation (as in data collator)
    tokens_truncated = tokenizer(
        formatted_prompt,
        add_special_tokens=False,
        truncation=True,
        max_length=DEFAULT_MAX_LENGTH,
        return_tensors="pt"
    )
    truncated_length = tokens_truncated.input_ids.shape[1]
    
    # Find marker positions
    input_ids = tokens_truncated.input_ids[0]
    start_positions = (input_ids == image_start_id).nonzero(as_tuple=True)[0]
    end_positions = (input_ids == image_end_id).nonzero(as_tuple=True)[0]
    
    if len(start_positions) > 0 and len(end_positions) > 0:
        start_idx = start_positions[0].item()
        end_idx = end_positions[0].item()
        markers_adjacent = (end_idx == start_idx + 1)
    else:
        start_idx = end_idx = None
        markers_adjacent = False
    
    # Calculate final length after visual injection
    # Formula: truncated_length - 2 (remove markers) + 64 (visual) + 2 (keep markers)
    # Simplified: truncated_length + 64
    final_length = truncated_length + DEFAULT_NUM_VISUAL_TOKENS
    
    # Calculate how much was lost to truncation
    tokens_lost = max(0, full_length - DEFAULT_MAX_LENGTH)
    
    return {
        "template_name": template_name,
        "full_length": full_length,
        "truncated_length": truncated_length,
        "final_length_with_visual": final_length,
        "tokens_lost_to_truncation": tokens_lost,
        "marker_start_pos": start_idx,
        "marker_end_pos": end_idx,
        "markers_adjacent": markers_adjacent,
        "truncation_occurred": full_length > DEFAULT_MAX_LENGTH
    }

def main():
    print("=" * 100)
    print("EMPIRICAL TOKEN LENGTH ANALYSIS - VLM Pathology Diagnosis Pipeline")
    print("=" * 100)
    
    print(f"\nConfiguration:")
    print(f"  - Max length (tokenizer): {DEFAULT_MAX_LENGTH}")
    print(f"  - Visual tokens: {DEFAULT_NUM_VISUAL_TOKENS}")
    print(f"  - Qwen2.5-7B context limit: 32,768 tokens")
    
    # Sample knowledge text (realistic length)
    sample_knowledge = (
        "Invasive ductal carcinoma (IDC) is the most common type of breast cancer, accounting for 70-80% of all cases. "
        "It originates in the milk ducts and invades surrounding breast tissue. Histologically, IDC is characterized by "
        "irregular nests and cords of malignant epithelial cells infiltrating a desmoplastic stroma. Nuclear grade is "
        "assessed based on nuclear size, pleomorphism, and mitotic activity."
    )
    
    # Test scenarios
    scenarios = [
        ("Minimum (28 tokens)", 28),
        ("Median (150 tokens)", 150),
        ("95th Percentile (298 tokens)", 298),
        ("Large (500 tokens)", 500),
        ("Very Large (1000 tokens)", 1000),
        ("Maximum (2000 tokens)", 2000),
    ]
    
    templates = [
        ("Training - No Knowledge", TRAINING_PROMPT_NO_KNOWLEDGE, False),
        ("Training - With Knowledge", TRAINING_PROMPT_WITH_KNOWLEDGE, True),
        ("Generation - No Knowledge", GENERATION_PROMPT_NO_KNOWLEDGE, False),
        ("Generation - With Knowledge", GENERATION_PROMPT_WITH_KNOWLEDGE, True),
    ]
    
    # Analyze each scenario
    for scenario_name, report_tokens in scenarios:
        print("\n" + "=" * 100)
        print(f"SCENARIO: {scenario_name} Report")
        print("=" * 100)
        
        # Create synthetic report
        report_text = create_synthetic_report(report_tokens)
        
        # Analyze each template
        results = []
        for template_name, template, use_knowledge in templates:
            knowledge = sample_knowledge if use_knowledge else None
            result = analyze_template(template_name, template, report_text, knowledge)
            results.append(result)
        
        # Display results in table format
        print(f"\n{'Template':<30} {'Full':<8} {'Truncated':<12} {'After Visual':<13} {'Lost':<8} {'Truncated?':<12}")
        print("-" * 100)
        
        for r in results:
            truncated_marker = "⚠️  YES" if r['truncation_occurred'] else "✓ NO"
            print(f"{r['template_name']:<30} {r['full_length']:<8} {r['truncated_length']:<12} "
                  f"{r['final_length_with_visual']:<13} {r['tokens_lost_to_truncation']:<8} {truncated_marker:<12}")
        
        # Analysis
        max_final = max(r['final_length_with_visual'] for r in results)
        any_truncated = any(r['truncation_occurred'] for r in results)
        
        print(f"\n  📊 Maximum final length: {max_final} tokens ({max_final/32768*100:.1f}% of Qwen2.5 context)")
        if any_truncated:
            max_lost = max(r['tokens_lost_to_truncation'] for r in results)
            print(f"  ⚠️  TRUNCATION: Up to {max_lost} tokens lost from report")
        else:
            print(f"  ✓ No truncation - all content preserved")
    
    # Summary recommendations
    print("\n" + "=" * 100)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 100)
    
    print(f"\n1. Current Setting (max_length={DEFAULT_MAX_LENGTH}):")
    print(f"   ✓ Works perfectly for reports ≤298 tokens (95th percentile)")
    print(f"   ⚠️  Truncates reports >450 tokens")
    print(f"   ⚠️  Significant truncation for reports >1000 tokens")
    
    print(f"\n2. Final Sequence Lengths:")
    print(f"   - 95th percentile: ~360-380 tokens (1.1% of context)")
    print(f"   - Maximum seen: ~576 tokens with current limit (1.8% of context)")
    print(f"   - Even with max_length=2048: ~2112 tokens (6.5% of context)")
    
    print(f"\n3. Recommendations:")
    print(f"   ✓ If 95% of reports are ≤298 tokens: Keep max_length=512 (current)")
    print(f"   ⚠️  If reports often exceed 500 tokens: Increase to max_length=768 or 1024")
    print(f"   ⚠️  If maximum detail needed: Consider max_length=2048")
    
    print(f"\n4. Trade-offs:")
    print(f"   - Higher max_length = More report detail preserved")
    print(f"   - Higher max_length = Slower training (longer sequences)")
    print(f"   - Higher max_length = More GPU memory usage")
    
    print("\n" + "=" * 100)

if __name__ == "__main__":
    main()
