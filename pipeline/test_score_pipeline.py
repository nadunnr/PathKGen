"""
test_score_pipeline.py — Verify prediction and reference extraction for NLU scoring.

Tests:
  1. extract_report_from_labels() returns only T-answer text (no ChatML markers)
  2. model.generate() decode returns only generated text (no prompt echoes)
  3. End-to-end: simulate a full training batch and verify both sides are clean

Run from the pipeline directory:
    python test_score_pipeline.py

No GPU required — all tests use CPU with the tokenizer only.
"""
import sys
import os
import torch

# Allow running from any working directory
sys.path.insert(0, os.path.dirname(__file__))

from transformers import AutoTokenizer
from config import (
    extract_report_from_labels,
    IMAGE_START_TOKEN, IMAGE_END_TOKEN,
    TRAINING_PROMPT_WSI_BENCH_KNOWLEDGE,
    TRAINING_PROMPT_WSI_BENCH,
    DEFAULT_LLM_PATH,
)
from data_collator import VLMDataCollator

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def check(name, condition, got=None, expected=None):
    if condition:
        print(f"  [{PASS}] {name}")
    else:
        print(f"  [{FAIL}] {name}")
        if expected is not None:
            print(f"           expected: {repr(expected)}")
        if got is not None:
            print(f"           got:      {repr(got)}")


# ─────────────────────────────────────────────────────────────────────────────
# Setup tokenizer
# ─────────────────────────────────────────────────────────────────────────────
print("Loading tokenizer from:", DEFAULT_LLM_PATH)
tokenizer = AutoTokenizer.from_pretrained(DEFAULT_LLM_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
special_tokens_dict = {'additional_special_tokens': [IMAGE_START_TOKEN, IMAGE_END_TOKEN]}
tokenizer.add_special_tokens(special_tokens_dict)
print("Tokenizer loaded.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: extract_report_from_labels — normal case (labels already masked)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("TEST 1: extract_report_from_labels (normal masked labels)")
print("=" * 60)

T_ANSWER = "The specimen shows moderately differentiated adenocarcinoma with focal necrosis."

# Simulate what the collator produces: only answer tokens survive (no -100 prefix)
answer_ids = tokenizer(T_ANSWER, add_special_tokens=False).input_ids
# Add the trailing <|im_end|> token the collator includes
im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
label_ids_tensor = torch.tensor(answer_ids + [im_end_id])

result = extract_report_from_labels(tokenizer, label_ids_tensor)
check("Returns T-answer text", T_ANSWER in result, got=result, expected=T_ANSWER)
check("No <|im_start|> in result", "<|im_start|>" not in result, got=result)
check("No <|im_end|> in result",   "<|im_end|>"   not in result, got=result)
check("No 'system' in result",     "system"        not in result, got=result)
check("No 'user' in result",       "user"          not in result, got=result)
check("Result is stripped",        result == result.strip(), got=result)
print()


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: extract_report_from_labels — safety net (markers accidentally present)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("TEST 2: extract_report_from_labels (safety net — markers present)")
print("=" * 60)

# Simulate edge case where the full prompt survived unmasked
full_prompt = f"<|im_start|>system\nYou are a pathology assistant.\n<|im_end|>\n<|im_start|>user\nQuestion text\n<|im_end|>\n<|im_start|>assistant\n{T_ANSWER}<|im_end|>"
full_ids = tokenizer(full_prompt, add_special_tokens=False).input_ids
label_ids_full = torch.tensor(full_ids)

result2 = extract_report_from_labels(tokenizer, label_ids_full)
check("Extracts only T-answer from full prompt", T_ANSWER in result2, got=result2, expected=T_ANSWER)
check("No prompt leakage (<|im_start|>)",  "<|im_start|>" not in result2, got=result2)
check("No 'system' returned",               "system"        not in result2, got=result2)
print()


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: VLMDataCollator label masking → verify extract_report_from_labels
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("TEST 3: Full collator → label mask → extract_report_from_labels")
print("=" * 60)

collator = VLMDataCollator(
    tokenizer=tokenizer,
    dataset_format='wsi-bench',
    max_length=512,
    use_knowledge_guidance=True
)

# Fake a batch item as dataset.__getitem__ would produce
QUESTION = "What is the diagnosis of this specimen?"
KNOWLEDGE = "Adenocarcinoma is a malignant neoplasm arising from glandular epithelium."

fake_sample = {
    'slide_embedding': torch.randn(1, 768),   # collator expects 'slide_embedding'
    'question': QUESTION,
    'answer': T_ANSWER,
    'knowledge_text': KNOWLEDGE,
    'id': 'test_sample_0',
}

batch = collator([fake_sample])
labels = batch['labels']  # shape [1, seq_len]

# Extract non-masked labels
label_ids = labels[0][labels[0] != -100]
print(f"  Total tokens: {labels.shape[1]}, Active label tokens: {label_ids.shape[0]}")

result3 = extract_report_from_labels(tokenizer, label_ids)
check("T-answer in extracted reference",      T_ANSWER   in result3,        got=result3)
check("Question NOT in extracted reference",  QUESTION   not in result3,    got=result3)
check("Knowledge NOT in extracted reference", KNOWLEDGE  not in result3,    got=result3)
check("No <|im_start|> in reference",        "<|im_start|>" not in result3, got=result3)
check("No <|im_end|> in reference",          "<|im_end|>"   not in result3, got=result3)
check("No 'system' in reference",            "system"        not in result3, got=result3)
print(f"  Extracted reference: {repr(result3)}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Simulated generate() output (no GPU — mock the decoder behaviour)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("TEST 4: generate() output decode simulation")
print("=" * 60)

def simulate_generate_decode(raw_output_text):
    """Simulate what model.generate() does to the output string."""
    generated_text = raw_output_text  # already skip_special_tokens=True equivalent

    if "<|im_start|>assistant" in generated_text:
        generated_text = generated_text.split("<|im_start|>assistant")[-1]
    elif "<|im_end|>" in generated_text:
        generated_text = generated_text.split("<|im_end|>")[0]

    return generated_text.strip()

# Case A: clean output (normal case with inputs_embeds)
clean_out = simulate_generate_decode(T_ANSWER)
check("Clean output unchanged", clean_out == T_ANSWER, got=clean_out, expected=T_ANSWER)

# Case B: output with trailing <|im_end|>
trailing_end = simulate_generate_decode(T_ANSWER + "<|im_end|>")
check("Trailing <|im_end|> stripped", trailing_end == T_ANSWER, got=trailing_end, expected=T_ANSWER)

# Case C: full ChatML echo (edge case — shouldn't happen with inputs_embeds)
full_echo = simulate_generate_decode(
    f"system\nYou are a pathologist.\nuser\nQuestion\n<|im_start|>assistant\n{T_ANSWER}<|im_end|>"
)
check("ChatML echo: only answer returned",
      T_ANSWER in full_echo and "system" not in full_echo,
      got=full_echo)

print()
print("=" * 60)
print("All tests complete.")
print("=" * 60)
