"""
Pipeline sanity tests. Run with:
    cd /home/nadun/wd/vlm_histo/pipeline && python test_pipeline.py

No GPU or model weights required. Tests are purely structural/logical.
"""

import sys
import os
import torch

# Add pipeline dir to path
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    GENERATION_PROMPT_NO_KNOWLEDGE,
    GENERATION_PROMPT_WITH_KNOWLEDGE,
    TRAINING_PROMPT_WSI_BENCH,
    TRAINING_PROMPT_WSI_BENCH_KNOWLEDGE,
    TRAINING_PROMPT_NO_KNOWLEDGE,
    TRAINING_PROMPT_WITH_KNOWLEDGE,
    IMAGE_START_TOKEN,
    IMAGE_END_TOKEN,
)
from data_collator import VLMDataCollator
from transformers import AutoTokenizer

LLM_PATH = "/home/nadun/wd/vlm_histo/checkpoints/Qwen2.5-7B-Instruct"

passed = 0
failed = 0

def ok(name):
    global passed
    passed += 1
    print(f"  [PASS] {name}")

def fail(name, detail=""):
    global failed
    failed += 1
    msg = f"  [FAIL] {name}"
    if detail:
        msg += f"\n         {detail}"
    print(msg)

def check(name, condition, detail=""):
    if condition:
        ok(name)
    else:
        fail(name, detail)

# ============================================================
print("\n" + "="*55)
print("1. PROMPT FORMAT TESTS (no weights needed)")
print("="*55)

FALLBACK_Q = "Analyze the whole slide histopathology image and provide a concise pathology diagnosis."
Q = "What is the primary diagnosis for this WSI?"
K = "Lung adenocarcinoma. Site: lung."

def build_prompt(question=None, knowledge_text=None):
    """Same logic as model.generate() when prompt=None."""
    q = question if question else FALLBACK_Q
    if knowledge_text is not None:
        return GENERATION_PROMPT_WITH_KNOWLEDGE.format(knowledge=knowledge_text, question=q)
    else:
        return GENERATION_PROMPT_NO_KNOWLEDGE.format(question=q)

print("\n  -- No knowledge, with question --")
p = build_prompt(question=Q, knowledge_text=None)
print(f"  Prompt snippet: ...{p[p.index(IMAGE_END_TOKEN):p.index(IMAGE_END_TOKEN)+120].strip()!r}...")
check("Q in prompt",         Q in p)
check("no {question} left",  "{question}" not in p)
check("no {knowledge} left", "{knowledge}" not in p)
check("image marker present", IMAGE_START_TOKEN in p)
check("assistant turn present", "<|im_start|>assistant" in p)

print("\n  -- With knowledge + question --")
p = build_prompt(question=Q, knowledge_text=K)
print(f"  Prompt snippet: ...{p[p.index(IMAGE_END_TOKEN):p.index(IMAGE_END_TOKEN)+180].strip()!r}...")
check("Q in prompt",         Q in p)
check("K in prompt",         K in p)
check("no unfilled placeholders", "{question}" not in p and "{knowledge}" not in p)
# Check order: image -> knowledge -> question
img_pos  = p.index(IMAGE_START_TOKEN)
know_pos = p.index(K)
q_pos    = p.index(Q)
print(f"  Positions: image={img_pos}, knowledge={know_pos}, question={q_pos}")
check("image < knowledge < question", img_pos < know_pos < q_pos)

print("\n  -- question=None -> fallback --")
p = build_prompt(question=None, knowledge_text=None)
print(f"  Prompt snippet: ...{p[p.index(IMAGE_END_TOKEN):p.index(IMAGE_END_TOKEN)+120].strip()!r}...")
check("fallback Q in prompt", FALLBACK_Q in p)
check("no placeholders left", "{question}" not in p)

# ============================================================
print("\n" + "="*55)
print("2. TRAIN/GENERATION SYSTEM PROMPT CONSISTENCY")
print("="*55)

def get_system(tmpl):
    s = tmpl.index("<|im_start|>system\n") + len("<|im_start|>system\n")
    e = tmpl.index("<|im_end|>")
    return tmpl[s:e].strip()

train_sys   = get_system(TRAINING_PROMPT_WSI_BENCH)
gen_sys     = get_system(GENERATION_PROMPT_NO_KNOWLEDGE)
train_sys_k = get_system(TRAINING_PROMPT_WSI_BENCH_KNOWLEDGE)
gen_sys_k   = get_system(GENERATION_PROMPT_WITH_KNOWLEDGE)

print(f"\n  No-knowledge training system: {train_sys!r}")
print(f"  No-knowledge   gen   system: {gen_sys!r}")
check("no-knowledge system prompts match", train_sys == gen_sys,
      f"train='{train_sys}' vs gen='{gen_sys}'")

print(f"\n  Knowledge training system: {train_sys_k!r}")
print(f"  Knowledge   gen   system: {gen_sys_k!r}")
check("knowledge system prompts match", train_sys_k == gen_sys_k,
      f"train='{train_sys_k}' vs gen='{gen_sys_k}'")

# ============================================================
print("\n" + "="*55)
print("3. COLLATOR LABEL MASKING (needs tokenizer)")
print("="*55)

print(f"\n  Loading tokenizer from {LLM_PATH} ...")
tokenizer = AutoTokenizer.from_pretrained(LLM_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({'additional_special_tokens': [IMAGE_START_TOKEN, IMAGE_END_TOKEN]})
print("  Tokenizer loaded.")

ANSWER   = "Invasive ductal carcinoma, grade 2."
QUESTION = "Describe the diagnostic features."
KNOW     = "Breast adenocarcinoma. Site: breast."

print("\n  -- WSI-Bench, no knowledge --")
col = VLMDataCollator(tokenizer=tokenizer, dataset_format='wsi-bench',
                      max_length=512, use_knowledge_guidance=False)
batch = [{'id': 's1', 'organ': 'breast', 'slide_embedding': torch.randn(128, 768),
          'question': QUESTION, 'answer': ANSWER}]
out = col(batch)
ids  = out['input_ids'][0]
lbls = out['labels'][0]
active_ids    = ids[lbls != -100]
active_text   = tokenizer.decode(active_ids, skip_special_tokens=True).strip()
num_active    = (lbls != -100).sum().item()
num_masked    = (lbls == -100).sum().item()
print(f"  Active label tokens  : {num_active}")
print(f"  Masked label tokens  : {num_masked}")
print(f"  Active decoded text  : {active_text!r}")
check("answer in active labels",  ANSWER in active_text, f"got: {active_text!r}")
check("prompt is masked",         num_masked > 10)
check("questions emitted in batch", 'questions' in out and out['questions'][0] == QUESTION)

print("\n  -- WSI-Bench, WITH knowledge --")
col_k = VLMDataCollator(tokenizer=tokenizer, dataset_format='wsi-bench',
                        max_length=512, use_knowledge_guidance=True)
batch_k = [{'id': 's1', 'organ': 'breast', 'slide_embedding': torch.randn(128, 768),
             'question': QUESTION, 'answer': ANSWER, 'knowledge_text': KNOW}]
out_k = col_k(batch_k)
ids_k  = out_k['input_ids'][0]
lbls_k = out_k['labels'][0]

# First knowledge token should be masked
know_first_tok = tokenizer(KNOW, add_special_tokens=False).input_ids[0]
positions = (ids_k == know_first_tok).nonzero(as_tuple=True)[0]
if len(positions):
    pos = positions[0].item()
    print(f"  First knowledge token ({know_first_tok}) at pos {pos}, label={lbls_k[pos].item()}")
    check("knowledge token is -100 in labels", lbls_k[pos].item() == -100)
else:
    fail("knowledge token found in input", "not found")

active_k    = ids_k[lbls_k != -100]
active_text_k = tokenizer.decode(active_k, skip_special_tokens=True).strip()
print(f"  Knowledge active text: {active_text_k!r}")
check("answer still active with knowledge", ANSWER in active_text_k)
check("knowledge_texts in batch", 'knowledge_texts' in out_k)
check("knowledge_text value correct", out_k['knowledge_texts'][0] == KNOW)

# ============================================================
print("\n" + "="*55)
print("SUMMARY")
print("="*55)
print(f"  Passed: {passed}")
print(f"  Failed: {failed}")
if failed:
    print(f"\n[!] {failed} test(s) FAILED")
    sys.exit(1)
else:
    print("\n  All tests passed!")
