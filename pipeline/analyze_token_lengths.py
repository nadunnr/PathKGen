
import json
import numpy as np
from transformers import AutoTokenizer
import os
import sys

# Add parent directory to path to import config
# We need to make sure we can import 'pipeline.config'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Actually, the file structure is:
# /home/nadun/wd/vlm_histo/pipeline/config.py
# If I run this script from /home/nadun/wd/vlm_histo/, then 'pipeline' is a module.
# Let's import directly.

try:
    from pipeline.config import (
        TRAINING_PROMPT_WSI_BENCH,
        TRAINING_PROMPT_WSI_BENCH_KNOWLEDGE
    )
except ImportError:
    # If running as script, try adding current dir to path
    sys.path.append(os.getcwd())
    from pipeline.config import (
        TRAINING_PROMPT_WSI_BENCH,
        TRAINING_PROMPT_WSI_BENCH_KNOWLEDGE
    )

# Configuration
MODEL_PATH = "/home/nadun/wd/vlm_histo/checkpoints/Qwen2.5-7B-Instruct"
TRAIN_SET = "/home/nadun/wd/datasets/wsi_bench/dataset/wsi_bench_Report_knowledge/WSI-Bench-train_Report_knowledge.json"
TEST_SET = "/home/nadun/wd/datasets/wsi_bench/dataset/wsi_bench_Report_knowledge/WSI-Bench-test_Report_knowledge.json"

def analyze_dataset(name, file_path, tokenizer):
    print(f"\nAnalyzing {name} set ({file_path})...")
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    question_lengths = []
    answer_lengths = []
    knowledge_lengths = []
    prompt_overhead_lengths = []
    
    # Use the KNOWLEDGE template
    template = TRAINING_PROMPT_WSI_BENCH_KNOWLEDGE

    # Handle if data is dict (knowledge datasets) or list (original)
    if isinstance(data, dict):
        items = list(data.values())
    else:
        items = data
    
    for item in items:
        # Get question/answer/knowledge
        question = item.get('question', '') or ''
        answer = item.get('T-answer', item.get('answer', '')) or ''
        knowledge = item.get('knowledge', '') or ''
        
        q_tokens = len(tokenizer.encode(question, add_special_tokens=False))
        a_tokens = len(tokenizer.encode(answer, add_special_tokens=False))
        k_tokens = len(tokenizer.encode(knowledge, add_special_tokens=False))
        
        # Calculate full prompt length (System + User wrapper + Question + Knowledge)
        # Template is:
        # ... Generic knowledge ...
        # {knowledge}
        # ...
        # {question}<|im_end|>
        # <|im_start|>assistant
        # {answer}<|im_end|>
        
        # Construct prompt string UP TO generated answer
        prompt_str = template.split("{answer}")[0].format(question=question, knowledge=knowledge)
        
        p_tokens = len(tokenizer.encode(prompt_str, add_special_tokens=True))
        
        question_lengths.append(q_tokens)
        answer_lengths.append(a_tokens)
        knowledge_lengths.append(k_tokens)
        prompt_overhead_lengths.append(p_tokens)
        
    visual_tokens = 128
    
    # Percentiles
    p95_q = np.percentile(question_lengths, 95)
    p95_a = np.percentile(answer_lengths, 95)
    p95_k = np.percentile(knowledge_lengths, 95)
    p95_p = np.percentile(prompt_overhead_lengths, 95)
    
    # Add visual tokens to prompt length for effective input
    effective_prompt_p95 = p95_p - 2 + visual_tokens # -2 for <image_start><image_end> which are replaced
    
    print(f"  Samples: {len(data)}")
    print(f"  Visual Tokens (Fixed): {visual_tokens}")
    print(f"  Question Tokens (95th/Max): {np.percentile(question_lengths, 95):.1f} / {np.max(question_lengths):.1f}")
    print(f"  Answer Tokens (95th/Max): {np.percentile(answer_lengths, 95):.1f} / {np.max(answer_lengths):.1f}")
    print(f"  Knowledge Tokens (95th/Max): {np.percentile(knowledge_lengths, 95):.1f} / {np.max(knowledge_lengths):.1f}")
    print(f"  Prompt Text Tokens (System + Image + Know/Quest) (95th/Max): {np.percentile(prompt_overhead_lengths, 95):.1f} / {np.max(prompt_overhead_lengths):.1f}")
    
    # Effective input length = Prompt text + Visual tokens - markers
    effective_input_lengths = [l + visual_tokens - 2 for l in prompt_overhead_lengths]
    print(f"  Effective Input Prompt Tokens (Prompt Text + 128 Visual - 2 markers) (95th/Max): {np.percentile(effective_input_lengths, 95):.1f} / {np.max(effective_input_lengths):.1f}")
    
    # Total sequence length
    total_lengths = [e + a for e, a in zip(effective_input_lengths, answer_lengths)]
    print(f"  Total Sequence Length (Effective Prompt + Answer) (95th/Max): {np.percentile(total_lengths, 95):.1f} / {np.max(total_lengths):.1f}")
    print("")


def main():
    print(f"Loading tokenizer from {MODEL_PATH}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    analyze_dataset("Val/Test", TEST_SET, tokenizer)
    analyze_dataset("Train", TRAIN_SET, tokenizer)

if __name__ == "__main__":
    main()
