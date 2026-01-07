import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration
model_id = "Qwen/Qwen2.5-7B-Instruct"
checkpoint_path = "/home/nadun/wd/vlm_histo/checkpoints/SlideChat_Weight/stage2_pth/mp_rank_00_model_states.pt" 

print("Loading base Qwen 2.5-7B and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

#Load in bfloat16 to match the SlideChat training precision
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load the SlideChat .pt weights
print(f"Loading weights from {checkpoint_path}...")
# map_location="cpu" is safer during the loading phase
state_dict = torch.load(checkpoint_path, map_location="cpu")

# Filter and Clean Keys
# XTuner/DeepSpeed saves weights with prefixes like 'model.' or 'module.'
# Standard Hugging Face models do not have these.
new_state_dict = {}
for key, value in state_dict.items():
    # Remove 'model.' prefix and skip the multimodal-only parts for now
    if key.startswith("model.") and "mm_projector" not in key and "slide_encoder" not in key:
        clean_key = key.replace("model.", "")
        new_state_dict[clean_key] = value

# 4. Inject Weights into the LLM
# strict=False is MANDATORY because we are skipping the projector/encoder keys
msg = model.load_state_dict(new_state_dict, strict=False)
print(f"Weight loading complete. Missing keys: {len(msg.missing_keys)}")
# Note: missing_keys should mostly be 'visual_encoder' or 'projector' related.

# 5. Simple Text Inference Check
print("\n--- Testing Model Output ---")
prompt = "Describe the role of a pathologist in cancer diagnosis."
messages = [
    {"role": "system", "content": "You are a helpful medical assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=100
)
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(f"\nResponse:\n{response.split('assistant')[-1].strip()}")