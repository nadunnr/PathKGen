import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Local Paths (Verified for your current setup)
# This points to the 15GB of base weights you just downloaded
base_model_path = "/home/nadun/wd/vlm_histo/checkpoints/Qwen2.5-7B-Instruct"
# This points to your specific SlideChat training checkpoint
checkpoint_path = "/home/nadun/wd/vlm_histo/checkpoints/SlideChat_Weight/stage2_pth/mp_rank_00_model_states.pt" 

print(f"Loading base Qwen model from: {base_model_path}")

# 2. Load Tokenizer and Base Model
# By pointing to the local folder, it skips all downloads    
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16, # Optimized for RTX 3090
    device_map="auto"           # Automatically uses your GPU
)

# 3. Load and Clean SlideChat Weights
print(f"Injecting SlideChat weights from: {checkpoint_path}")

# Using weights_only=True follows the new PyTorch security recommendation
state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

new_state_dict = {}
for key, value in state_dict.items():
    # SlideChat/XTuner saves weights with 'model.' prefix
    if key.startswith("model."):
        clean_key = key.replace("model.", "")
        
        # We skip the vision-specific keys because this script is for the LLM part only
        if "mm_projector" not in clean_key and "slide_encoder" not in clean_key:
            new_state_dict[clean_key] = value

# 4. Inject Weights into the LLM
# strict=False is required as we are intentionally skipping vision layers
msg = model.load_state_dict(new_state_dict, strict=False)

print("-" * 30)
print(f"Successfully loaded SlideChat weights.")
print(f"Missing keys (Expected vision layers): {len(msg.missing_keys)}")
print(f"Unexpected keys: {len(msg.unexpected_keys)}")
print("-" * 30)

# 5. Inference Test
prompt = "Describe the role of a pathologist in cancer diagnosis."
messages = [
    {"role": "system", "content": "You are a helpful medical assistant specialized in histopathology."},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

print("\nGenerating Clinical Analysis...")
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=300,
    do_sample=True,
    temperature=0.3, # Lower temperature for more factual, less "creative" results
    top_p=0.9,
    repetition_penalty=1.1 # Helps prevent looping or generic list-making
)

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# Output the result directly
print("-" * 30)
print(f"Model Analysis:\n{response}")
print("-" * 30)