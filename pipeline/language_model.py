import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Paths
base_model_path = "/home/nadun/wd/vlm_histo/checkpoints/Qwen2.5-7B-Instruct"
# Path to your SlideChat .pt file
checkpoint_path = "/home/nadun/wd/vlm_histo/checkpoints/SlideChat_Weight/stage2_pth/mp_rank_00_model_states.pt" 

print(f"1. Loading base Qwen 2.5 model architecture from: {base_model_path}")

# 2. Initializing with Default Weights
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16, 
    device_map="auto"           
)

# 3. Loading SlideChat Weights
print(f"2. Overriding default weights with SlideChat checkpoint: {checkpoint_path}")
# Loading onto CPU first to prevent VRAM spikes during mapping
state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

new_state_dict = {}
for key, value in state_dict.items():
    # SlideChat/XTuner saves weights with 'model.' prefix; we strip it for the HuggingFace model
    if key.startswith("model."):
        clean_key = key.replace("model.", "")
        # Skip vision-only parameters for this text-only inference test
        if "mm_projector" not in clean_key and "slide_encoder" not in clean_key:
            new_state_dict[clean_key] = value

# 4. Inject and Confirm
# strict=False is necessary as we are omitting the vision layers in this branch
msg = model.load_state_dict(new_state_dict, strict=False)

print("-" * 30)
if len(msg.unexpected_keys) == 0 and len(msg.missing_keys) > 0:
    print("STATUS: SUCCESS. Model is now running on SLIDECHAT WEIGHTS.")
    print(f"Note: {len(msg.missing_keys)} base layers (mostly vision-related) were not overwritten.")
else:
    print("STATUS: WARNING. Mismatched weights detected. Verify the 'model.' prefix logic.")
print("-" * 30)

# 5. Inference (Paragraph Completion)
prompt = "Histopathological Analysis Report: Pathological examination of the tissue specimen reveals"
inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)

print("Generating Analysis...")
generated_ids = model.generate(
    **inputs,
    max_new_tokens=250,
    do_sample=True,
    temperature=0.4, 
    repetition_penalty=1.2,
    eos_token_id=tokenizer.eos_token_id
)

response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

print("-" * 30)
print(f"Report Output (SlideChat Weighted):\n{response}")
print("-" * 30)