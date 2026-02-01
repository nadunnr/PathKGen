"""
Configuration file for VLM training and inference.

Single source of truth for prompts and other shared settings.
Change the prompt here once, and it updates everywhere.
"""

# ============================================================================
# PROMPT CONFIGURATION
# ============================================================================

# Image marker tokens (will be added as special tokens to tokenizer)
IMAGE_START_TOKEN = "<image_start>"
IMAGE_END_TOKEN = "<image_end>"

# Training prompt templates (include {report} placeholder)

# Without knowledge guidance
TRAINING_PROMPT_NO_KNOWLEDGE = """<|im_start|>system
You are an expert pathology AI assistant specialized in analyzing whole slide images (WSI) from histopathology specimens. You generate accurate, concise diagnostic reports based on visual analysis.<|im_end|>
<|im_start|>user
<image_start><image_end>
Analyze the whole slide histopathology image shown above and provide a concise single-paragraph pathology diagnosis.<|im_end|>
<|im_start|>assistant
{report}<|im_end|>"""

# With knowledge guidance (includes {knowledge} placeholder)
TRAINING_PROMPT_WITH_KNOWLEDGE = """<|im_start|>system
You are an expert pathology AI assistant specialized in analyzing whole slide images (WSI) from histopathology specimens. You integrate medical knowledge with visual analysis to generate accurate, concise diagnostic reports.<|im_end|>
<|im_start|>user
<image_start><image_end>

Medical Knowledge Context:
{knowledge}

Using the whole slide histopathology image and the medical knowledge provided, generate a concise single-paragraph pathology diagnosis.<|im_end|>
<|im_start|>assistant
{report}<|im_end|>"""

# Generation/Inference prompts (no {report} placeholder)

# Without knowledge guidance
GENERATION_PROMPT_NO_KNOWLEDGE = """<|im_start|>system
You are an expert pathology AI assistant specialized in analyzing whole slide images (WSI) from histopathology specimens. You generate accurate, concise diagnostic reports based on visual analysis.<|im_end|>
<|im_start|>user
<image_start><image_end>
Analyze the whole slide histopathology image shown above and provide a concise single-paragraph pathology diagnosis.<|im_end|>
<|im_start|>assistant
"""

# With knowledge guidance (includes {knowledge} placeholder)
GENERATION_PROMPT_WITH_KNOWLEDGE = """<|im_start|>system
You are an expert pathology AI assistant specialized in analyzing whole slide images (WSI) from histopathology specimens. You integrate medical knowledge with visual analysis to generate accurate, concise diagnostic reports.<|im_end|>
<|im_start|>user
<image_start><image_end>

Medical Knowledge Context:
{knowledge}

Using the whole slide histopathology image and the medical knowledge provided, generate a concise single-paragraph pathology diagnosis.<|im_end|>
<|im_start|>assistant
"""

# Legacy template for backward compatibility
TRAINING_PROMPT_TEMPLATE = TRAINING_PROMPT_NO_KNOWLEDGE
GENERATION_PROMPT = GENERATION_PROMPT_NO_KNOWLEDGE

# ============================================================================
# HELPER FUNCTIONS FOR EVALUATION
# ============================================================================

def extract_report_from_labels(tokenizer, label_ids):
    """
    Extract the ground truth report text from tokenized labels.
    
    The labels contain the full training prompt in ChatML format:
    <|im_start|>system...  
    <|im_start|>user...
    <|im_start|>assistant
    REPORT_TEXT<|im_end|>
    
    This function extracts just the REPORT_TEXT part for NLU metric evaluation.
    
    Args:
        tokenizer: HuggingFace tokenizer
        label_ids: Tensor of label token IDs (with -100 already filtered out)
        
    Returns:
        str: The ground truth report text
    """
    # Decode full text (keep special tokens to find markers)
    full_text = tokenizer.decode(label_ids, skip_special_tokens=False)
    
    # Find assistant response section
    assistant_marker = "<|im_start|>assistant"
    end_marker = "<|im_end|>"
    
    if assistant_marker in full_text:
        # Split at assistant marker and take everything after
        parts = full_text.split(assistant_marker, 1)
        if len(parts) > 1:
            after_assistant = parts[1]
            
            # Find the end marker
            if end_marker in after_assistant:
                report = after_assistant.split(end_marker, 1)[0]
                return report.strip()
            else:
                # No end marker found, return everything after assistant
                return after_assistant.strip()
    
    # Fallback: decode without special tokens and return
    # This handles old format or if markers not found
    return tokenizer.decode(label_ids, skip_special_tokens=True).strip()


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Visual embedding settings
DEFAULT_VISUAL_DIM = 768  # TITAN encoder output dimension
DEFAULT_NUM_VISUAL_TOKENS = 64  # Number of visual tokens to project to
DEFAULT_LLM_HIDDEN_DIM = 3584  # Qwen 2.5 7B hidden dimension

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Sequence length settings
DEFAULT_MAX_LENGTH = 512  # Maximum input sequence length (prompt + report)
DEFAULT_MAX_NEW_TOKENS = 350  # Maximum tokens to generate during validation
INFERENCE_MAX_NEW_TOKENS = 400  # Maximum tokens for standalone inference

# Generation parameters
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_REPETITION_PENALTY = 1.2

# Knowledge guidance settings
DEFAULT_USE_KNOWLEDGE_GUIDANCE = False
DEFAULT_MAX_KNOWLEDGE_TOKENS = 64

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

# Default paths (can be overridden by command-line arguments)
DEFAULT_LLM_PATH = "/home/nadun/wd/vlm_histo/checkpoints/Qwen2.5-7B-Instruct"
DEFAULT_SLIDECHAT_CHECKPOINT = "/home/nadun/wd/vlm_histo/checkpoints/SlideChat_Weight/stage2_pth/mp_rank_00_model_states.pt"
DEFAULT_DATA_JSON = "/home/nadun/wd/datasets/tcga_report_histgen/primary_site_splits/breast_dataset_split_0.7_0.1_0.2.json"
DEFAULT_STAGE1_OUTPUT = "/home/nadun/wd/vlm_histo/checkpoints/stage1"
DEFAULT_STAGE2_OUTPUT = "/home/nadun/wd/vlm_histo/checkpoints/stage2"
