"""
Configuration file for VLM training and inference.

Single source of truth for prompts and other shared settings.
Change the prompt here once, and it updates everywhere.
"""

# ============================================================================
# PROMPT CONFIGURATION
# ============================================================================

# Training prompt template (includes {report} placeholder)
TRAINING_PROMPT_TEMPLATE = "Provide a concise single-paragraph pathology diagnosis:\n\n{report}"

# Generation/Inference prompt (no placeholder)
GENERATION_PROMPT = "Provide a concise single-paragraph pathology diagnosis:\n\n"

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

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

# Default paths (can be overridden by command-line arguments)
DEFAULT_LLM_PATH = "/home/nadun/wd/vlm_histo/checkpoints/Qwen2.5-7B-Instruct"
DEFAULT_SLIDECHAT_CHECKPOINT = "/home/nadun/wd/vlm_histo/checkpoints/SlideChat_Weight/stage2_pth/mp_rank_00_model_states.pt"
DEFAULT_DATA_JSON = "/home/nadun/wd/datasets/tcga_report_histgen/primary_site_splits/breast_dataset_split_0.7_0.1_0.2.json"
DEFAULT_STAGE1_OUTPUT = "/home/nadun/wd/vlm_histo/checkpoints/stage1"
DEFAULT_STAGE2_OUTPUT = "/home/nadun/wd/vlm_histo/checkpoints/stage2"
