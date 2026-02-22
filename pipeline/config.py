"""
Configuration file for VLM training and inference.

Single source of truth for prompts and other shared settings.
Change the prompt here once, and it updates everywhere.
"""
import os
from dataclasses import dataclass

# ============================================================================
# PROMPT CONFIGURATION
# ============================================================================

# Image marker tokens (will be added as special tokens to tokenizer)
IMAGE_START_TOKEN = "<image_start>"
IMAGE_END_TOKEN = "<image_end>"

# ============================================================================
# PROMPT TEMPLATES - HISTGEN FORMAT (Legacy)
# ============================================================================

# Training prompt without knowledge guidance (HistGen format)
TRAINING_PROMPT_NO_KNOWLEDGE = """<|im_start|>system
You are a pathology assistant specialized in analyzing whole slide images of tissue samples. Provide detailed, accurate diagnostic observations based on microscopic features.<|im_end|>
<|im_start|>user
<image_start><image_end>
Please analyze this whole slide image and provide a detailed diagnostic report describing the key histopathological features and diagnosis.<|im_end|>
<|im_start|>assistant
{report}<|im_end|>"""

# Training prompt with knowledge guidance (HistGen format)
TRAINING_PROMPT_WITH_KNOWLEDGE = """<|im_start|>system
You are a pathology assistant specialized in analyzing whole slide images of tissue samples. Provide detailed, accurate diagnostic observations based on microscopic features.<|im_end|>
<|im_start|>user
<image_start><image_end>
Generic knowledge regarding histology type (96.5% accurate). The site is exact.
{knowledge}
Use to help, but verify visually.

Please analyze this whole slide image and provide a detailed diagnostic report describing the key histopathological features and diagnosis.<|im_end|>
<|im_start|>assistant
{report}<|im_end|>"""

# ============================================================================
# PROMPT TEMPLATES - WSI-BENCH FORMAT (Recommended)
# ============================================================================

# WSI-Bench format - uses dataset's question field
TRAINING_PROMPT_WSI_BENCH = """<|im_start|>system
You are a pathology assistant specialized in analyzing whole slide images of tissue samples. Provide detailed, accurate diagnostic observations based on microscopic features.<|im_end|>
<|im_start|>user
<image_start><image_end>
{question}<|im_end|>
<|im_start|>assistant
{answer}<|im_end|>"""

# WSI-Bench with knowledge guidance
TRAINING_PROMPT_WSI_BENCH_KNOWLEDGE = """<|im_start|>system
You are a pathology assistant specialized in analyzing whole slide images of tissue samples. Provide detailed, accurate diagnostic observations based on microscopic features.<|im_end|>
<|im_start|>user
<image_start><image_end>
Generic knowledge regarding histology type (97.52% accurate). The site is exact.
{knowledge}
Use to help, but verify visually.

{question}<|im_end|>
<|im_start|>assistant
{answer}<|im_end|>"""

# ============================================================================
# DATASET PATHS
# ============================================================================

# WSI-Bench dataset paths (pre-split into train/test)
WSI_BENCH_TRAIN = "/home/nadun/wd/datasets/wsi_bench/dataset/wsi_bench_Report_with_features/wsi_bench_Report_train.json"
WSI_BENCH_TEST = "/home/nadun/wd/datasets/wsi_bench/dataset/wsi_bench_Report_with_features/wsi_bench_Report_test.json"

# WSI-Bench dataset paths WITH KNOWLEDGE
WSI_BENCH_TRAIN_KNOWLEDGE = "/home/nadun/wd/datasets/wsi_bench/dataset/WSI-Bench-train_Report_knowledge.json"
WSI_BENCH_TEST_KNOWLEDGE = "/home/nadun/wd/datasets/wsi_bench/dataset/WSI-Bench-test_Report_knowledge.json"

# HistGen dataset path (contains splits internally)
HISTGEN_DATASET = "/home/nadun/wd/datasets/WSIBench/processed/histgen_dataset.json"  # Example path

# ============================================================================
# INFERENCE/GENERATION PROMPTS
# ============================================================================

# Generation/Inference prompts (no {report} placeholder)

# Without knowledge guidance
GENERATION_PROMPT_NO_KNOWLEDGE = """<|im_start|>system
You are a pathology assistant specialized in analyzing whole slide images of tissue samples. Provide detailed, accurate diagnostic observations based on microscopic features.<|im_end|>
<|im_start|>user
<image_start><image_end>
Analyze the whole slide histopathology image shown above and provide a concise single-paragraph pathology diagnosis.<|im_end|>
<|im_start|>assistant
"""

# With knowledge guidance (includes {knowledge} placeholder)
GENERATION_PROMPT_WITH_KNOWLEDGE = """<|im_start|>system
You are a pathology assistant specialized in analyzing whole slide images of tissue samples. Provide detailed, accurate diagnostic observations based on microscopic features.<|im_end|>
<|im_start|>user
<image_start><image_end>
Generic knowledge regarding histology type (97.52% accurate). The site is exact.
{knowledge}
Use to help, but verify visually.

Generate a concise single-paragraph pathology diagnosis.<|im_end|>
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
DEFAULT_NUM_VISUAL_TOKENS = 128  # Default number of visual tokens
DEFAULT_VISUAL_DIM = 768  # TITAN encoder feature dimension per token
DEFAULT_LLM_HIDDEN_DIM = 3584  # Qwen 2.5 7B hidden dimension

# NOTE: Number of visual tokens is now DYNAMIC based on encoder output
# Typical values: 128 (TITAN), 256, 576 (Gigapath)
# Determined automatically from feature file shape [num_tokens, 768]

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Sequence length settings
DEFAULT_MAX_LENGTH = 768  # Maximum input sequence length (prompt + report) - increased for knowledge context
DEFAULT_MAX_NEW_TOKENS = 512  # Maximum tokens to generate during validation (increased for safety)
INFERENCE_MAX_NEW_TOKENS = 512  # Maximum tokens for standalone inference

# Generation parameters
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_REPETITION_PENALTY = 1.2

# Knowledge guidance settings
DEFAULT_USE_KNOWLEDGE_GUIDANCE = False
# ============================================================================
# PATH CONFIGURATION
# ============================================================================

# Default paths (can be overridden by command-line arguments)
DEFAULT_LLM_PATH = "/home/nadun/wd/vlm_histo/checkpoints/Qwen2.5-7B-Instruct"
UMLS_LLM_PATH    = "/home/nadun/wd/vlm_histo/checkpoints/Qwen-UMLS-7B-Instruct"  # Medical-domain fine-tuned
DEFAULT_SLIDECHAT_CHECKPOINT = "/home/nadun/wd/vlm_histo/checkpoints/SlideChat_Weight/stage2_pth/mp_rank_00_model_states.pt"
DEFAULT_DATA_JSON = "/home/nadun/wd/datasets/tcga_report_histgen/primary_site_splits/breast_dataset_split_0.7_0.1_0.2.json"
DEFAULT_STAGE1_OUTPUT = "/home/nadun/wd/vlm_histo/checkpoints/stage1"
DEFAULT_STAGE2_OUTPUT = "/home/nadun/wd/vlm_histo/checkpoints/stage2"



@dataclass
class Stage1Config:
    """Configuration for Stage 1 contrastive alignment training."""
    
    # ---- Data ----
    train_csv: str = "/home/nadun/wd/data/dataset/report_dataset_train.csv"
    test_csv: str = "/home/nadun/wd/data/dataset/report_dataset_test.csv"
    slide_features_dir: str = "/home/nadun/wd/data/slide_features"  # Has train/ and test/ subdirs
    text_embeddings_dir: str = "/home/nadun/wd/data/text_embeddings"  # Has train/ and test/ subdirs
    
    
    # ---- TITAN Model ----
    titan_model_name: str = "MahmoodLab/TITAN"
    titan_dim: int = 768  # TITAN token dimension (128 tokens × 768-dim each)
    
    # ---- Projection Heads ----
    hidden_dim: int = 3584        # Hidden dimension of MLP projectors
    alignment_dim: int = 3584     # Output dimension of shared alignment space
    
    # ---- Contrastive Loss (WSI-LLaVA paper) ----
    temperature: float = 0.02   # InfoNCE temperature (paper: τ=0.02)
    
    # ---- Training (WSI-LLaVA paper) ----
    lr: float = 1e-3            # Learning rate (paper: 0.001)
    weight_decay: float = 0.01
    batch_size: int = 64        # Batch size (paper: 64)
    epochs: int = 50            # Total epochs (paper: 50)
    warmup_ratio: float = 0.03  # Warmup fraction of total steps
    
    # ---- Text Encoding ----
    context_length: int = 128   # TITAN text context length
    
    # ---- Checkpointing ----
    checkpoint_dir: str = "./checkpoints/stage1"
    save_every_n_epochs: int = 5
    
    # ---- System ----
    device: str = "cuda"
    num_workers: int = 4
    seed: int = 42
    log_interval: int = 10      # Log every N batches
    
    def __post_init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
