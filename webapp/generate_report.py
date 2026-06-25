import sys
import torch
import h5py
import os
import gc
sys.path.insert(0, '/home/nadun/wd/PathKGen/pipeline') 
from config import DEFAULT_LLM_PATH, DEFAULT_VISUAL_DIM, DEFAULT_NUM_VISUAL_TOKENS

# Avoid very slow optional-package scans in transformers caused by injected paths.
os.environ["PYTHONPATH"] = ""
os.environ.setdefault("PYTHONNOUSERSITE", "1")


SLIDE_EMB = '/home/nadun/wd/PathKGen/data/slide_features/test/TCGA-2A-AAYO-01Z-00-DX1.A5BB3437-956A-4A42-AC28-1217B8016970.h5'
CKPT_PTH = '/home/nadun/wd/PathKGen/checkpoints/q7b_best/s2/best_model.pt'
HISTO = 'Thymoma'


def build_question(histo: str) -> str:
    # Match pipeline inference style by passing a concrete question to the model.
    return (
        f"Analyze this whole slide histopathology image for {histo} and "
        "provide a concise pathology diagnosis."
    )


def load_model_checkpoint(ckpt_pth, llm_path=DEFAULT_LLM_PATH):
    # Import lazily so startup prints appear before heavy transformers import.
    from model import VisionLanguageModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model on CPU to save GPU memory during loading
    print(f"\nInitializing model (Stage 2 - QLoRA)...")
    model = VisionLanguageModel(
        llm_path=llm_path,
        slidechat_checkpoint=None,  # Weights are in checkpoint
        visual_dim=DEFAULT_VISUAL_DIM,
        num_visual_tokens=DEFAULT_NUM_VISUAL_TOKENS,
        use_qlora=True,  # Stage 2 uses QLoRA
        freeze_llm=True,  # Inference only
        gradient_checkpointing=False,
        device=device
    )
    
    # Load checkpoint
    print(f"Loading checkpoint from: {ckpt_pth}")
    checkpoint = torch.load(ckpt_pth, map_location="cpu", weights_only=False)
    
    # Load projection weights
    model.projection.load_state_dict(checkpoint["projection_state_dict"])
    print("  ✓ Projection weights loaded")
    
    # Load LoRA weights
    if "lora_state_dict" in checkpoint:
        model.llm.load_state_dict(checkpoint["lora_state_dict"], strict=False)
        print("  ✓ LoRA weights loaded")
    else:
        print("  ! Warning: 'lora_state_dict' not found in checkpoint")
    
    # Log checkpoint metadata if available
    if "epoch" in checkpoint:
        print(f"  Checkpoint epoch: {checkpoint['epoch'] + 1}")
    if "val_loss" in checkpoint and checkpoint["val_loss"] is not None:
        print(f"  Checkpoint val loss: {checkpoint['val_loss']:.4f}")
    
    # Free checkpoint memory
    del checkpoint
    torch.cuda.empty_cache()
    gc.collect()
    
    # Avoid calling .to(device) for quantized/device-mapped LLMs because this can
    # trigger an expensive full-model move and appear as a stall.
    if getattr(model, "use_qlora", False):
        model.device = str(device)
        model.projection = model.projection.to(device)
    else:
        model = model.to(device)
    model.eval()
    
    return model, device


def load_slide_features(slide_emb_path, feature_key='token'):
    """
    Load slide embeddings from H5 file.
    
    Args:
        slide_emb_path: Path to the H5 file containing slide features
        feature_key: Key in H5 file where features are stored (default: 'token')
        
    Returns:
        slide_embedding: torch.Tensor of shape [num_tokens, feature_dim]
    """
    if not os.path.exists(slide_emb_path):
        raise FileNotFoundError(f"Feature file not found: {slide_emb_path}")
    
    print(f"Loading slide embeddings from: {slide_emb_path}")
    with h5py.File(slide_emb_path, 'r') as h5_file:
        if feature_key not in h5_file:
            available_keys = list(h5_file.keys())
            print(f"Available keys in H5 file: {available_keys}")
            raise KeyError(
                f"Feature key '{feature_key}' not found in H5 file: {slide_emb_path}"
            )
        
        slide_features = h5_file[feature_key][:]
        
        # Validate shape
        if slide_features.ndim != 2:
            raise ValueError(
                f"Expected 2D features [num_tokens, feature_dim], "
                f"got shape {slide_features.shape}"
            )
        
        slide_embedding = torch.tensor(slide_features, dtype=torch.float32)
        print(f"  Loaded features shape: {slide_embedding.shape}")
    
    return slide_embedding


def generate_report(slide_emb=SLIDE_EMB, ckpt_pth=CKPT_PTH, histo=HISTO, feature_key='token'):
    """
    Generate a pathology report from a whole slide image embedding.
    
    Args:
        slide_emb: Path to H5 file containing slide features
        ckpt_pth: Path to the checkpoint file with model weights
        histo: Histology type (e.g., 'Thymoma', 'Carcinoma', etc.)
        feature_key: Key in H5 file where the selected slide embedding is stored
        
    Returns:
        report: Generated pathology report as string
    """
    print(f"\n{'='*70}")
    print(f"Generating Report for Histology: {histo}")
    print(f"{'='*70}\n")
    
    # Load model
    model, device = load_model_checkpoint(ckpt_pth)
    
    # Load slide features
    slide_features = load_slide_features(slide_emb, feature_key=feature_key)
    
    # Ensure batch dimension [1, num_tokens, feature_dim]
    if slide_features.dim() == 2:
        slide_features = slide_features.unsqueeze(0)
    
    if not histo or not str(histo).strip():
        raise ValueError("histo must be a non-empty string")

    question_text = build_question(histo.strip())

    print(f"Generating report with prompt context for histology: {histo}\n")
    
    # Generate report
    with torch.no_grad():
        report = model.generate(
            visual_features=slide_features,
            question=question_text,
            knowledge_text=None,
            max_new_tokens=512,
            temperature=0.0,  # Greedy decoding for reproducibility
            top_p=1.0,
            num_beams=1,
            no_repeat_ngram_size=3,
        )
    
    print("=" * 70)
    print("GENERATED REPORT:")
    print("=" * 70)
    print(report)
    print("=" * 70 + "\n")
    
    return report


if __name__ == "__main__":
    # Run the report generation with default values.
    report = generate_report(
        slide_emb=SLIDE_EMB,
        ckpt_pth=CKPT_PTH,
        histo=HISTO,
        feature_key='token',
    )
    print("Report generation complete!")