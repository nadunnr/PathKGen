"""
Inference script for generating pathology reports from WSI slides.

Usage:
    # Single slide
    python pipeline/inference.py \
        --checkpoint /path/to/checkpoint.pt \
        --feature_path /path/to/slide.h5 \
        --output_path report.txt

    # Batch inference on test set
    python pipeline/inference.py \
        --checkpoint /path/to/checkpoint.pt \
        --data_json /path/to/dataset.json \
        --split test \
        --output_dir /path/to/outputs
"""

import os
import argparse
import json
import h5py
import torch
from tqdm import tqdm
from pathlib import Path

# Local imports
from model import VisionLanguageModel
from dataset import WSI_Report_Dataset
from config import (GENERATION_PROMPT_NO_KNOWLEDGE, GENERATION_PROMPT_WITH_KNOWLEDGE,
                    DEFAULT_LLM_PATH, DEFAULT_SLIDECHAT_CHECKPOINT, DEFAULT_VISUAL_DIM,
                    DEFAULT_NUM_VISUAL_TOKENS, INFERENCE_MAX_NEW_TOKENS,
                    DEFAULT_TEMPERATURE, DEFAULT_TOP_P, 
                    DEFAULT_REPETITION_PENALTY)


def parse_args():
    parser = argparse.ArgumentParser(description="VLM Inference for Report Generation")
    
    # Model arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--llm_path", type=str,
                       default=DEFAULT_LLM_PATH,
                       help="Path to base LLM")
    parser.add_argument("--slidechat_checkpoint", type=str,
                       default=DEFAULT_SLIDECHAT_CHECKPOINT,
                       help="Path to SlideChat weights")
    parser.add_argument("--visual_dim", type=int, default=DEFAULT_VISUAL_DIM,
                       help="Dimension of slide features")
    parser.add_argument("--num_visual_tokens", type=int, default=DEFAULT_NUM_VISUAL_TOKENS,
                       help="Number of visual tokens")
    parser.add_argument("--use_qlora", action="store_true",
                       help="Whether checkpoint uses QLoRA (Stage 2)")
    parser.add_argument("--skip_slidechat_init", action="store_true",
                       help="Skip loading SlideChat weights (use this for inference with trained checkpoints)")
    
    # Dataset arguments
    parser.add_argument("--feature_path", type=str, default=None,
                       help="Path to single H5 feature file")
    parser.add_argument("--feature_key", type=str, default="tokens",
                       help="Key in H5 file for features")
    parser.add_argument("--data_json", type=str, default=None,
                       help="Path to dataset JSON for batch inference")
    parser.add_argument("--dataset_format", type=str, default="wsi-bench",
                       choices=["histgen", "wsi-bench"],
                       help="Dataset format")
    parser.add_argument("--split", type=str, default="test",
                       help="Dataset split to use (histgen only)")
    parser.add_argument("--use_knowledge_guidance", action="store_true",
                       help="Enable knowledge graph guidance for report generation")
    
    # Generation arguments
    parser.add_argument("--max_new_tokens", type=int, default=INFERENCE_MAX_NEW_TOKENS,
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                       help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=DEFAULT_TOP_P,
                       help="Nucleus sampling parameter")
    parser.add_argument("--repetition_penalty", type=float, default=DEFAULT_REPETITION_PENALTY,
                       help="Repetition penalty")
    
    # Output arguments
    parser.add_argument("--output_path", type=str, default=None,
                       help="Output path for single report")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for batch reports")
    parser.add_argument("--save_json", action="store_true",
                       help="Save results as JSON (batch mode)")

    # Single-slide options
    parser.add_argument("--question", type=str, default=None,
                       help="Question for single-slide inference (WSI-Bench format)")
    parser.add_argument("--knowledge_text", type=str, default=None,
                       help="Knowledge text for single-slide inference (if --use_knowledge_guidance)")

    # Misc
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for inference (use 1 for variable generation length)")
    
    return parser.parse_args()


def load_slide_features(feature_path: str, feature_key: str = "features") -> torch.Tensor:
    """Load slide features from H5 file."""
    with h5py.File(feature_path, 'r') as h5_file:
        features = torch.tensor(h5_file[feature_key][:], dtype=torch.float32)
    return features


def single_inference(args, model, device):
    """Run inference on a single slide."""
    print(f"Loading features from: {args.feature_path}")
    features = load_slide_features(args.feature_path, args.feature_key)

    # Build question and knowledge for generation
    question = getattr(args, 'question', None)
    knowledge_text = args.knowledge_text if args.use_knowledge_guidance else None


    print("Generating report...")
    report = model.generate(
        visual_features=features.to(device),
        prompt=None,
        question=question,
        knowledge_text=knowledge_text,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty
    )

    # Print to console
    print("\n" + "="*60)
    print("Generated Report:")
    print("="*60)
    print(report)
    print("="*60)

    # Save to file if specified
    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"\n✓ Report saved to: {args.output_path}")




def batch_inference(args, model, device):
    """Run inference on dataset split."""
    print(f"Loading dataset from: {args.data_json}")
    
    split = None if args.dataset_format == 'wsi-bench' else args.split
    dataset = WSI_Report_Dataset(
        json_path=args.data_json,
        dataset_format=args.dataset_format,
        split=split,
        feature_key=args.feature_key,
        use_knowledge_guidance=args.use_knowledge_guidance
    )
    
    print(f"Processing {len(dataset)} slides...")
    
    results = []

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    for i in tqdm(range(len(dataset)), desc="Generating reports"):
        sample = dataset[i]
        slide_id = sample['id']
        features = sample['slide_embedding']
        ground_truth = sample.get('answer', sample.get('report', ''))
        question = sample.get('question', None)
        knowledge_text = sample.get('knowledge_text', None) if args.use_knowledge_guidance else None
        
        # Generate report
        generated_report = model.generate(
            visual_features=features.to(device),
            prompt=None,
            question=question,
            knowledge_text=knowledge_text,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty
        )
        
        result = {
            'id': slide_id,
            'organ': sample.get('organ', 'unknown'),
            'question': question,
            'generated_report': generated_report,
            'ground_truth': ground_truth
        }
        results.append(result)
        
        # Save individual report
        if args.output_dir:
            report_path = Path(args.output_dir) / f"{slide_id}_report.txt"
            with open(report_path, 'w') as f:
                f.write(f"Slide ID: {slide_id}\n")
                f.write(f"Organ: {sample.get('organ', 'unknown')}\n")
                if question:
                    f.write(f"Question: {question}\n")
                f.write("\n" + "="*60 + "\n")
                f.write("Generated Report:\n")
                f.write("="*60 + "\n")
                f.write(generated_report)
                f.write("\n\n" + "="*60 + "\n")
                f.write("Ground Truth:\n")
                f.write("="*60 + "\n")
                f.write(ground_truth)
    
    # Save JSON summary
    if args.save_json and args.output_dir:
        json_path = Path(args.output_dir) / "results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {json_path}")
    
    print(f"\n✓ Generated {len(results)} reports")
    if args.output_dir:
        print(f"✓ Individual reports saved to: {args.output_dir}")


def main():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Validate input arguments
    if args.feature_path is None and args.data_json is None:
        raise ValueError("Must specify either --feature_path or --data_json")
    
    if args.feature_path and args.output_path is None:
        print("Warning: No output path specified for single inference")
    
    if args.data_json and args.output_dir is None:
        print("Warning: No output directory specified for batch inference")
    
    # Auto-detect checkpoint type (Stage 1 or Stage 2)
    print(f"\nChecking checkpoint type: {args.checkpoint}")
    checkpoint_cpu = torch.load(args.checkpoint, map_location='cpu')
    has_lora = 'lora_state_dict' in checkpoint_cpu
    
    if has_lora and not args.use_qlora:
        print("Detected Stage 2 checkpoint (QLoRA). Setting use_qlora=True")
        args.use_qlora = True
    elif not has_lora and args.use_qlora:
        print("Warning: use_qlora=True but checkpoint doesn't have LoRA weights")
        args.use_qlora = False
    
    # Clear checkpoint from memory
    del checkpoint_cpu
    torch.cuda.empty_cache()
    
    # Initialize model
    print("\n" + "="*60)
    print("Loading VLM Model for Inference")
    print("="*60)
    
    # For inference with trained checkpoint, skip SlideChat to save memory
    slidechat_path = None if args.skip_slidechat_init else args.slidechat_checkpoint
    if args.skip_slidechat_init:
        print("Skipping SlideChat initialization (will load from checkpoint)")
    
    model = VisionLanguageModel(
        llm_path=args.llm_path,
        slidechat_checkpoint=slidechat_path,
        visual_dim=args.visual_dim,
        num_visual_tokens=args.num_visual_tokens,
        use_qlora=args.use_qlora,
        freeze_llm=True,  # Not needed for inference
        device=device
    )
    
    # Load checkpoint weights
    print(f"\nLoading checkpoint weights from: {args.checkpoint}")
    model.load_checkpoint(args.checkpoint, load_optimizer=False)
    model.eval()
    
    # Clear cache
    torch.cuda.empty_cache()
    
    print("\n✓ Model loaded successfully!")
    
    # Run inference
    if args.feature_path:
        single_inference(args, model, device)
    else:
        batch_inference(args, model, device)
    
    print("\n" + "="*60)
    print("Inference Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
