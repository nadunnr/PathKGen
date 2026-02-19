"""
Standalone evaluation script for VLM checkpoints.

Evaluates a trained model checkpoint with full NLP metrics (BLEU, METEOR, ROUGE-L).

Usage:
    # Evaluate Stage 1 checkpoint
    python evaluate_metrics.py \
        --checkpoint checkpoints/stage1/.../best_model.pt \
        --stage 1
    
    # Evaluate Stage 2 checkpoint
    python evaluate_metrics.py \
        --checkpoint checkpoints/stage2/.../best_model.pt \
        --stage 2 \
        --lora_rank 64
    
    # Save results to file
    python evaluate_metrics.py \
        --checkpoint checkpoints/stage1/.../best_model.pt \
        --stage 1 \
        --output_file results.json
"""

import argparse
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

# Local imports
from model import VisionLanguageModel
from dataset import WSI_Report_Dataset
from data_collator import VLMDataCollator
from config import (GENERATION_PROMPT, TRAINING_PROMPT_TEMPLATE,
                    DEFAULT_LLM_PATH, DEFAULT_SLIDECHAT_CHECKPOINT,
                    DEFAULT_DATA_JSON, DEFAULT_VISUAL_DIM, 
                    DEFAULT_NUM_VISUAL_TOKENS, DEFAULT_MAX_LENGTH,
                    DEFAULT_MAX_NEW_TOKENS,
                    extract_report_from_labels)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VLM checkpoint with full metrics")
    
    # Required arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint (.pt file)")
    parser.add_argument("--stage", type=int, choices=[1, 2], required=True,
                       help="Stage 1 (projection only) or Stage 2 (projection + QLoRA)")
    
    # Data arguments
    parser.add_argument("--data_json", type=str, default=DEFAULT_DATA_JSON,
                       help="Path to dataset JSON")
    parser.add_argument("--dataset_format", type=str, default="histgen", choices=["histgen", "wsi-bench"],
                       help="Dataset format: 'histgen' (nested dict with splits) or 'wsi-bench' (flat list)")
    parser.add_argument("--split", type=str, default="test",
                       help="Dataset split to evaluate on (default: test). Leave empty for wsi-bench format.")
    parser.add_argument("--feature_key", type=str, default="features",
                       help="Key in H5 files for slide features")
    
    # Model arguments
    parser.add_argument("--llm_path", type=str, default=DEFAULT_LLM_PATH,
                       help="Path to base LLM")
    parser.add_argument("--slidechat_checkpoint", type=str, default=DEFAULT_SLIDECHAT_CHECKPOINT,
                       help="Path to SlideChat weights")
    parser.add_argument("--visual_dim", type=int, default=DEFAULT_VISUAL_DIM,
                       help="Dimension of slide features")
    parser.add_argument("--num_visual_tokens", type=int, default=DEFAULT_NUM_VISUAL_TOKENS,
                       help="Number of visual tokens")
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH,
                       help="Maximum sequence length")
    
    # Generation arguments
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Sampling temperature (0.0 = greedy decoding for evaluation)")
    parser.add_argument("--top_p", type=float, default=1.0,
                       help="Nucleus sampling top_p (1.0 = no nucleus sampling)")
    
    # Stage 2 specific arguments
    parser.add_argument("--lora_rank", type=int, default=64,
                       help="LoRA rank (Stage 2 only)")
    parser.add_argument("--lora_alpha", type=int, default=128,
                       help="LoRA alpha (Stage 2 only)")
    
    # Evaluation arguments
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="DataLoader workers")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Path to save evaluation results (JSON)")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (default: cuda if available)")
    
    return parser.parse_args()


@torch.no_grad()
def validate(model, dataloader, device, tokenizer, args):
    """Validate the model with loss and generation metrics."""
    model.eval()
    total_loss = 0
    num_samples = 0
    all_predictions = []
    all_references = []
    
    pbar = tqdm(dataloader, desc="Evaluating")
    
    for batch in pbar:
        # Process each sample individually
        visual_features = batch['visual_features']
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        batch_size = visual_features.shape[0]
        batch_loss_sum = 0
        
        # Process samples one at a time
        for i in range(batch_size):
            # Move single sample to device
            single_visual = visual_features[i:i+1].to(device)
            single_input = input_ids[i:i+1].to(device)
            single_mask = attention_mask[i:i+1].to(device)
            single_labels = labels[i:i+1].to(device)
            
            # Forward pass for single sample
            outputs = model(
                visual_features=single_visual,
                input_ids=single_input,
                attention_mask=single_mask,
                labels=single_labels
            )
            
            sample_loss = outputs['loss'].item()
            batch_loss_sum += sample_loss
            
            # Free memory immediately
            del outputs, single_input, single_mask, single_labels
            torch.cuda.empty_cache()
            
            # Generate prediction
            # Get knowledge text if available
            knowledge_text = None
            if 'knowledge_texts' in batch:
                knowledge_text = batch['knowledge_texts'][i]
            
            pred_text = model.generate(
                visual_features=single_visual,
                prompt=None,  # Let model auto-select based on knowledge_text
                knowledge_text=knowledge_text,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p
            )
            all_predictions.append(pred_text)
            
            # Extract ground truth report from labels
            label_ids = labels[i][labels[i] != -100]
            ref_text = extract_report_from_labels(tokenizer, label_ids)
            all_references.append(ref_text)
            
            # Clear cache after each sample
            del single_visual
            torch.cuda.empty_cache()
        
        avg_batch_loss = batch_loss_sum / batch_size
        total_loss += batch_loss_sum
        num_samples += batch_size
        
        pbar.set_postfix({'loss': f'{avg_batch_loss:.4f}'})
    
    avg_loss = total_loss / num_samples
    
    # Compute metrics
    from metrics import compute_metrics
    metrics = compute_metrics(all_predictions, all_references)
    
    return avg_loss, metrics, all_predictions, all_references


def main():
    args = parse_args()
    
    # Set device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("=" * 60)
    print(f"Evaluating VLM Checkpoint (Stage {args.stage})")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Split: {args.split}")
    print()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create validation dataset
    print("Loading dataset...")
    # Handle split for different formats
    if args.dataset_format == 'wsi-bench':
        # WSI-Bench: already pre-split, no split argument needed
        val_split = None
    else:
        # HistGen: use provided split
        val_split = args.split
    
    val_dataset = WSI_Report_Dataset(
        json_path=args.data_json,
        dataset_format=args.dataset_format,
        split=val_split,
        feature_key=args.feature_key
    )
    print(f"  Loaded {len(val_dataset)} samples from '{args.split if val_split else 'pre-split file'}' split")
    
    
    # Create data collator
    data_collator = VLMDataCollator(
        tokenizer=tokenizer,
        dataset_format=args.dataset_format,
        max_length=args.max_length
    )
    
    # Create validation dataloader
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=data_collator,
        pin_memory=True
    )
    
    # Load model
    print("\nInitializing model...")
    if args.stage == 1:
        # Stage 1: Projection only, LLM frozen
        model = VisionLanguageModel(
            llm_path=args.llm_path,
            visual_dim=args.visual_dim,
            num_visual_tokens=args.num_visual_tokens,
            slidechat_checkpoint=args.slidechat_checkpoint,
            use_qlora=False,
            freeze_llm=True,
            gradient_checkpointing=False
        ).to(device)
    else:
        # Stage 2: Projection + QLoRA
        model = VisionLanguageModel(
            llm_path=args.llm_path,
            visual_dim=args.visual_dim,
            num_visual_tokens=args.num_visual_tokens,
            slidechat_checkpoint=args.slidechat_checkpoint,
            use_qlora=True,
            freeze_llm=False,
            gradient_checkpointing=False
        ).to(device)
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Load weights (different keys for Stage 1 vs Stage 2)
    model.projection.load_state_dict(checkpoint['projection_state_dict'])
    if args.stage == 2 and 'lora_state_dict' in checkpoint:
        model.llm.load_state_dict(checkpoint['lora_state_dict'], strict=False)
    
    print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
    if 'val_loss' in checkpoint:
        print(f"  Checkpoint val loss: {checkpoint['val_loss']:.4f}")
    
    # Evaluate
    print("\n" + "=" * 60)
    print("Running Evaluation")
    print("=" * 60)
    
    val_loss, metrics, predictions, references = validate(
        model, val_loader, device, tokenizer, args
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Loss: {val_loss:.4f}")
    print(f"\nNLP Metrics:")
    print(f"  BLEU-1: {metrics.get('bleu1', 0):.4f}")
    print(f"  BLEU-2: {metrics.get('bleu2', 0):.4f}")
    print(f"  BLEU-3: {metrics.get('bleu3', 0):.4f}")
    print(f"  BLEU-4: {metrics.get('bleu4', 0):.4f}")
    print(f"  METEOR: {metrics.get('meteor', 0):.4f}")
    print(f"  ROUGE-L: {metrics.get('rouge_l', 0):.4f}")
    
    # Save results if requested
    if args.output_file:
        results = {
            'checkpoint': args.checkpoint,
            'stage': args.stage,
            'split': args.split,
            'num_samples': len(val_dataset),
            'val_loss': val_loss,
            'metrics': metrics,
            'generation_params': {
                'max_new_tokens': args.max_new_tokens,
                'temperature': args.temperature,
                'top_p': args.top_p
            }
        }
        
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to: {args.output_file}")
    
    # Print sample predictions
    print("\n" + "=" * 60)
    print("Sample Predictions (first 3)")
    print("=" * 60)
    for i in range(min(3, len(predictions))):
        print(f"\n--- Sample {i+1} ---")
        print(f"Reference: {references[i][:200]}...")
        print(f"Prediction: {predictions[i][:200]}...")
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
