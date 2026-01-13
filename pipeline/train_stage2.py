"""
Stage 2 Training: Projection + LLM with QLoRA

Fine-tunes both the projection layer and LLM using QLoRA for memory efficiency.

Usage:
    python pipeline/train_stage2.py \
        --stage1_checkpoint /path/to/stage1/best_model.pt \
        --epochs 5 \
        --batch_size 2 \
        --lr 2e-5
"""

import os
import argparse
import json
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer

# Local imports
from model import VisionLanguageModel
from dataset import WSI_Report_Dataset
from data_collator import VLMDataCollator


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2 Training: Projection + QLoRA")
    
    # Data arguments
    parser.add_argument("--data_json", type=str, 
                       default="/home/nadun/wd/vlm_histo/data/curated_breast_dataset.json",
                       help="Path to dataset JSON")
    parser.add_argument("--feature_key", type=str, default="features",
                       help="Key in H5 files for slide features")
    
    # Model arguments
    parser.add_argument("--llm_path", type=str,
                       default="/home/nadun/wd/vlm_histo/checkpoints/Qwen2.5-7B-Instruct",
                       help="Path to base LLM")
    parser.add_argument("--slidechat_checkpoint", type=str,
                       default="/home/nadun/wd/vlm_histo/checkpoints/SlideChat_Weight/stage2_pth/mp_rank_00_model_states.pt",
                       help="Path to SlideChat weights")
    parser.add_argument("--stage1_checkpoint", type=str, required=True,
                       help="Path to Stage 1 checkpoint (for projection weights)")
    parser.add_argument("--visual_dim", type=int, default=768,
                       help="Dimension of slide features")
    parser.add_argument("--num_visual_tokens", type=int, default=32,
                       help="Number of visual tokens")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=5,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Training batch size (small for QLoRA)")
    parser.add_argument("--val_batch_size", type=int, default=4,
                       help="Validation batch size")
    parser.add_argument("--lr", type=float, default=2e-5,
                       help="Learning rate for LLM LoRA")
    parser.add_argument("--projection_lr", type=float, default=1e-4,
                       help="Learning rate for projection layer")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps (increase for smaller batch size)")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                       help="Enable gradient checkpointing to save memory")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="DataLoader workers")
    
    # QLoRA arguments
    parser.add_argument("--lora_rank", type=int, default=64,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=128,
                       help="LoRA alpha")
    
    # Checkpointing
    parser.add_argument("--output_dir", type=str, 
                       default="/home/nadun/wd/vlm_histo/checkpoints/stage2",
                       help="Output directory for checkpoints")
    parser.add_argument("--save_every", type=int, default=1,
                       help="Save checkpoint every N epochs")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Resume training from Stage 2 checkpoint")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--log_every", type=int, default=10,
                       help="Log every N steps")
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


def train_epoch(model, dataloader, optimizer, scheduler, epoch, args, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
    
    for step, batch in enumerate(pbar):
        # Move batch to device
        visual_features = batch['visual_features'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = model(
            visual_features=visual_features,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs['loss']
        
        # Scale loss for gradient accumulation
        loss = loss / args.gradient_accumulation_steps
        loss.backward()
        
        # Optimizer step
        if (step + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * args.gradient_accumulation_steps
        
        # Update progress bar
        if step % args.log_every == 0:
            pbar.set_postfix({
                'loss': f'{loss.item() * args.gradient_accumulation_steps:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


@torch.no_grad()
def validate(model, dataloader, epoch, args, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
    
    for batch in pbar:
        visual_features = batch['visual_features'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            visual_features=visual_features,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        total_loss += outputs['loss'].item()
        pbar.set_postfix({'val_loss': f'{outputs["loss"].item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save args
    with open(os.path.join(args.output_dir, 'train_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize datasets
    print("Loading datasets...")
    train_dataset = WSI_Report_Dataset(
        json_path=args.data_json,
        split='train',
        feature_key=args.feature_key
    )
    
    val_dataset = WSI_Report_Dataset(
        json_path=args.data_json,
        split='test',
        feature_key=args.feature_key
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Initialize collator
    collator = VLMDataCollator(
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True
    )
    
    # Initialize model with QLoRA
    print("\n" + "="*60)
    print("Initializing VLM for Stage 2 Training (QLoRA)")
    print("="*60)
    
    model = VisionLanguageModel(
        llm_path=args.llm_path,
        slidechat_checkpoint=args.slidechat_checkpoint,
        visual_dim=args.visual_dim,
        num_visual_tokens=args.num_visual_tokens,
        use_qlora=True,    # Stage 2: Use QLoRA
        freeze_llm=False,  # Stage 2: Train LLM with LoRA
        gradient_checkpointing=args.gradient_checkpointing,
        device=device
    )
    
    # Load Stage 1 projection weights
    print(f"\nLoading Stage 1 projection weights from: {args.stage1_checkpoint}")
    checkpoint = torch.load(args.stage1_checkpoint, map_location=device)
    model.projection.load_state_dict(checkpoint['projection_state_dict'])
    print("✓ Projection weights loaded from Stage 1")
    
    total_params, trainable_params = model.count_parameters()
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Optimizer with separate learning rates
    # LoRA parameters: lower LR
    # Projection parameters: higher LR
    param_groups = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if p.requires_grad and 'projection' in n],
            'lr': args.projection_lr,
            'name': 'projection'
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if p.requires_grad and 'projection' not in n],
            'lr': args.lr,
            'name': 'lora'
        }
    ]
    
    optimizer = AdamW(param_groups, weight_decay=args.weight_decay)
    
    print(f"\nOptimizer parameter groups:")
    for group in optimizer.param_groups:
        n_params = sum(p.numel() for p in group['params'])
        print(f"  {group['name']}: {n_params:,} params, lr={group['lr']:.2e}")
    
    # Learning rate scheduler
    num_training_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=1e-7)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume_from is not None:
        print(f"\nResuming from checkpoint: {args.resume_from}")
        start_epoch, optimizer_state = model.load_checkpoint(args.resume_from, load_optimizer=True)
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        start_epoch += 1
    
    # Training loop
    print("\n" + "="*60)
    print("Starting Stage 2 Training")
    print("="*60)
    
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, epoch, args, device)
        history['train_loss'].append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, epoch, args, device)
        history['val_loss'].append(val_loss)
        
        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt')
            model.save_checkpoint(save_path, epoch, optimizer.state_dict())
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.output_dir, 'best_model.pt')
            model.save_checkpoint(best_path, epoch, optimizer.state_dict())
            print(f"  ✓ New best model saved! (val_loss: {val_loss:.4f})")
    
    # Save final model
    final_path = os.path.join(args.output_dir, 'final_model.pt')
    model.save_checkpoint(final_path, args.epochs - 1, optimizer.state_dict())
    
    # Save training history
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*60)
    print("Stage 2 Training Complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
