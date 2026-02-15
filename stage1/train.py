"""Stage 1 Training: WSI-Text Contrastive Alignment.

Follows WSI-LLaVA Stage 1:
- Freeze TITAN image & text encoders
- Train MLP projection heads via InfoNCE loss
- Align WSI and text embeddings in shared space
"""

import os
import sys
import json
import torch
import torch.nn as nn
import logging
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoModel
from tqdm import tqdm

from config import Stage1Config
from models import Stage1AlignmentModel, InfoNCELoss
from dataset import (
    AlignmentDataset,
    collate_fn,
    load_dataset_json,
    precompute_text_embeddings,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)



def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_retrieval_accuracy(logits: torch.Tensor):
    """Compute i2t and t2i retrieval accuracy from similarity matrix."""
    labels = torch.arange(logits.shape[0], device=logits.device)
    i2t_acc = (logits.argmax(dim=1) == labels).float().mean().item()
    t2i_acc = (logits.T.argmax(dim=1) == labels).float().mean().item()
    return i2t_acc, t2i_acc



def train_one_epoch(
    model: Stage1AlignmentModel,
    criterion: InfoNCELoss,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str,
    epoch: int,
    log_interval: int = 10,
):
    model.train()
    
    total_loss = 0.0
    total_i2t = 0.0
    total_t2i = 0.0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        tokens = batch['tokens'].to(device)
        text_emb = batch['text_embeddings'].to(device)
        
        # Forward and loss
        img_tokens, txt_proj = model(tokens, text_emb)
        loss, logits = criterion(img_tokens, txt_proj)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.trainable_parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        # Metrics
        i2t_acc, t2i_acc = compute_retrieval_accuracy(logits)
        B = tokens.shape[0]
        
        total_loss += loss.item() * B
        total_i2t += i2t_acc * B
        total_t2i += t2i_acc * B
        total_samples += B
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'i2t': f'{i2t_acc:.3f}',
            't2i': f'{t2i_acc:.3f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
        })
        
        if (batch_idx + 1) % log_interval == 0:
            logger.info(
                f"Epoch {epoch} [{batch_idx+1}/{len(dataloader)}] "
                f"loss={loss.item():.4f}  i2t={i2t_acc:.3f}  t2i={t2i_acc:.3f}"
            )
    
    return {
        'loss': total_loss / total_samples,
        'i2t_acc': total_i2t / total_samples,
        't2i_acc': total_t2i / total_samples,
        'avg_acc': (total_i2t + total_t2i) / (2 * total_samples),
    }


@torch.no_grad()
def validate(
    model: Stage1AlignmentModel,
    criterion: InfoNCELoss,
    dataloader: DataLoader,
    device: str,
):
    model.eval()
    
    total_loss = 0.0
    total_i2t = 0.0
    total_t2i = 0.0
    total_samples = 0
    
    for batch in dataloader:
        tokens = batch['tokens'].to(device)
        text_emb = batch['text_embeddings'].to(device)
        
        img_tokens, txt_proj = model(tokens, text_emb)
        loss, logits = criterion(img_tokens, txt_proj)
        
        i2t_acc, t2i_acc = compute_retrieval_accuracy(logits)
        B = tokens.shape[0]
        
        total_loss += loss.item() * B
        total_i2t += i2t_acc * B
        total_t2i += t2i_acc * B
        total_samples += B
    
    if total_samples == 0:
        return {'loss': 0, 'i2t_acc': 0, 't2i_acc': 0, 'avg_acc': 0}
    
    return {
        'loss': total_loss / total_samples,
        'i2t_acc': total_i2t / total_samples,
        't2i_acc': total_t2i / total_samples,
        'avg_acc': (total_i2t + total_t2i) / (2 * total_samples),
    }


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, best_loss, cfg, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'best_loss': best_loss,
        'config': {
            'titan_dim': cfg.titan_dim,
            'hidden_dim': cfg.hidden_dim,
            'alignment_dim': cfg.alignment_dim,
        },
    }, path)


# ───────────────────────── main ────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 1: WSI-Text Contrastive Alignment")
    parser.add_argument('--data_json', type=str, default=None)
    parser.add_argument('--feature_dir', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--hidden_dim', type=int, default=None)
    parser.add_argument('--alignment_dim', type=int, default=None)
    parser.add_argument('--checkpoint_dir', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()
    
    # Build config from CLI args
    cfg = Stage1Config()
    for key, val in vars(args).items():
        if val is not None and hasattr(cfg, key):
            setattr(cfg, key, val)
    cfg.__post_init__()
    
    set_seed(cfg.seed)
    
    logger.info("=" * 60)
    logger.info("Stage 1: WSI-Text Contrastive Alignment")
    logger.info("=" * 60)
    logger.info(f"Data: {cfg.data_json}")
    logger.info(f"Architecture: TITAN tokens → zero-pad → CLIP → token-wise MLP")
    logger.info(f"Dims: hidden={cfg.hidden_dim}, align={cfg.alignment_dim}")
    logger.info(f"Training: lr={cfg.lr}, bs={cfg.batch_size}, epochs={cfg.epochs}")
    logger.info(f"Temperature: {cfg.temperature}")
    logger.info("=" * 60)
    
    # Load TITAN for text encoding
    logger.info("Loading TITAN model...")
    titan = AutoModel.from_pretrained(cfg.titan_model_name, trust_remote_code=True)
    titan = titan.to(cfg.device).eval()
    for p in titan.parameters():
        p.requires_grad = False
    
    # Load data
    train_samples = load_dataset_json(cfg.data_json, split="train")
    
    # Override feature paths if specified
    if cfg.feature_dir:
        for s in train_samples:
            s['feature_path'] = os.path.join(cfg.feature_dir, os.path.basename(s['feature_path']))
    
    # Load validation set if available
    try:
        val_samples = load_dataset_json(cfg.data_json, split="val")
        if cfg.feature_dir:
            for s in val_samples:
                s['feature_path'] = os.path.join(cfg.feature_dir, os.path.basename(s['feature_path']))
    except KeyError:
        val_samples = None
        logger.info("No validation split found.")
    
    # Pre-compute text embeddings
    logger.info("Pre-computing text embeddings...")
    train_text_embs = precompute_text_embeddings(
        [s['report'] for s in train_samples], titan,
        device=cfg.device, batch_size=cfg.batch_size, context_length=cfg.context_length,
    )
    
    if val_samples:
        val_text_embs = precompute_text_embeddings(
            [s['report'] for s in val_samples], titan,
            device=cfg.device, batch_size=cfg.batch_size, context_length=cfg.context_length,
        )
    
    # Free TITAN from GPU
    del titan
    torch.cuda.empty_cache()
    
    # Build datasets & dataloaders
    train_dataset = AlignmentDataset(
        samples=train_samples,
        text_embeddings=train_text_embs,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    logger.info(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    
    val_loader = None
    if val_samples:
        val_dataset = AlignmentDataset(
            samples=val_samples,
            text_embeddings=val_text_embs,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        logger.info(f"Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    
    # Build model, optimizer, scheduler, loss
    model = Stage1AlignmentModel(
        titan_dim=cfg.titan_dim,
        text_dim=cfg.titan_dim,
        hidden_dim=cfg.hidden_dim,
        alignment_dim=cfg.alignment_dim,
        clip_model_name=cfg.clip_model_name,
        freeze_clip=cfg.freeze_clip,
    ).to(cfg.device)
    
    model.print_trainable_params()
    
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )
    
    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.lr,
        total_steps=total_steps,
        pct_start=cfg.warmup_ratio,
        anneal_strategy='cos',
    )
    
    criterion = InfoNCELoss(temperature=cfg.temperature)
    
    # Resume from checkpoint if requested
    start_epoch = 1
    best_loss = float('inf')
    if args.resume:
        ckpt = torch.load(args.resume, map_location=cfg.device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if ckpt.get('scheduler_state_dict'):
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_loss = ckpt.get('best_loss', float('inf'))
        logger.info(f"Resumed from epoch {ckpt['epoch']}, best_loss={best_loss:.4f}")
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(start_epoch, cfg.epochs + 1):
        train_metrics = train_one_epoch(
            model, criterion, train_loader, optimizer, scheduler,
            cfg.device, epoch, cfg.log_interval,
        )
        
        logger.info(
            f"Epoch {epoch}/{cfg.epochs}  "
            f"train_loss={train_metrics['loss']:.4f}  "
            f"i2t={train_metrics['i2t_acc']:.3f}  "
            f"t2i={train_metrics['t2i_acc']:.3f}  "
            f"avg={train_metrics['avg_acc']:.3f}"
        )
        
        # Validation
        if val_loader:
            val_metrics = validate(model, criterion, val_loader, cfg.device)
            logger.info(
                f"         val_loss={val_metrics['loss']:.4f}  "
                f"i2t={val_metrics['i2t_acc']:.3f}  "
                f"t2i={val_metrics['t2i_acc']:.3f}  "
                f"avg={val_metrics['avg_acc']:.3f}"
            )
            current_loss = val_metrics['loss']
        else:
            current_loss = train_metrics['loss']
        
        # Save best
        if current_loss < best_loss:
            best_loss = current_loss
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                train_metrics, best_loss, cfg,
                os.path.join(cfg.checkpoint_dir, "alignment_best.pt"),
            )
            logger.info(f"  -> New best! Saved alignment_best.pt (loss={best_loss:.4f})")
        
        # Save periodic
        if epoch % cfg.save_every_n_epochs == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                train_metrics, best_loss, cfg,
                os.path.join(cfg.checkpoint_dir, f"alignment_epoch_{epoch}.pt"),
            )
    
    # Final save
    save_checkpoint(
        model, optimizer, scheduler, cfg.epochs,
        train_metrics, best_loss, cfg,
        os.path.join(cfg.checkpoint_dir, "alignment_final.pt"),
    )
    
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"Best loss: {best_loss:.4f}")
    logger.info(f"Checkpoints: {cfg.checkpoint_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
