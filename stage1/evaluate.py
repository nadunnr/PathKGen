"""
Evaluate Stage 1 alignment quality with retrieval metrics.

Metrics:
- Image→Text retrieval: Recall@1, Recall@5
- Text→Image retrieval: Recall@1, Recall@5
- Mean retrieval accuracy
"""

import argparse
import json
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import logging

from config import Stage1Config
from models import Stage1AlignmentModel
from dataset import (
    AlignmentDataset, collate_fn, load_dataset_csv,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@torch.no_grad()
def compute_retrieval_metrics(model, dataloader, device):
    """
    Compute full retrieval metrics over the dataset.
    
    Collects all projected embeddings, then computes similarity matrix 
    and recall@K.
    """
    model.eval()
    all_img = []
    all_txt = []
    
    for batch in dataloader:
        tokens = batch['tokens'].to(device)
        text_emb = batch['text_embeddings'].to(device)
        
        img_proj, txt_proj = model(tokens, text_emb)
        
        # Mean-pool image tokens if needed
        if img_proj.dim() == 3:
            img_proj = img_proj.mean(dim=1)  # [B, N, D] -> [B, D]
            img_proj = F.normalize(img_proj, p=2, dim=-1)
        
        all_img.append(img_proj.cpu())
        all_txt.append(txt_proj.cpu())
    
    all_img = torch.cat(all_img, dim=0)  # [N, D]
    all_txt = torch.cat(all_txt, dim=0)  # [N, D]
    
    # Full similarity matrix
    sim = all_img @ all_txt.T  # [N, N]
    N = sim.shape[0]
    labels = torch.arange(N)
    
    results = {}
    for k in [1, 5, 10]:
        if k > N:
            continue
        # Image → Text 
        topk_i2t = sim.topk(k, dim=1).indices
        i2t_recall = (topk_i2t == labels.unsqueeze(1)).any(dim=1).float().mean().item()
        
        # Text → Image
        topk_t2i = sim.T.topk(k, dim=1).indices
        t2i_recall = (topk_t2i == labels.unsqueeze(1)).any(dim=1).float().mean().item()
        
        results[f'i2t_R@{k}'] = i2t_recall
        results[f't2i_R@{k}'] = t2i_recall
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Stage 1 alignment")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help="Path to alignment checkpoint (.pt)")
    parser.add_argument('--test_csv', type=str, 
                        default="/home/sandun/nadun/data/dataset/report_dataset_test.csv")
    parser.add_argument('--slide_features_dir', type=str,
                        default="/home/sandun/nadun/data/slide_features")
    parser.add_argument('--text_embeddings_dir', type=str,
                        default="/home/sandun/nadun/data/text_embeddings")
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model_cfg = ckpt['config']
    logger.info(f"Loaded checkpoint from epoch {ckpt['epoch']}")
    
    # Build model (handle both old and new config formats)
    model = Stage1AlignmentModel(
        titan_dim=model_cfg.get('titan_dim', model_cfg.get('input_dim', 768)),
        text_dim=model_cfg.get('text_dim', model_cfg.get('titan_dim', 768)),
        hidden_dim=model_cfg['hidden_dim'],
        alignment_dim=model_cfg['alignment_dim'],
    ).to(args.device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    # Load data from CSV
    logger.info(f"Loading {args.split} data from CSV...")
    samples = load_dataset_csv(
        csv_path=args.test_csv,
        split=args.split,
    )
    
    # Build dataset
    dataset = AlignmentDataset(
        samples=samples,
        slide_features_dir=os.path.join(args.slide_features_dir, args.split),
        text_embeddings_dir=os.path.join(args.text_embeddings_dir, args.split),
    )
    loader = DataLoader(
        dataset, batch_size=32, shuffle=False,
        collate_fn=collate_fn, num_workers=2,
    )
    
    # Evaluate
    metrics = compute_retrieval_metrics(model, loader, args.device)
    
    logger.info(f"\nRetrieval results on '{args.split}' split ({len(dataset)} samples):")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")


if __name__ == '__main__':
    main()
