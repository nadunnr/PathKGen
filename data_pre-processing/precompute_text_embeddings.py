"""Standalone script to pre-compute and save text embeddings using TITAN."""

import argparse
import os
import h5py
import pandas as pd
import torch
from transformers import AutoModel
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@torch.no_grad()
def encode_texts_batch(texts, titan_model, device, context_length=128):
    """Encode a batch of texts with TITAN."""
    input_ids = titan_model.text_encoder.tokenizer(texts, context_length=context_length)
    input_ids = input_ids.to(device)
    
    with torch.autocast('cuda', torch.float16):
        embeddings = titan_model.encode_text(input_ids, normalize=False)
    
    return embeddings.float().cpu()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_csv', type=str, required=False, 
                        default='/home/nadun/wd/datasets/wsi_bench/dataset/report_dataset_test.csv')
    parser.add_argument('--output_dir', type=str, required=False, 
                        default='/home/nadun/wd/datasets/wsi_bench/text_embeddings/test')
    parser.add_argument('--titan_model', type=str, default='MahmoodLab/TITAN')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--context_length', type=int, default=128,
                        help='TITAN text context length (default: 128)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--id_column', type=str, default='tcga_id',
                        help='Column name for sample IDs')
    parser.add_argument('--text_column', type=str, default='report_content',
                        help='Column name for text content')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    samples = pd.read_csv(args.data_csv).to_dict(orient='records')
    logger.info(f"Loaded {len(samples)} samples from {args.data_csv}")
    
    # Load TITAN
    logger.info("Loading TITAN model...")
    titan = AutoModel.from_pretrained(args.titan_model, trust_remote_code=True)
    titan = titan.to(args.device)
    titan.eval()
    
    # Process in batches and save individually
    logger.info(f"Encoding and saving to {args.output_dir} (format: .pt)")
    
    for i in tqdm(range(0, len(samples), args.batch_size), desc="Encoding batches"):
        batch_samples = samples[i:i + args.batch_size]
        batch_texts = [s[args.text_column] for s in batch_samples]
        batch_ids = [s[args.id_column] for s in batch_samples]
        
        # Encode batch
        embeddings = encode_texts_batch(
            batch_texts, titan, args.device, args.context_length
        )  # [B, 768]
        
        for j, (sample_id, emb) in enumerate(zip(batch_ids, embeddings)):
            out_path = os.path.join(args.output_dir, f"{sample_id}.h5")
            with h5py.File(out_path, 'w') as f:
                f.create_dataset('features', data=emb.numpy(), dtype='float32')

    
    logger.info(f"Done! Saved {len(samples)} text embeddings to {args.output_dir}")


if __name__ == '__main__':
    main()
