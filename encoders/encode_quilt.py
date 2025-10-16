import pandas as pd
from pathlib import Path
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

model_cfg = 'conch_ViT-B-16'
#4checkpoint_path = '/checkpoints/CONCH/pytorch_model.bin'
checkpoint_path = "/home/nadun/hdd/literature-guided-digital-histopathology/checkpoints/conch/pytorch_model.bin"

model, preprocess = create_model_from_pretrained(model_cfg, checkpoint_path)
_ = model.eval()

def conch_encode_img(image_path, image_only, model_cfg='conch_ViT-B-16'):
    
    image = Image.open(image_path)
    image = preprocess(image).unsqueeze(0)

    with torch.inference_mode():
        if image_only:
            #For image-only tasks, it is common to directly use the representation before the projection head and l2-normalization.
            # This is done by setting **proj_contrast=False** and **normalize=False**.
            image_embs = model.encode_image(image, proj_contrast=False, normalize=False)
        else:
            image_embs = model.encode_image(image)

    #print(f"image embeddings: {image_embs.shape}")

    return image_embs
 

def conch_encode_text(texts, model_cfg='conch_ViT-B-16'):

    tokenizer = get_tokenizer() # load tokenizer
    text_tokens = tokenize(texts=texts, tokenizer=tokenizer) # tokenize the text
    text_embs = model.encode_text(text_tokens)

    #print(f"text embeddings: text_embs.shape", text_embs.shape)

    return text_embs

class HistopathTextImageDataset(Dataset):
    def __init__(self, csv_path, images_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        # self.image_encoder = image_encoder
        # self.text_encoder = text_encoder
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_rel_path = row["image_path"]
        caption = row["caption"]
        img_full_path = self.images_dir / img_rel_path
        img_embedding = conch_encode_img(img_full_path, image_only=False).detach()
        text_embeds = conch_encode_text([caption]).detach()

        return img_embedding, text_embeds

def calculate_similarity(img_embeds, text_embeds):
    image_embs = img_embeds.squeeze(1)
    text_embeds = text_embeds.squeeze(1)

    image_embs = image_embs / image_embs.norm(dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    sim = torch.sum(image_embs * text_embeds, dim=-1)  # (batch_size,)
    return sim  # or sim.unsqueeze(1)


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Hardcode your paths for now
datapath = Path("/home/nadun/hdd/QUILT1M")
csv_path = datapath / "quilt_1M_lookup.csv"
img_path = datapath / "quilt_1m"

dataset = HistopathTextImageDataset(csv_path=csv_path, images_dir=img_path)
# dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

# def safe_collate(batch):
#     imgs, captions = zip(*batch)
#     imgs = torch.stack(imgs, dim=0)  # default, no out= usage
#     return imgs, list(captions)

def safe_collate(batch):
    imgs, texts = zip(*batch)
    # imgs is tuple of tensors (each [1,512] or [C, H, W] etc.)
    imgs = torch.stack(imgs, dim=0)  # simple stacking
    texts = torch.stack(texts, dim=0)  # if they are tensors of same shape
    return imgs, texts

dataloader = DataLoader(dataset, batch_size=4096, shuffle=False,
                        num_workers=0, collate_fn=safe_collate)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

count = 0
n = 10


import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch

def plot_tsne_embeddings(
    img_embs,   # np array shape (N, d)
    text_embs,  # np array shape (N, d)
    itr,
    title_prefix="t-SNE",
    save_prefix="tsne",
    show_plot=False,
    save_path=None
):
    """
    Produce three plots:
      1. Combined image + text embeddings in same 2D space (colored)
      2. Only image embeddings in 2D
      3. Only text embeddings in 2D

    img_embs, text_embs: same embedding dimension d, shape (N, d).
    """
    assert img_embs.shape[1] == text_embs.shape[1], "Embedding dims must match"
    N = img_embs.shape[0]

    # Stack for combined
    X_combined = np.concatenate([img_embs, text_embs], axis=0)  # (2N, d)
    labels = np.array([0] * N + [1] * N)  # 0=image, 1=text

    # Run t-SNE on combined
    tsne = TSNE(n_components=2, random_state=42, init="pca", perplexity=30)
    X2_combined = tsne.fit_transform(X_combined)

    # Split back
    X2_img = X2_combined[:N]
    X2_txt = X2_combined[N:]

    # 1) Combined plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X2_img[:, 0], X2_img[:, 1], c='blue', label='image', s=10, alpha=0.7)
    plt.scatter(X2_txt[:, 0], X2_txt[:, 1], c='red', label='text', s=10, alpha=0.7)
    plt.legend()
    plt.title(f"{title_prefix} (image + text)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    combined_file = f"encoders/results/{save_prefix}_combined_itr{itr}.png"
    plt.savefig(combined_file, dpi=300, bbox_inches="tight")
    print("Saved combined t-SNE to", combined_file)
    if show_plot:
        plt.show()
    plt.close()

    # 2) Image-only plot
    plt.figure(figsize=(6, 5))
    plt.scatter(X2_img[:, 0], X2_img[:, 1], c='blue', label='image only', s=10, alpha=0.7)
    plt.legend()
    plt.title(f"{title_prefix} (image only)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    img_file = f"encoders/results/{save_prefix}_image_itr{itr}.png"
    plt.savefig(img_file, dpi=300, bbox_inches="tight")
    print("Saved image-only t-SNE to", img_file)
    if show_plot:
        plt.show()
    plt.close()

    # 3) Text-only plot
    plt.figure(figsize=(6, 5))
    plt.scatter(X2_txt[:, 0], X2_txt[:, 1], c='red', label='text only', s=10, alpha=0.7)
    plt.legend()
    plt.title(f"{title_prefix} (text only)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    txt_file = f"encoders/results/{save_prefix}_text_itr{itr}.png"
    plt.savefig(txt_file, dpi=300, bbox_inches="tight")
    print("Saved text-only t-SNE to", txt_file)
    if show_plot:
        plt.show()
    plt.close()

    return X2_combined, X2_img, X2_txt



for imgs, captions in dataloader:

    #if count == n: break
    print("Iteration", count)

    # captions is a list of strings
    #print("Batch imgs:", imgs.shape)
    #print("Caption length", captions.shape)

    similarity = calculate_similarity(imgs, captions)
    average_similarity = similarity.mean().item()
    print("Average similarity:", average_similarity)
    # Here you would tokenize captions and pass them through the text encoder
    # tokenized = tokenizer(captions, ...)
    # text_embeds = text_encoder(**tokenized)
    # img_embeds = image_encoder(imgs)

    # Example usage:
    # Suppose you have torch tensors `emb_img` and `emb_txt` of shape (N, d)
    emb_img_np = imgs.squeeze(1).cpu().numpy()
    emb_txt_np = captions.squeeze(1).cpu().numpy()

    result_save_path = f"encoders/results/tsne_img_text_{count}.png"

    plot_tsne_embeddings(
    emb_img_np,   # np array shape (N, d)
    emb_txt_np,  # np array shape (N, d)
    itr=count,
    title_prefix="t-SNE",
    save_prefix="tsne",
    show_plot=False,
    save_path=None)

    #break
    count += 1
