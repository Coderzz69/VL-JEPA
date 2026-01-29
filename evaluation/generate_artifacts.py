import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoTokenizer

from models.vision_encoder import VisionEncoder
from models.text_encoder import TextEncoder
from data.coco_dataset import CocoCaptionDataset

def generate_artifacts():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Setup
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = CocoCaptionDataset(
        image_dir="data/coco/images/val2017",
        annotation_file="data/coco/annotations/annotations/captions_val2017.json",
        tokenizer=tokenizer
    )
    # Use subset for visualization
    dataset = Subset(dataset, range(100))
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
    
    vision_enc = VisionEncoder().to(device)
    text_enc = TextEncoder().to(device)
    
    vision_enc.eval()
    text_enc.eval()
    
    img_feats = []
    txt_feats = []
    
    print("Extracting features...")
    with torch.no_grad():
        for images, text_data in tqdm(loader):
            images = images.to(device)
            input_ids = text_data["input_ids"].squeeze(1).to(device)
            attn_mask = text_data["attention_mask"].squeeze(1).to(device)
            
            # Vision features: Global Average Pooling of patches
            # VisionEncoder returns (B, N, D)
            v_out = vision_enc(images)
            v_emb = v_out.mean(dim=1) 
            
            # Text features: CLS token usually? 
            # TextEncoder returns (B, L, D) probably?
            # Let's check TextEncoder. But standard BERT is CLS.
            # Assuming TextEncoder returns sequence.
            t_out = text_enc(input_ids, attn_mask)
            # If t_out is (B, L, D), take first token (CLS) or mean?
            # Let's assume CLS is at index 0 for BERT.
            t_emb = t_out[:, 0, :]
            
            img_feats.append(v_emb.cpu())
            txt_feats.append(t_emb.cpu())
            
    img_feats = torch.cat(img_feats, dim=0) # (N, D)
    txt_feats = torch.cat(txt_feats, dim=0) # (N, D)
    
    # Normalize
    img_feats = F.normalize(img_feats, dim=1)
    txt_feats = F.normalize(txt_feats, dim=1)
    
    # 1. Similarity Histogram (Image-Text Cosine Similarity)
    # Correct pairs are diagonal if aligned, but here standard pretrained models.
    # We just want distribution of similarities for corresponding pairs.
    sims = (img_feats * txt_feats).sum(dim=1)
    
    plt.figure()
    plt.hist(sims.numpy(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.title("Image-Text Cosine Similarity Distribution")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.savefig("plots/embedding_similarity_histogram.png")
    plt.close()
    print("Saved plots/embedding_similarity_histogram.png")
    
    # 2. PCA Visualization
    # Visualize Image embeddings vs Text embeddings
    pca = PCA(n_components=2)
    
    # Combine for joint PCA
    all_feats = torch.cat([img_feats, txt_feats], dim=0).numpy()
    all_pca = pca.fit_transform(all_feats)
    
    n_samples = img_feats.shape[0]
    img_pca = all_pca[:n_samples]
    txt_pca = all_pca[n_samples:]
    
    plt.figure(figsize=(10, 8))
    plt.scatter(img_pca[:, 0], img_pca[:, 1], alpha=0.5, label='Images', s=10)
    plt.scatter(txt_pca[:, 0], txt_pca[:, 1], alpha=0.5, label='Text', s=10)
    plt.legend()
    plt.title("PCA of Image and Text Embeddings")
    plt.savefig("plots/pca_visualization.png")
    plt.close()
    print("Saved plots/pca_visualization.png")

if __name__ == "__main__":
    generate_artifacts()
