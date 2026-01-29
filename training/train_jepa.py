
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt

from data.coco_dataset import CocoCaptionDataset
from models.vision_encoder import VisionEncoder
from models.text_encoder import TextEncoder
from models.predictor import JEPAPredictor
from models.target_encoders import build_target_encoder
from training.masking import mask_vision_patches
from training.ema import ema_update

# -----------------------------
# Training function (Refactored for Ablations)
# -----------------------------
def train_jepa_run(mask_ratio=0.6, epochs=2, save_name="jepa_training_loss"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting training with mask_ratio={mask_ratio} on {device}")
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    dataset = CocoCaptionDataset(
        image_dir="data/coco/images/val2017",
        annotation_file="data/coco/annotations/annotations/captions_val2017.json",
        tokenizer=tokenizer
    )
    
    # Use subset for speed/demo
    dataset = torch.utils.data.Subset(dataset, range(100))
    
    loader = DataLoader(
        dataset,
        batch_size=10, # Match demo batch size
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    loss_history = []
    tau = 0.996
    
    # Re-initialize models/optimizer per run
    vision_online = VisionEncoder().to(device)
    text_online = TextEncoder().to(device)
    predictor = JEPAPredictor().to(device)
    
    vision_target = build_target_encoder(vision_online).to(device)
    text_target = build_target_encoder(text_online).to(device)
    
    for p in vision_online.parameters(): p.requires_grad = False
    for p in text_online.parameters(): p.requires_grad = False
    
    mask_token = torch.nn.Parameter(torch.zeros(1, 1, 768, device=device))
    torch.nn.init.normal_(mask_token, std=0.02)
    
    optimizer = torch.optim.AdamW(list(predictor.parameters()) + [mask_token], lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        for images, text in tqdm(loader, desc=f"Epoch {epoch+1} (M={mask_ratio})"):
            images = images.to(device)
            text = {k: v.squeeze(1).to(device) for k, v in text.items()}
            
            with torch.no_grad():
                vision_patches = vision_online(images)
                text_tokens = text_online(text["input_ids"], text["attention_mask"])
            
            visible_patches, _, mask_idx = mask_vision_patches(vision_patches, mask_ratio=mask_ratio)
            
            context = []
            masked_counts = []
            
            for b in range(len(visible_patches)):
                num_masked = len(mask_idx[b])
                masked_counts.append(num_masked)
                mask_tokens_exp = mask_token.squeeze(0).expand(num_masked, -1)
                
                # Check dimensions for cat. vision_patches (N, D). text_tokens (M, D).
                ctx = torch.cat([visible_patches[b], text_tokens[b], mask_tokens_exp], dim=0)
                context.append(ctx)
            
            context = torch.stack(context)
            
            with torch.cuda.amp.autocast():
                pred = predictor(context)
                with torch.no_grad():
                    target_patches = vision_target(images)
                
                loss = 0.0
                for b in range(len(mask_idx)):
                    num_masked = masked_counts[b]
                    # Pred is (B, L, D). We want last num_masked tokens.
                    pred_b = pred[b, -num_masked:] 
                    target_b = target_patches[b, mask_idx[b]]
                    loss += 1 - F.cosine_similarity(pred_b, target_b).mean()
                
                loss /= len(mask_idx)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            ema_update(vision_target, vision_online, tau)
            ema_update(text_target, text_online, tau)
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}: JEPA loss = {avg_loss:.4f}")
        
    # Save plot
    os.makedirs("plots", exist_ok=True)
    plt.plot(loss_history, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("JEPA Loss")
    plt.title(f"VL-JEPA Loss (Mask Ratio {mask_ratio})")
    plt.savefig(f"plots/{save_name}.png")
    plt.close()
    
    return loss_history[-1]

if __name__ == "__main__":
    train_jepa_run(mask_ratio=0.6, epochs=2, save_name="jepa_training_loss")
