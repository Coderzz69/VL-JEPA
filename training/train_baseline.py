import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data.coco_dataset import CocoCaptionDataset
from models.baseline import BaselineVL

# -----------------------------
# Device
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -----------------------------
# Tokenizer
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# -----------------------------
# Dataset & Loader
# -----------------------------
dataset = CocoCaptionDataset(
    image_dir="data/coco/images/val2017",
    annotation_file="data/coco/annotations/annotations/captions_val2017.json",
    tokenizer=tokenizer
)

loader = DataLoader(
    dataset,
    batch_size=4,          # SAFE for 4GB GPU
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

# -----------------------------
# Model
# -----------------------------
model = BaselineVL().to(device)

# Freeze encoders (baseline = control experiment)
for p in model.vision.parameters():
    p.requires_grad = False

for p in model.text.parameters():
    p.requires_grad = False

# -----------------------------
# Optimizer (train projections only)
# -----------------------------
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)

# -----------------------------
# Mixed Precision (memory saver)
# -----------------------------
scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

# -----------------------------
# Training Loop
# -----------------------------
epochs = 3

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for images, text in loader:
        images = images.to(device, non_blocking=True)
        text = {k: v.squeeze(1).to(device, non_blocking=True) for k, v in text.items()}

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            v, t = model(images, text)
            loss = 1 - F.cosine_similarity(v, t).mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

print("Baseline training complete.")
