import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models.vision_encoder import VisionEncoder
from data.coco_classification_dataset import CocoClassificationDataset

def train_linear_probe():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Dataset
    dataset = CocoClassificationDataset(
        image_dir="data/coco/images/val2017",
        annotation_file="data/coco/annotations/annotations/instances_val2017.json"
    )
    
    # Fast proxy: subset 100 samples for demo
    dataset = Subset(dataset, range(100))

    loader = DataLoader(
        dataset,
        batch_size=10,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # Models
    encoder = VisionEncoder().to(device)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    # Linear Classifier
    classifier = nn.Linear(768, 80).to(device) # 80 COCO classes
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    epochs = 2
    acc_history = []

    print("Starting Linear Probe Training...")
    
    for epoch in range(epochs):
        total_correct = 0
        total_samples = 0
        
        for images, labels in tqdm(loader, desc=f"Epoch {epoch+1}"):
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                features = encoder(images) 
                # encoder returns (B, N, D), we need global pool?
                # VisionEncoder returns (B, N, D) and includes CLS token usually?
                # Let's check VisionEncoder. It uses patch_embed + blocks + norm.
                # Usually standard ViT outputs sequence.
                # Let's simple perform mean pooling.
                features = features.mean(dim=1)

            logits = classifier(features)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

        acc = total_correct / total_samples
        acc_history.append(acc)
        print(f"Epoch {epoch+1}: Accuracy = {acc:.4f}")

    # Save Plot
    os.makedirs("plots", exist_ok=True)
    plt.plot(acc_history, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Linear Probe Accuracy (Frozen Encoder)")
    plt.savefig("plots/linear_probe_accuracy.png")
    plt.close()
    
    print(f"Final Accuracy: {acc_history[-1]:.4f}")
    print("Plot saved to plots/linear_probe_accuracy.png")

if __name__ == "__main__":
    train_linear_probe()
