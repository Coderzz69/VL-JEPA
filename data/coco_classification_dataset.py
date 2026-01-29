import json
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CocoClassificationDataset(Dataset):
    def __init__(self, image_dir, annotation_file):
        with open(annotation_file, "r") as f:
            data = json.load(f)

        self.image_dir = image_dir
        self.images = {img["id"]: img["file_name"] for img in data["images"]}
        self.categories = {cat["id"]: i for i, cat in enumerate(data["categories"])}
        self.num_classes = len(self.categories)

        # Filter annotations to only keep those with a valid category
        self.samples = []
        for ann in data["annotations"]:
            img_id = ann["image_id"]
            cat_id = ann["category_id"]
            if img_id in self.images and cat_id in self.categories:
                self.samples.append((img_id, self.categories[cat_id]))

        # Limit to unique image-label pairs to avoid duplicate training signal
        # Use a simple strategy: take the first label found for an image, or multi-label?
        # For simple linear probe, let's treat it as single-label multi-class (just pick one)
        # or better: duplicate images for multiple labels. Standard approach for linear probe on COCO is multi-label BCE or single label cross entropy.
        # Given "simple prototype", let's replicate image for each label (standard classification style)
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, label = self.samples[idx]
        filename = self.images[img_id]
        path = os.path.join(self.image_dir, filename)
        
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)
