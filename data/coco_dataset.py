import json
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CocoCaptionDataset(Dataset):
    def __init__(self, image_dir, annotation_file, tokenizer=None):
        with open(annotation_file, "r") as f:
            data = json.load(f)

        self.image_dir = image_dir
        self.annotations = data["annotations"]
        self.images = {img["id"]: img["file_name"] for img in data["images"]}
        self.tokenizer = tokenizer

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_path = os.path.join(self.image_dir, self.images[ann["image_id"]])
        image = self.transform(Image.open(img_path).convert("RGB"))
        caption = ann["caption"]

        if self.tokenizer:
            caption = self.tokenizer(
                caption,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=32
            )

        return image, caption