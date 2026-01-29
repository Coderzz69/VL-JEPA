import torch
import torch.nn as nn
import timm
from transformers import AutoModel

class BaselineVL(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()

        self.vision = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=0
        )

        self.text = AutoModel.from_pretrained("bert-base-uncased")

        self.text_proj = nn.Linear(768, embed_dim)
        self.vision_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, images, text_inputs):
        v = self.vision(images)
        v = self.vision_proj(v)

        t = self.text(**text_inputs).last_hidden_state[:, 0]
        t = self.text_proj(t)

        return v, t
