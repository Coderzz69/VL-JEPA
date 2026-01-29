import timm
import torch.nn as nn

class VisionEncoder(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()

        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=0
        )

        self.embed_dim = embed_dim

    def forward(self, images):
        """
        Returns patch embeddings: (B, N, D)
        """
        x = self.vit.patch_embed(images)
        x = self.vit.pos_drop(x + self.vit.pos_embed[:, 1:, :])

        for blk in self.vit.blocks:
            x = blk(x)

        x = self.vit.norm(x)
        return x
