import torch
import torch.nn as nn

class JEPAPredictor(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                batch_first=True,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                activation="gelu"
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        x: (B, N_context, D)
        returns: (B, N_context, D)
        """
        for layer in self.layers:
            x = layer(x)

        return self.norm(x)
