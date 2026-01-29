import torch.nn as nn
from transformers import AutoModel

class TextEncoder(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()

        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        self.proj = nn.Linear(768, embed_dim)

    def forward(self, input_ids, attention_mask):
        """
        Returns token embeddings: (B, T, D)
        """
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        x = out.last_hidden_state
        x = self.proj(x)
        return x
