import torch
import torch.nn as nn
import numpy as np


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        # MatMul
        attn = torch.bmm(q, k.transpose(1, 2))
        # Scale
        attn = attn / self.temperature

        if mask is not None:
            # Mask
            attn = attn.masked_fill(mask, -np.inf)

        # Softmax
        attn = self.softmax(attn)
        # Dropout
        attn = self.dropout(attn)
        # MatMul
        output = torch.bmm(attn, v)

        return output, attn
