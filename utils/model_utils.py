import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:,  0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        # x = x + self.pe[:x]
        return self.pe[x]


if __name__ == '__main__':
    encod = PositionalEncoding(d_model=8, max_len=4)
    ip_tensor = torch.randint(0, 4, size=(5, 1))
    print(ip_tensor)
    print(encod(ip_tensor))
    print(encod(ip_tensor).shape)
