import torch
from torch import nn

"""
Layer normalization between each layer in encoder/decoder block
Args:
    d_embedding: dimension of embedding vectors
    eps: added to layer statistic of standard deviation

Parameters (learnable) for calibrating normalization:
    gamma: scaling factor
    beta: offset factor
"""

class LayerNorm(nn.Module):
    def __init__(self, d_embedding, eps=1e-10):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_embedding))
        self.beta = nn.Parameter(torch.zeros(d_embedding))
        self.eps = eps

    def forward(self, x):
        """
        Inputs size:
            x: (batch_size, seq_len, d_embedding)
        Output size:
            out: (batch_size, seq_len, d_embedding)
        """
        mean = x.mean(-1, keepdim=True) # -1 specifies we're performing these across last dim of tensor, which is d_embedding
        var = x.var(-1, unbiased=False, keepdim=True)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out

