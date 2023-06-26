import torch
from torch import nn
import math

"""
Creates pe (position encoding) tensor of size = (1, max_seq_len, d_embedding)

Args:
    d_embedding: dimension of embedding vectors
    max_seq_len: max num of tokens (words) in input seq (sentence)
"""

class PositionEncoder(nn.Module):
    def __init__(self, d_embedding, max_seq_len):
        super().__init__()
        self.d_embedding = d_embedding

        pe = torch.zeros(max_seq_len, d_embedding, requires_grad=False)
        for pos in range(max_seq_len):
            for i in range(0, d_embedding, 2):
                pe[pos,i] = math.sin(pos / 10000 ** (2*i/d_embedding))
                pe[pos,i+1] = math.cos(pos / 10000 ** (2*(i+1)/d_embedding))

        pe.unsqueeze(0) # add dimension to torch tensor -> now size
        self.register_buffer('pe', pe) # register pe tensor as non-learnable to not update it during training


    def forward(self, x):
        """
        Args:
            x: input sequence tensor (semantics encoded)
            out: input tensor plus pe tensor (semantics + position encoded)
        """
        seq_len = x.size(1)
        out = x + self.pe[:seq_len,:]
        return out
    

        #https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec#d554 -> has :seq along dim 1 of pe



