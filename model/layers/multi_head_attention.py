import torch
from torch import nn
import torch.nn.functional as F
import math

"""
Computes multi-head attention
Args:
    d_embedding: dimension of embedding vectors
    n_heads: number of heads
"""

class MultiHeadAttention(nn.Module):
    def __init__(self, d_embedding, n_heads):
        super().__init__()
        self.d_embedding = d_embedding
        self.n_heads = n_heads
        self.d_single_head = int(d_embedding / n_heads)

        # projection matrices
        self.w_q = nn.Linear(self.d_embedding, self.d_embedding) # concat of w_q_i's described in the paper
        self.w_k = nn.Linear(self.d_embedding, self.d_embedding)
        self.w_v = nn.Linear(self.d_embedding, self.d_embedding)
        self.w_o = nn.Linear(d_embedding, d_embedding)
    

    def forward(self, q, k, v, mask=None):
        """
        Inputs size:
            q, k, v: (batch_size, seq_len, d_embedding)
        Output size:
            out: (batch_size, seq_len, d_embedding)
        """
        # 1. pass through linear layer (multiply by matrices)
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. split by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v) # resulting q, k, v have size=(batch_size, n_heads, seq_len, d_single_head)

        # 3. compute scaled dot-product attention
        weighted_v = self.attention(q, k, v, mask=mask)

        # 4. concat
        concat = self.concat(weighted_v) # concat has size (batch_size, seq_len, d_embedding)

        # 5. output layer
        out = self.w_o(concat) 

        return out # out has size (batch_size, seq_len, d_embedding)


    """
    Helper methods: computes scaled dot-product attention
    """
    def attention(self, q, k, v, mask): # note how to handle mask
        """
        Helper method: compute scaled dot-product attention
        Inputs (q, k, v) size: (batch_size, n_heads, seq_len, d_single_head)
        Output size: (batch_size, n_heads, seq_len, d_single_head)
        """
        scores = torch.matmul(q, k.transpose(-1, -2)) # 4d matrix multiplication is just batched 2d multiplication
        scores = scores / math.sqrt(self.d_single_head) # score has size=(batch_size, n_heads, seq_len, seq_len)

        if mask is not None:
            scores = scores.masked_fill(mask==0, value=-1e30 if scores.dtype == torch.float32 else -1e+4) # where mask is true, change value to -inf 

        scores = F.softmax(scores, dim=-1)
        weighted_v = torch.matmul(scores, v)
        return weighted_v
    
    def split(self, tensor):
        """
        Helper method: splits q, k, v tensors by number of heads
        Input size:(batch_size, seq_len, d_embedding)
        Output size: (batch_size, n_heads, seq_len, d_single_head)
        """
        batch_size, seq_len, d_embedding = tensor.size()
        tensor = tensor.view(batch_size, seq_len, self.n_heads, self.d_single_head) # convert tensors from size=(batch_size, seq_len, d_embedding) to size=(batch_size, seq_len, n_heads, d_single_head)
        tensor = tensor.transpose(1,2) # swap axis 1 and 2 to get size=(batch_size, n_heads, seq_len, d_single_head)
        return tensor
    
    def concat(self, tensor):
        """
        Helper function: concat (opposite of split)
        Input size: (batch_size, n_heads, seq_len, d_single_head)
        Output size: (batch_size, seq_len, d_embedding)
        """
        batch_size, n_heads, seq_len, d_single_head = tensor.size()
        tensor = tensor.transpose(1, 2)
        tensor = tensor.contiguous().view(batch_size, seq_len, self.d_embedding)
        return tensor

