from torch import nn

from model.layers.multi_head_attention import MultiHeadAttention
from model.layers.feed_forward import FeedForward
from model.layers.layer_norm import LayerNorm


"""
Single encoder block
"""
class EncoderBlock(nn.Module):

    def __init__(self, d_embedding, ff_d_hidden, n_heads, p_drop):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_embedding=d_embedding, n_heads=n_heads)
        self.drop1 = nn.Dropout(p=p_drop)
        self.norm1 = LayerNorm(d_embedding=d_embedding)

        self.ff = FeedForward(d_embedding=d_embedding, d_hidden=ff_d_hidden, p_drop=p_drop)
        self.drop2 = nn.Dropout(p=p_drop)
        self.norm2 = LayerNorm(d_embedding=d_embedding)

    def forward(self, x, src_mask):
        """
        Inputs size:
            x: (batch_size, seq_len, d_embedding)
        Output size: 
            x: (batch_size, seq_len, d_embedding)
        """
        # 1. self attention
        _x = x
        x = self.self_attention(q=x, k=x, v=x, mask=src_mask)
        x = self.drop1(x)
        x = self.norm1(x + _x)

        # 2. feed forward
        _x = x
        x = self.ff(x)
        x = self.drop2(x)
        x = self.norm2(x + _x)

        return x


