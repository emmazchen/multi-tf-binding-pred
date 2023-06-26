from torch import nn

from model.layers.multi_head_attention import MultiHeadAttention
from model.layers.feed_forward import FeedForward
from model.layers.layer_norm import LayerNorm


class CrossAttentionBlock(nn.Module):

    def __init__(self, d_embedding, n_heads, ff_d_hidden, p_drop):
        super().__init__()
        self.cross_attentn = MultiHeadAttention(d_embedding=d_embedding,n_heads=n_heads)
        self.drop = nn.Dropout(p=p_drop)
        self.norm = LayerNorm(d_embedding=d_embedding)    

    def forward(self, q, kv, crossattentn_mask):
        _q = q
        q = self.cross_attentn(q=q, k=kv, v=kv, mask=crossattentn_mask) #output has size (q_seq_len, d_embedding)
        q = self.drop(q)
        q = self.norm(q + _q)

        return q
