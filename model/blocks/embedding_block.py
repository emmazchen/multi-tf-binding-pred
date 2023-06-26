from torch import nn

from model.layers.embedding.token_embedder import TokenEmbedder
from model.layers.embedding.position_encoder import PositionEncoder

class EmbeddingBlock(nn.Module):
    def __init__(self, d_embedding, vocab_size, max_seq_len, p_drop):
        super().__init__()
        self.token_emb = TokenEmbedder(vocab_size=vocab_size, d_embedding=d_embedding)
        self.position_enc = PositionEncoder(d_embedding=d_embedding, max_seq_len=max_seq_len)
        self.drop = nn.Dropout(p=p_drop)

    def forward(self, x):
        """
        Inputs size:
            x: (batch_size, seq_len)
        Output size: 
            x: (batch_size, seq_len, d_embedding)
        """
        x = self.token_emb(x)
        x = self.position_enc(x)
        x = self.drop(x)
        return x