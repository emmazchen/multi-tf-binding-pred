from torch import nn
from model.blocks.embedding_block import EmbeddingBlock
from model.blocks.encoder_block import EncoderBlock

class Encoder(nn.Module):
    def __init__(self, n_blocks, src_vocab_size, max_seq_len, d_embedding, ff_d_hidden, n_heads, p_drop):
        super().__init__()
        self.n_blocks=n_blocks
        self.emb = EmbeddingBlock(d_embedding=d_embedding, vocab_size=src_vocab_size, max_seq_len=max_seq_len, p_drop=p_drop)
        self.enc_blocks = nn.ModuleList([EncoderBlock(d_embedding=d_embedding, ff_d_hidden=ff_d_hidden, n_heads=n_heads, p_drop=p_drop) 
                                         for i in range(n_blocks)])
    
    def forward(self, x, src_mask):
        x = self.emb(x) # embed

        for block in self.enc_blocks: # pass through all the enc blocks
            x = block(x, src_mask=src_mask)

        return x
