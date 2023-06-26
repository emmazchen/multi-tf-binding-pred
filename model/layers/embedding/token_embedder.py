from torch import nn

"""
Generate vector embeddings from tokens

Args:
    vocab_size: size of vocabulary
    d_embedding: dimension of embedding vectors
"""

class TokenEmbedder(nn.Module):
    def __init__(self, vocab_size, d_embedding):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_embedding, padding_idx=0) #padding_idx=0 tells embedder to embed padding (0s) as 0s
    
    
    def forward(self, x):
        """
        Inputs size:
            x: (batch_size, seq_len)
        Output size: 
            x: (batch_size, seq_len, d_embedding)
        """
        return self.embed(x)
    

