from torch import nn

"""
Feedforward output from attention layers
Args:
    d_embedding: dimension of embedding vectors
    d_hidden: dimension of hidden layer
    p_drop: drop-out probability
"""

class FeedForward(nn.Module):
    def __init__(self, d_embedding, d_hidden, p_drop=0.1):
        super().__init__()
        self.l1 = nn.Linear(d_embedding, d_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=p_drop) #added drop out
        self.l2 = nn.Linear(d_hidden, d_embedding)

    def forward(self, x):
        """
        Inputs size:
            x: (batch_size, seq_len, d_embedding)
        Output size:
            out: (batch_size, seq_len, d_embedding)
        """
        x = self.l1(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.l2(x)
        return out

