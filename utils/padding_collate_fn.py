from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np

"""
For pretrained pair model with esm and nucleotide transformer. Pads prot and dna separately
"""
def padding_collate_pretrainedpair(data):
    (x, y) = zip(*data)
    prot, dna = zip(*x)
    prot = pad_sequence(prot, batch_first=True, padding_value=1) # pad with 1s because autotokenizers' pad_token_idx set to 1 (default) -> have updated pad_idx in config so later when making masks, we know this is pad_idx
    dna = pad_sequence(dna, batch_first=True, padding_value=1) # 0 as token id was used for <unk> in dna autotokenizer and <cls> in prot autotokenizer
    x = zip(prot,dna)
    y = torch.tensor(y, dtype=torch.float32)
    return x, y


"""
Custom collate function that pads sequences of variable length in batch
"""
def padding_collate(data):
    (x, y) = zip(*data)
    x = pad_sequence(x, batch_first=True)
    y = torch.tensor(y, dtype=torch.float32)
    return x, y

