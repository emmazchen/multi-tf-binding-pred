#%%
from torch.utils.data import Dataset
import torch
import numpy as np
from transformers import AutoTokenizer
    

class PretrainedPairDataset(Dataset):
    """ Takes in list of tuples in format (prot, dna, prob) that are integer encoded in my own system, outputs auto-tokenized data ready for pre-trained model"""
    def __init__(self, data):
        (prot,dna,prob) = zip(*data)

        # create lists of strings for auto-tokenizers
        all_prot=[]
        for prot_seq in prot:
            prot_str=""
            for aa in prot_seq:
                prot_str+=chr(aa+64) 
            all_prot.append(prot_str)

        dna_hash = {31:'A', 32:'C', 33:'G', 34:'T'}
        all_dna = []
        for dna_seq in dna:
            dna_str=""
            for nuc in dna_seq:
                dna_str+=dna_hash[nuc]
            all_dna.append(dna_str)
        self.label = prob

        # auto-tokenize
        dna_tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species")
        prot_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        # set pad token idx to 0
        self.dna = dna_tokenizer.batch_encode_plus(all_dna)["input_ids"]
        self.prot = prot_tokenizer.batch_encode_plus(all_prot)["input_ids"]

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        return ((torch.tensor(self.prot[index]), torch.tensor(self.dna[index])), torch.tensor(self.label[index])) # make each seq (and label) a tensor instead of list



class ConcatedDataset(Dataset):
    """ Takes in list of tuples in format (prot, dna, prob), returns list of tuples in form (prot_dna, prob), where 'prot_dna' is concat of prot, special token, and dna """
    def __init__(self, data):
        (prot,dna,prob) = zip(*data)
        # these are lists of tensors
        self.prot = [torch.tensor(p) for p in prot]
        self.dna = [torch.tensor(d) for d in dna]
        self.label = [torch.tensor(y) for y in prob]
        # concat prot and dna seqs together with separator in between
        concat = [torch.tensor(x, dtype=torch.int32) for x in [np.concatenate((p,np.array([100]),d)) for p,d in zip(prot,dna)]] #separator is 100
        # sort from longest to shortest
        #concat.sort(key = lambda x: -len(x))
        self.prot_dna = concat

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        return (self.prot_dna[index], self.label[index])
    


class SeqDataset(Dataset):
    """ Takes in two lists, __getitem__ returns tuples """
    def __init__(self, seq, label):
        self.seq=seq
        self.label=label

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        return torch.tensor(self.seq[index]), torch.tensor(self.label[index], dtype=torch.float32)


# %%
