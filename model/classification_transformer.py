import torch
import torch.nn as nn

from model.blocks.embedding_block import EmbeddingBlock
from model.blocks.encoder_block import EncoderBlock
from model.models.encoder import Encoder

from transformers import EsmModel, AutoModelForMaskedLM
from model.blocks.cross_attention_block import CrossAttentionBlock

class PretrainedPairModel(nn.Module):
    def __init__(self, mconfigs):
        super().__init__()
        self.pad_idx = mconfigs['pad_idx'] # both prot and dna autotokenizer used same pad_idx (=1)
        self.prot_encoder = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
        self.dna_encoder = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species")
        # notify dropout and batch norm we're in eval mode
        self.prot_encoder.eval()
        self.dna_encoder.eval()
        # freeze pretrained encoders' parameters
        for param in self.prot_encoder.parameters():
            param.requires_grad = False
        for param in self.dna_encoder.parameters():
            param.requires_grad = False
        # reshape
        self.reshape_prot = nn.Linear(mconfigs['prot_demb'], mconfigs['cross_attentn_kwargs']['d_embedding'])
        self.reshape_dna = nn.Linear(mconfigs['dna_demb'], mconfigs['cross_attentn_kwargs']['d_embedding'])
        # cross attention
        self.prot_cross_attentn = CrossAttentionBlock(**mconfigs['cross_attentn_kwargs'])
        self.dna_cross_attentn = CrossAttentionBlock(**mconfigs['cross_attentn_kwargs'])
        # self attention
        self.prot_self_attentn = EncoderBlock(**mconfigs["self_attentn_kwargs"])
        self.dna_self_attentn = EncoderBlock(**mconfigs["self_attentn_kwargs"])
        # output layers
        self.mlp = nn.Sequential(
            nn.Linear(**mconfigs['mlp']['l1_kwargs']),
            nn.ReLU(),
            nn.Linear(**mconfigs['mlp']['l2_kwargs']),
            nn.ReLU(),
            nn.Linear(**mconfigs['mlp']['l3_kwargs'])
        )

    def forward(self, x):
        # separate data and make into tensors of size (batchsize * maxseqlen_inbatch)
        prot_l, dna_l = zip(*x)
        prot, dna =torch.stack(prot_l).to(device="cuda"), torch.stack(dna_l).to(device="cuda")

        # make masks
        prot_2dmask, dna_2dmask = self.huggingface_attentn_mask(prot, self.pad_idx), self.huggingface_attentn_mask(dna, self.pad_idx) # masks for pretrained encoders. both prot and dna autotokenizer used same pad_idx (=1)
        protq_mask = self.make_pad_mask(q=prot,k=dna, pad_idx=self.pad_idx)
        dnaq_mask = self.make_pad_mask(q=dna,k=prot, pad_idx=self.pad_idx) # masks for cross attention
        prot_mask = self.make_pad_mask(q=prot,k=prot, pad_idx=self.pad_idx)
        dna_mask = self.make_pad_mask(q=dna,k=dna, pad_idx=self.pad_idx) # masks for self attention
        
        # pass into pretrained encoders to get embeddings
        prot_embedding = self.prot_encoder(prot, attention_mask = prot_2dmask).last_hidden_state #torch.FloatTensor of shape (batch_size, prot_seq_len, prot_demb=320)
        dna_embedding = self.dna_encoder(dna, attention_mask = dna_2dmask, output_hidden_states=True)['hidden_states'][-1] #(batch_size, dna_seq_len, dna_demb=2560)
        #reshape
        prot_embedding = self.reshape_prot(prot_embedding) #shape=(batchsize, max_prot_seq_len, d_embedding for cross attentn)
        dna_embedding = self.reshape_dna(dna_embedding) #shape=(batchsize, max_dna_seq_len, d_embedding)
        # cross attention
        prot_embedding = self.prot_cross_attentn(q=prot_embedding, kv=dna_embedding, crossattentn_mask=protq_mask)
        dna_embedding = self.dna_cross_attentn(q=dna_embedding, kv=prot_embedding, crossattentn_mask=dnaq_mask)
        # self attention
        prot_embedding = self.prot_self_attentn(x=prot_embedding, src_mask=prot_mask)
        dna_embedding = self.dna_self_attentn(x=dna_embedding, src_mask=dna_mask)
        # flatten along seq_len dimension
        prot = torch.mean(prot_embedding, 1) #shape=(batchsize, d_embedding)
        dna = torch.mean(dna_embedding, 1) 
        # concat the two along axis 1
        x = torch.cat((prot, dna), 1) #shape=(batchsize, 2*d_embedding)
        out = self.mlp(x)
        return out

    """
    Helper methods to make masks
    """
    def huggingface_attentn_mask(self, seqs, pad_idx):
        """ Huggingface transformers' forward()'s attention_mask parameter needs to be 
        size (batch_size, sequence_length)
        1 for tokens that are not masked (no padding) and 0 for tokens that are masked (padding) -> same as before
        """
        mask = seqs.ne(pad_idx)
        return mask
    
    def make_pad_mask(self, q, k, pad_idx):
        """
        True where no padding
        False where padding

        Input size:
            q, k: (batch_size, seq_len_q), (batch_size, seq_len_k)
        Output size:
            mask: (batch_size, 1, seq_len_q, seq_len_k)
            we apply mask on each of the heads, which is why it has size 1 along dim 1
        """
        len_q, len_k = q.size(1), k.size(1) # max_seq_len for q and k, i.e. the len to which all seq are padded to

        q_2d = q.ne(pad_idx)
        k_2d = k.ne(pad_idx) # these are 2d: (batch_size, src_seq_len)

        # make 4d, then expand along dims to reach final size (batch_size, 1, len_q, len_k)
        k = k_2d.unsqueeze(1).unsqueeze(2) 
        q = q_2d.unsqueeze(1).unsqueeze(3) # k is now (batch_size, 1, 1, len_k), q is now (batch_size, 1, len_q, 1)
        k = k.repeat(1, 1, len_q, 1)
        q = q.repeat(1, 1, 1, len_k) # k and q now (batch_size, 1, len_q, len_k)

        # effect of matmul -> only valid when both are not padding
        mask = k & q 
        return mask
    

class AblateCrossModel(nn.Module):
    def __init__(self, mconfigs):
        super().__init__()
        self.pad_idx = mconfigs['pad_idx'] # both prot and dna autotokenizer used same pad_idx (=1)
        self.prot_encoder = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
        self.dna_encoder = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species")
        # notify dropout and batch norm we're in eval mode
        self.prot_encoder.eval()
        self.dna_encoder.eval()
        # freeze pretrained encoders' parameters
        for param in self.prot_encoder.parameters():
            param.requires_grad = False
        for param in self.dna_encoder.parameters():
            param.requires_grad = False
        # reshape
        self.reshape_prot = nn.Linear(mconfigs['prot_demb'], mconfigs['cross_attentn_kwargs']['d_embedding'])
        self.reshape_dna = nn.Linear(mconfigs['dna_demb'], mconfigs['cross_attentn_kwargs']['d_embedding'])
        # # cross attention
        # self.prot_cross_attentn = CrossAttentionBlock(**mconfigs['cross_attentn_kwargs'])
        # self.dna_cross_attentn = CrossAttentionBlock(**mconfigs['cross_attentn_kwargs'])
        # # self attention
        # self.prot_self_attentn = EncoderBlock(**mconfigs["self_attentn_kwargs"])
        # self.dna_self_attentn = EncoderBlock(**mconfigs["self_attentn_kwargs"])
        # # output layers
        self.mlp = nn.Sequential(
            nn.Linear(**mconfigs['mlp']['l1_kwargs']),
            nn.ReLU(),
            nn.Linear(**mconfigs['mlp']['l2_kwargs']),
            nn.ReLU(),
            nn.Linear(**mconfigs['mlp']['l3_kwargs'])
        )

    def forward(self, x):
        # separate data and make into tensors of size (batchsize * maxseqlen_inbatch)
        prot_l, dna_l = zip(*x)
        prot, dna =torch.stack(prot_l).to(device="cuda"), torch.stack(dna_l).to(device="cuda")

        # make masks
        prot_2dmask, dna_2dmask = self.huggingface_attentn_mask(prot, self.pad_idx), self.huggingface_attentn_mask(dna, self.pad_idx) # masks for pretrained encoders. both prot and dna autotokenizer used same pad_idx (=1)
        protq_mask = self.make_pad_mask(q=prot,k=dna, pad_idx=self.pad_idx)
        dnaq_mask = self.make_pad_mask(q=dna,k=prot, pad_idx=self.pad_idx) # masks for cross attention
        prot_mask = self.make_pad_mask(q=prot,k=prot, pad_idx=self.pad_idx)
        dna_mask = self.make_pad_mask(q=dna,k=dna, pad_idx=self.pad_idx) # masks for self attention
        
        # pass into pretrained encoders to get embeddings
        prot_embedding = self.prot_encoder(prot, attention_mask = prot_2dmask).last_hidden_state #torch.FloatTensor of shape (batch_size, prot_seq_len, prot_demb=320)
        dna_embedding = self.dna_encoder(dna, attention_mask = dna_2dmask, output_hidden_states=True)['hidden_states'][-1] #(batch_size, dna_seq_len, dna_demb=2560)
        #reshape
        prot_embedding = self.reshape_prot(prot_embedding) #shape=(batchsize, max_prot_seq_len, d_embedding for cross attentn)
        dna_embedding = self.reshape_dna(dna_embedding) #shape=(batchsize, max_dna_seq_len, d_embedding)
        # # cross attention
        # prot_embedding = self.prot_cross_attentn(q=prot_embedding, kv=dna_embedding, crossattentn_mask=protq_mask)
        # dna_embedding = self.dna_cross_attentn(q=dna_embedding, kv=prot_embedding, crossattentn_mask=dnaq_mask)
        # # self attention
        # prot_embedding = self.prot_self_attentn(x=prot_embedding, src_mask=prot_mask)
        # dna_embedding = self.dna_self_attentn(x=dna_embedding, src_mask=dna_mask)
        # # flatten along seq_len dimension
        prot = torch.mean(prot_embedding, 1) #shape=(batchsize, d_embedding)
        dna = torch.mean(dna_embedding, 1) 
        # concat the two along axis 1
        x = torch.cat((prot, dna), 1) #shape=(batchsize, 2*d_embedding)
        out = self.mlp(x)
        return out

    """
    Helper methods to make masks
    """
    def huggingface_attentn_mask(self, seqs, pad_idx):
        """ Huggingface transformers' forward()'s attention_mask parameter needs to be 
        size (batch_size, sequence_length)
        1 for tokens that are not masked (no padding) and 0 for tokens that are masked (padding) -> same as before
        """
        mask = seqs.ne(pad_idx)
        return mask
    

    def make_pad_mask(self, q, k, pad_idx):
        """
        True where no padding
        False where padding

        Input size:
            q, k: (batch_size, seq_len_q), (batch_size, seq_len_k)
        Output size:
            mask: (batch_size, 1, seq_len_q, seq_len_k)
            we apply mask on each of the heads, which is why it has size 1 along dim 1
        """
        len_q, len_k = q.size(1), k.size(1) # max_seq_len for q and k, i.e. the len to which all seq are padded to

        q_2d = q.ne(pad_idx)
        k_2d = k.ne(pad_idx) # these are 2d: (batch_size, src_seq_len)

        # make 4d, then expand along dims to reach final size (batch_size, 1, len_q, len_k)
        k = k_2d.unsqueeze(1).unsqueeze(2) 
        q = q_2d.unsqueeze(1).unsqueeze(3) # k is now (batch_size, 1, 1, len_k), q is now (batch_size, 1, len_q, 1)
        k = k.repeat(1, 1, len_q, 1)
        q = q.repeat(1, 1, 1, len_k) # k and q now (batch_size, 1, len_q, len_k)

        # effect of matmul -> only valid when both are not padding
        mask = k & q 
        return mask
    

class AblateSelfModel(nn.Module):
    def __init__(self, mconfigs):
        super().__init__()
        self.pad_idx = mconfigs['pad_idx'] # both prot and dna autotokenizer used same pad_idx (=1)
        self.prot_encoder = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
        self.dna_encoder = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species")
        # notify dropout and batch norm we're in eval mode
        self.prot_encoder.eval()
        self.dna_encoder.eval()
        # freeze pretrained encoders' parameters
        for param in self.prot_encoder.parameters():
            param.requires_grad = False
        for param in self.dna_encoder.parameters():
            param.requires_grad = False
        # reshape
        self.reshape_prot = nn.Linear(mconfigs['prot_demb'], mconfigs['cross_attentn_kwargs']['d_embedding'])
        self.reshape_dna = nn.Linear(mconfigs['dna_demb'], mconfigs['cross_attentn_kwargs']['d_embedding'])
        # cross attention
        self.prot_cross_attentn = CrossAttentionBlock(**mconfigs['cross_attentn_kwargs'])
        self.dna_cross_attentn = CrossAttentionBlock(**mconfigs['cross_attentn_kwargs'])
        # self attention
        # self.prot_self_attentn = EncoderBlock(**mconfigs["self_attentn_kwargs"])
        # self.dna_self_attentn = EncoderBlock(**mconfigs["self_attentn_kwargs"])
        # output layers
        self.mlp = nn.Sequential(
            nn.Linear(**mconfigs['mlp']['l1_kwargs']),
            nn.ReLU(),
            nn.Linear(**mconfigs['mlp']['l2_kwargs']),
            nn.ReLU(),
            nn.Linear(**mconfigs['mlp']['l3_kwargs'])
        )

    def forward(self, x):
        # separate data and make into tensors of size (batchsize * maxseqlen_inbatch)
        prot_l, dna_l = zip(*x)
        prot, dna =torch.stack(prot_l).to(device="cuda"), torch.stack(dna_l).to(device="cuda")

        # make masks
        prot_2dmask, dna_2dmask = self.huggingface_attentn_mask(prot, self.pad_idx), self.huggingface_attentn_mask(dna, self.pad_idx) # masks for pretrained encoders. both prot and dna autotokenizer used same pad_idx (=1)
        protq_mask = self.make_pad_mask(q=prot,k=dna, pad_idx=self.pad_idx)
        dnaq_mask = self.make_pad_mask(q=dna,k=prot, pad_idx=self.pad_idx) # masks for cross attention
        prot_mask = self.make_pad_mask(q=prot,k=prot, pad_idx=self.pad_idx)
        dna_mask = self.make_pad_mask(q=dna,k=dna, pad_idx=self.pad_idx) # masks for self attention
        
        # pass into pretrained encoders to get embeddings
        prot_embedding = self.prot_encoder(prot, attention_mask = prot_2dmask).last_hidden_state #torch.FloatTensor of shape (batch_size, prot_seq_len, prot_demb=320)
        dna_embedding = self.dna_encoder(dna, attention_mask = dna_2dmask, output_hidden_states=True)['hidden_states'][-1] #(batch_size, dna_seq_len, dna_demb=2560)
        #reshape
        prot_embedding = self.reshape_prot(prot_embedding) #shape=(batchsize, max_prot_seq_len, d_embedding for cross attentn)
        dna_embedding = self.reshape_dna(dna_embedding) #shape=(batchsize, max_dna_seq_len, d_embedding)
        # cross attention
        prot_embedding = self.prot_cross_attentn(q=prot_embedding, kv=dna_embedding, crossattentn_mask=protq_mask)
        dna_embedding = self.dna_cross_attentn(q=dna_embedding, kv=prot_embedding, crossattentn_mask=dnaq_mask)
        # self attention
        # prot_embedding = self.prot_self_attentn(x=prot_embedding, src_mask=prot_mask)
        # dna_embedding = self.dna_self_attentn(x=dna_embedding, src_mask=dna_mask)
        # flatten along seq_len dimension
        prot = torch.mean(prot_embedding, 1) #shape=(batchsize, d_embedding)
        dna = torch.mean(dna_embedding, 1) 
        # concat the two along axis 1
        x = torch.cat((prot, dna), 1) #shape=(batchsize, 2*d_embedding)
        out = self.mlp(x)
        return out

    """
    Helper methods to make masks
    """
    def huggingface_attentn_mask(self, seqs, pad_idx):
        """ Huggingface transformers' forward()'s attention_mask parameter needs to be 
        size (batch_size, sequence_length)
        1 for tokens that are not masked (no padding) and 0 for tokens that are masked (padding) -> same as before
        """
        mask = seqs.ne(pad_idx)
        return mask
    

    def make_pad_mask(self, q, k, pad_idx):
        """
        True where no padding
        False where padding

        Input size:
            q, k: (batch_size, seq_len_q), (batch_size, seq_len_k)
        Output size:
            mask: (batch_size, 1, seq_len_q, seq_len_k)
            we apply mask on each of the heads, which is why it has size 1 along dim 1
        """
        len_q, len_k = q.size(1), k.size(1) # max_seq_len for q and k, i.e. the len to which all seq are padded to

        q_2d = q.ne(pad_idx)
        k_2d = k.ne(pad_idx) # these are 2d: (batch_size, src_seq_len)

        # make 4d, then expand along dims to reach final size (batch_size, 1, len_q, len_k)
        k = k_2d.unsqueeze(1).unsqueeze(2) 
        q = q_2d.unsqueeze(1).unsqueeze(3) # k is now (batch_size, 1, 1, len_k), q is now (batch_size, 1, len_q, 1)
        k = k.repeat(1, 1, len_q, 1)
        q = q.repeat(1, 1, 1, len_k) # k and q now (batch_size, 1, len_q, len_k)

        # effect of matmul -> only valid when both are not padding
        mask = k & q 
        return mask
    




#%%
class ClassificationTransformer(nn.Module):
    def __init__(self, mconfigs):
        super().__init__()
        self.pad_idx = mconfigs['pad_idx']
        self.encoder = Encoder(**mconfigs['encoder'])
        self.avg_pool_reduce_dim = torch.mean # average along dimension to get (batch_size, d_embedding)
        self.mlp = nn.Sequential(
            nn.Linear(mconfigs['encoder']['d_embedding'],mconfigs['mlp']['l1']['out_features']),
            nn.ReLU(),
            nn.Linear(**mconfigs['mlp']['l2']),
            nn.ReLU(),
            nn.Linear(**mconfigs['mlp']['l3'])
        )
    
    def forward(self, x):
        """
        Input size:
            x: (batch_size, seq_len)
        Output size:
            out: (batch_size, num_classes)
        """
        # make mask for self attention in encoder
        src_mask = self.make_pad_mask(x, x, self.pad_idx)
 
        x = self.encoder(x, src_mask)
        x = self.avg_pool_reduce_dim(x, 1)
        out = self.mlp(x)
        return out


    """
    Helper methods to make masks
    """
    def make_pad_mask(self, q, k, pad_idx):
        """
        True where no padding
        False where padding

        Input size:
            q, k: (batch_size, seq_len_q), (batch_size, seq_len_k)
        Output size:
            mask: (batch_size, 1, seq_len_q, seq_len_k)
            we apply mask on each of the heads, which is why it has size 1 along dim 1
        """
        len_q, len_k = q.size(1), k.size(1) # max_seq_len for q and k, i.e. the len to which all seq are padded to

        q_2d = q.ne(pad_idx)
        k_2d = k.ne(pad_idx) # these are 2d: (batch_size, src_seq_len)

        # make 4d, then expand along dims to reach final size (batch_size, 1, len_q, len_k)
        k = k_2d.unsqueeze(1).unsqueeze(2) 
        q = q_2d.unsqueeze(1).unsqueeze(3) # k is now (batch_size, 1, 1, len_k), q is now (batch_size, 1, len_q, 1)
        k = k.repeat(1, 1, len_q, 1)
        q = q.repeat(1, 1, 1, len_k) # k and q now (batch_size, 1, len_q, len_k)

        # effect of matmul -> only valid when both are not padding
        mask = k & q 
        return mask

#%%

from torch.nn import TransformerEncoderLayer, TransformerEncoder
from model.blocks.embedding_block import EmbeddingBlock
class TorchTransformer(nn.Module):
    def __init__(self, mconfigs):
        super().__init__()
        self.pad_idx = mconfigs['pad_idx']
        self.emb = EmbeddingBlock(mconfigs['encoder']['d_embedding'], mconfigs['encoder']['src_vocab_size'], mconfigs['encoder']['max_seq_len'], mconfigs['encoder']['p_drop'])
        encoder_layer = TransformerEncoderLayer(d_model=mconfigs['encoder']['d_embedding'], nhead=mconfigs['encoder']['n_heads'])
        self.encoder = TransformerEncoder(encoder_layer, mconfigs['encoder']['n_blocks'])
        self.avg_pool_reduce_dim = torch.mean # average along dimension to get (batch_size, d_embedding)
        self.mlp = nn.Sequential(
            nn.Linear(mconfigs['encoder']['d_embedding'],mconfigs['mlp']['l1']['out_features']),
            nn.ReLU(),
            nn.Linear(**mconfigs['mlp']['l2']),
            nn.ReLU(),
            nn.Linear(**mconfigs['mlp']['l3'])
        )
    
    def forward(self, x):
        """
        Input size:
            x: (batch_size, seq_len)
        Output size:
            out: (batch_size, num_classes)
        """
        # make mask for self attention in encoder
        src_mask = self.make_pad_mask_for_torch(x, self.pad_idx) #src mask should be (batchsize, seqlen)
        x = self.emb(x)
        x = torch.transpose(x, 0, 1) # convert from batch first
        x = self.encoder(x.float(), src_key_padding_mask=src_mask)
        x = torch.transpose(x, 0, 1) # convert back to batch first
        x = self.avg_pool_reduce_dim(x, 1)
        out = self.mlp(x)
        return out


    """
    Helper methods to make masks
    """
    def make_pad_mask_for_torch(self, k, pad_idx):
        """
        True where no padding
        False where padding

        Input size:
            k: (batch_size, seq_len)
        Output size:
            mask: (batch_size,  seq_len)
        """
        mask = k.ne(pad_idx)  # 0s where it's padding
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask




