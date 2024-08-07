import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .Warpformer.Layers import EncoderLayer


class Warpformer(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(self, input_dim, num_types=2, d_model=16, d_inner=16, n_layers=3, n_head=3, d_k=8, d_v=8, dropout=0.5, *args, **kwargs):
        
        super().__init__()
        self.d_model = d_model
        self.embed_time = d_model
        d_inner = input_dim
        num_types = input_dim
        # event type embedding
        self.event_enc = Event_Encoder(d_model, num_types)
        self.type_matrix = torch.tensor([int(i) for i in range(1,num_types+1)]).to('cuda:1')
        self.type_matrix = rearrange(self.type_matrix, 'k -> 1 1 k')
        self.num_types = num_types
        warp_num = [0, 12]
        warp_layer_num = 2
        self.no_warping = False
        self.full_attn = True

        self.warpformer_layer_stack = nn.ModuleList([
            WarpformerLayer(int(warp_num[i]), n_head, d_model, d_inner, d_k, d_v, dropout)
            for i in range(warp_layer_num)])
        
        self.value_enc = Value_Encoder(hid_units=d_inner, output_dim=d_model, num_type=num_types)
        self.learn_time_embedding = Time_Encoder(self.embed_time, d_inner)
        
        self.w_t = nn.Linear(1, num_types, bias=False)

        self.linear = nn.Linear(d_model*warp_layer_num, d_model)

    def forward(self, event_time, event_value, non_pad_mask):
        """ Encode event sequences via masked self-attention. """
        '''
        non_pad_mask: [B,L,K]
        slf_attn_mask: [B,K,LQ,LK], the values to be masked are set to True
        len_pad_mask: [B,L], pick the longest length and mask the remains
        '''
        # embedding
        tem_enc_k = self.learn_time_embedding(event_time, non_pad_mask)  # [B,L,1,D], [B,L,K,D]
        tem_enc_k = rearrange(tem_enc_k, 'b l k d -> b k l d') # [B,K,L,D]

        value_emb = self.value_enc(event_value, non_pad_mask)
        value_emb = rearrange(value_emb, 'b l k d -> b k l d') # [B,K,L,D]
        
        self.type_matrix = self.type_matrix.to(non_pad_mask.device)
        # event_emb = self.type_matrix * non_pad_mask
        event_emb = self.type_matrix
        event_emb = self.event_enc(event_emb) 
        event_emb = rearrange(event_emb, 'b l k d -> b k l d') # [B,K,L,D]

        
        h0 = value_emb + event_emb
        z0 = torch.mean(h0, dim=1)    
        return z0
    
    
class WarpformerLayer(nn.Module):
    def __init__(self, new_l, n_head, d_model, d_inner, d_k, d_v, dropout):
        super().__init__()
        self.new_l = new_l
        self.num_types = 23
        d_inner = d_inner
        d_model = d_model
        
        self.time_split = [i for i in range(self.new_l+1)] # for hour aggregation

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(3)])

    
    def hour_aggregate(self, event_time, h0, non_pad_mask):
        # time_split: [new_l]
        # event_time: [B,L]
        # h0: [B,K,L,D]
        # non_pad_mask: [B,K,L]
        new_l = len(self.time_split)-1
        b, k, l, dim = h0.shape
        
        event_time_k = repeat(event_time, 'b l -> b k l', k=self.num_types)
        new_event_time = torch.zeros((b,self.num_types,new_l)).to(h0.device)
        new_h0 = torch.zeros((b,self.num_types,new_l, dim)).to(h0.device)
        new_pad_mask = torch.zeros((b,self.num_types,new_l)).to(h0.device)
        almat = torch.zeros((b,l,new_l)).to(h0.device) # [B,L,S]
        
        # for each time slot
        for i in range(len(self.time_split)-1):
            idx = (event_time_k.ge(self.time_split[i]) & event_time_k.lt(self.time_split[i+1])) # [B,K,L]
            total = torch.sum(idx, dim=-1) # [B,K]
            total[total==0] = 1
            
            tmp_h0 = h0 * idx.unsqueeze(-1) # [B,K,L,D]
            tmp_h0 = rearrange(tmp_h0, 'b k l d -> (b k) d l')
            tmp_h0 = F.max_pool1d(tmp_h0, tmp_h0.size(-1)).squeeze() # [BK,D,1]
            new_h0[:,:,i,:] = rearrange(tmp_h0, '(b k) d -> b k d', b=b)
            almat[:,:,i] = (event_time.ge(self.time_split[i]) & event_time.lt(self.time_split[i+1])) # [B,L]

            new_event_time[:,:,i] = torch.sum(event_time_k * idx, dim=-1) / total
            new_pad_mask[:,:,i] = torch.sum(non_pad_mask * idx, dim=-1) / total
        
        almat = repeat(almat, 'b l s -> b k l s', k=k)
        return new_h0, new_event_time, new_pad_mask, almat

    
    def forward(self, h0, non_pad_mask, event_time):
        
        z0, _, new_pad_mask, almat = self.hour_aggregate(event_time, h0, non_pad_mask)
        
        for enc_layer in self.layer_stack:
            z0, _, _ = enc_layer(z0, non_pad_mask=new_pad_mask)
        
        return z0, new_pad_mask, almat
    
    
class Value_Encoder(nn.Module):
    def __init__(self, hid_units, output_dim, num_type):
        self.hid_units = hid_units
        self.output_dim = output_dim
        self.num_type = num_type
        super(Value_Encoder, self).__init__()

        self.encoder = nn.Linear(1, output_dim)

    def forward(self, x, non_pad_mask):
        non_pad_mask = rearrange(non_pad_mask, 'b l k -> b l k 1')
        x = rearrange(x, 'b l k -> b l k 1')
        x = self.encoder(x)
        return x * non_pad_mask

class Event_Encoder(nn.Module):
    def __init__(self, d_model, num_types):
        super(Event_Encoder, self).__init__()
        self.event_emb = nn.Embedding(num_types+1, d_model, padding_idx=0)

    def forward(self, event):
        # event = event * self.type_matrix
        event_emb = self.event_emb(event.long())
        return event_emb

class Time_Encoder(nn.Module):
    def __init__(self, embed_time, num_types):
        super(Time_Encoder, self).__init__()
        self.periodic = nn.Linear(1, embed_time - 1)
        self.linear = nn.Linear(1, 1)
        self.k_map = nn.Parameter(torch.ones(1,1,num_types,embed_time))

    def forward(self, tt, non_pad_mask):
        non_pad_mask = rearrange(non_pad_mask, 'b l k -> b l k 1')
        if tt.dim() == 3:  # [B,L,K]
            tt = rearrange(tt, 'b l k -> b l k 1')
        else: # [B,L]
            tt = rearrange(tt, 'b l -> b l 1 1')
        
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        out = torch.cat([out1, out2], -1) # [B,L,1,D]
        out = torch.mul(out, self.k_map)
        # return out * non_pad_mask # [B,L,K,D]
        return out