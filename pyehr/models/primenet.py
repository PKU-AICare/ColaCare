import math
import string
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


MAX_VAL = 1e4
MIN_VAL = 1e-12


class multiTimeAttention(nn.Module):
    
    def __init__(self, input_dim, nhidden=16, embed_time=16, num_heads=1):
        super(multiTimeAttention, self).__init__()
        assert embed_time % num_heads == 0
        self.embed_time = embed_time
        self.embed_time_k = embed_time // num_heads
        self.h = num_heads
        self.dim = input_dim
        self.nhidden = nhidden
        self.linears = nn.ModuleList([nn.Linear(embed_time, embed_time), # to embed the query
                                      nn.Linear(embed_time, embed_time), # to embed the key
                                      nn.Linear(input_dim*num_heads, nhidden)]) # to embed attention weighted values
        
    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        dim = value.size(-1)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)
        if mask is not None:
            scores = scores.masked_fill(mask.to(query.device).unsqueeze(-3) == 0, -1e9)
        p_attn = F.softmax(scores, dim = -2)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.sum(p_attn.to(query.device)*value.unsqueeze(-3).to(query.device), -2), p_attn.to(query.device)
    
    
    def forward(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        batch, seq_len, dim = value.size()
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        value = value.unsqueeze(1)

        # zip will pair the 3 linear layers with (query, key)
        # Since there are 3 linear layers but only 2 elements in the tupe (query, key),
        # only the first two linear layers will be mapped to this tuple
        # (first linear layer, query), (second linear layer, key)
        # so the list has two elements
        # input query passed through the first linear layer -> output becomes first element of the list (query)
        # input key passed through the second linear layer -> output becomes second element of the list (key)
        # query, key = [2, 3]
        # so 2 is assigned to query and 3 is assigned to key
        # had there been a, b, c = [1, 2] -> ValueError: not enough values to unpack (expected 3, got 2)
        # had there been a, b = [1, 2, 3] -> ValueError: too many values to unpack (expected 2)
        # look into learn_time_emb.py for further clarification on this line
        query, key = [l(x).view(x.size(0), -1, self.h, self.embed_time_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]

        x, _ = self.attention(query, key, value, mask, dropout)
        x = x.transpose(1, 2).contiguous() \
             .view(batch, -1, self.h * dim)
        return self.linears[-1](x)


class BertPooler(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Tanh()

    def forward(self, first_token_tensor):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class PrimeNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_time=32, num_heads=1, freq=10, learn_emb=True, dropout=0.3, pooling='bert', cls_query=torch.linspace(0, 1., 128), **kwargs):
        super(PrimeNet, self).__init__()
       
        assert embed_time % num_heads == 0
        self.freq = freq
        self.embed_time = embed_time
        self.learn_emb = learn_emb
        self.dim = input_dim
        self.hidden_dim = hidden_dim
        self.cls_query =  cls_query
        self.time_att = multiTimeAttention(self.dim, self.hidden_dim, self.embed_time, num_heads)

        self.pos_emb = nn.Embedding(512, self.hidden_dim)  ## +1 for cls token
        self.transformer = TransformerBlock(self.hidden_dim, num_heads, self.hidden_dim, dropout=dropout)

        self.cls_emb = nn.Embedding(1, self.hidden_dim)
        self.pooling = pooling

        self.pooler = BertPooler(hidden_dim)

        assert self.pooling in ['ave', 'att', 'bert']

        if self.learn_emb:
            self.periodic = nn.Linear(1, self.embed_time-1)
            self.linear = nn.Linear(1, 1)
    
    def learn_time_embedding(self, tt):
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)
        
    def time_embedding(self, pos, d_model):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model)
        position = 48.*pos.unsqueeze(2)
        # print(self.freq)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(torch.log(torch.tensor(self.freq, dtype=torch.float32)) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
    
    def encode(self, x, is_pooling=False):
        out = x

        if self.pooling == 'att' or self.pooling == 'bert':

            batch_size = out.size(0)
            # cls_tokens = torch.zeros(batch_size).to(out.device)
            cls_tokens = torch.zeros(batch_size).to(out.device).long()
            cls_repr = self.cls_emb(cls_tokens).view(batch_size, 1, -1)  # (batch_size, 1, nhidden)

        if self.pooling == 'bert':
            out = torch.cat([cls_repr, out], dim=1)

        if is_pooling:
            out = self.transformer(out)
            if self.pooling == 'ave':
                out = torch.mean(out, dim=1)  #Ave Pooling
            elif self.pooling == 'att':
                out = out.permute(0, 2, 1) # (batch_size, seq_len, nhidden) -> (batch_size, nhidden, seq_len)
                weights = F.softmax(torch.bmm(cls_repr, out), dim=-1) # (batch_size, 1, seq_len)
                out = torch.sum(out * weights, dim=-1) # (batch_size, nhidden)
            else: # bert
                out = out[ : , 0]

            return self.pooler(out)

        else:
            positions = torch.arange(out.shape[1]).long().unsqueeze(0).repeat([out.shape[0], 1])
            out = out + self.pos_emb(positions.to(out.device))
            out = self.transformer(out)
            if self.pooling == 'bert':
                # return out[1:]  ## remove cls token
                return out[:, 1:]
            else:
                return out
    
    def forward(self, x, query_time_steps=None):
        
        # x : (batch x num_seq), seq_len, (input_dim x 2)
        # time_steps : (batch x num_seq), seq_len
        # query_time_steps : (batch x num_seq), seq_len, input_dim

        device = x.device
        x = x.float()
        time_steps = x[:, :, -1]

        if query_time_steps is None:
            query_time_steps = time_steps
        
        if self.learn_emb:
            key = self.learn_time_embedding(time_steps)
            time_query = self.learn_time_embedding(query_time_steps)
            cls_query = self.learn_time_embedding(self.cls_query.unsqueeze(0).to(device))
        else:
            key = self.time_embedding(time_steps.cpu(), self.embed_time).to(device)
            time_query = self.time_embedding(query_time_steps.cpu(), self.embed_time).to(device)
            cls_query = self.time_embedding(self.cls_query.unsqueeze(0), self.embed_time).to(device)
        
        # time_query -> irregular time representation
        # cls_query -> corresponding regular time representation
        cls_out = self.time_att(cls_query, key, x) 

        # since cls_out comes from cls_query, the learnt feature representation is for the corresponding regular time-series
        # cls_out is pooled ('ave', 'att', 'bert') and the output is used for CL representation and sequence classification
        cls_out = self.encode(cls_out, is_pooling=True)
        
        return cls_out
    
    
##################
# Self-Attention #
##################

class Attention(nn.Module):
    "Compute 'Scaled Dot Product Attention"
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -MAX_VAL)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    "Take in model size and number of heads."
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)

#########
# Utils #
#########

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.activation(self.w_1(x)))

class GELU(nn.Module):
    "Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU"
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        inv = (var + self.eps).rsqrt() * self.a_2
        return x * inv + (self.b_2 - mean * inv)

class SublayerConnection(nn.Module):
    "A residual connection followed by a layer norm."
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # sublayer is a single or a combination of learnable NN layers
        # sublayer(x) returns the output after passing x through the sublayer
        # the output of sublayer(x) is added to x for residual connection
        return self.norm(x + self.dropout(sublayer(x))) 

class OutputLayer(nn.Module):
    "Ouptut Layer for BERT model"
    def __init__(self, hidden_dim):
        super(OutputLayer, self).__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.activation = GELU()
        self.layer_norm = LayerNorm(hidden_dim)

    def forward(self, x):
        return self.layer_norm(self.activation(self.linear(x)))

####################
# TransformerBlock #
####################

class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=0.2)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)

    def forward(self, x, mask=None):
        # x = x + self.attention(x)
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))   # use residual connection
        
        # x = x + self.feed_forward(x)
        x = self.output_sublayer(x, self.feed_forward)
        return x