import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn


class Attention(nn.Module):
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = torch.softmax(scores, dim=-1)
        if mask is not None:
            p_attn = p_attn.masked_fill(mask == 0, 0)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList(
            [nn.Linear(d_model, d_model, bias=False) for _ in range(3)]
        )
        self.output_linear = nn.Linear(d_model, d_model, bias=False)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linear_layers, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        if mask is not None:
            mask = mask.unsqueeze(1)
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x, mask=None):
        x = self.w_2(self.dropout(self.activation(self.w_1(x))))
        if mask is not None:
            mask = mask.sum(dim=-1) > 0
            x[~mask] = 0
        return x


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class TransformerBlock(nn.Module):
    """Transformer block.

    MultiHeadedAttention + PositionwiseFeedForward + SublayerConnection

    Args:
        hidden: hidden size of transformer.
        attn_heads: head sizes of multi-head attention.
        dropout: dropout rate.
    """

    def __init__(self, hidden, attn_heads, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(
            d_model=hidden, d_ff=4 * hidden, dropout=dropout
        )
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        """Forward propagation.

        Args:
            x: [batch_size, seq_len, hidden]
            mask: [batch_size, seq_len, seq_len]

        Returns:
            A tensor of shape [batch_size, seq_len, hidden]
        """
        x = self.input_sublayer(x, lambda _x: self.attention(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, lambda _x: self.feed_forward(_x, mask=mask))
        return self.dropout(x)


class TransformerLayer(nn.Module):
    """Transformer layer.

    Paper: Ashish Vaswani et al. Attention is all you need. NIPS 2017.

    This layer is used in the Transformer model. But it can also be used
    as a standalone layer.

    Args:
        input_dim: the hidden feature size.
        heads: the number of attention heads. Default is 1.
        dropout: dropout rate. Default is 0.5.
        num_layers: number of transformer layers. Default is 1.

    Examples:
        >>> from pyhealth.models import TransformerLayer
        >>> input = torch.randn(3, 128, 64)  # [batch size, sequence len, input_dim]
        >>> layer = TransformerLayer(64)
        >>> emb, cls_emb = layer(input)
        >>> emb.shape
        torch.Size([3, 128, 64])
        >>> cls_emb.shape
        torch.Size([3, 64])
    """

    def __init__(self, input_dim, heads=1, dropout=0.5, num_layers=1):
        super(TransformerLayer, self).__init__()
        self.transformer = nn.ModuleList(
            [TransformerBlock(input_dim, heads, dropout) for _ in range(num_layers)]
        )

    def forward(
        self, x: torch.tensor, mask: Optional[torch.tensor] = None
    ) -> Tuple[torch.tensor, torch.tensor]:
        """Forward propagation.

        Args:
            x: a tensor of shape [batch size, sequence len, input_dim].
            mask: an optional tensor of shape [batch size, sequence len], where
                1 indicates valid and 0 indicates invalid.

        Returns:
            emb: a tensor of shape [batch size, sequence len, input_dim],
                containing the output features for each time step.
            cls_emb: a tensor of shape [batch size, input_dim], containing
                the output features for the first time step.
        """
        if mask is not None:
            mask = torch.einsum("ab,ac->abc", mask, mask)
        for transformer in self.transformer:
            x = transformer(x, mask)
        emb = x
        cls_emb = x[:, 0, :]
        return emb, cls_emb


class Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, heads=1, dropout=0.5, num_layers=1, **kwargs):
        super(Transformer, self).__init__()
        self.hidden_dim = hidden_dim
        self.transformer_layer = TransformerLayer(input_dim, heads, dropout, num_layers)
        self.proj = nn.Linear(input_dim, hidden_dim)
    def forward(self, x, mask=None):
        batch_size, time_steps, _ = x.size()
        # out = torch.zeros((batch_size, time_steps, self.hidden_dim))
        # for cur_time in range(time_steps):
        #     cur_x = x[:, :cur_time+1, :]
        #     cur_mask = mask[:, :cur_time+1]
        #     _, cur_out = self.transformer_layer(cur_x, cur_mask) # use cls token
        #     cur_out = self.proj(cur_out)
        #     out[:, cur_time, :] = cur_out
        # return out
        _, out = self.transformer_layer(x, mask) # use cls token
        out = self.proj(out)
        return out