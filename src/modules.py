import logging
import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class CrossmodalTransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        attn_dropout,
        res_dropout,
        relu_dropout,
        n_layer,
        attn_mask,
    ):
        super(CrossmodalTransformerEncoder, self).__init__()
        self.attn_mask = attn_mask
        self.layers = nn.ModuleList([])
        for layer in range(n_layer):
            new_layer = TransformerDecoderBlock(
                d_model, nhead, dim_feedforward, attn_dropout, res_dropout, relu_dropout
            )
            self.layers.append(new_layer)

    def forward(self, src, tgt):
        for layer in self.layers:
            tgt = layer(src, tgt, src_mask=self.attn_mask)
        return tgt


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        attn_dropout,
        res_dropout,
        relu_dropout,
        n_layer,
        attn_mask,
    ):
        super(TransformerEncoder, self).__init__()
        self.attn_mask = attn_mask
        self.layers = nn.ModuleList([])
        for layer in range(n_layer):
            new_layer = TransformerEncoderBlock(
                d_model, nhead, dim_feedforward, attn_dropout, res_dropout, relu_dropout
            )
            self.layers.append(new_layer)

    def forward(self, src):
        for layer in self.layers:
            src = layer(src, x_attn_mask=self.attn_mask)
        return src


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, attn_dropout, res_dropout, relu_dropout):
        """
        Args:
            d_model: the number of expected features in the input (required).
            nhead: the number of heads in the multiheadattention models (required).
            dim_feedforward: the dimension of the feedforward network model (required).
            attn_dropout: the dropout value for multihead attention (required).
            res_dropout: the dropout value for residual connection (required).
            relu_dropout: the dropout value for relu (required).
        """
        super(TransformerEncoderBlock, self).__init__()
        self.transformer = TransformerBlock(d_model, nhead, attn_dropout, res_dropout)
        self.feedforward = FeedForwardBlock(d_model, dim_feedforward, res_dropout, relu_dropout)

    def forward(self, x, x_key_padding_mask=None, x_attn_mask=None):
        """
        x : input of the encoder layer -> (L, B, d)
        """
        x = self.transformer(x, x, x, key_padding_mask=x_key_padding_mask, attn_mask=x_attn_mask)
        x = self.feedforward(x)
        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, attn_dropout, res_dropout, relu_dropout):
        """
        Args:
            d_model: the number of expected features in the input (required).
            nhead: the number of heads in the multiheadattention models (required).
            dim_feedforward: the dimension of the feedforward network model (required).
            attn_dropout: the dropout value for multihead attention (required).
            res_dropout: the dropout value for residual connection (required).
            relu_dropout: the dropout value for relu (required).
        """
        super(TransformerDecoderBlock, self).__init__()
        # self.transformer1 = TransformerBlock(d_model, nhead, attn_dropout, res_dropout)
        self.transformer2 = TransformerBlock(d_model, nhead, attn_dropout, res_dropout)
        self.feedforward = FeedForwardBlock(d_model, dim_feedforward, res_dropout, relu_dropout)

    def forward(
        self,
        src,
        tgt,
        src_mask=None,
        src_key_padding_mask=None,
        tgt_mask=None,
        tgt_key_padding_mask=None,
    ):
        """
        src : output from the encoder layer(query) -> (L, B, d)
        tgt : input from the decoder layer(key, value) -> (L, B, d)
        """
        """
        tgt = self.transformer1(
            tgt, tgt, tgt, key_padding_mask=tgt_key_padding_mask, attn_mask=tgt_mask
        )
        """
        x = self.transformer2(
            tgt, src, src, key_padding_mask=src_key_padding_mask, attn_mask=src_mask
        )
        x = self.feedforward(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, attn_dropout, res_dropout):
        super(TransformerBlock, self).__init__()
        self.res_dropout = res_dropout
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=attn_dropout)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=True):
        mask = get_future_mask(query, key).to("cuda") if attn_mask else None
        x = self.self_attn(query, key, value, key_padding_mask=key_padding_mask, attn_mask=mask)[0]
        x = query + F.dropout(x, self.res_dropout, self.training)
        x = self.layernorm(x)
        return x


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, dim_feedforward, res_dropout, relu_dropout):
        super(FeedForwardBlock, self).__init__()
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x):
        x2 = self.linear2(F.dropout(F.relu(self.linear1(x)), self.relu_dropout, self.training))
        x = F.relu(x + F.dropout(x2, self.res_dropout, self.training))
        x = self.layernorm(x)
        return x


def get_future_mask(q, k=None):
    dim_query = q.shape[0]
    dim_key = dim_query if k is None else k.shape[0]
    future_mask = torch.triu(torch.ones(dim_query, dim_key), diagonal=1)
    future_mask = future_mask.masked_fill(future_mask == 1.0, float("-inf")).masked_fill(
        future_mask == 0.0, 1.0
    )
    return future_mask
