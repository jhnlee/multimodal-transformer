import logging
import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class CrossmodalTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, n_layer):
        super(CrossmodalTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([])
        for layer in range(n_layer):
            new_layer = TransformerDecoderBlock(d_model, nhead, dim_feedforward, dropout)
            self.layers.append(new_layer)

    def forward(self, src, tgt):
        for layer in self.layers:
            tgt = layer(src, tgt)
        return tgt


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, n_layer):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([])
        for layer in range(n_layer):
            new_layer = TransformerEncoderBlock(d_model, nhead, dim_feedforward, dropout)
            self.layers.append(new_layer)

    def forward(self, src):
        for layer in self.layers:
            src = layer(src)
        return src


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        """
        Args:
            d_model: the number of expected features in the input (required).
            nhead: the number of heads in the multiheadattention models (required).
            dim_feedforward: the dimension of the feedforward network model (required).
            dropout: the dropout value (required).
        """
        super(TransformerEncoderBlock, self).__init__()
        self.transformer = TransformerBlock(d_model, nhead, dropout)
        self.feedforward = FeedForwardBlock(d_model, dim_feedforward, dropout)

    def forward(self, x, x_key_padding_mask=None, x_attn_mask=None):
        """
        x : input of the encoder layer -> (L, B, d)
        """
        x = self.transformer(x, x, x, key_padding_mask=x_key_padding_mask, attn_mask=x_attn_mask)
        x = self.feedforward(x)
        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        """
        Args:
            d_model: the number of expected features in the input (required).
            nhead: the number of heads in the multiheadattention models (required).
            dim_feedforward: the dimension of the feedforward network model (required).
            dropout: the dropout value (required).
        """
        super(TransformerDecoderBlock, self).__init__()
        # self.transformer1 = TransformerBlock(d_model, nhead, dropout)
        self.transformer2 = TransformerBlock(d_model, nhead, dropout)
        self.feedforward = FeedForwardBlock(d_model, dim_feedforward, dropout)

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
    def __init__(self, d_model, nhead, dropout):
        super(TransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        x = self.self_attn(
            query, key, value, key_padding_mask=key_padding_mask, attn_mask=attn_mask
        )[0]
        x = query + self.dropout(x)
        x = self.layernorm(x)
        return x


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout):
        super(FeedForwardBlock, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x):
        x2 = self.linear2(self.dropout1(F.relu(self.linear1(x))))
        x = F.relu(x + self.dropout2(x2))
        x = self.layernorm(x)
        return x
