import os
import sys

import torch
import math
import torch.nn as nn
from backbone.neural import MultiHeadedAttention, PositionwiseFeedForward


class Classifier(nn.Module):
    def __init__(self, window_size):
        super(Classifier, self).__init__()
        if window_size == 1:
            conv_kernel_size = 2
            flat_size = 256
        else:
            conv_kernel_size = window_size*2-2
            flat_size = 256 * 3

        # block1
        if window_size == 1:
            self.block1 = nn.Sequential(
                nn.Conv1d(in_channels=768, out_channels=256, kernel_size=conv_kernel_size),
                nn.LayerNorm([256, 1]),
                nn.ReLU(),
            )
        else:
            self.block1 = nn.Sequential(
                nn.Conv1d(in_channels=768, out_channels=256, kernel_size=conv_kernel_size),
                nn.LayerNorm([256, 3]),
                nn.ReLU(),
            )
        
        self.block2 = nn.Sequential(
            nn.Linear(flat_size, 1),
        )
        
    def forward(self, x):
        x = x.transpose(1, 2).contiguous() # B * N * C --> B * C * N
        out = self.block1(x)
        batch_size = out.size(0)
        out = out.view(batch_size, -1)
        out = self.block2(out)
        return out.view(-1)


class LinearClassifier(nn.Module):
    def __init__(self, window_size):
        super(LinearClassifier, self).__init__()
        flat_size = 768 * window_size * 2
        # self.linear_layer = nn.Sequential(
        #     nn.Linear(flat_size, 32),
        #     nn.LayerNorm([32]),
        #     nn.ReLU(),
        #     nn.Linear(32, 1)
        # )
        self.linear_layer = nn.Sequential(
            nn.Linear(flat_size, 1),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        out = self.linear_layer(x)
        return out.view(-1)

class PositionalEncoding(nn.Module):
    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if (step):
            emb = emb + self.pe[:, step][:, None, :]
        else:
            emb = emb + self.pe[:, :emb.size(1)]
        emb = self.dropout(emb)
        return emb

    def get_emb(self, emb):
        return self.pe[:, :emb.size(1)]


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, query, inputs, mask):
        if (iter != 0):
            input_norm = self.layer_norm(inputs)
        else:
            input_norm = inputs

        mask = mask.unsqueeze(1)
        context = self.self_attn(input_norm, input_norm, input_norm, mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class SepTransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout, num_inter_layers=0):
        super(SepTransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_inter_layers = num_inter_layers
        self.pos_emb = PositionalEncoding(dropout, d_model)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout) for _ in range(num_inter_layers)])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.wo = nn.Linear(d_model, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, top_vecs, mask):
        """ See :obj:`EncoderBase.forward()`"""
        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        pos_emb = self.pos_emb.pe[:, :n_sents]
        x = top_vecs * mask[:, :, None].float()
        x = x + pos_emb

        for i in range(self.num_inter_layers):
            x = self.transformer_inter[i](i, x, x, ~mask)  # all_sents * max_tokens * dim

        # !!TODO!!
        # Layer Norm or not?
        out = self.layer_norm(x)
        # sent_scores = self.sigmoid(self.wo(x))
        # sent_scores = sent_scores.squeeze(-1) * mask.float()
        return out
