import torch
from timm.models.layers import trunc_normal_
from torch import nn
from lib.models.attention import SelfAttention
from einops import rearrange, repeat, reduce


class TextEncoder(nn.Module):

    def __init__(self, cfg):
        super(TextEncoder, self).__init__()
        tokens = cfg.TOKENS
        input_dim = cfg.INPUT_DIM
        dim = cfg.OUTPUT_DIM
        depth = cfg.DEPTH
        heads = cfg.HEADS
        dropout = cfg.DROPOUT
        drop_path = cfg.DROPPATH

        self.rnn = getattr(nn, cfg.RNN.NAME)(
            input_dim, input_dim // 2 if cfg.RNN.BIDIRECTIONAL else input_dim,
            num_layers=cfg.RNN.NUM_LAYERS, bidirectional=cfg.RNN.BIDIRECTIONAL, batch_first=True
        )
        self.sa_block = nn.ModuleList(
            [SelfAttention(input_dim, heads, dropout=dropout, drop_path=drop_path) for _ in range(depth)]
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, dim)
        )

        self.pos_embed = nn.Parameter(torch.zeros(tokens, input_dim))
        trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x):
        b, *_ = x.shape

        x = self.rnn(x)[0]

        pos_embed = repeat(self.pos_embed, "... -> b ...", b=b)
        # x = x + pos_embed

        for sa in self.sa_block:
            x = sa(x)
        x = self.proj(x)
        return x
