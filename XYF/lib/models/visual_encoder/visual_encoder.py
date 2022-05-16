import torch
from timm.models.layers import trunc_normal_
from torch import nn
from lib.models.attention import SelfAttention
from einops import rearrange, repeat, reduce


class VisualEncoder(nn.Module):

    def __init__(self, cfg):
        super(VisualEncoder, self).__init__()
        tokens = cfg.TOKENS
        input_dim = cfg.INPUT_DIM
        dim = cfg.OUTPUT_DIM
        depth = cfg.DEPTH
        heads = cfg.HEADS
        dropout = cfg.DROPOUT
        drop_path = cfg.DROPPATH

        self.proj = nn.Linear(input_dim, dim)
        self.sa_block = nn.ModuleList([SelfAttention(dim, heads, dropout=dropout, drop_path=drop_path) for _ in range(depth)])

        self.pos_embed = nn.Parameter(torch.zeros(tokens, dim))
        trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x):
        b, *_ = x.shape
        x = self.proj(x)

        pos_embed = repeat(self.pos_embed, "... -> b ...", b=b)
        x = x + pos_embed

        for sa in self.sa_block:
            x = sa(x)
        return x
