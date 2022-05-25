import torch
from timm.models.layers import trunc_normal_
from torch import nn
from lib.models.utils.attention import CrossAttention, Attention
from einops import repeat


class Sequential(nn.Module):

    def __init__(self, cfg):
        super(Sequential, self).__init__()
        tokens = cfg.TOKENS
        dim = cfg.DIM
        depth = cfg.DEPTH
        heads = cfg.HEADS
        dropout = cfg.DROPOUT
        drop_path = cfg.DROPPATH

        self.sa_block = nn.ModuleList([nn.Sequential(
                Attention(dim, heads=heads, dropout=dropout),
                nn.LayerNorm(dim)
            ) for _ in range(depth)
        ])
        self.ca_v_block = nn.ModuleList(
            [CrossAttention(dim, dim, heads=heads, dropout=dropout, drop_path=drop_path) for _ in range(depth)]
        )
        self.ca_t_block = nn.ModuleList(
            [CrossAttention(dim, dim, heads=heads, dropout=dropout, drop_path=drop_path) for _ in range(depth)]
        )

        self.latent = nn.Parameter(torch.zeros(tokens, dim))
        trunc_normal_(self.latent, std=.02)

    def forward(self, v_f, t_f):
        b, *_ = v_f.shape
        latent = repeat(self.latent, '... -> b ...', b=b)

        for sa, ca_v, ca_t in zip(self.sa_block, self.ca_v_block, self.ca_t_block):
            latent = sa(latent)
            latent = ca_v(latent, v_f)
            latent = ca_t(latent, t_f)

        return latent
