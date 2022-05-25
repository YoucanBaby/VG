import torch
from timm.models.layers import trunc_normal_, DropPath
from torch import nn
from lib.models.utils.attention import CrossAttention, Attention
from einops import repeat


class CrossAttentionNoMLP(nn.Module):
    def __init__(self, q_dim, kv_dim, heads=8, head_dim=64, dropout=0., drop_path=0.):
        super().__init__()
        self.q_norm = nn.LayerNorm(q_dim)
        self.kv_norm = nn.LayerNorm(kv_dim)
        self.attention = Attention(q_dim, kv_dim, heads=heads, head_dim=head_dim, dropout=dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, q, kv, mask=None):
        if mask is not None:
            b, m_h, m_w = mask.shape
            kv = kv[:, :m_h]

        q, kv = self.q_norm(q), self.kv_norm(kv)
        x = q + self.drop_path(self.attention(q, kv))
        return x


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
                nn.LayerNorm(dim),
                Attention(dim)
            ) for _ in range(depth)
        ])
        self.ca_v_block = nn.ModuleList([CrossAttentionNoMLP(dim, dim) for _ in range(depth)])
        self.ca_t_block = nn.ModuleList([CrossAttention(dim, dim) for _ in range(depth)])

        self.latent = nn.Parameter(torch.zeros(tokens, dim))
        trunc_normal_(self.latent, std=.02)
        self.pos_embed = nn.Parameter(torch.zeros(tokens, dim))

    def forward(self, v_f, t_f):
        b, *_ = v_f.shape
        latent = repeat(self.latent, '... -> b ...', b=b)
        pos_embed = repeat(self.pos_embed, '... -> b ...', b=b)

        latent = latent + pos_embed

        for sa, ca_v, ca_t in zip(self.sa_block, self.ca_v_block, self.ca_t_block):
            latent = sa(latent)
            latent = ca_v(latent, v_f)
            latent = ca_t(latent, t_f)

        return latent
