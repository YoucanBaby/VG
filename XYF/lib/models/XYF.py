from einops import rearrange, repeat, reduce
from timm.models.layers import trunc_normal_
import torch
from torch import nn
import torch.nn.functional as F

import lib.models.visual_encoder as visual_encoder
import lib.models.text_encoder as text_encoder
import lib.models.decoder as decoder
import lib.models.prediction as prediction
from lib.models.utils.attention import MLP, SelfAttention, CrossAttention


class Encoder(nn.Module):
    def __init__(self, num_tokens, in_dim, dim=384, depth=2):
        super(Encoder, self).__init__()
        self.linear = nn.Linear(in_dim, dim)
        self.sa_block = nn.ModuleList([SelfAttention(dim) for _ in range(depth)])

        self.pos_embed = nn.Parameter(torch.zeros(num_tokens, dim))
        trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x, mask=None):
        b, *_ = x.shape
        device = x.device
        pos_embed = repeat(self.pos_embed, "... -> b ...", b=b, device=device)

        x = self.linear(x)

        x = x + pos_embed

        for sa in self.sa_block:
            x = sa(x, mask)
        return x


class Decoder(nn.Module):
    def __init__(self, num_tokens=100, dim=384, depth=3):
        super().__init__()

        self.sa_block = nn.ModuleList(
            [SelfAttention(dim, dim) for _ in range(depth)]
        )
        self.ca_v_block = nn.ModuleList(
            [CrossAttention(dim, dim) for _ in range(depth)]
        )
        self.ca_t_block = nn.ModuleList(
            [CrossAttention(dim, dim) for _ in range(depth)]
        )

        self.latent = nn.Parameter(torch.zeros(num_tokens, dim))
        trunc_normal_(self.latent, std=.02)

    def forward(self, v_feat, v_mask, t_feat, t_mask):
        b, *_ = v_feat.shape
        latent = repeat(self.latent, '... -> b ...', b=b)

        for sa, ca_v, ca_t in zip(self.sa_block, self.ca_v_block, self.ca_t_block):
            latent = sa(latent)
            latent = ca_v(latent, v_feat, v_mask)
            latent = ca_t(latent, t_feat, t_mask)

        return latent


class Prediction(nn.Module):
    def __init__(self, dim=384):
        super().__init__()
        self.to_time = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 2)
        )
        self.to_score = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        times = self.to_time(x)

        scores = self.to_score(x)
        scores = reduce(scores, 'b n d -> b n 1', 'mean')

        preds = torch.cat([times, scores], dim=-1)

        return preds


class XYF(nn.Module):

    def __init__(self, cfg):
        super(XYF, self).__init__()
        self.cfg = cfg

        self.v_encoder = Encoder(num_tokens=768, in_dim=1024)
        self.t_encoder = Encoder(num_tokens=46, in_dim=300)
        self.decoder = Decoder()
        self.prediction = Prediction()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, v_input, v_mask, t_input, t_mask):
        # v_input.shape: [B, 8416, 1024], v_mask.shape: [B, ?, 1024]
        # t_input.shape: [B, 46, 300], t_mask.shape: [B, ?, 300]
        # v_f.shape: [B, 8416, 384]
        # t_f.shape: [B, 46, 384]

        v_feat = self.v_encoder(v_input, v_mask)
        t_feat = self.t_encoder(t_input, t_mask)

        latent = self.decoder(v_feat, v_mask, t_feat, t_mask)

        preds = self.prediction(latent)
        return preds
