from einops import rearrange, repeat, reduce
from timm.models.layers import trunc_normal_
import torch
from torch import nn
import torch.nn.functional as F
import random

from lib.models.utils.attention import SelfAttention, CrossAttention
from lib.models.utils.prediction import Prediction


class Encoder(nn.Module):
    def __init__(self, num_tokens, in_dim, dim=384, depth=1):
        super(Encoder, self).__init__()
        self.linear = nn.Linear(in_dim, dim)
        self.sa_block = nn.ModuleList([SelfAttention(dim) for _ in range(depth)])

        self.pos_embed = nn.Parameter(torch.zeros(num_tokens, dim))
        trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x, mask=None):
        b, *_ = x.shape
        pos_embed = repeat(self.pos_embed, "... -> b ...", b=b)

        x = self.linear(x)

        x = x + pos_embed

        for sa in self.sa_block:
            x = sa(x, mask)
        return x


class Decoder1(nn.Module):
    def __init__(self, num_tokens=100, dim=384, depth=4):
        super().__init__()

        self.ca = CrossAttention(dim, dim)
        self.sa_block = nn.ModuleList(
            [SelfAttention(dim) for _ in range(depth)]
        )

        self.latent = nn.Parameter(torch.zeros(num_tokens, dim))
        trunc_normal_(self.latent, std=.02)

    def forward(self, v_feat, t_feat, v_mask=None, t_mask=None):
        x = torch.cat([v_feat, t_feat], dim=-1)
        b, *_ = v_feat.shape
        latent = repeat(self.latent, '... -> b ...', b=b)

        latent = self.ca_v(latent, v_feat, v_mask)

        for sa in self.sa_block:
            latent = sa(latent)
        return latent


class Decoder(nn.Module):
    def __init__(self, dim=384, depth=3):
        super().__init__()
        self.depth = depth

        self.sa_block = nn.ModuleList(
            [SelfAttention(dim) for _ in range(depth)]
        )
        self.ca = CrossAttention(dim, dim)

        self.pos_embed = nn.Parameter(torch.zeros(396, dim))
        self.latent = nn.Parameter(torch.zeros(100, dim))
        self._init_parameters()

    def _init_parameters(self):
        with torch.no_grad():
            trunc_normal_(self.pos_embed, std=.02)
            trunc_normal_(self.latent, std=.02)

    def forward(self, v_feat, t_feat, v_mask=None, t_mask=None):
        b, *_ = v_feat.shape
        pos_embed = repeat(self.pos_embed, "... -> b ...", b=b)
        latent = repeat(self.latent, "... -> b ...", b=b)

        x = torch.cat([v_feat, t_feat], dim=1)
        x = x + pos_embed

        for sa in self.sa_block:
            x = sa(x)
        latent = self.ca(latent, x)
        return latent


class XYF(nn.Module):

    def __init__(self):
        super(XYF, self).__init__()

        self.v_encoder = Encoder(num_tokens=384, in_dim=1024)
        self.rnn = nn.GRU(300, 150, num_layers=3, bidirectional=True, batch_first=True)
        self.t_encoder = Encoder(num_tokens=12, in_dim=300)
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

    def forward(self, v_input, t_input, v_mask=None, t_mask=None):
        # v_input.shape: [B, 384, 1024], v_mask.shape: [B, 384, 1024]
        # t_input.shape: [B, 12, 300], t_mask.shape: [B, 12, 300]
        # v_f.shape: [B, 384, 384]
        # t_f.shape: [B, 12, 384]

        v_feat = self.v_encoder(v_input, v_mask)

        # self.rnn.flatten_parameters()
        # t_input = self.rnn(t_input)[0]
        t_feat = self.t_encoder(t_input, t_mask)

        latent = self.decoder(v_feat, t_feat, v_mask, t_mask)

        preds = self.prediction(latent)
        return preds
