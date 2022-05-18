import torch
from timm.models.layers import trunc_normal_
from torch import nn
from lib.models.attention import SelfAttention, CrossAttention, Attention
from einops import rearrange, repeat, reduce


class FFN(nn.Module):

    def __init__(self, cfg):
        super(FFN, self).__init__()
        input_dim = cfg.INPUT_DIM

        self.to_out = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 3)
        )

    def forward(self, x):
        x = self.to_out(x)
        return x
