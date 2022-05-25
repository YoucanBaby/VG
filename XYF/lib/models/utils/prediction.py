import torch
from torch import nn
from einops import reduce


class Prediction(nn.Module):
    def __init__(self, dim=384):
        super().__init__()
        self.to_out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.to_out(x)
