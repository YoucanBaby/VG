import torch
from torch import nn
from einops import reduce


class FFN(nn.Module):

    def __init__(self, cfg):
        super(FFN, self).__init__()
        input_dim = cfg.INPUT_DIM
        hidden_dim = cfg.HIDDEN_DIM

        self.to_time = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2)
        )
        self.to_score = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        times = self.to_time(x)

        scores = self.to_score(x)
        scores = reduce(scores, 'b n d -> b n 1', 'mean')

        preds = torch.cat([times, scores], dim=-1)

        return preds
