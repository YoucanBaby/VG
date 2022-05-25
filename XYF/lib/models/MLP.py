import torch
from torch import nn
from einops import reduce

from lib.models.utils.attention import SelfAttention


class MLP(nn.Module):
    '''用于测试输入和输出
    '''

    def __init__(self, cfg):
        super(MLP, self).__init__()
        self.cfg = cfg

        self.visual_ffn = nn.Linear(1024, 384)
        self.visual_encoder = nn.ModuleList(
            [SelfAttention(384, 8, dropout=0., drop_path=0.) for _ in range(2)]
        )
        self.text_ffn = nn.Linear(300, 384)
        self.text_encoder = nn.ModuleList(
            [SelfAttention(384, 8, dropout=0., drop_path=0.) for _ in range(2)]
        )

        self.decoder = nn.ModuleList(
            [SelfAttention(384, 8, dropout=0., drop_path=0.) for _ in range(4)]
        )

        self.to_time = nn.Sequential(
            nn.Linear(384, 384),
            nn.GELU(),
            nn.Linear(384, 384),
            nn.GELU(),
            nn.Linear(384, 2)
        )
        self.to_score = nn.Sequential(
            nn.Linear(384, 384),
            nn.GELU(),
            nn.Linear(384, 384)
        )

    def forward(self, visual_input, textual_input):
        # visual_input.shape: [B, 不定长, 1024], textual_input.shape: [B, 25, 300],
        # TODO 把txt和video的token都padding到相同大小试试
        # TODO 为什么输出的每一列的值都一样?

        v_f = self.visual_ffn(visual_input)
        for sa in self.visual_encoder:
            v_f = sa(v_f)

        t_f = self.text_ffn(textual_input)
        for sa in self.text_encoder:
            t_f = sa(t_f)

        x = torch.cat((v_f, t_f), dim=1)
        for sa in self.decoder:
            x = sa(x)

        times = self.to_time(x)
        scores = self.to_score(x)
        scores = reduce(scores, 'b n d -> b n 1', 'mean')
        preds = torch.cat([times, scores], dim=-1)

        return preds
