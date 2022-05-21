from einops import rearrange
from timm.models.layers import trunc_normal_
from torch import nn


class MLP(nn.Module):

    def __init__(self, cfg):
        super(MLP, self).__init__()
        self.cfg = cfg
        input_dim = cfg.MODEL

        # TODO 写个MLP测试一下数据

        # self.visual_encoder = nn.ModuleList(
        #     [SelfAttention(dim, heads, dropout=dropout, drop_path=drop_path) for _ in range(depth)]
        # )


    def forward(self, visual_input, textual_input):
        # visual_input.shape: [B, 不定长, 1024], textual_input.shape: [B, 25, 300],
        # TODO 把txt和video的token都padding到相同大小试试
        # TODO 为什么输出的每一列的值都一样?

        v_f = self.visual_encoder(visual_input)
        # v_f.shape: [B, 不定长, 384]

        t_f = self.text_encoder(textual_input)
        # t_f.shape: [B, 不定长, 384]

        latent = self.decoder(v_f, t_f)

        preds = self.prediction(latent)
        return preds
