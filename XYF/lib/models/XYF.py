from einops import rearrange
from timm.models.layers import trunc_normal_
from torch import nn

import lib.models.visual_encoder as visual_encoder
import lib.models.text_encoder as text_encoder
import lib.models.decoder as decoder
import lib.models.prediction as prediction


class MLPwithMask(nn.Module):
    def __init__(self, m_h):
        super(MLPwithMask, self).__init__()
        nn.linear()

    def forward(self, visual_input, visual_mask):
        pass


class VisualEncoder(nn.Module):
    def __init__(self, cfg):
        super(VisualEncoder, self).__init__()

    def forward(self, visual_input, visual_mask):
        v_f = self.mlp(visual_input, visual_mask)


class XYF(nn.Module):

    def __init__(self, cfg):
        super(XYF, self).__init__()
        self.cfg = cfg

        self.visual_encoder = getattr(visual_encoder, cfg.VISUAL_ENCODER.NAME)(cfg.VISUAL_ENCODER.PARAMS)
        self.text_encoder = getattr(text_encoder, cfg.TEXT_ENCODER.NAME)(cfg.TEXT_ENCODER.PARAMS)
        self.decoder = getattr(decoder, cfg.DECODER.NAME)(cfg.DECODER.PARAMS)
        self.prediction = getattr(prediction, cfg.PREDICTION.NAME)(cfg.PREDICTION.PARAMS)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, visual_input, textual_input, visual_mask, textual_mask):
        # visual_input.shape: [B, 8416, 1024], textual_input.shape: [B, 46, 300],

        v_f = self.visual_encoder(visual_input)
        # v_f.shape: [B, 不定长, 384]

        t_f = self.text_encoder(textual_input)
        # t_f.shape: [B, 不定长, 384]

        latent = self.decoder(v_f, t_f)

        preds = self.prediction(latent)
        return preds