from einops import rearrange
from timm.models.layers import trunc_normal_
from torch import nn

import lib.models.visual_encoder as visual_encoder
import lib.models.text_encoder as text_encoder
import lib.models.decoder as decoder
from lib.core.config import cfg
from lib.models.utils.prediction import Prediction


class GTR(nn.Module):

    def __init__(self):
        super(GTR, self).__init__()
        self.cfg = cfg

        self.visual_encoder = getattr(visual_encoder,
                                      cfg.MODEL.VISUAL_ENCODER.NAME)(cfg.MODEL.VISUAL_ENCODER.PARAMS)
        self.text_encoder = getattr(text_encoder,
                                    cfg.MODEL.TEXT_ENCODER.NAME)(cfg.MODEL.TEXT_ENCODER.PARAMS)
        self.decoder = getattr(decoder,
                               cfg.MODEL.DECODER.NAME)(cfg.MODEL.DECODER.PARAMS)
        self.prediction = Prediction(dim=cfg.MODEL.PARAMS.DIM)

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
        # v_input.shape: [B, 384, 1024], t_input.shape: [B, 46, 300],

        v_f = self.visual_encoder(v_input)
        # v_f.shape: [B, 384, 320]

        t_f = self.text_encoder(t_input)
        # t_f.shape: [B, 18, 320]

        latent = self.decoder(v_f, t_f)

        preds = self.prediction(latent)
        return preds
