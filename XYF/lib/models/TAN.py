import torch
from torch import nn
import lib.models.clip_modules as clip_modules
import lib.models.prop_modules as prop_modules
import lib.models.map_modules as map_modules
import lib.models.fusion_modules as fusion_modules
import lib.models.pred_modules as pred_modules


class MultiScale_TAN(nn.Module):

    def __init__(self, cfg):
        super(MultiScale_TAN, self).__init__()
        self.cfg = cfg

        self.clip_module = getattr(clip_modules, cfg.CLIP_MODULE.NAME)(cfg.CLIP_MODULE.PARAMS)
        self.prop_module = getattr(prop_modules, cfg.PROP_MODULE.NAME)(cfg.PROP_MODULE.PARAMS)
        if getattr(fusion_modules, cfg.FUSION_MODULE.NAME, None):
            self.fusion_module = getattr(fusion_modules, cfg.FUSION_MODULE.NAME)(cfg.FUSION_MODULE.PARAMS)
        self.map_modules = nn.ModuleList()
        self.pred_modules = nn.ModuleList()

        for _ in range(cfg.PARAMS.NUM_SCALES):
            self.map_modules.append(getattr(map_modules, cfg.MAP_MODULE.NAME)(cfg.MAP_MODULE.PARAMS))
            self.pred_modules.append(getattr(pred_modules, cfg.PRED_MODULE.NAME)(cfg.PRED_MODULE.PARAMS))

    def forward(self, textual_input, textual_mask, visual_input, visual_mask):
        # textual_input.shape: [B, 25, 300],    textual_mask.shape: [B, 25, 1]
        # visual_input.shape: [B, 4096, 384],   visual_mask.shape: [B, 1, 384]
        if 'GROUPS' in self.cfg.CLIP_MODULE.PARAMS:
            clip_input = torch.cat([visual_input for _ in range(self.cfg.CLIP_MODULE.PARAMS.GROUPS)], dim=1)
        else:
            # clip_input.shape: [B, 4096, 384]
            clip_input = visual_input

        # vis_h.shape: torch.Size([B, 512, 64]), clip_mask.shape: torch.Size([B, 1, 64])
        vis_h, clip_mask = self.clip_module(clip_input, visual_mask)

        # prop_hs.shape: List([B, 512, 64, 16], ...), map_masks.shape: List([B, 1, 64, 16], ...)
        prop_hs, map_masks = self.prop_module(vis_h, clip_mask)

        assert len(prop_hs) == len(map_masks) == self.cfg.PARAMS.NUM_SCALES
        predictions = []
        for idx, (prop_h, map_mask) in enumerate(zip(prop_hs, map_masks)):
            fused_h, map_mask = self.fusion_module(textual_input, textual_mask, prop_h, map_mask)
            map_h, map_mask = self.map_modules[idx](fused_h, map_mask)
            prediction, map_mask = self.pred_modules[idx](map_h, map_mask)
            predictions.append(prediction * map_mask)

        # predictions.shape: List([B, 1, 64, 16], [B, 1, 32, 16], [B, 1, 16, 16])

        return predictions, map_masks
