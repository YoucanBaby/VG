import torch
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer
from lib.models.TAN import MultiScale_TAN
from lib.datasets.dataset import MomentLocalizationDataset
from lib.core.config import cfg, update_config
from lib.core.utils import AverageMeter, create_logger
import lib.models as models
import lib.models.loss as loss
from moment_localization.run import parse_args, reset_config


def main():
    # input = torch.ones([2, 2])
    # mask = torch.zeros([2, 2], dtype=torch.bool)
    # print(mask)
    # mask[range(2), range(2)] = 1
    # print(mask)
    #
    # target = torch.randn([2, 2])
    # loss = F.binary_cross_entropy(input, target)
    # loss_value = F.binary_cross_entropy_with_logits(input, target)
    #
    # print(input.masked_select(mask))
    #
    # print(target)
    # print(loss, torch.sum(loss))
    # print(loss_value)

    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.autograd.set_detect_anomaly(True)

    reset_config(cfg, args)

    model = MultiScale_TAN(cfg)

    textual_input = torch.randn((1, 25, 300))
    textual_mask = torch.randn((1, 25, 1))
    visual_input = torch.randn((1, 4096, 384))
    visual_mask = torch.randn((1, 1, 384))

    res = model(textual_input, textual_mask, visual_input, visual_mask)


if __name__ == '__main__':
    main()