import torch
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer


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
    x = torch.randn((3, 3))
    mask = torch.tensor([
            [0, 0, 0],
            [0, 1, 1],
            [0, 1, 1],
    ])
    output = VisionTransformer(x, mask=mask)


if __name__ == '__main__':
    main()