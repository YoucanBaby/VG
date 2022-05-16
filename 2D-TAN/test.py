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
    # x = torch.randn((3, 3))
    # mask = torch.tensor([
    #         [0, 0, 0],
    #         [0, 1, 1],
    #         [0, 1, 1],
    # ])
    # output = VisionTransformer(x, mask=mask)
    tious = [0.1, 0.3, 0.5, 0.7]
    recalls = [1, 5]

    res = [['R{}@{}'.format(i, j) for i in recalls for j in tious]]
    print(res)

    max_result = [[[0, 0], [1, 0]], [[2, 0], [3, 0]], [[4, 0], [5, 0]], [[6, 0], [7, 0]]]
    res = ['{:.02f}'.format(max_result[i][j][0]) for j in range(len(recalls)) for i in range(len(tious))]
    print(res)

    for i in range(len(tious)):
        for j in range(len(recalls)):
            print(max_result[i][j][0])


if __name__ == '__main__':
    main()