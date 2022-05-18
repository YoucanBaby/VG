from torch.utils.data import TensorDataset
import torch
from torch.utils.data import DataLoader
import math


def main():
    # x = torch.tensor([
    #     [0, 1, 2],
    #     [3, 4, 5],
    #     [6, 7, 8],
    #     [9, 10, 11]
    # ])
    # gt = torch.tensor([11, 22, 33, 44])
    #
    # # 封装数据a与标签b
    # train_dataset = TensorDataset(x, gt)
    #
    # print('=' * 160)
    # print([t for t in train_dataset])
    #
    # train_loader = DataLoader(train_dataset,
    #                           batch_size=2,
    #                           shuffle=True)
    #
    # for cur_iter, sample in enumerate(train_loader):
    #     x, gt = sample
    #     print('=' * 160)
    #     print('cur_iter: {}'.format(cur_iter))
    #     print('x: {}'.format(x))
    #     print('gt: {}'.format(gt))
    x = torch.tensor([1, math.e, math.e ** 2, 10])
    print(torch.log(x))

if __name__ == '__main__':
    main()
