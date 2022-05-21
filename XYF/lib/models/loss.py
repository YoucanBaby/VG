import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce

from lib.core.config import cfg


def get_best_pred(preds, gt, duration):
    '''
    返回最好的预测结果的下标
    :param preds: torch.size([b, 100, 3])
    :param gt: list.size([b, 2])
    :param duration: list.size([b])
    :param cfg:
    :return: torch.size([b, 3])
    '''

    def _iou(pred, gt):
        '''
        :param pred: torch.size([b, 100, 2])
        :param gt: torch.size([b, 100, 2])
        :return: torch.size([b, 100])
        '''
        n = (pred[:, :, 1] - pred[:, :, 0]).clamp(min=0)
        m = (gt[:, :, 1] - gt[:, :, 0]).clamp(min=0)

        inter_left = torch.max(pred[:, :, 0], gt[:, :, 0])
        inter_right = torch.min(pred[:, :, 1], gt[:, :, 1])
        inter = (inter_right - inter_left).clamp(min=0)

        union = n + m - inter

        return inter / union

    def _match_loss(score, pred, gt):
        '''
        :param score: torch.size([b, 100])
        :param pred: torch.size([b, 100, 2])
        :param gt: torch.size([b, 100, 2])
        :param cfg:
        :return: torch.size([b, 100])
        '''
        cost_l1, cost_iou = cfg.LOSS.PARAMS.COST_L1, cfg.LOSS.PARAMS.COST_IOU

        l1 = torch.abs(pred - gt)
        l1_loss = cost_l1 * (l1[:, :, 0] + l1[:, :, 1])

        iou = _iou(pred, gt).clamp(0).add(1e-8).clamp(0, 1)
        iou_loss = cost_iou * (- torch.log(iou))

        loss = - score + l1_loss + iou_loss

        # message = 'score: {}'.format(torch.max(score))
        # message += 'l1_loss: {}'.format(torch.max(l1_loss))
        # message += 'iou_loss: {}'.format(torch.max(iou_loss))
        # print(message)

        return loss

    b, n, _ = preds.shape

    gt = repeat(gt, 'b d -> b n d', n=n)
    duration = repeat(duration, 'b d -> b n d', n=n)

    norm_score = preds[:, :, 2] / 100
    norm_times = preds[:, :, :2] / duration
    norm_gt = gt / duration

    match_loss = _match_loss(norm_score, norm_times, norm_gt)
    index = torch.argmin(match_loss, dim=1)

    best_pred = torch.stack([preds[i, j] for i, j in enumerate(index)])

    return best_pred


class SetLoss(nn.Module):
    def __init__(self):
        super(SetLoss, self).__init__()
        self.cost_l1 = cfg.LOSS.PARAMS.COST_L1
        self.cost_iou = cfg.LOSS.PARAMS.COST_IOU

    def iou(self, pred, gt):
        n = (pred[:, 1] - pred[:, 0]).clamp(0, 1)
        m = (gt[:, 1] - gt[:, 0]).clamp(0, 1)

        inter_left = torch.max(pred[:, 0], gt[:, 0]).clamp(0, 1)
        inter_right = torch.min(pred[:, 1], gt[:, 1]).clamp(0, 1)
        inter = (inter_right - inter_left).clamp(0, 1)

        union = n + m - inter

        iou = inter / union
        return iou.clamp(0, 1)

    def iou_loss(self, pred, gt):
        iou = self.iou(pred, gt)
        iou = iou.add(1e-2).clamp(0, 1)
        return -torch.log(iou)

    def forward(self, score, pred, gt):
        # score = score.add(1e-2).clamp(0, 1)
        # score_loss = -torch.log(score)
        score = score.clamp(0, 1)
        l1_loss = self.cost_l1 * torch.abs(pred[:, 0] - gt[:, 0]) + torch.abs(pred[:, 1] - gt[:, 1])
        iou_loss = self.cost_iou * self.iou_loss(pred, gt)

        # loss = -score + l1_loss + iou_loss

        loss = l1_loss + iou_loss
        return loss, score, l1_loss, iou_loss


def many_to_one_loss(preds, gt, duration):
    '''
    :param preds: torch.size([b, 100, 3])
    :param gt: list.size([b, 2])
    :param duration: list.size([b])
    :param cfg:
    :return: torch.size([1])
    '''
    device = preds.device
    b, *_ = preds.shape

    gt = torch.tensor(gt, requires_grad=True).to(device, non_blocking=True)
    duration = torch.tensor(duration, requires_grad=True).to(device, non_blocking=True)
    duration = repeat(duration, 'b -> b 2')

    best_pred = get_best_pred(preds, gt, duration)

    norm_score = (best_pred[:, 2] / 100).clamp(min=0, max=1)
    norm_times = (best_pred[:, :2] / duration).clamp(min=0, max=1)
    norm_gt = gt / duration

    criterion = SetLoss()
    loss, score, l1_loss, iou_loss = criterion(norm_score, norm_times, norm_gt)

    loss_value = loss.sum() / b
    score_value = score.sum() / b

    # print(score, score_value.sum(), b)

    l1_loss_value = l1_loss.sum() / b
    iou_loss_value = iou_loss.sum() / b

    loss_dict = {'loss': loss_value,
                 'score': score_value,
                 'l1_loss': l1_loss_value,
                 'iou_loss': iou_loss_value}

    return loss_dict
