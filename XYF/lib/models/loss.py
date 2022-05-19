import torch
import torch.nn.functional as F
from einops import rearrange, repeat, reduce

from lib.core.config import cfg


def get_iou_2d(pred, gt):
    '''
    :param pred: torch.size([b, 2])
    :param gt: torch.size([b, 2])
    :return: torch.size([b])
    '''
    n = (pred[:, 1] - pred[:, 0]).clamp(min=0)
    m = (gt[:, 1] - gt[:, 0]).clamp(min=0)

    inter_left = torch.max(pred[:, 0], gt[:, 0]).clamp(min=0)
    inter_right = torch.min(pred[:, 1], gt[:, 1]).clamp(min=0)
    inter = (inter_right - inter_left).clamp(min=0)

    union = n + m - inter

    return inter / union


def get_loss(score, pred, gt):
    '''
    :param score: torch.size([b])
    :param pred: torch.size([b, 2])
    :param gt: torch.size([b, 2])
    :return: torch.size([b])
    '''
    cost_l1, cost_iou = cfg.LOSS.PARAMS.COST_L1, cfg.LOSS.PARAMS.COST_IOU

    l1 = torch.abs(pred - gt)
    l1_loss = cost_l1 * (l1[:, 0] + l1[:, 1]) ** 0.5

    iou = get_iou_2d(pred, gt).clamp(0, 1).add(1e-4)
    iou_loss = cost_iou * (- torch.log(iou)) ** 0.5

    score = score.clamp(0, 1).add(1e-4)
    score_loss = (- torch.log(score)) ** 0.5

    # TODO loss可以再改一改
    # TODO RuntimeError: Function 'PowBackward0' returned nan values in its 0th output.
    loss = score_loss + l1_loss + iou_loss

    # b = score.shape[0]
    # message = ''
    # message += 'score_loss: {:.2f}  '.format(score_loss.sum().item() / b)
    # message += 'l1_loss: {:.2f}  '.format(l1_loss.sum().item() / b)
    # message += 'iou_loss: {:.2f}  '.format(iou_loss.sum().item() / b)
    # message += 'loss: {:.2f}'.format(loss.sum().item() / b)
    # print(message)

    return loss


def get_iou_3d(pred, gt):
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


def get_match_loss(score, pred, gt):
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

    iou = get_iou_3d(pred, gt).clamp(0, 1).add(1e-8)
    iou_loss = cost_iou * (- torch.log(iou))

    loss = - score + l1_loss + iou_loss
    return loss


def get_best_pred(preds, gt, duration):
    '''
    返回最好的预测结果的下标
    :param preds: torch.size([b, 100, 3])
    :param gt: list.size([b, 2])
    :param duration: list.size([b])
    :param cfg:
    :return: torch.size([b, 3])
    '''
    device = preds.device
    b, n, _ = preds.shape

    gt = repeat(gt, 'b d -> b n d', n=n)
    duration = repeat(duration, 'b d -> b n d', n=n)

    norm_times = (preds[:, :, :2] / duration).clamp(min=0, max=1)
    norm_score = (preds[:, :, 2] / 100).clamp(min=0, max=1)
    norm_gt = gt / duration

    match_loss = get_match_loss(norm_score, norm_times, norm_gt)
    index = torch.argmin(match_loss, dim=1)

    best_pred = torch.stack([preds[i, j] for i, j in enumerate(index)])

    return best_pred


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

    gt = torch.tensor(gt, device=device, requires_grad=False)

    duration = torch.tensor(duration, device=device, requires_grad=False)
    duration = repeat(duration, 'b -> b 2')

    best_pred = get_best_pred(preds, gt, duration)

    norm_times = (best_pred[:, :2] / duration).clamp(min=0, max=1)
    norm_score = (best_pred[:, 2] / 100).clamp(min=0, max=1)
    norm_gt = gt / duration

    loss = get_loss(norm_score, norm_times, norm_gt)

    loss_value = loss.sum() / b
    return loss_value
