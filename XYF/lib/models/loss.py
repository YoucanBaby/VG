import torch
import torch.nn.functional as F
from einops import rearrange, repeat, reduce


def get_iou_2d(pred, gt):
    n = (pred[:, 1] - pred[:, 0]).clamp(min=0, max=1)
    m = (gt[:, 1] - gt[:, 0]).clamp(min=0, max=1)

    inter_left = torch.max(pred[:, 0], gt[:, 0]).clamp(min=0, max=1)
    inter_right = torch.min(pred[:, 1], gt[:, 1]).clamp(min=0, max=1)
    inter = (inter_right - inter_left).clamp(min=0, max=1)

    union = n + m - inter

    return inter / union


def get_loss_2d(score, pred, gt, cfg):
    cost_l1, cost_iou = cfg.COST_L1, cfg.COST_IOU

    l1 = torch.abs(pred - gt)
    l1 = l1[:, 0] + l1[:, 1]
    iou = get_iou_2d(pred, gt)

    loss = - torch.log(score.clamp(min=1e-1)) - score + cost_l1 * l1 + cost_iou * iou
    return loss


def get_iou_3d(pred, gt):
    n = pred[:, :, 1] - pred[:, :, 0]
    m = gt[:, :, 1] - gt[:, :, 0]

    inter_left = torch.max(pred[:, :, 0], gt[:, :, 0])
    inter_right = torch.min(pred[:, :, 1], gt[:, :, 1])
    inter = (inter_right - inter_left).clamp(min=0)

    union = n + m - inter

    return inter / union


def get_box_loss_3d(score, pred, gt, cfg):
    cost_l1, cost_iou = cfg.COST_L1, cfg.COST_IOU

    l1 = torch.abs(pred - gt)
    l1 = l1[:, :, 0] + l1[:, :, 1]
    iou = get_iou_3d(pred, gt)

    loss = - score + cost_l1 * l1 + cost_iou * iou
    return loss


def get_best_pred(preds, gt, duration, cfg):
    '''
    返回最好的预测结果
    :param preds: torch.size([b, 100, 3])
    :param gt: list.size([b, 2])
    :param duration: list.size([b])
    :param cfg:
    :return: torch.size([b, 3])
    '''
    device = preds.device
    b, n, _ = preds.shape

    gt = torch.tensor(gt, device=device, requires_grad=False)
    gt = repeat(gt, 'b d -> b n d', n=n)

    duration = torch.tensor(duration, device=device, requires_grad=False)
    duration = repeat(duration, 'b -> b n 2', n=n)

    norm_times = preds[:, :, :2] / duration
    norm_score = preds[:, :, 2] / 100
    norm_gt = gt / duration

    box_loss = get_box_loss_3d(norm_score, norm_times, norm_gt, cfg)
    index = torch.argmin(box_loss, dim=1)

    best_pred = torch.stack([preds[i, j] for i, j in enumerate(index)])

    return best_pred


def many_to_one_loss(best_pred, gt, duration, cfg):
    '''
    :param best_pred: torch.size([b, 3])
    :param gt: list.size([b, 2])
    :param duration: list.size([b])
    :param cfg:
    :return: torch.size([1])
    '''
    device = best_pred.device
    b, _ = best_pred.shape

    gt = torch.tensor(gt, device=device, requires_grad=False)

    duration = torch.tensor(duration, device=device, requires_grad=False)
    duration = repeat(duration, 'b -> b 2')

    norm_times = best_pred[:, :2] / duration
    norm_score = best_pred[:, 2] / 100
    norm_gt = gt / duration

    loss = get_loss_2d(norm_score, norm_times, norm_gt, cfg)

    loss_value = loss.sum() / b
    return loss_value
