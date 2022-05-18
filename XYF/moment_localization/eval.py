import json
import argparse
import numpy as np
from terminaltables import AsciiTable

from lib.core.config import cfg, update_config
from lib.core.utils import iou


def get_iou(pred, gt):
    '''
    :param pred: list.size([100, 2])
    :param gt: list.size([2])
    :return:
    '''
    MAX = 1e8

    pred, gt = np.array(pred), np.array(gt)
    a = np.clip(pred[:, 1] - pred[:, 0], 0, MAX)
    b = np.clip(gt[1] - gt[0], 0, MAX)

    inter_left = np.clip(pred[:, 0] - gt[0], 0, MAX)
    inter_right = np.clip(pred[:, 1] - gt[1], 0, MAX)

    inter = np.clip(inter_right - inter_left, 0, MAX)

    union = a + b - inter

    return inter / union


def evaluate(preds, annotations):
    '''
    :param preds: list.size([b, 100, 2])
    :param annotations: list.size([b, 2])
    :return:
    '''
    tious = cfg.TEST.TIOU
    recalls = cfg.TEST.RECALL

    eval_result = [[[] for _ in recalls] for _ in tious]
    average_iou = []

    # TODO eval这一部分没有完全懂
    for pred, data in zip(preds, annotations):
        gt = data['times']
        iou = get_iou(pred, gt)

        iou = np.sort(iou, axis=-1)
        average_iou.append(np.mean(iou, axis=-1))

        for i, t in enumerate(tious):
            for j, r in enumerate(recalls):
                eval_result[i][j].append((iou > t)[:r].any())

    eval_result = np.array(eval_result).mean(axis=-1)
    miou = np.mean(average_iou)

    # {'ranks': [[R1@0.1, R5@0.1], [R1@0.3, R5@0.3], [R1@0.5, R5@0.5], [R1@0.7, R5@0.7]], 'mIoU': [mean IoU]}
    result = {'ranks': eval_result, 'mIoU': miou}
    return result


def display_results(result, title=None):
    '''返回结果表格'''
    tious = cfg.TEST.TIOU
    recalls = cfg.TEST.RECALL

    display_data = [['R{}@{}'.format(i, j) for i in recalls for j in tious] + ['mIoU']]
    ranks = result['ranks'] * 100
    miou = result['mIoU'] * 100
    display_data.append(['{:.02f}'.format(ranks[j][i]) for i in range(len(recalls)) for j in range(len(tious))]
                        + ['{:.02f}'.format(miou)])
    table = AsciiTable(display_data, title)
    for i in range(len(tious) * len(recalls)):
        table.justify_columns[i] = 'center'
    return table.table


def display_max_results(max_result, title=None):
    '''返回最大结果表格'''
    tious = cfg.TEST.TIOU
    recalls = cfg.TEST.RECALL

    display_data = [['R{}@{}'.format(i, j) for i in recalls for j in tious]]
    display_data.append(
        ['{:.02f}'.format(max_result[i][j][0] * 100) for j in range(len(recalls)) for i in range(len(tious))]
    )
    display_data.append(
        ['{}'.format(max_result[i][j][1]) for j in range(len(recalls)) for i in range(len(tious))]
    )
    table = AsciiTable(display_data, title)
    for i in range(len(tious) * len(recalls)):
        table.justify_columns[i] = 'center'
    return table.table


def parse_args():
    parser = argparse.ArgumentParser(description='Train localization network')

    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    args, rest = parser.parse_known_args()

    # update config
    update_config(args.cfg)

    parser.add_argument('--verbose', default=False, action="store_true", help='print progress bar')
    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.verbose:
        config.VERBOSE = args.verbose


if __name__ == '__main__':
    args = parse_args()
    reset_config(cfg, args)
    train_data = json.load(open('/data/home2/hacker01/Data/DiDeMo/train_data.json', 'r'))
    val_data = json.load(open('/data/home2/hacker01/Data/DiDeMo/val_data.json', 'r'))

    moment_frequency_dict = {}
    for d in train_data:
        times = [t for t in d['times']]
        for time in times:
            time = tuple(time)
            if time not in moment_frequency_dict.keys():
                moment_frequency_dict[time] = 0
            moment_frequency_dict[time] += 1

    prior = sorted(moment_frequency_dict, key=moment_frequency_dict.get, reverse=True)
    prior = [list(item) for item in prior]
    prediction = [prior for d in val_data]

    evaluate(prediction, val_data)
