from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pprint
import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from fvcore.nn import flop_count, flop_count_str, FlopCountAnalysis
import math

import eval

from lib.datasets.dataset import MomentLocalizationDataset
from lib.core.config import cfg, update_config
from lib.core.utils import AverageMeter, create_logger
from lib.utils.comm import synchronize
import lib.models as models
import lib.models.utils.loss as loss


def collate_fn(batch):
    batch_anno_idx = [b['anno_idx'] for b in batch]
    batch_video_id = [b['video_id'] for b in batch]
    batch_duration = [b['duration'] for b in batch]
    batch_description = [b['description'] for b in batch]

    batch_v_feat = [b['v_feat'] for b in batch]
    batch_v_mask = [b['v_mask'] for b in batch]
    batch_t_feat = [b['t_feat'] for b in batch]
    batch_t_mask = [b['t_mask'] for b in batch]

    batch_gt = [b['gt'] for b in batch]

    batch_data = {
        'batch_anno_idx': batch_anno_idx,
        'batch_video_id': batch_video_id,
        'batch_duration': batch_duration,
        'batch_description': batch_description,

        'batch_v_feat': nn.utils.rnn.pad_sequence(batch_v_feat, batch_first=True),
        'batch_v_mask': batch_v_mask,
        'batch_t_feat': nn.utils.rnn.pad_sequence(batch_t_feat, batch_first=True),
        'batch_t_mask': batch_t_mask,

        'batch_gt': batch_gt
    }
    return batch_data


def network(sample, model, optimizer=None):
    duration = sample['batch_duration']
    gt = sample['batch_gt']

    v_input = sample['batch_v_feat']
    v_mask = sample['batch_v_mask']

    t_input = sample['batch_t_feat']
    t_mask = sample['batch_t_mask']

    if cfg.DATASET.SAMPLE_FEATURE:
        preds = model(v_input, t_input)
    else:
        preds = model(v_input, t_input, v_mask, t_mask)

    loss_dict = getattr(loss, cfg.LOSS.NAME)(preds, gt, duration)
    loss_value = loss_dict['loss']

    if model.training:
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
        # torch.cuda.synchronize()

    return preds, loss_dict


def train_epoch(train_loader, model, optimizer, verbose=False):
    model.train()
    annotations = train_loader.dataset.annotations

    if True:
        loss_meter = AverageMeter()
        score_meter = AverageMeter()
        l1_loss_meter = AverageMeter()
        iou_loss_meter = AverageMeter()

    preds_dict = {}
    if verbose:
        pbar = tqdm(total=len(train_loader), dynamic_ncols=True)

    for cur_iter, sample in enumerate(train_loader):
        preds, loss_dict = network(sample, model, optimizer)
        if True:
            loss_meter.update(loss_dict['loss'].item(), 1)
            score_meter.update(loss_dict['score'].item(), 1)
            l1_loss_meter.update(loss_dict['l1_loss'].item(), 1)
            iou_loss_meter.update(loss_dict['iou_loss'].item(), 1)
        preds_dict.update(
            {
                idx: pred
                for idx, pred in zip(sample['batch_anno_idx'], preds.detach().cpu().numpy().tolist())
            }
        )

        if cur_iter % 30 == 0:
            message = 'lr: {:.7f}; '.format(optimizer.param_groups[0]['lr'])
            message += 'avg_loss: {:.2f}; '.format(loss_meter.avg)
            message += 'score: {:.2f}; '.format(score_meter.avg)
            message += 'l1_loss: {:.2f}; '.format(l1_loss_meter.avg)
            message += 'iou_loss: {:.2f}'.format(iou_loss_meter.avg)
            print(message)

            sorted_annotations = [annotations[key] for key in sorted(preds_dict.keys())]
            sorted_preds = [preds_dict[key] for key in sorted(preds_dict.keys())]
            result = eval.evaluate(sorted_preds, sorted_annotations)
            table_message = eval.display_results(result, 'performance on training set')
            print(table_message)

            print('preds[0, :3, :]: {}'.format(preds[0, :3, :]))
            print('preds[6, :3, :]: {}'.format(preds[6, :3, :]))

        if args.debug:
            return

        if verbose:
            pbar.update(1)

    if verbose:
        pbar.close()

    sorted_annotations = [annotations[key] for key in sorted(preds_dict.keys())]
    sorted_preds = [preds_dict[key] for key in sorted(preds_dict.keys())]
    result = eval.evaluate(sorted_preds, sorted_annotations)

    return loss_meter.avg, result


@torch.no_grad()
def test_epoch(test_loader, model, verbose=False, save_results=False):
    model.eval()

    if True:
        loss_meter = AverageMeter()
        score_meter = AverageMeter()
        l1_loss_meter = AverageMeter()
        iou_loss_meter = AverageMeter()

    preds_dict = {}
    saved_dict = {}

    if verbose:
        pbar = tqdm(total=len(test_loader), dynamic_ncols=True)

    for cur_iter, sample in enumerate(test_loader):
        preds, loss_dict = network(sample, model)
        if True:
            loss_meter.update(loss_dict['loss'].item(), 1)
            score_meter.update(loss_dict['score'].item(), 1)
            l1_loss_meter.update(loss_dict['l1_loss'].item(), 1)
            iou_loss_meter.update(loss_dict['iou_loss'].item(), 1)
        preds_dict.update(
            {
                idx: timestamp
                for idx, timestamp in zip(sample['batch_anno_idx'], preds.detach().cpu().numpy().tolist())
            }
        )
        saved_dict.update({idx: {'vid': vid, 'preds': preds, 'description': description}
                           for idx, vid, description, preds in zip(sample['batch_anno_idx'],
                                                                       sample['batch_video_id'],
                                                                       sample['batch_description'],
                                                                       preds.detach().cpu().numpy().tolist())})
        if verbose:
            pbar.update(1)

    if verbose:
        pbar.close()

    annotations = test_loader.dataset.annotations
    sorted_annotations = [annotations[key] for key in sorted(preds_dict.keys())]
    sorted_preds = [preds_dict[key] for key in sorted(preds_dict.keys())]

    saved_dict = [saved_dict[key] for key in sorted(saved_dict.keys())]
    if save_results:
        if not os.path.exists('results/{}'.format(cfg.DATASET.NAME)):
            os.makedirs('results/{}'.format(cfg.DATASET.NAME))
        torch.save(saved_dict,
                   'results/{}/{}-{}.pkl'.format(cfg.DATASET.NAME, os.path.basename(args.cfg).split('.yaml')[1],
                                                 test_loader.dataset.split))

    result = eval.evaluate(sorted_preds, sorted_annotations)

    return loss_meter.avg, result


def train(cfg, verbose):
    logger, final_output_dir, tensorboard_dir = create_logger(cfg, args.cfg, cfg.TAG)
    logger.info('\n' + pprint.pformat(args))
    logger.info('\n' + pprint.pformat(cfg))

    # cudnn related setting
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC

    init_epoch = 0
    model = getattr(models, cfg.MODEL.NAME)()
    if cfg.MODEL.CHECKPOINT and cfg.TRAIN.CONTINUE:
        init_epoch = int(os.path.basename(cfg.MODEL.CHECKPOINT)[5:9]) + 1
        model_checkpoint = torch.load(cfg.MODEL.CHECKPOINT)
        model.load_state_dict(model_checkpoint)
        print(f"loading checkpoint: {cfg.MODEL.CHECKPOINT}")
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.nn.DataParallel(model)      # ???????????????DDP
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
            find_unused_parameters=True, broadcast_buffers=False,
        )
    model = model.to(device)

    # print FLOPs and Parameters
    if True:
        num_v_tokens, num_t_tokens = cfg.DATASET.MAX_VIS_TOKENS, cfg.DATASET.MAX_TXT_TOKENS
        v_feat, v_mask = torch.zeros([1, num_v_tokens, 1024], device=device), torch.zeros([1, num_v_tokens, 1024], device=device)
        t_feat, t_mask = torch.zeros([1, num_t_tokens, 300], device=device), torch.zeros([1, num_t_tokens, 300], device=device)
        if cfg.DATASET.SAMPLE_FEATURE:
            count_dict, *_ = flop_count(model, (v_feat, t_feat))
        else:
            count_dict, *_ = flop_count(model, (v_feat, t_feat, v_mask, t_mask))
        count = sum(count_dict.values())
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info(flop_count_str(FlopCountAnalysis(model, (v_feat, t_feat))))
        logger.info('{:<30}  {:.1f} GFlops'.format('number of FLOPs: ', count))
        logger.info('{:<30}  {:.1f} MB'.format('number of params: ', n_parameters / 1000 ** 2))

    # TODO lr_schedule: AdamW with Momentum
    if cfg.OPTIM.NAME == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.OPTIM.PARAMS.LR)
    elif cfg.OPTIM.NAME == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=cfg.OPTIM.PARAMS.LR)
    elif cfg.OPTIM.NAME == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=cfg.OPTIM.PARAMS.LR)
    elif cfg.OPTIM.NAME == 'AdamW':
        optimizer = optim.AdamW(model.parameters(),
                                lr=cfg.OPTIM.PARAMS.LR,
                                weight_decay=cfg.OPTIM.PARAMS.WEIGHT_DECAY)

        t = cfg.LR_SCHEDULER.PARAMS.WARM_UP_EPOCH
        T = cfg.TRAIN.MAX_EPOCH
        lr_lambda = lambda epoch: (0.9 * epoch / t + 0.1) if epoch < t else 0.1 if 0.5 * (
                1 + math.cos(math.pi * (epoch - t) / (T - t))) < 0.1 else 0.5 * (
                1 + math.cos(math.pi * (epoch - t) / (T - t)))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        raise NotImplementedError

    train_dataset = MomentLocalizationDataset(cfg.DATASET, 'train')
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.TRAIN.BATCH_SIZE,
                              shuffle=cfg.TRAIN.SHUFFLE,
                              num_workers=cfg.WORKERS,
                              pin_memory=True,
                              drop_last=True,
                              collate_fn=collate_fn)

    if not cfg.DATASET.NO_VAL:
        val_dataset = MomentLocalizationDataset(cfg.DATASET, 'val')
        val_loader = DataLoader(val_dataset,
                                batch_size=cfg.TEST.BATCH_SIZE,
                                shuffle=False,
                                num_workers=cfg.WORKERS,
                                pin_memory=True,
                                collate_fn=collate_fn)

    test_dataset = MomentLocalizationDataset(cfg.DATASET, 'test')
    test_loader = DataLoader(test_dataset,
                             batch_size=cfg.TEST.BATCH_SIZE,
                             shuffle=False,
                             num_workers=cfg.WORKERS,
                             pin_memory=True,
                             collate_fn=collate_fn)

    # [[score1, score2], [epoch1, epoch2]]
    max_metric = [[[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]]
    writer = SummaryWriter(tensorboard_dir)

    for cur_epoch in range(init_epoch, cfg.TRAIN.MAX_EPOCH):
        train_avg_loss, train_result = train_epoch(train_loader, model, optimizer, verbose)

        loss_message = '\n' + 'epoch: {};'.format(cur_epoch)
        loss_message += ' lr: {};'.format(optimizer.param_groups[0]['lr'])
        loss_message += ' train loss {:.2f}'.format(train_avg_loss)

        table_message = '\n' + eval.display_results(train_result, 'performance on training set')

        if not cfg.DATASET.NO_VAL:
            val_avg_loss, val_result = test_epoch(val_loader, model, verbose)
            loss_message += ' val loss: {:.4f};'.format(val_avg_loss)
            table_message += '\n' + eval.display_results(val_result, 'performance on validation set')
            writer.add_scalar('val_avg_loss', val_avg_loss, cur_epoch)

        # test_result: {'ranks': [[R1@0.1, R5@0.1], [R1@0.3, R5@0.3], [R1@0.5, R5@0.5], [R1@0.7, R5@0.7]], 'mIoU': [mean IoU]}
        test_avg_loss, test_result = test_epoch(test_loader, model, verbose)
        loss_message += ' test loss: {:.4f}'.format(test_avg_loss)
        table_message += '\n' + eval.display_results(test_result, 'performance on testing set')

        message = loss_message + table_message
        logger.info(message)

        # save max metrics and tensorboard
        if True:
            tious = cfg.TEST.TIOU
            recalls = cfg.TEST.RECALL

            for i in range(len(tious)):
                for j in range(len(recalls)):
                    if test_result['ranks'][i, j] > max_metric[i][j][0]:
                        max_metric[i][j][0], max_metric[i][j][1] = test_result['ranks'][i, j], cur_epoch

            test_max_result = eval.display_max_results(max_metric, 'max score and epoch')
            logger.info(test_max_result)

            writer.add_scalar('train_avg_loss', train_avg_loss, cur_epoch)
            writer.add_scalar('test_avg_loss', test_avg_loss, cur_epoch)
            writer.add_scalar('mIoU', test_result['mIoU'], cur_epoch)

            for i in range(len(tious)):
                for j in range(len(recalls)):
                    writer.add_scalar(f'R{recalls[j]}@{tious[i]}', test_result['ranks'][i, j], cur_epoch)

        # save model
        if not args.no_save:
            # test_result['ranks'][0] == [R1@0.1, R5@0.1]
            saved_model_filename = os.path.join(cfg.MODEL_DIR, '{}/{}/epoch{:04d}-{:.4f}-{:.4f}.pkl'.format(
                cfg.DATASET.NAME, os.path.basename(args.cfg).split('.yaml')[0],
                cur_epoch, test_result['ranks'][0, 0], test_result['ranks'][0, 1]))

            # os.path.dirname(path), ??????????????????????????????
            root_folder1 = os.path.dirname(saved_model_filename)
            root_folder2 = os.path.dirname(root_folder1)
            root_folder3 = os.path.dirname(root_folder2)
            if not os.path.exists(root_folder3):
                print('Make directory %s ...' % root_folder3)
                os.mkdir(root_folder3)
            if not os.path.exists(root_folder2):
                print('Make directory %s ...' % root_folder2)
                os.mkdir(root_folder2)
            if not os.path.exists(root_folder1):
                print('Make directory %s ...' % root_folder1)
                os.mkdir(root_folder1)

            torch.save(model.module.state_dict(), saved_model_filename)

        scheduler.step()


def test(cfg, split):
    # cudnn related setting, ignore it
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC

    model = getattr(models, cfg.MODEL.NAME)(cfg.MODEL)

    if os.path.exists(cfg.MODEL.CHECKPOINT):
        model_checkpoint = torch.load(cfg.MODEL.CHECKPOINT)
        model.load_state_dict(model_checkpoint)
    else:
        raise ("checkpoint not exists")

    model = torch.nn.DataParallel(model)
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dataset = MomentLocalizationDataset(cfg.DATASET, split)
    dataloader = DataLoader(dataset,
                            batch_size=cfg.TEST.BATCH_SIZE,
                            shuffle=False,
                            num_workers=cfg.WORKERS,
                            pin_memory=True,
                            collate_fn=collate_fn)
    avg_loss, result = test_epoch(dataloader, model, True, save_results=True)

    print(' val loss {:.4f}'.format(avg_loss))
    print(eval.display_results(result, 'performance on {} set'.format(split)))


def parse_args():
    parser = argparse.ArgumentParser(description='Train localization network')

    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    args, rest = parser.parse_known_args()

    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--seed', help='seed', default=0, type=int)
    parser.add_argument('--gpus', help='gpus', type=str)
    parser.add_argument('--workers', help='num of dataloader workers', type=int)
    parser.add_argument('--dataDir', help='data path', type=str)
    parser.add_argument('--modelDir', help='model path', type=str)
    parser.add_argument('--logDir', help='log path', type=str)
    parser.add_argument('--verbose', default=False, action="store_true", help='print progress bar')
    parser.add_argument('--tag', help='tags shown in log', type=str)
    parser.add_argument('--mode', default='train', help='run test epoch only')
    parser.add_argument('--split', help='test split', type=str)
    parser.add_argument('--no_save', default=True, action="store_true", help='don\'t save checkpoint')
    parser.add_argument('--debug', default=False, type=bool)
    # parser.add_argument('local_rank')
    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers is not None:
        config.WORKERS = args.workers
    if args.dataDir:
        config.DATASET.DATA_DIR = os.path.join(args.dataDir, config.DATASET.DATA_DIR)
    if args.modelDir:
        config.MODEL_DIR = args.modelDir
    if args.logDir:
        config.LOG_DIR = args.logDir
    if args.tag:
        config.TAG = args.tag
    if args.debug:
        config.debug = args.debug


if __name__ == '__main__':
    args = parse_args()
    reset_config(cfg, args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.autograd.set_detect_anomaly(True)
    torch.backends.cudnn.deterministic = True

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = False
    # args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    if args.mode == 'train':
        train(cfg, args.verbose)
    else:
        test(cfg, args.split)
