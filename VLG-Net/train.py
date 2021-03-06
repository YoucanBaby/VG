import argparse
import os

import torch
import random
from torch import optim
from torch import multiprocessing
from pathlib import Path
import numpy as np

multiprocessing.set_sharing_strategy('file_system')

from lib.config import get_cfg_defaults, set_hps_cfg
from lib.data import make_data_loader
from lib.engine.inference import inference
from lib.engine.trainer import do_train
from lib.modeling import build_model
from lib.utils.checkpoint import VLGCheckpointer
from lib.utils.comm import synchronize, get_rank, cleanup
from lib.utils.imports import import_file
from lib.utils.logger import setup_logger
from lib.utils.miscellaneous import mkdir, save_config

from mock import Mock
import logging
from pytorch_model_summary import summary
from tensorboardX import SummaryWriter

from prettytable import PrettyTable


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    train_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        train_params += param
    print(table)
    print('{:<30}  {:.1f} MB'.format('number of params: ', train_params / 1000 ** 2))


def load_pretrained_graph_weights(model, cfg, logger):
    # check dimension to load correct model:
    if cfg.MODEL.VLG.FEATPOOL.HIDDEN_SIZE == 256:
        path = './datasets/gcnext_warmup/gtad_best_256.pth.tar'
    elif cfg.MODEL.VLG.FEATPOOL.HIDDEN_SIZE == 512:
        path = './datasets/gcnext_warmup/gtad_best_512.pth.tar'

    logger.info('Load pretrained model from {}'.format(path))
    pretrained_dict = torch.load(path)['state_dict']
    pretrained_keep = dict()  # manually copy the weight
    if '256' in path:
        layer_name = 'module.x_1d_b'
        for i in range(cfg.MODEL.VLG.FEATPOOL.NUM_AGGREGATOR_LAYERS):
            pretrained_keep[f'context_aggregator.{i}.tconvs.0.weight'] = pretrained_dict[
                'module.x_1d_b.2.tconvs.0.weight']
            pretrained_keep[f'context_aggregator.{i}.tconvs.0.bias'] = pretrained_dict['module.x_1d_b.2.tconvs.0.bias']
            pretrained_keep[f'context_aggregator.{i}.tconvs.2.weight'] = pretrained_dict[
                'module.x_1d_b.2.tconvs.2.weight']
            pretrained_keep[f'context_aggregator.{i}.tconvs.2.bias'] = pretrained_dict['module.x_1d_b.2.tconvs.2.bias']
            pretrained_keep[f'context_aggregator.{i}.tconvs.4.weight'] = pretrained_dict[
                'module.x_1d_b.2.tconvs.4.weight']
            pretrained_keep[f'context_aggregator.{i}.tconvs.4.bias'] = pretrained_dict['module.x_1d_b.2.tconvs.4.bias']
            pretrained_keep[f'context_aggregator.{i}.sconvs.0.weight'] = pretrained_dict[
                'module.x_1d_b.2.fconvs.0.weight']
            pretrained_keep[f'context_aggregator.{i}.sconvs.0.bias'] = pretrained_dict['module.x_1d_b.2.fconvs.0.bias']
            pretrained_keep[f'context_aggregator.{i}.sconvs.2.weight'] = pretrained_dict[
                'module.x_1d_b.2.fconvs.2.weight']
            pretrained_keep[f'context_aggregator.{i}.sconvs.2.bias'] = pretrained_dict['module.x_1d_b.2.fconvs.2.bias']
            pretrained_keep[f'context_aggregator.{i}.sconvs.4.weight'] = pretrained_dict[
                'module.x_1d_b.2.fconvs.4.weight']
            pretrained_keep[f'context_aggregator.{i}.sconvs.4.bias'] = pretrained_dict['module.x_1d_b.2.fconvs.4.bias']
    elif '512' in path:
        layer_name = 'module.backbone1'
        for i in range(cfg.MODEL.VLG.FEATPOOL.NUM_AGGREGATOR_LAYERS):
            pretrained_keep[f'context_aggregator.{i}.tconvs.0.weight'] = pretrained_dict[
                'module.backbone1.2.tconvs.0.weight']
            pretrained_keep[f'context_aggregator.{i}.tconvs.0.bias'] = pretrained_dict[
                'module.backbone1.2.tconvs.0.bias']
            pretrained_keep[f'context_aggregator.{i}.tconvs.2.weight'] = pretrained_dict[
                'module.backbone1.2.tconvs.2.weight']
            pretrained_keep[f'context_aggregator.{i}.tconvs.2.bias'] = pretrained_dict[
                'module.backbone1.2.tconvs.2.bias']
            pretrained_keep[f'context_aggregator.{i}.tconvs.4.weight'] = pretrained_dict[
                'module.backbone1.2.tconvs.4.weight']
            pretrained_keep[f'context_aggregator.{i}.tconvs.4.bias'] = pretrained_dict[
                'module.backbone1.2.tconvs.4.bias']
            pretrained_keep[f'context_aggregator.{i}.sconvs.0.weight'] = pretrained_dict[
                'module.backbone1.2.sconvs.0.weight']
            pretrained_keep[f'context_aggregator.{i}.sconvs.0.bias'] = pretrained_dict[
                'module.backbone1.2.sconvs.0.bias']
            pretrained_keep[f'context_aggregator.{i}.sconvs.2.weight'] = pretrained_dict[
                'module.backbone1.2.sconvs.2.weight']
            pretrained_keep[f'context_aggregator.{i}.sconvs.2.bias'] = pretrained_dict[
                'module.backbone1.2.sconvs.2.bias']
            pretrained_keep[f'context_aggregator.{i}.sconvs.4.weight'] = pretrained_dict[
                'module.backbone1.2.sconvs.4.weight']
            pretrained_keep[f'context_aggregator.{i}.sconvs.4.bias'] = pretrained_dict[
                'module.backbone1.2.sconvs.4.bias']
    else:
        raise ValueError('Specify hidden size in feature file name')

    model.load_state_dict(pretrained_keep, strict=False)
    return model


def train(cfg, writer, local_rank, distributed, args):
    model = build_model(cfg)

    logger = logging.getLogger("model.trainer")

    ### GTAD pretraining
    if cfg.MODEL.PRETRAINV:
        model = load_pretrained_graph_weights(model, cfg, logger)

    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    count_parameters(model)

    # TODO ?????????????????????AdamW, ?????????warmup-cosine
    optimizer = optim.Adam(model.parameters(), lr=cfg.SOLVER.LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.SOLVER.LR_STEP_SIZE, gamma=cfg.SOLVER.LR_GAMMA)

    # Deprecated, to be removed.
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True, broadcast_buffers=False,
        )

    save_to_disk = get_rank() == 0
    checkpointer = VLGCheckpointer(
        cfg, model, optimizer, scheduler, cfg.OUTPUT_DIR, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(f='', use_latest=False)
    arguments = {"epoch": 1}
    arguments.update(extra_checkpoint_data)

    data_loader = make_data_loader(cfg, is_train=True, is_distributed=distributed)

    data_loader_val = None
    data_loader_test = None
    test_period = cfg.SOLVER.TEST_PERIOD
    if test_period > 0:
        if len(cfg.DATASETS.VAL) != 0:
            data_loader_val = make_data_loader(cfg, is_train=False, is_distributed=distributed, is_for_period=True)
        else:
            logger.info('Please specify validation dataset in config file for performance evaluation during training')
        data_loader_test = make_data_loader(cfg, is_train=False, is_distributed=distributed)

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    do_train(
        cfg,
        writer,
        model,
        data_loader,
        data_loader_val,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        test_period,
        arguments,
        dataset_name=cfg.DATASETS['TRAIN'][0],
        data_loader_test=data_loader_test[0],
    )

    checkpointer.cleanup_data()
    return model


def test(cfg, model, distributed):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()
    dataset_names = cfg.DATASETS.TEST
    data_loaders_test = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for dataset_name, data_loaders_test in zip(dataset_names, data_loaders_test):
        inference(
            model,
            data_loaders_test,
            dataset_name=dataset_name,
            nms_thresh=cfg.TEST.NMS_THRESH,
            device=cfg.MODEL.DEVICE,
            name=cfg.DATASETS['TEST'][0],
        )
        synchronize()


def main():
    parser = argparse.ArgumentParser(description="VLG")
    parser.add_argument("--config-file", default="configs/activitynet.yml", type=str, metavar="FILE",
                        help="path to config file")
    parser.add_argument("--local_rank", type=int, default=0)

    parser.add_argument('--enable-tb', action='store_true',
                        help="Enable tensorboard logging")
    parser.add_argument('--hps', type=Path, default=Path('non-existent'),
                        help='yml file defining the range of hps to be used in training (randomly sampled)')
    parser.add_argument('--debug', type=bool, default=False)

    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg = set_hps_cfg(cfg, args.hps)
    cfg.TEST.BATCH_SIZE = cfg.SOLVER.BATCH_SIZE
    cfg.freeze()

    # fix seeds for reproducibility
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    if cfg.OUTPUT_DIR:
        mkdir(cfg.OUTPUT_DIR)

    writer = None
    if args.enable_tb:
        try:
            writer = SummaryWriter(f'{cfg.OUTPUT_DIR}/tensorboard')
        except:
            writer = None

    logger = setup_logger("config", cfg.OUTPUT_DIR, get_rank())
    logging.error("?????????????????????, ????????????logger.info???")

    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded config from: {}".format(args.config_file))

    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    save_config(cfg, output_config_path)

    model = train(cfg, writer, args.local_rank, args.distributed, args)

    # ?????????????????????
    if len(cfg.DATASETS.TEST) != 0:
        best_checkpoint = f"{cfg.OUTPUT_DIR}/model_best_epoch.pth"
        if os.path.isfile(best_checkpoint):
            model.load_state_dict(torch.load(best_checkpoint))
        test(cfg, model, args.distributed)
        synchronize()

    if args.distributed:
        cleanup()


if __name__ == "__main__":
    main()
