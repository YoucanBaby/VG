from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import yaml
from easydict import EasyDict as edict

cfg = edict()

cfg.WORKERS = 16
cfg.LOG_DIR = ''
cfg.MODEL_DIR = ''
cfg.RESULT_DIR = ''

# CUDNN related params
cfg.CUDNN = edict()
cfg.CUDNN.BENCHMARK = True
cfg.CUDNN.DETERMINISTIC = False
cfg.CUDNN.ENABLED = True

# DATASET related params
cfg.DATASET = edict()
cfg.DATASET.DATA_DIR = ''
cfg.DATASET.NAME = ''
cfg.DATASET.VIS_INPUT_TYPE = ''
cfg.DATASET.TXT_INPUT_TYPE = ''
cfg.DATASET.OUTPUT_TYPE = ''
cfg.DATASET.ALIGNMENT = ''
cfg.DATASET.NO_VAL = False
cfg.DATASET.NO_TEST = False
cfg.DATASET.TIME_UNIT = None
cfg.DATASET.NUM_FRAMES = 256
cfg.DATASET.SPLIT = ''
cfg.DATASET.NORMALIZE = True
cfg.DATASET.FPS = 25
cfg.DATASET.SAMPLE_FEATURE = True
cfg.DATASET.MAX_VIS_TOKENS = 0
cfg.DATASET.MAX_TXT_TOKENS = 0

# grounding model related params
cfg.MODEL = edict()
cfg.MODEL.NAME = ''
cfg.MODEL.PARAMS = None
cfg.MODEL.CHECKPOINT = ''  # The checkpoint for the best performance
cfg.MODEL.VISUAL_ENCODER = edict()
cfg.MODEL.VISUAL_ENCODER.NAME = ''
cfg.MODEL.VISUAL_ENCODER.PARAMS = None
cfg.MODEL.TEXT_ENCODER = edict()
cfg.MODEL.TEXT_ENCODER.NAME = ''
cfg.MODEL.TEXT_ENCODER.PARAMS = None
cfg.MODEL.TEXT_ENCODER.RNN = None
cfg.MODEL.DECODER = edict()
cfg.MODEL.DECODER.NAME = ''
cfg.MODEL.DECODER.PARAMS = None
cfg.MODEL.PREDICTION = edict()
cfg.MODEL.PREDICTION.NAME = ''
cfg.MODEL.PREDICTION.PARAMS = None

# OPTIM
cfg.OPTIM = edict()
cfg.OPTIM.NAME = ''
cfg.OPTIM.PARAMS = edict()

# LR_SCHEDULER
cfg.LR_SCHEDULER = edict()
cfg.LR_SCHEDULER.NAME = ''
cfg.LR_SCHEDULER.PARAMS = edict()

# train
cfg.TRAIN = edict()
cfg.TRAIN.MAX_EPOCH = 100
cfg.TRAIN.BATCH_SIZE = 64
cfg.TRAIN.SHUFFLE = True
cfg.TRAIN.CONTINUE = False

cfg.LOSS = edict()
cfg.LOSS.NAME = 'many_to_one_loss'
cfg.LOSS.PARAMS = None

# test
cfg.TEST = edict()
cfg.TEST.RECALL = []
cfg.TEST.TIOU = []
cfg.TEST.BATCH_SIZE = 1
cfg.TEST.NMS_THRESH = 0.4
cfg.TEST.TOP_K = 10
cfg.TEST.EVAL_TRAIN = False
cfg.TEST.SCORE_SORT = True


def _update_dict(cfg, value):
    for k, v in value.items():
        if k in cfg:
            if 'PARAMS' in k:
                cfg[k] = v
            elif isinstance(v, dict):
                _update_dict(cfg[k], v)
            else:
                cfg[k] = v
        else:
            raise ValueError("{} not exist in config.py".format(k))


def update_config(config_file):
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
        for k, v in exp_config.items():
            if k in cfg:
                if isinstance(v, dict):
                    _update_dict(cfg[k], v)
                else:
                    cfg[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))
