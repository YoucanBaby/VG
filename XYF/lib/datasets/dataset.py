""" Dataset loader for the Charades-STA dataset """
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
import torchtext
import torchvision
import os
import json
import csv
import numpy as np
import h5py
import random
from nltk.tokenize import word_tokenize
# random.seed(0)
import nltk

from lib.core.utils import iou, ioa
from lib.datasets.transforms import feature_temporal_sampling

nltk.download('punkt')


class DatasetBase(data.Dataset):
    def __init__(self, cfg, split):
        super(DatasetBase, self).__init__()
        self.cfg = cfg
        self.split = split

        self.annotations = None
        getattr(self, "get_{}_annotations".format(self.cfg.NAME))()

    def __len__(self):
        return len(self.annotations)

    def get_video_frames(self, video_id):
        if self.cfg.NAME == "charades":
            video_path = os.path.join(self.cfg.DATA_DIR, 'Charades_v1_480', video_id + '.mp4')
        else:
            raise NotImplementedError
        pts, fps = torchvision.io.read_video_timestamps(video_path, pts_unit='sec')
        vframes = torchvision.io.read_video(video_path, start_pts=pts[0], end_pts=pts[-1], pts_unit='sec')[0]
        return vframes

    def get_video_features(self, video_id):
        file_path = os.path.join(self.cfg.DATA_DIR, '{}.hdf5'.format(self.cfg.VIS_INPUT_TYPE))
        with h5py.File(file_path, 'r') as hdf5_file:
            features = torch.from_numpy(hdf5_file[video_id][:]).float()
        if self.cfg.NORMALIZE:
            features = F.normalize(features, dim=1)
        mask = torch.ones(features.shape[0], 1)
        return features, mask

    def __getitem__(self, index):
        return self.get_item(index)

    def get_item(self, index):
        raise NotImplementedError


class MomentLocalizationDataset(DatasetBase):
    def __init__(self, cfg, split):
        super(MomentLocalizationDataset, self).__init__(cfg, split)
        self.annotations = sorted(self.annotations, key=lambda anno: anno['duration'], reverse=True)

        # 通过vocab.stoi得到一个字典，返回词表中每个词的索引
        # 通过vocab.stoi['<unk>']返回得到词表中对应词的索引
        # 为<unk>在vocab.vectors中增加一个token
        vocab = torchtext.vocab.pretrained_aliases["glove.6B.300d"](
            cache=os.path.join(self.cfg.DATA_DIR, '.vector_cache'))
        vocab.itos.extend(['<unk>'])
        vocab.stoi['<unk>'] = vocab.vectors.shape[0]
        vocab.vectors = torch.cat([vocab.vectors, torch.zeros(1, vocab.dim)], dim=0)

        word_embedding = nn.Embedding.from_pretrained(vocab.vectors)

        self.vocab = vocab
        self.word_embedding = word_embedding

    def get_sentence_features(self, description):
        if self.cfg.TXT_INPUT_TYPE == 'glove':
            word_idxs = torch.tensor([self.vocab.stoi.get(w.lower(), 400000) for w in description.split()],
                                     dtype=torch.long)
            word_vectors = self.word_embedding(word_idxs)
        else:
            raise NotImplementedError
        return word_vectors, torch.ones(word_vectors.shape[0], 1)

    def get_activitynet_annotations(self):
        with open(os.path.join(self.cfg.DATA_DIR, '{}.json'.format(self.split)), 'r') as f:
            annotations = json.load(f)
        anno_pairs = []
        missing_videos = []  # 'v_0dkIbKXXFzI'
        for vid, video_anno in annotations.items():
            if vid in missing_videos:
                continue
            duration = video_anno['duration']
            for timestamp, sentence in zip(video_anno['timestamps'], video_anno['sentences']):
                if timestamp[0] < timestamp[1]:
                    anno_pairs.append(
                        {
                            'video': vid,
                            'duration': duration,
                            'times': [max(timestamp[0], 0), min(timestamp[1], duration)],
                            'description': ' '.join(word_tokenize(sentence)),
                        }
                    )
        self.annotations = anno_pairs

    def get_charades_annotations(self):
        durations = {}
        with open(os.path.join(self.cfg.DATA_DIR, 'Charades_v1_{}.csv'.format(self.split))) as f:
            reader = csv.DictReader(f)
            for row in reader:
                durations[row['id']] = float(row['length'])

        anno_file = open(os.path.join(self.cfg.DATA_DIR, "charades_sta_{}.txt".format(self.split)), 'r')
        annotations = []
        for line in anno_file:
            anno, sent = line.split("##")
            sent = sent.split('.\n')[0]
            vid, s_time, e_time = anno.split(" ")
            duration = durations[vid]
            s_time = float(s_time)
            e_time = min(float(e_time), duration)
            if s_time < e_time:
                annotations.append(
                    {'video': vid, 'times': [s_time, e_time], 'description': sent,
                     'duration': duration})
        anno_file.close()
        self.annotations = annotations

    def get_tacos_annotations(self):
        with open(os.path.join(self.cfg.DATA_DIR, '{}.json'.format(self.split)), 'r') as f:
            annotations = json.load(f)
        anno_pairs = []
        # [{'video': 's26-d26', 'duration': 1402.17, 'times': [6.36, 19.08], 'description': 'He took out pot'}, ...]
        for vid, video_anno in annotations.items():
            duration = video_anno['num_frames'] / video_anno['fps']
            for timestamp, sentence in zip(video_anno['timestamps'], video_anno['sentences']):
                if timestamp[0] < timestamp[1]:
                    anno_pairs.append(
                        {
                            'video': os.path.splitext(vid)[0],
                            'duration': duration,
                            'times': [max(timestamp[0] / video_anno['fps'], 0),
                                      min(timestamp[1] / video_anno['fps'], duration)],
                            'description': ' '.join(word_tokenize(sentence)),
                        }
                    )
        self.annotations = anno_pairs

    def sample_item(self, x, max_tokens):
        x_h, _ = x.shape
        if x_h <= max_tokens:
            x = F.pad(x, [0, 0, 0, max_tokens - x_h])
        else:
            random_list = random.sample(range(x_h), max_tokens)
            random_list.sort()
            x = torch.stack([x[i] for i in random_list])
        return x

    def get_item(self, index):
        # index = 752#2548#3951#837#3951
        video_id = self.annotations[index]['video']
        duration = self.annotations[index]['duration']
        description = self.annotations[index]['description']
        v_feat, v_mask = self.get_video_features(video_id)
        t_feat, t_mask = self.get_sentence_features(description)
        gt = self.annotations[index]['times']

        if self.cfg.SAMPLE_FEATURE:
            v_feat = self.sample_item(v_feat, self.cfg.MAX_VIS_TOKENS)       # [384, 4096]
            t_feat = self.sample_item(t_feat, self.cfg.MAX_TXT_TOKENS)       # [36, 300]
            v_mask, t_mask = None, None
        else:
            v_h, t_h = v_feat.shape[0], t_feat.shape[0]
            # v_feat, vis_mask填充为[384, 4096]
            v_feat = F.pad(v_feat, [0, 0, 0, self.cfg.MAX_VIS_TOKENS - v_h])
            t_feat = F.pad(t_feat, [0, 0, 0, self.cfg.MAX_TXT_TOKENS - t_h])
            # t_feat, txt_mask填充为[36, 300]
            v_mask = F.pad(v_mask, [0, 0, 0, self.cfg.MAX_VIS_TOKENS - v_h], value=0)
            t_mask = F.pad(t_mask, [0, 0, 0, self.cfg.MAX_TXT_TOKENS - t_h], value=0)

        item = {
            'anno_idx': index,
            'video_id': video_id,
            'duration': duration,
            'description': description,

            'v_feat': v_feat,
            'v_mask': v_mask,

            't_feat': t_feat,
            't_mask': t_mask,

            'gt': gt
        }
        return item


if __name__ == '__main__':
    file_path = os.path.join('data/TACoS', '{}.hdf5'.format('vgg_fc7'))

    # 查看视频最大token, 查看视频所有的token
    if False:
        max_tokens = 0
        token_list = []
        with open(os.path.join('data/TACoS', 'train.json')) as json_file:
            annotation = json.load(json_file)
            for video_id in annotation.keys():
                video_id = video_id.split('.')[0]
                with h5py.File(file_path, 'r') as hdf5_file:
                    features = torch.from_numpy(hdf5_file[video_id][:]).float()
                    max_tokens = max(max_tokens, features.shape[0])
                    token_list.append(features.shape[0])
        token_list.sort()
        print(max_tokens)
        print(features.shape)
        print(token_list, token_list[len(token_list) // 2])

    # 查看text的最大token, 查看text所有的token
    if False:
        vocab = torchtext.vocab.pretrained_aliases["glove.6B.300d"](
            cache=os.path.join('data/TACoS', '.vector_cache'))
        vocab.itos.extend(['<unk>'])
        vocab.stoi['<unk>'] = vocab.vectors.shape[0]
        vocab.vectors = torch.cat([vocab.vectors, torch.zeros(1, vocab.dim)], dim=0)
        word_embedding = nn.Embedding.from_pretrained(vocab.vectors)

        max_tokens = 0
        token_list = []
        with open(os.path.join('data/TACoS', 'train.json')) as json_file:
            annotation = json.load(json_file)
            for video_id in annotation.keys():
                for description in annotation[video_id]['sentences']:
                    word_idxs = torch.tensor([vocab.stoi.get(w.lower(), 400000) for w in description.split()],
                                             dtype=torch.long)
                    word_vectors = word_embedding(word_idxs)
                    max_tokens = max(max_tokens, word_vectors.shape[0])
                    token_list.append(word_vectors.shape[0])
            print(max_tokens)
            token_list.sort()
            print(token_list, token_list[len(token_list) // 2])
