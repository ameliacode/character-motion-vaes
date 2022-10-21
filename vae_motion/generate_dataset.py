import sys, os
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))

import glob, time, copy

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from utils.logging import Logger

class MocapDataset(Dataset):
    def __init__(self, split='train',
                 data_paths=None,
                 split_by='dataset',
                 splits_path=None,
                 train_frac=0.8, val_frac=0.1,
                 sample_num_frames=10,
                 step_frames_in=1,
                 step_frames_out=1,
                 frames_out_step_size=1,
                 data_rot_rep='aa',
                 data_return_config='smpl+joints+contacts',
                 return_global=False,
                 only_global=False,
                 data_noise_std=0.0,
                 deterministic_train=False,
                 custom_split=None
                 ):
        super(MocapDataset, self).__init__()

        self.data_roots = data_paths
        self.splits_path = splits_path
        self.split = split
        self.split_by = split_by

        self.train_frac = train_frac
        self.val_frac = val_frac

        self.sample_num_frames = sample_num_frames
        self.step_frames_in = step_frames_in
        self.step_frames_out = step_frames_out
        self.frames_out_step_size = frames_out_step_size
        self.rot_rep = data_rot_rep
        self.only_global = only_global
        self.return_global = return_global or self.only_global
        self.noise_std = data_noise_std
        self.return_cfg = 0 #?
        self.deterministic_train = deterministic_train
        self.custom_split = custom_split

        Logger.log('This split contains %d sequences (that meet the duration criteria).' % (self.num_seq))
        Logger.log('The dataset contains %d sub-sequences in total.' % (self.data_len))