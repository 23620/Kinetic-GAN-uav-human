import os
import sys
import numpy as np
import random
import pickle

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from . import tools

# 骨架节点的连接对，用于计算骨骼特征
coco_pairs = [(1, 6), (2, 1), (3, 1), (4, 2), (5, 3), (6, 7), (7, 1), (8, 6), (9, 7), (10, 8), (11, 9),
                (12, 6), (13, 7), (14, 12), (15, 13), (16, 14), (17, 15)]

class Feeder(Dataset):
    """ Feeder for UAV-Human skeleton-based action synthesis
    Arguments:
        data_path: the path to '.npy' data, the shape of real data should be (N, C, T, V, M)
        label_path: the path to label '.npy' data
        p_interval: interval for valid frame cropping
        window_size: temporal window size to which the sequence will be resized
        bone: whether to use bone features
        vel: whether to use velocity features
    """
    def __init__(self,
                 data_path: str,
                 label_path: str,
                 p_interval: list = [0.95],
                 window_size: int = 64,
                 bone: bool = False,
                 vel: bool = False):
        super(Feeder, self).__init__()
        self.data_path = data_path
        self.label_path = label_path
        self.p_interval = p_interval
        self.window_size = window_size
        self.bone = bone
        self.vel = vel

        # 加载数据
        self.load_data()

    def load_data(self):
        # 加载 npy 格式的数据
        self.data = np.load(self.data_path, allow_pickle=True)
        self.label = np.load(self.label_path, allow_pickle=True)
        self.sample_name = ['sample_' + str(i) for i in range(len(self.data))]
        
        # 获取最大最小值以便后续标准化使用
        self.max, self.min = self.data.max(), self.data.min()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        # 获取数据，形状为 (N, C, T, V, M)
        data_numpy = self.data[idx]  # (C, T, V, M)
        label = self.label[idx]

        # 计算有效帧数 (非零帧)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        if valid_frame_num == 0:
            # 如果没有有效帧，返回全零张量
            return np.zeros((3, self.window_size, 17, 1)), label

        # 对有效帧进行裁剪并缩放到指定窗口大小
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)

        # 如果启用骨骼特征，则计算骨骼特征
        if self.bone:
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in coco_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy

        # 如果启用速度特征，则计算关节点速度
        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        # 中心化处理
        data_numpy = data_numpy - np.tile(data_numpy[:, :, 0:1, :], (1, 1, data_numpy.shape[2], 1))

        data_numpy = data_numpy[:,:,:,0]

        # 返回数据
        #print("data_numpy.shape:", data_numpy.shape)
        return data_numpy, label

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)