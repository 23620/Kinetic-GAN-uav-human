# sys
import os
import sys
import numpy as np
import random
import pickle

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

# visualization
import time


class Feeder(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
    """

    def __init__(self,
                 data_path,
                 label_path,
                 classes,
                 mmap=True):
        self.data_path = data_path
        self.label_path = label_path
        self.classes = np.array(classes)
        self.load_data(mmap)

    def load_data(self, mmap):
        # data: N C V T M

        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        self.N, self.C, self.T, self.V, self.M = self.data.shape
        self.annotations = self.get_annotations()

    def get_annotations(self):
        annotations = []
        for l in self.label:
            labels = 1*(self.classes == l)
            annotations.append(labels)
        return annotations


    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index][:,:,:,0])
        label = self.annotations[index]
        label = torch.FloatTensor(label)


        return data_numpy, label