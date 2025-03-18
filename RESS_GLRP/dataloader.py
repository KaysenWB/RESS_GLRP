import os
import pickle
import random
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt



class Dataset_ship(Dataset):
    def __init__(self, args):
        super(Dataset, self).__init__()
        self.args = args
        self.root = args.data_root
        self.rate = args.data_rate
        self.read_data()

    def read_data(self):
        print('Loading_data')
        f = open(self.root, 'rb')
        self.data_raw = pickle.load(f)
        f.close()
        print('Loading_finish: {} batch'.format(int(len(self.data_raw))))
        self.border1 = int(len(self.data_raw) * self.rate[0] / 10)
        self.border2 = int(len(self.data_raw) * (self.rate[0] + self.rate[1])/ 10)

    def pointer(self, flag):
        if flag == 'train':
            self.data = self.data_raw[:self.border1]
        elif flag == 'val':
            self.data = self.data_raw[self.border1: self.border2]
        elif flag == 'test':
            self.data = self.data_raw[self.border2:]
        elif flag == 'pred':
            self.data = [self.data_raw[-2]]

    def get_length(self, flag):
        if flag == 'train':
            return self.border1
        elif flag == 'test' or flag == 'val':
            return self.border2 - self.border1


    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def batch_operation(self, batch_data, ifpred=False):

        batch, Adj = batch_data # B2
        batch = batch[:self.args.seq_length, :, :]
        real_tra = batch.copy()
        Adj =Adj[:self.args.seq_length, :, :]

        # zero shifting
        s = batch[self.args.obs_length - 1]  # observed point
        shift_value = np.repeat(s[np.newaxis,:], self.args.seq_length, 0)
        shift = batch - shift_value # the meaning of 'batch - shift_value' is change coordinate origin to the observed point,  change coordinate.

        # standard
        mean = batch.mean(axis=(0, 1), keepdims=True)
        std = batch.std(axis=(0, 1), keepdims=True)
        batch = (batch-mean)/std


        batch_data = (batch, shift, Adj)

        if ifpred:
            return batch_data, real_tra
        else:
            return batch_data