#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 16:29:09 2021

@author: aggelen
"""
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

#%% Common
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% Dataset & Dataloader
class ExampleDataset(torch.utils.data.Dataset):
    def __init__(self, params):
        print('>> Preparing Dataset ...')
        self.data_len = params['data_len']

        print('>> Dataset created!')

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):

        item = None
        return item

    def create_data_loader(self, batch_size):
        self.data_loader = torch.utils.data.DataLoader(self,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 # num_workers=8,
                                                 # pin_memory=True,
                                                 )

#%% Network
class ExampleNetwork(nn.Module):
    def __init__(self):
        super(ExampleNetwork, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(1, 64, 3),
                                   nn.BatchNorm2d(64),
                                   nn.PReLU())

    def forward(self, x):
        y = self.conv1(x)

        # The forward method should return a dict.
        return {'y': y}

#%% Loss
class ExampleLoss(nn.Module):
    def __init__(self):
        super(ExampleLoss, self).__init__()

    def forward(self, estimated, target):
        # The forward method takes estimated and target parameters.
        loss = None
        return loss