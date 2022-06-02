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

#%% Network
class ExampleNetwork(nn.Module):
    def __init__(self, model_params):
        super(ExampleNetwork, self).__init__()

        self.enc = nn.Sequential(nn.Linear(16,64),
                                 nn.ELU())
        
        self.reg = nn.Linear(64,1)
        self.to(device)

    def forward(self, x):
        y = self.enc(x)
        y = self.reg(y)
        return y

