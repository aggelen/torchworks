#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 16:29:09 2021

@author: aggelen
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class Loss(nn.Module):
    def __init__(self):
        pass
    
class DataSet(Dataset):
    def __init__(self):
        pass
    
data_loader = DataLoader()
    
    
class ExampleNetwork(nn.Module):
    def __init__(self):
        self.conv = nn.Conv1d(1, 1, 1)
        
        self.loss = Loss()
        self.data_loader = data_loader
        
        self.set_optimizer()
    
    def set_optimizer(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, betas=(0.9,0.999))
    
    def forward(self, x):
        return self.conv(x)