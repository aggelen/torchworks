#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% Data
class ExampleDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_params):
        self.data_len = dataset_params['data_len']
        self.vector_size = dataset_params['vector_size']
        self.fw_data = torch.rand(self.data_len, self.vector_size)
        self.target_data = (torch.rand(self.data_len, 1) > 0.5).type(torch.FloatTensor)
        
    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        return {'forward_data': self.fw_data[idx].flatten().to(device), 'target_data': self.target_data[idx].to(device)}

    def data_loader(self, batch_size):
        return torch.utils.data.DataLoader(self,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           )
