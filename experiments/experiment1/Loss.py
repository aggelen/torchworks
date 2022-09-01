#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% Loss
class ExampleLoss(nn.Module):
    def __init__(self, loss_params):
        super(ExampleLoss, self).__init__()
        self.criteria = nn.MSELoss()
        self.to(device)

    def forward(self, estimated, target):
        # The forward method takes estimated and target parameters.
        return self.criteria(estimated, target)