#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 13:52:23 2021

@author: aggelen
"""

from torchworks import Experiment

exp1 = Experiment(exp_path='experiments/experiment1')
exp1.train()
exp1.plot_loss_hist()
