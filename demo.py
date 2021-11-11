#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 13:52:23 2021

@author: aggelen
"""

from TorchWorks import Experiment
from Networks.ExampleNetwork import ExampleNetwork

model = ExampleNetwork
hyper_params = {}
experiment_params = {'no_epoch': 10}

exp0 = Experiment(network=model, hyper_params=hyper_params, experiment_params=experiment_params)

exp0.train()

# exp0.test()

# exp0.accuracy()
# exp0.rmse()

# exp0.save('path_to_save')