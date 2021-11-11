#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 13:52:23 2021

@author: aggelen
"""

from TorchWorks import Experiment

model = None
hyper_params = {}
experiment_params = {}

exp0 = Experiment(network=model, hyper_params=hyper_params, experiment_params=experiment_params)

# exp0.train()
# exp0.test()

# exp0.accuracy()
# exp0.rmse()

# exp0.save('path_to_save')