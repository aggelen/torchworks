#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 13:51:50 2021

@author: Aykut Görkem Gelen
"""

class Experiment:
    def __init__(self, network, hyper_params, experiment_params):
        self.network = network
        self.hyper_params = hyper_params
        self.experiment_params = experiment_params
        
    