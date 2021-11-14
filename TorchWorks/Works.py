#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 13:51:50 2021

@author: Aykut Görkem Gelen
@author: Eyyüp Yıldız

"""
import torch
from tqdm import tqdm

class Experiment:
    def __init__(self, network, experiment_params):
        self.network = network
        
        self.no_epoch = experiment_params['no_epoch']

        #Optimizer
        if experiment_params['optimizer'] == 'default':
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-4, betas=(0.9,0.999))
        else:
            self.optimizer = experiment_params['optimizer']

        self.loss = experiment_params['loss']
        self.data_loader = experiment_params['data_loader']

    def train(self):
        self.network.train()            #FIXME! is this really needed?
        
        for k in range(self.no_epoch):
            running_loss = 0.0
            
            for batch_id, batch_data in tqdm(enumerate(self.data_loader)):
                
                forward_data, target_data = batch_data
                
                self.optimizer.zero_grad()
                
                #forward + backward + optimize
                forward_output = self.network(forward_data)
                
                loss = self.loss(forward_output, target_data)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                
            # print statistics
            print('RunningLoss: {} @ Epoch #{}'.format(running_loss / self.data_loder.__len__(), k+1))

        print('Train Completed!')

    def test(self):
        self.network.eval()
        pass

    def validation(self):
        pass