#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 13:51:50 2021

@author: Aykut Görkem Gelen
@author: Eyyüp Yıldız

"""
import os
import json
import torch
import itertools
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import importlib
from os.path import exists
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Experiment:
    def __init__(self, exp_path):
        
        self.experiment_loader(exp_path)
        
        self.last_epoch = 0
        self.loss_hist = []
        self.best_loss = 1e6

        

        # print('No Params in the Model: {}'.format(self.get_n_params()))
        
    def experiment_loader(self, exp_path):
        print('>> Experiment Loader')
        
        config_exists = exists(os.path.join(exp_path, 'params.json'))
        if config_exists:
            self.params = self.load_json(os.path.join(exp_path, 'params.json'))
            print('Params: OK')
        else:
            raise Exception("Can't find 'params.json' in experiment path") 
            
        model_exists = exists(os.path.join(exp_path, 'Model.py'))
        if model_exists:
            module = importlib.import_module('experiments.' + self.params['exp_name'] + '.Model')
            model_class = getattr(module, self.params['model_name'])
            self.model = model_class(self.params['model_params'])
            print('Model: OK')
        else:
            raise Exception("Can't find 'Model.py' in experiment path") 
        
        data_exists = exists(os.path.join(exp_path, 'Model.py'))
        if data_exists:
            module = importlib.import_module('experiments.' + self.params['exp_name'] + '.Data')
            data_class = getattr(module, self.params['dataset_name'])
            self.dataset = data_class(self.params['dataset_params'])
            print('Dataset: OK')
        else:
            raise Exception("Can't find 'Data.py' in experiment path") 
            
        loss_exists = exists(os.path.join(exp_path, 'Loss.py'))
        if loss_exists:
            module = importlib.import_module('experiments.' + self.params['exp_name'] + '.Loss')
            loss_class = getattr(module, self.params['loss_name'])
            self.loss = loss_class(self.params['loss_params'])
            print('Loss: OK')
        else:
            raise Exception("Can't find 'Data.py' in experiment path") 
            
        if self.params['load_cp']:
            self.load_checkpoint(self.params['cp_path'])
            
        # opt_exists = exists(os.path.join(exp_path, 'Optimizer.py'))
        # if opt_exists:
        #     module = importlib.import_module('experiments.' + self.params['exp_name'] + '.Optimizer')
        #     loss_class = getattr(module, self.params['optimizer_name'])
        #     self.loss = loss_class(self.params['optimizer_params'])
        #     print('Optimizer: OK')
        # else:
        #     raise Exception("Can't find 'Optimizer.py' in experiment path") 
        
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.params['learning_rate'],
                                          betas=self.params['betas'],
                                          weight_decay=self.params['weight_decay'])
        
    @staticmethod
    def load_json(json_path):
        with open(json_path) as json_file:
            data = json.load(json_file)
         
        return data

    def train(self):
        print(">> Train")
        self.model.train()  
        data_loader = self.dataset.data_loader(self.params['batch_size'])
        
        for k in range(self.params['no_epochs']):
            running_loss = 0.0
            
            for batch_id, batch_data in tqdm(enumerate(data_loader)):
                forward_data, target_data = batch_data['forward_data'], batch_data['target_data']

                self.optimizer.zero_grad()
 
                forward_output = self.model(forward_data)
                loss = self.loss(forward_output, target_data)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            # print statistics
            epoch_loss = running_loss / (self.dataset.__len__())
            
            print('RunningLoss: {} @ Epoch #{}'.format(epoch_loss, k+1))

            #bestlos / save / log
            self.loss_hist.append(epoch_loss)
            
            if True:    #save every:
                print('Autosave @ Loss: {} @ Epoch #{}'.format(epoch_loss, k+1))
                self.best_loss = epoch_loss
                self.save_checkpoint(self.params['cp_path'])
                
    def save_checkpoint(self, chekpoint_path):
        torch.save({
            'last_epoch': self.last_epoch,
            'model_state_dict': self.model.state_dict(),
            'loss_state_dict': self.loss.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_hist': self.loss_hist,
            'best_loss': self.best_loss}, chekpoint_path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path,  map_location=torch.device(device))
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.loss.load_state_dict(checkpoint['loss_state_dict'], strict=True)
        
        self.optimizer.__setstate__({'state': defaultdict(dict)})
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.last_epoch = checkpoint['last_epoch']
        self.loss_hist = checkpoint['loss_hist']
        self.best_loss = checkpoint['best_loss']

    def plot_loss_hist(self):
        plt.plot(self.loss_hist)
        plt.xlabel('No Epochs')
        plt.ylabel('Train Loss')