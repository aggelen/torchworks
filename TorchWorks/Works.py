#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 13:51:50 2021

@author: Aykut Görkem Gelen
@author: Eyyüp Yıldız

"""
import torch
from tqdm import tqdm
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Experiment:
    def __init__(self, network, experiment_params):
        self.last_epoch = 0
        self.loss_hist = []
        self.best_loss = 1e6

        self.max_depth = 400

        if 'scheduler' in experiment_params:
            self.scheduler = experiment_params['scheduler']
        else:
            self.scheduler = None

        if 'ext_lr' in experiment_params:
            self.ext_lr = experiment_params['ext_lr']
        else:
            self.ext_lr = False

        self.network = network

        self.no_epoch = experiment_params['no_epoch']
        self.lr = experiment_params['lr']
        self.cp_path = experiment_params['checkpoint_path']

        #Optimizer
        if experiment_params['optimizer'] == 'default':
            self.opt_mode = 'default'
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr, betas=(0.9,0.999))
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                        mode='min',
                                                                        factor=0.25,
                                                                        patience=4,
                                                                        verbose=True)
            print('>> Optimizer set! LR:{}'.format(self.lr))
            self.dual_lr = False
        elif experiment_params['optimizer'] == 'dual':
            self.opt_mode = 'dual'

            self.dual_params = experiment_params['dual_params']
            self.dual_lr = experiment_params['dual_lr']

            s0, f0 = experiment_params['optimizer1'][0]
            s1, f1 = experiment_params['optimizer2'][0]

            lr1, lr2 = experiment_params['optimizer1'][1], experiment_params['optimizer2'][1]
            params = list(self.network.parameters())
            p0 = params[s0:f0]
            #FIXME!
            if f1 == -1:
                p1 = params[s1:]
            else:
                p1 = params[s1:f1]

            self.optimizer1 = torch.optim.Adam(p0, lr=lr1, betas=(0.9,0.999))
            self.optimizer2 = torch.optim.Adam(p1, lr=lr2, betas=(0.9,0.999))

            print('>> Optimizers set! Opt1 LR:{}, Opt2 LR:{}'.format(lr1, lr2))
        else:
            self.optimizer = experiment_params['optimizer']

        self.loss = experiment_params['loss']
        self.dataset = experiment_params['dataset']
        self.data_loader = self.dataset.data_loader

        self.cp_load_flag = experiment_params['load_checkpoint']

        if self.cp_load_flag:
            if self.ext_lr:
                self.load_checkpoint(self.cp_path, self.ext_lr)
            else:
                self.load_checkpoint(self.cp_path)

        print('No Params in the Model: {}'.format(self.get_n_params(self.network)))

    def train(self):
        self.network.train()            #FIXME! is this really needed?

        for k in range(self.no_epoch):

            if self.scheduler == 'half_every10':
                if k%10 == 0:
                    # if k%10 == 0 and k!=0:
                    self.lr /= 2
                    print('LR halved! New LR: {}'.format(self.lr))

            running_loss = 0.0
            running_loss1 = 0.0
            running_loss2 = 0.0
            for batch_id, batch_data in tqdm(enumerate(self.data_loader)):

                forward_data, target_data = batch_data

                if self.opt_mode == 'dual':
                    self.optimizer1.zero_grad()
                    self.optimizer2.zero_grad()
                else:
                    self.optimizer.zero_grad()

                #forward + backward + optimize
                forward_output = self.network(forward_data)

                if self.opt_mode == 'dual':
                    loss1, loss2 = self.loss(forward_output, target_data)
                    loss1.backward()
                    loss2.backward()
                    self.optimizer1.step()
                    self.optimizer2.step()
                    running_loss1 += loss1.item()
                    running_loss2 += loss2.item()

                    running_loss += loss1.item() + loss2.item()
                else:

                    loss = self.loss(forward_output, target_data)
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item()

            # print statistics
            epoch_loss = running_loss / self.data_loader.__len__()
            print('RunningLoss: {} @ Epoch #{}'.format(epoch_loss, k+1))
            if self.opt_mode == 'dual':
                epoch_loss1 = running_loss1 / self.data_loader.__len__()
                epoch_loss2 = running_loss2 / self.data_loader.__len__()
                print('RunningLoss1: {} RunningLoss2: {} @ Epoch #{}'.format(epoch_loss1, epoch_loss2, k+1))

            #bestlos / save / log
            self.loss_hist.append(epoch_loss)
            if epoch_loss < self.best_loss:
                print('Autosave @ Loss: {} @ Epoch #{}'.format(epoch_loss, k+1))
                self.best_loss = epoch_loss
                self.save_checkpoint(self.cp_path)

            self.scheduler.step(epoch_loss)

        print('Train Completed!')

    def save_checkpoint(self, chekpoint_path):
        if self.opt_mode == 'dual':
            torch.save({
                'epoch': self.last_epoch,
                'model_state_dict': self.network.state_dict(),
                'optimizer1_state_dict': self.optimizer1.state_dict(),
                'optimizer2_state_dict': self.optimizer2.state_dict(),
                'loss_hist': self.loss_hist,
                'best_loss': self.best_loss,
                'last_lr': self.lr,
                'last_lr_opt1': self.optimizer1.param_groups[0]['lr'],
                'last_lr_opt2': self.optimizer2.param_groups[0]['lr'],
                }, chekpoint_path)
        else:
            torch.save({
                'epoch': self.last_epoch,
                'model_state_dict': self.network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss_hist': self.loss_hist,
                'best_loss': self.best_loss,
                'last_lr': self.lr,
                }, chekpoint_path)

    def load_checkpoint(self, path, ext_lr=False):
        checkpoint = torch.load(path,  map_location=torch.device(device))

        self.network.load_state_dict(checkpoint['model_state_dict'])

        if self.opt_mode == 'dual':
            self.optimizer1.__setstate__({'state': defaultdict(dict)})
            self.optimizer2.__setstate__({'state': defaultdict(dict)})
            self.optimizer1.load_state_dict(checkpoint['optimizer1_state_dict'])
            self.optimizer2.load_state_dict(checkpoint['optimizer2_state_dict'])
        else:
            self.optimizer.__setstate__({'state': defaultdict(dict)})
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.last_epoch = checkpoint['epoch']
        self.loss_hist = checkpoint['loss_hist']
        self.best_loss = checkpoint['best_loss']

        if ext_lr:
            self.lr = ext_lr
            self.optimizer.param_groups[0]['lr'] = self.lr
            print('LR set externally: {}'.format(self.lr))

        elif self.dual_lr:
            self.optimizer1.param_groups[0]['lr'] = self.dual_params[0]
            self.optimizer2.param_groups[0]['lr'] = self.dual_params[1]
            print('>> CP load with lr1: {}, lr2:{}'.format(self.dual_params[0], self.dual_params[1]))
        else:
            self.lr = checkpoint['last_lr']
            if self.opt_mode == 'dual':
                self.optimizer1.param_groups[0]['lr'] = checkpoint['last_lr_opt1']
                self.optimizer2.param_groups[0]['lr'] = checkpoint['last_lr_opt2']
            else:
                self.optimizer.param_groups[0]['lr'] = self.lr

    def test(self, idx=False):
        self.network.eval()
        if isinstance(idx, int):
            #single forward with specified idx
            forward_data, target_data = self.dataset.__getitem__(idx)
            forward_output = self.network(forward_data)

            loss = self.loss(forward_output, target_data)
            print('Loss: {}'.format(loss))

            return forward_output
        else:
            #test all data?
            pass
    
    #FIXME! a temporary experiment special function, to be removed 
    def create_imgs(self, path):
        import torchvision
        self.network.eval()
        for idx in tqdm(range(900)):
            forward_data, _ = self.dataset.__getitem__(idx)
            raw_events = forward_data['raw_events']
            gray_img = self.network.e2v.generate_gray_image(raw_events)

            grid = torchvision.transforms.functional.to_pil_image(gray_img[0])
            grid.save(path + '{:06d}.png'.format(idx))

    def validation(self):
        pass

    def print_model_structure(self):
        i = 0
        for name, param in self.network.named_parameters():
            if param.requires_grad:
                # param.data
                print(i, name)
                i += 1

    def get_n_params(self):
        # No of params in a model
        pp=0
        for p in list(self.network.parameters()):
            nn=1
            for s in list(p.size()):
                nn = nn*s
            pp += nn
        return pp