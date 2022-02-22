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
import itertools
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Experiment:
    def __init__(self, network, experiment_params):
        self.last_epoch = 0
        self.loss_hist = []
        self.eval_hist = []
        self.best_loss = 1e6

        self.l1_lambda = 0.00001

        self.network = network

        self.ext_lr = experiment_params['ext_lr']
        self.batch_size = experiment_params['batch_size']
        self.no_epoch = experiment_params['no_epoch']
        self.lr = experiment_params['lr']
        self.cp_path = experiment_params['checkpoint_path']
        self.loss = experiment_params['loss']

        # params = [self.network.parameters(), self.loss.parameters()]

        # self.optimizer = torch.optim.Adam(itertools.chain(*params), lr=self.lr, betas=(0.9,0.999))
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr, betas=(0.9,0.999))

        # self.optimizer.add_param_group({'params': self.loss.parameters(), 'lr': 1e-3})

        print('>> Optimizer set! LR:{}'.format(self.lr))


        self.dataset = experiment_params['dataset']
        self.data_loader = self.dataset.data_loader

        self.cp_load_flag = experiment_params['load_checkpoint']

        if self.cp_load_flag:
            if self.ext_lr:
                self.load_checkpoint(self.cp_path, self.ext_lr)
            else:
                self.load_checkpoint(self.cp_path)

        print('No Params in the Model: {}'.format(self.get_n_params()))

    def train(self):
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
        #                                                             mode='min',
        #                                                             factor=0.1,
        #                                                             patience=4,
        #                                                             verbose=True)
        
        self.network.train()            #FIXME! is this really needed?
        # subset = torch.utils.data.Subset(self.dataset, np.arange(500))
        subset = torch.utils.data.Subset(self.dataset, np.array(self.dataset.train_idx))
        data_loader = torch.utils.data.DataLoader(subset, batch_size=self.batch_size, shuffle=True)

        for k in range(self.no_epoch):
            running_loss = 0.0
            for batch_id, batch_data in tqdm(enumerate(data_loader)):
                forward_data, target_data = batch_data['forward_data'], batch_data['target_data']

                self.optimizer.zero_grad()
                #forward + backward + optimize
                forward_output = self.network(forward_data)
                loss = self.loss(forward_output, target_data)

                l1_norm = sum(p.abs().sum() for p in self.network.parameters())
                loss = loss + self.l1_lambda * l1_norm

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                # print(loss.item())

            # print statistics
            epoch_loss = running_loss / (subset.__len__()*self.batch_size)
            # epoch_loss = running_loss / 300
            print('RunningLoss: {} @ Epoch #{}'.format(epoch_loss, k+1))

            #bestlos / save / log
            self.loss_hist.append(epoch_loss)

            eval_loss = self.validate()
            self.eval_hist.append(eval_loss)

            # self.scheduler.step(eval_loss)

            if epoch_loss < self.best_loss:
                print('Autosave @ Loss: {} @ Epoch #{}'.format(epoch_loss, k+1))
                self.best_loss = epoch_loss
                self.save_checkpoint(self.cp_path)

            # print('Sigmas: {}/{}'.format(list(self.loss.parameters())[0][0],
            #                               list(self.loss.parameters())[0][1]))

        print('Train Completed!')

    # def evaluate(self):
    #     eval_subs = np.concatenate([np.arange(560,610), np.arange(610,640), np.arange(710,780)])
    #     subset = torch.utils.data.Subset(self.dataset, eval_subs)
    #     data_loader = torch.utils.data.DataLoader(subset, batch_size=self.batch_size, shuffle=True)

    #     self.network.eval()
    #     running_loss = 0.0
    #     for batch_id, batch_data in tqdm(enumerate(data_loader)):
    #         forward_data, target_data = batch_data['forward_data'], batch_data['target_data']
    #         forward_output = self.network(forward_data)
    #         loss = self.loss(forward_output, target_data)
    #         running_loss += loss.item()
    #     # print statistics
    #     epoch_loss = running_loss / (self.dataset.__len__()/self.batch_size)
    #     # epoch_loss = running_loss / 20
    #     print('Eval Loss: {}'.format(epoch_loss))
    #     return epoch_loss

    def evaluate(self, save_path):
        eval_subs = np.arange(1,1001)
        # eval_subs = np.array(list(self.dataset.test_idx))
        subset = torch.utils.data.Subset(self.dataset, eval_subs)
        # subset = torch.utils.data.Subset(self.dataset, np.array(self.dataset.valid_idx))
        data_loader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False)

        est_quats = []
        est_transs = []

        self.network.eval()
        # running_loss = 0.0
        for batch_id, batch_data in tqdm(enumerate(data_loader)):
            forward_data, target_data = batch_data['forward_data'], batch_data['target_data']
            estimated = self.network(forward_data)

            est_quat = estimated['quat'].detach().cpu().numpy()
            est_trans = estimated['trans'].detach().cpu().numpy()

            est_quats.append(est_quat)
            est_transs.append(est_trans)

        np.save(save_path + '/quats.npy', est_quats, allow_pickle=True)
        np.save(save_path + '/transs.npy', est_transs, allow_pickle=True)

        print('Evaluate Finished')

    def save_checkpoint(self, chekpoint_path):
        torch.save({
            'epoch': self.last_epoch,
            'model_state_dict': self.network.state_dict(),
            # 'loss_state_dict': self.loss.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_hist': self.loss_hist,
            'eval_hist': self.eval_hist,
            'best_loss': self.best_loss,
            'last_lr': self.lr,
            }, chekpoint_path)

    def load_checkpoint(self, path, ext_lr=False):
        # device = torch.device('cpu')
        checkpoint = torch.load(path,  map_location=torch.device(device))

        self.network.load_state_dict(checkpoint['model_state_dict'], strict=True)

        # self.loss.load_state_dict(checkpoint['loss_state_dict'], strict=True)

        self.optimizer.__setstate__({'state': defaultdict(dict)})
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.last_epoch = checkpoint['epoch']
        self.loss_hist = checkpoint['loss_hist']
        self.eval_hist = checkpoint['eval_hist']
        self.best_loss = checkpoint['best_loss']

        if ext_lr:
            self.lr = ext_lr
            self.optimizer.param_groups[0]['lr'] = self.lr
            print('LR set externally: {}'.format(self.lr))
        else:
            self.lr = checkpoint['last_lr']
            print('Saved LR: {}'.format(self.lr))
            self.optimizer.param_groups[0]['lr'] = self.lr

    def test(self, idx=False):
        self.network.eval()

        subset = torch.utils.data.Subset(self.dataset, [idx])
        data_loader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False)

        if isinstance(idx, int):
            #single forward with specified idx
            for batch_id, batch_data in enumerate(data_loader):
                forward_data, target_data = batch_data['forward_data'], batch_data['target_data']

                forward_output = self.network(forward_data)

                loss = self.loss(forward_output, target_data)

                l1_norm = sum(p.abs().sum() for p in self.network.parameters())
                loss = loss + self.l1_lambda * l1_norm

                print('Loss: {}'.format(loss.item()))

            return forward_output
        else:
            #test all data?
            # subset = torch.utils.data.Subset(self.dataset, np.arange(400,600))
            # data_loader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False)

            mean_loss = 0
            i = 0
            for batch_id, batch_data in tqdm(enumerate(self.data_loader)):
                if batch_id > 400:
                    forward_data, target_data = batch_data['forward_data'], batch_data['target_data']
                    forward_output = self.network(forward_data)
                    loss = self.loss(forward_output, target_data)
                    mean_loss += loss.item()
                    i += 1

            mean_loss /= i
            print('Mean Loss: {}'.format(mean_loss))


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

    def validate(self):
        self.network.eval()
        val_loss = 0.0
        # eval_subs = np.concatenate([np.arange(560,610), np.arange(610,640), np.arange(710,780)])
        # subset = torch.utils.data.Subset(self.dataset, eval_subs)
        subset = torch.utils.data.Subset(self.dataset, np.array(self.dataset.valid_idx))
        data_loader = torch.utils.data.DataLoader(subset, batch_size=self.batch_size, shuffle=False)
        for batch_id, batch_data in tqdm(enumerate(data_loader)):
            forward_data, target_data = batch_data['forward_data'], batch_data['target_data']
            forward_output = self.network(forward_data)
            loss = self.loss(forward_output, target_data)

            l1_norm = sum(p.abs().sum() for p in self.network.parameters())
            loss = loss + self.l1_lambda * l1_norm

            val_loss += loss.item()

        #val_loss /= subset.__len__()
        val_loss = val_loss / (subset.__len__()*self.batch_size)

        print('Validation Loss: {}'.format(val_loss))
        return val_loss

    def mse(self):
        self.network.eval()
        mse = torch.nn.MSELoss()
        
        subset = torch.utils.data.Subset(self.dataset, np.array(self.dataset.test_idx))
        data_loader = torch.utils.data.DataLoader(subset, batch_size=self.batch_size, shuffle=False)
        
        print('No of test data: {}'.format(data_loader.__len__()))
        
        est_q, est_t, tgt_q, tgt_t = [], [], [], []
        for batch_id, batch_data in tqdm(enumerate(data_loader)):
            forward_data, target_data = batch_data['forward_data'], batch_data['target_data']
            forward_output = self.network(forward_data)

            est_q.append(forward_output['quat'].detach())
            est_t.append(forward_output['trans'].detach())
            
            tgt_q.append(target_data['quat'].squeeze(1))
            tgt_t.append(target_data['trans'].squeeze(1))

        est_q = torch.vstack(est_q)
        est_t = torch.vstack(est_t)
        tgt_q = torch.vstack(tgt_q)
        tgt_t = torch.vstack(tgt_t)

        # from pyquaternion import Quaternion

        # quat_tgt = Quaternion(tgt_q[0].detach().numpy())
        # quat_est = Quaternion(est_q[0].detach().numpy())

        # Quaternion.absolute_distance(quat_tgt, quat_est)

        mse_t = mse(est_t, tgt_t)
        mse_q = mse(est_q, tgt_q)

        print('> MSE Metric Results')

        print('> MSE Trans: {} \n> MSE Quat: {}'.format(mse_t, mse_q*512))
        print('> Total: {}'.format(mse_t, mse_q*512))
        #     loss = self.loss(forward_output, target_data)
        #     val_loss += loss.item()

        # val_loss /= subset.__len__()
        # print('Validation Loss: {}'.format(val_loss))

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