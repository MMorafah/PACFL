import numpy as np
import copy 

import torch 
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.data import DatasetSplit 
from ..pruning.structured import *

class Client_Sub_S(object): 
    def __init__(self, name, model, local_bs, local_ep, lr, momentum, device, mask_ch, mask_fc, cfg_prune,
                 pruning_target_fc, in_ch, ks, args,
                 train_ds=None, train_idxs=None, test_ds = None, test_idxs = None):
        
        self.name = name 
        self.net = model
        self.local_bs = local_bs
        self.local_ep = local_ep
        self.lr = lr 
        self.momentum = momentum 
        self.device = device 
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(train_ds, train_idxs), batch_size=self.local_bs, shuffle=True)
        self.ldr_test = DataLoader(DatasetSplit(test_ds, test_idxs), batch_size=200)
        self.mask_ch = mask_ch
        self.mask_fc = mask_fc
        self.cfg_prune = cfg_prune
        self.pruning_target_fc = pruning_target_fc
        self.in_ch = in_ch
        self.ks = ks
        self.args = args
        self.acc_best = 0 
        self.count = 0 
        self.pruned_total = 0
        self.pruned_ch = 0
        self.pruned_fc = 0
        self.pruned_ch_rtonet = 0
        self.pruned_fc_rtonet = 0
        self.save_best = True 
        
    def train(self, percent_ch, percent_fc, dist_thresh_ch, dist_thresh_fc, acc_thresh, 
              net_glob, is_print=False):
        
        self.net.to(self.device)
        self.net.train()
        
        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=self.momentum)
        
        epoch_loss = []
        mch1 = copy.deepcopy(self.mask_ch)
        mch2 = copy.deepcopy(self.mask_ch)
        
        mfc1 = copy.deepcopy(self.mask_fc)
        mfc2 = copy.deepcopy(self.mask_fc)
        for iteration in range(self.local_ep):
            batch_loss = []
            #for name, param in net.named_parameters():
                #print(f'Name: {name}, NAN: {np.mean(np.isnan(param.detach().cpu().numpy()))}, INF: {np.mean(np.isinf(param.detach().cpu().numpy()))}')

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                self.net.zero_grad()
                optimizer.zero_grad()
                log_probs = self.net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                
                if self.args.sr:
                    updateBN(self.net, self.args)
                    
                # Freezing Pruned weights by making their gradients Zero
                step_fc = 0
                for name, p in self.net.named_parameters():
                    if 'weight' in name and 'fc' in name:
                        tensor = p.data.cpu().numpy()
                        grad_tensor = p.grad.data.cpu().numpy()
                        if step_fc == 0: 
                            temp_mask = np.zeros_like(grad_tensor)
                            end_mask = self.mask_ch[-1].cpu().numpy()
                            idx0 = np.squeeze(np.argwhere(np.asarray(end_mask)))
                            if idx0.size == 1:
                                idx0 = np.resize(idx0,(1,))
                                
                            for i in range(len(idx0)):
                                ix0 = idx0[i]
                                size = self.ks*self.ks
                                temp_mask[:,i*size:i*size+size] = self.mask_fc[step_fc][:,ix0*size:ix0*size+size]
                            grad_tensor = grad_tensor * temp_mask
                        else: 
                            grad_tensor = grad_tensor * self.mask_fc[step_fc]
                        p.grad.data = torch.from_numpy(grad_tensor).to(self.device)
                        
                        step_fc = step_fc + 1 
                        
                optimizer.step()
                batch_loss.append(loss.item())
                
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
            if iteration+1 == 1: 
                cfg1, cfg_mask1, mch1 = fake_prune_ch(percent_ch, copy.deepcopy(self.net), self.cfg_prune)
                mfc1 = fake_prune_fc(percent_fc, copy.deepcopy(self.net), copy.deepcopy(self.mask_ch), 
                                     copy.deepcopy(self.mask_fc), self.ks)
            if iteration+1 == 5:     
                cfg2, cfg_mask2, mch2 = fake_prune_ch(percent_ch, copy.deepcopy(self.net), self.cfg_prune)
                mfc2 = fake_prune_fc(percent_fc, copy.deepcopy(self.net), copy.deepcopy(self.mask_ch), 
                                     copy.deepcopy(self.mask_fc), self.ks)
        
        if self.save_best: 
            _, acc = self.eval_test(self.net)
            if acc > self.acc_best:
                self.acc_best = acc
                
        dist_fc = dist_masks(mfc1, mfc2)
        dist_ch = 1 - float(torch.sum(mch2==mch1)) / mch2.size(0)
        if is_print:
            print(f'Distance FC: {dist_fc}, Distance Channel: {dist_ch}')
        
        ## Un-Structured Prunning: Fully Connected Layers 
        state_dict = copy.deepcopy(self.net.state_dict())
        final_mask_fc = copy.deepcopy(self.mask_fc)
        
        if dist_fc > dist_thresh_fc and self.pruned_total < self.pruning_target_fc: 
            if (self.pruning_target_fc - self.pruned_total < percent_fc): 
                print(f'.... IMPOSING PRUNING To Reach Target....')
                percent_fc = ((((100 - self.pruned_total) - (100 - self.pruning_target_fc))/(100 - self.pruned_total))
                              * 100)
                if percent_fc > 16: 
                    percent_fc = percent_fc - 5
                if percent_fc <= 0: 
                    percent_fc = 0.01
                mfc2 = fake_prune_fc(percent_fc, copy.deepcopy(self.net), copy.deepcopy(self.mask_ch), 
                                     copy.deepcopy(self.mask_fc), self.ks)

            old_dict = copy.deepcopy(self.net.state_dict())
            new_dict = real_prune_fc(copy.deepcopy(self.net), copy.deepcopy(self.mask_ch), copy.deepcopy(mfc2), self.ks)
            self.net.load_state_dict(new_dict)
            _, acc = self.eval_test(self.net)

            if acc > acc_thresh: 
                if is_print:
                    print(f'Un-Structured Pruned!')
                state_dict = new_dict
                final_mask_fc = mfc2 
            else: 
                state_dict = old_dict 
                final_mask_fc = copy.deepcopy(self.mask_fc)

        self.net.load_state_dict(state_dict)
        
        ## Structured Prunning: Convolutional + BatchNorm Layers
        state_dict = copy.deepcopy(self.net.state_dict())
        final_mask_ch = copy.deepcopy(self.mask_ch)
        final_net = copy.deepcopy(self.net)
        if dist_ch > dist_thresh_ch: 
            if is_print:
                print(f'New Model: {cfg2}')
            newnet = real_prune_ch(copy.deepcopy(self.net), cfg2, cfg_mask2, self.ks, 
                                        self.in_ch, self.device, self.args)
            _, acc = self.eval_test(newnet)

            if acc > acc_thresh:
                if is_print:
                    print(f'Structured Pruned!')
                #state_dict = new_dict
                state_dict = copy.deepcopy(newnet.state_dict())
                #print(f'self.mask: {self.mask}, cfg_mask2: {cfg_mask2}')
                final_mc, final_mfc = update_mask_ch_fc(copy.deepcopy(self.mask_ch), 
                                                        copy.deepcopy(final_mask_fc), cfg_mask2, self.ks)
                final_mask_ch = copy.deepcopy(final_mc)
                final_mask_fc = copy.deepcopy(final_mfc)
                final_net = copy.deepcopy(newnet)
                
            del newnet 
        
        del self.net 
        
        self.net = final_net
        self.mask_ch = final_mask_ch
        self.mask_fc = final_mask_fc 
        
        out= print_pruning(copy.deepcopy(self.net), net_glob, is_print)
        
        self.pruned_total = out[0]
        self.pruned_ch = out[1]
        self.pruned_fc = out[2]
        self.pruned_ch_rtonet = out[3]
        self.pruned_fc_rtonet = out[4]
        
        return sum(epoch_loss) / len(epoch_loss)
 
    def get_mask_ch(self):
        return self.mask_ch 
    def get_mask_fc(self):
        return self.mask_fc 
    def get_pruned_total(self):
        return self.pruned_total
    def get_pruned_ch(self):
        return self.pruned_ch
    def get_pruned_fc(self):
        return self.pruned_fc
    def get_pruned_ch_rtonet(self):
        return self.pruned_ch_rtonet
    def get_pruned_fc_rtonet(self):
        return self.pruned_fc_rtonet
    def get_count(self):
        return self.count
    def get_net(self):
        return self.net
    def get_state_dict(self):
        return self.net.state_dict()
    def set_state_dict(self, state_dict):
        self.net.load_state_dict(state_dict)
    def get_best_acc(self):
        return self.acc_best

    def eval_test(self, model):
        model.to(self.device)
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.ldr_test:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        test_loss /= len(self.ldr_test.dataset)
        accuracy = 100. * correct / len(self.ldr_test.dataset)
        return test_loss, accuracy
    
    def eval_train(self, model):
        model.to(self.device)
        model.eval()
        train_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.ldr_train:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                train_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        train_loss /= len(self.ldr_train.dataset)
        accuracy = 100. * correct / len(self.ldr_train.dataset)
        return train_loss, accuracy