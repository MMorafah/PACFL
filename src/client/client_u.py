import numpy as np
import copy 

import torch 
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.data import DatasetSplit 
from ..pruning.unstructured import *

class Client_Sub_Un(object):
    def __init__(self, name, model, local_bs, local_ep, lr, momentum, device, mask, pruning_target, 
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
        self.mask = mask 
        self.pruning_target = pruning_target
        self.acc_best = 0 
        self.count = 0 
        self.pruned = 0 
        self.save_best = True 
        
    def train(self, percent, dist_thresh, acc_thresh, is_print = False):
        self.net.to(self.device)
        self.net.train()
        
        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=self.momentum)
        
        epoch_loss = []
        m1 = copy.deepcopy(self.mask)
        m2 = copy.deepcopy(self.mask)
        for iteration in range(self.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                self.net.zero_grad()
                optimizer.zero_grad()
                log_probs = self.net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                
                # Freezing Pruned weights by making their gradients Zero
                step = 0
                for name, p in self.net.named_parameters():
                    if 'weight' in name:
                        tensor = p.data.cpu().numpy()
                        grad_tensor = p.grad.data.cpu().numpy()
                        #grad_tensor = np.where(tensor < EPS, 0, grad_tensor)
                        grad_tensor = grad_tensor * self.mask[step]
                        p.grad.data = torch.from_numpy(grad_tensor).to(self.device)
                        step = step + 1 
                        
                optimizer.step()
                batch_loss.append(loss.item())
                
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
            if iteration+1 == 1: 
                m1 = fake_prune(percent, copy.deepcopy(self.net), copy.deepcopy(self.mask))
            if iteration+1 == 5:     
                m2 = fake_prune(percent, copy.deepcopy(self.net), copy.deepcopy(self.mask))
        
        if self.save_best: 
            _, acc = self.eval_test()
            if acc > self.acc_best:
                self.acc_best = acc 
                
        dist = dist_masks(m1, m2)
        if is_print:
            print(f'Distance: {dist}')
        
        state_dict = copy.deepcopy(self.net.state_dict())
        final_mask = copy.deepcopy(self.mask)
        
        if dist > dist_thresh and self.pruned < self.pruning_target: 
            if (self.pruning_target - self.pruned < percent): 
                print(f'..IMPOSING PRUNING To Reach Target PRUNING..')
                #print(f'user prune: {user_pruned}')
                percent = ((((100 - self.pruned) - (100 - self.pruning_target))/(100 - self.pruned)) * 100)
                #print(f'Percent {percent}')
                if percent > 5: 
                    percent = 5
                m2 = fake_prune(percent, copy.deepcopy(self.net), copy.deepcopy(self.mask))

            old_dict = copy.deepcopy(self.net.state_dict())
            new_dict = real_prune(copy.deepcopy(self.net), m2)
            self.net.load_state_dict(new_dict)
            _, acc = self.eval_test()
            if is_print:
                print(f'acc after pruning: {acc}')
            if acc > acc_thresh: 
                if is_print:
                    print(f'Pruned! acc after pruning {acc}')
                state_dict = new_dict 
                final_mask = m2 
            else: 
                if is_print:
                    print(f'Not Pruned!!!')
                state_dict = old_dict 
                final_mask = copy.deepcopy(self.mask)
                
        self.net.load_state_dict(state_dict)
        self.mask = copy.deepcopy(final_mask) 
        self.pruned, _ = print_pruning(copy.deepcopy(self.net), is_print)
        
        return sum(epoch_loss) / len(epoch_loss)
    
    def get_state_dict(self):
        return self.net.state_dict()
    def get_mask(self):
        return self.mask 
    def get_best_acc(self):
        return self.acc_best
    def get_pruning(self):
        return self.pruned
    def get_count(self):
        return self.count
    def get_net(self):
        return self.net
    def set_state_dict(self, state_dict):
        self.net.load_state_dict(state_dict)

    def eval_test(self):
        self.net.to(self.device)
        self.net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.ldr_test:
                data, target = data.to(self.device), target.to(self.device)
                output = self.net(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        test_loss /= len(self.ldr_test.dataset)
        accuracy = 100. * correct / len(self.ldr_test.dataset)
        return test_loss, accuracy
    
    def eval_train(self):
        self.net.to(self.device)
        self.net.eval()
        train_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.ldr_train:
                data, target = data.to(self.device), target.to(self.device)
                output = self.net(data)
                train_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        train_loss /= len(self.ldr_train.dataset)
        accuracy = 100. * correct / len(self.ldr_train.dataset)
        return train_loss, accuracy