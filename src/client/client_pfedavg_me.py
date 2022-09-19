import numpy as np
import copy 
import math

import torch 
from torch import nn, optim
import torch.nn.functional as F

class Client_PFedMe(object):
    def __init__(self, name, model, local_bs, local_ep, lr, momentum, device,
                 train_dl_local = None, test_dl_local = None):
        
        self.name = name 
        self.net = model
        self.local_bs = local_bs
        self.local_ep = local_ep
        self.lr = lr 
        self.momentum = momentum 
        self.device = device
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = train_dl_local
        self.ldr_test = test_dl_local
        self.acc_best = 0 
        self.count = 0 
        self.save_best = True
        self.K = 5
        self.lam = 15
        self.personal_lr = 0.09
#         self.K = 5
#         self.lam = 15
#         self.personal_lr = 0.01
        
    def train(self, is_print = False):
        self.net.to(self.device)
        self.net.train()
        
        optimizer = pFedMeOptimizer(self.net.parameters(), lr=self.personal_lr, lam=self.lam)

        epoch_loss = []
        w_local = copy.deepcopy(list(self.net.parameters()))
        for iteration in range(self.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                
                for _ in range(self.K):
                    optimizer.zero_grad()
                    log_probs = self.net(images)
                    loss = self.loss_func(log_probs, labels)
                    loss.backward()
                    personalized_model_bar, _ = optimizer.step(w_local)

                for new_param, localweight, w in zip(personalized_model_bar, w_local, self.net.parameters()):
                    w.data = localweight.data - self.lam * self.lr * (localweight.data - new_param.data)

                batch_loss.append(loss.item())
                
            if batch_loss != []:
                epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
#         if self.save_best: 
#             _, acc = self.eval_test()
#             if acc > self.acc_best:
#                 self.acc_best = acc 
#         return sum(epoch_loss) / len(epoch_loss)

        if epoch_loss == []:
            return 0
        else:
            return sum(epoch_loss) / len(epoch_loss)
    
    def get_state_dict(self, keep_vars=False):
        return self.net.state_dict(keep_vars=keep_vars)
    def get_best_acc(self):
        return self.acc_best
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

class pFedMeOptimizer(optim.Optimizer):
    def __init__(self, params, lr=0.01, lam=15 , mu=0.001):
        #self.local_weight_updated = local_weight # w_i,K
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, lam=lam, mu=mu)
        super(pFedMeOptimizer, self).__init__(params, defaults)
    
    def step(self, local_weight_updated, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        weight_update = local_weight_updated.copy()
        for group in self.param_groups:
            for p, localweight in zip( group['params'], weight_update):
                p.data = p.data - group['lr'] * (p.grad.data + group['lam'] * (p.data - localweight.data) + group['mu']*p.data)
        return  group['params'], loss

def eval_test(net, args, ldr_test): 
    net.to(args.device)
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in ldr_test:
            data, target = data.to(args.device), target.to(args.device)
            output = net(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
    test_loss /= len(ldr_test.dataset)
    accuracy = 100. * correct / len(ldr_test.dataset)
    return test_loss, accuracy