import numpy as np
import copy 

import torch 
from torch import nn, optim
import torch.nn.functional as F

class Client_Scaffold(object):
    def __init__(self, name, model, local_bs, local_ep, lr, momentum, device, 
                 train_dl_local=None, test_dl_local=None, c_local=None):
        
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
        
        #self.c_global = c_global
        self.c_local = c_local
        
        for key in self.c_local.keys():
            self.c_local[key] = self.c_local[key].to(self.device)
        
    def train(self, w_glob, c_global, is_print = False):
        self.net.to(self.device)
        self.net.train()
        
        for key in c_global.keys():
            c_global[key] = c_global[key].to(self.device)
    
        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=0)
        
        cnt=0
        epoch_loss = []
        for iteration in range(self.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                self.net.zero_grad()
                #optimizer.zero_grad()
                log_probs = self.net(images)
                loss = self.loss_func(log_probs, labels)
                
                loss.backward()         
                optimizer.step()
                
                net_para = self.net.state_dict()
                for key in net_para:
                    net_para[key] = net_para[key].to(self.device) - self.lr * (c_global[key] - self.c_local[key])
                self.net.load_state_dict(net_para)
                
                cnt+=1 
                batch_loss.append(loss.item())
                
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
        c_new = copy.deepcopy(self.c_local)
        c_delta = copy.deepcopy(self.c_local)
        net_para = self.net.state_dict()
        for key in net_para:
            c_new[key] = c_new[key].to(self.device) - c_global[key].to(self.device) + (w_glob[key].to(self.device) - net_para[key].to(self.device))/(cnt* self.lr)
            c_delta[key] = c_new[key].to(self.device) - self.c_local[key].to(self.device)
            
        self.c_local = copy.deepcopy(c_new)
        
#         if self.save_best: 
#             _, acc = self.eval_test()
#             if acc > self.acc_best:
#                 self.acc_best = acc 
        
        return sum(epoch_loss)/len(epoch_loss), c_delta
    
    def get_state_dict(self):
        return self.net.state_dict()
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
    
    def eval_test_glob(self, glob_dl):
        self.net.to(self.device)
        self.net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in glob_dl:
                data, target = data.to(self.device), target.to(self.device)
                output = self.net(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        test_loss /= len(glob_dl.dataset)
        accuracy = 100. * correct / len(glob_dl.dataset)
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