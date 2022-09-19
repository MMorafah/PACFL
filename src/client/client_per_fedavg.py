import numpy as np
import copy

import torch 
from torch import nn, optim
import torch.nn.functional as F

class Client_Per_FedAvg(object):
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
        
    def train(self, is_print = False):
        self.net.to(self.device)
        self.net.train()
        
        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=0)
        beta = 0.001

        epoch_loss = []
        for iteration in range(self.local_ep):
            dataloader_iterator = iter(self.ldr_train)
            batch_loss = []
            for batch_idx in range(len(self.ldr_train) // 2):
                temp_w = copy.deepcopy(self.get_state_dict())

                images, labels = next(dataloader_iterator)
                images, labels = images.to(self.device), labels.to(self.device)
                self.net.zero_grad()
                #optimizer.zero_grad()
                log_probs = self.net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                adjust_learning_rate(optimizer, self.lr)
                optimizer.step()

                batch_loss.append(loss.item())

                images, labels = next(dataloader_iterator)
                images, labels = images.to(self.device), labels.to(self.device)
                self.net.zero_grad()
                #optimizer.zero_grad()
                log_probs = self.net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                
                self.set_state_dict(temp_w)
                adjust_learning_rate(optimizer, beta)
                optimizer.step()

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

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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