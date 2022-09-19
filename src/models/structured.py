import numpy as np 

from torch import nn
import torch.nn.functional as F
import torch.nn.init as init

## Defining MODELS For Structured Pruning   

class LeNetBN5Mnist(nn.Module):
    def __init__(self, cfg=None, ks=5):
        super(LeNetBN5Mnist, self).__init__()
        if cfg == None: 
            self.cfg = [10, 'M', 20, 'M'] 
        else: 
            self.cfg = cfg
            
        self.ks = ks 
        self.main = nn.Sequential()
        self.make_layers(self.cfg, True) 
        
        #self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        #self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        
        self._initialize_weights()
    
    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 1
        idx_maxpool = 1 
        idx_bn = 1
        idx_conv = 1 
        idx_relu = 1
        for v in self.cfg:
            if v == 'M':
                layers += [('maxpool{}'.format(idx_maxpool), nn.MaxPool2d(kernel_size=2, stride=2))]
                idx_maxpool += 1 
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=self.ks)
                if batch_norm:
                    layers += [('conv{}'.format(idx_conv), conv2d), ('bn{}'.format(idx_bn), nn.BatchNorm2d(v)),
                               ('relu{}'.format(idx_relu), nn.ReLU(inplace=True))]
                    idx_bn += 1 
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                idx_conv += 1
                idx_relu += 1 
                in_channels = v
        
        [self.main.add_module(n, l) for n, l in layers]
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    
    def forward(self, x):
        x = self.main(x)
        #print(x.shape)
        #x = x.view(-1, self.cfg[-2] * self.ks * self.ks)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class LeNetBN5Cifar(nn.Module):
    def __init__(self, nclasses = 10, cfg=None, ks=5):
        super(LeNetBN5Cifar, self).__init__()
        if cfg == None: 
            self.cfg = [6, 'M', 16, 'M'] 
        else: 
            self.cfg = cfg
    
        self.ks = ks 
        fc_cfg = [120, 84, 100]
        
        self.main = nn.Sequential()
        self.make_layers(self.cfg, True)        
        
        self.fc1 = nn.Linear(self.cfg[-2] * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, nclasses)
        
        self._initialize_weights()
    
    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        idx_maxpool = 1 
        idx_bn = 1
        idx_conv = 1 
        idx_relu = 1
        for v in self.cfg:
            if v == 'M':
                layers += [('maxpool{}'.format(idx_maxpool), nn.MaxPool2d(kernel_size=2, stride=2))]
                idx_maxpool += 1 
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=self.ks)
                if batch_norm:
                    layers += [('conv{}'.format(idx_conv), conv2d), ('bn{}'.format(idx_bn), nn.BatchNorm2d(v)),
                               ('relu{}'.format(idx_relu), nn.ReLU(inplace=True))]
                    idx_bn += 1 
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                idx_conv += 1
                idx_relu += 1 
                in_channels = v
        
        [self.main.add_module(n, l) for n, l in layers]
    
    def forward(self, x):
        #x = self.main.conv1(x)
        x = self.main(x)
        
        #print(x.shape)
        #print(self.cfg[2])
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
        return

def updateBN(mymodel, args):
    for m in mymodel.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(args.s*torch.sign(m.weight.data))  # L1       
    return