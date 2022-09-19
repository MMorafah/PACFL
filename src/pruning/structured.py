import numpy as np
from scipy import linalg
from scipy.spatial import distance

import torch
import torch.nn as nn

from ..models.structured import *

def make_init_mask_fc(model):
    '''
    Makes the initial pruning mask for the given model's fully connected layers. 
    For example, for LeNet-5 architecture it return a list of 3 arrays, each array is 
    the same size of each fc layer's weights and with all 1 entries. We do not prune bias
    
    :param model: a pytorch model 
    :return mask: a list of pruning masks of fc layers
    '''
    step = 0
    for m in model.modules(): 
        if isinstance(m, nn.Linear):
            step = step + 1
            
    mask = [None]* step 
    
    step = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            tensor = m.weight.data.cpu().numpy()
            mask[step] = np.ones_like(tensor)
            print(f'step {step}, shape: {mask[step].shape}')
            step = step + 1
            
    return mask

def make_init_mask_ch(model): 
    '''
    Makes the initial pruning mask for the given model's convolutional layers. For example, 
    for LeNet-5 architecture it return a list of 2 arrays, 
    each array is the same size of each conv layer's weights and with all 1 entries. We do not prune bias
    
    :param model: a pytorch model 
    :return mask: a list of pruning masks of conv layers 
    '''
    total = 0
    cfg_mask = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]
            _m = torch.ones(m.weight.data.shape[0])
            cfg_mask.append(_m)
    #mask = torch.ones(total)
    return cfg_mask


def fake_prune_fc(percent, model, mask_ch, mask_fc, ks):
    '''
    This function derives the new pruning mask of fc layers, it put 0 for the weights under the given percentile
    
    :param percent: pruning percent 
    :param model: a pytorch model 
    :param mask_ch: the pruning mask of conv layers 
    :param mask_fc: the pruning mask of fc layers 
    :param ks: kernel size of the model 
    
    :return mask: updated pruning mask of fc layers 
    '''
    step = 0
    for m in model.modules():
        # We do not prune bias term
        if isinstance(m, nn.Linear):
            tensor = m.weight.data.cpu().numpy()
            if step == 0: 
                temp_mask = np.zeros_like(tensor) # [120, 400] --> [120, 250]
                msk = mask_ch[-1].cpu().numpy()
                
                idx0 = np.squeeze(np.argwhere(np.asarray(msk)))
                if idx0.size == 1:
                    idx0 = np.resize(idx0,(1,))
                    
                for i in range(len(idx0)):
                    ix = idx0[i]
                    temp_mask[:,i*ks*ks:i*ks*ks+ks*ks] = mask_fc[step][:,ix*ks*ks:ix*ks*ks+ks*ks]
                    
                alive = tensor[np.nonzero(tensor * temp_mask)] # flattened array of nonzero values
            else: 
                alive = tensor[np.nonzero(tensor * mask_fc[step])] # flattened array of nonzero values

            percentile_value = np.percentile(abs(alive), percent)
            
            # Convert Tensors to numpy and calculate
            weight_dev = m.weight.device
            if step == 0:
                new_mask = np.where(abs(tensor) < percentile_value, 0, temp_mask)
                for i in range(len(idx0)):
                    ix = idx0[i]
                    mask_fc[step][:,ix*ks*ks:ix*ks*ks+ks*ks] = new_mask[:,i*ks*ks:i*ks*ks+ks*ks]
            else: 
                new_mask = np.where(abs(tensor) < percentile_value, 0, mask_fc[step]) 
                # Apply new mask
                mask_fc[step] = new_mask
                
            step += 1

    return mask_fc

def fake_prune_ch(percent, model, cfg_prune): 
    '''
    This function derives the new pruning mask of conv layers, it put 0 for the channels with the p% lowest bn's
    weights 
    
    :param percent: pruning percent 
    :param model: a pytorch model 
    :param cfg_prune: final pruning config. It does not go lower than this 
    
    :return cfg: updated new config of the model after deriving pruning 
    :return cfg_mask: updated list of pruning mask for each conv layer 
    :return mask: updated pruning mask of conv layers all stacked together into a vector 
    '''
    total = 0
    bns = []  
    #bn = torch.zeros(total)
    index = 0
    step = 0 
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            #print(f'Shape: {m.weight.data.shape}')
            bns.append(torch.zeros(m.weight.data.shape[0]))
            bns[step] = m.weight.data.abs().clone()
            
            #size = m.weight.data.shape[0]
            #bn[index:(index+size)] = m.weight.data.abs().clone()
            
            total += m.weight.data.shape[0]
            #index += size
            step += 1 

    y = []
    idx = []
    thre_index = []
    thre = []
    for i in range(len(bns)): 
        y0, i0 = torch.sort(bns[i])
        thre_index0 = int(len(bns[i]) * percent)
        thre0 = y0[thre_index0]
        
        y.append(y0)
        idx.append(i0)
        thre_index.append(thre_index0)
        thre.append(thre0)
    
    pruned = 0
    cfg = []
    cfg_mask = []
    mask = torch.zeros(total)
    index = 0 
    step = 0 
    idx_cfg = 0
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            weight_copy = m.weight.data.clone()
            size = m.weight.data.numel()
            #print(f'Size: {size}') 
            assert isinstance(cfg_prune[idx_cfg], int)
            
            if size > cfg_prune[idx_cfg]:
                _mask = weight_copy.abs().gt(thre[step]).float().cuda()
                #print(torch.sum(_mask))
                if torch.sum(_mask) < cfg_prune[idx_cfg]: 
                    #print('Not Pruning Due To Below Threshold')
                    _mask = torch.ones_like(weight_copy).cuda()
                    
                pruned = pruned + _mask.shape[0] - torch.sum(_mask)
            else: 
                #print('Not Pruning')
                _mask = torch.ones_like(weight_copy).cuda() 
            
            step += 1 
            idx_cfg += 1 
            cfg.append(int(torch.sum(_mask)))
            cfg_mask.append(_mask.clone())
            mask[index:(index+size)] = _mask.view(-1)
            index += size
            #print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                #format(k, _mask.shape[0], int(torch.sum(_mask))))
        elif isinstance(m, nn.MaxPool2d):
            cfg.append('M')
            idx_cfg += 1

    #pruned_ratio = pruned/total

    return cfg, cfg_mask, mask 

def update_mask_ch_fc(glob_mask_ch, glob_mask_fc, new_mask_ch, ks): 
    '''
    After pruning channels, the first fc connected layer connecting the last conv layer will be pruned as well. 
    To handel this dimension mis-match, this function updates the 1st fc layer's pruning mask based on the 
    pruned channels. 
    
    :param glob_mask_ch: the global pruning mask of channels which keep track of the pruned channels corresponding
    to the original model 
    
    :param glob_mask_fc: the global pruning mask of fc layers which keep track of the pruned fc layers
    corresponding to the original model 
    
    :param new_mask_ch: the updated new pruning mask of channels 
    :param ks: kernel size of the model 
    
    :return glob_mask_ch, glob_mask_fc: updated global pruning masks of channels and fc layers 
    '''
    for k in range(len(glob_mask_ch)):
        glob_msk = glob_mask_ch[k]
        new_msk = new_mask_ch[k]
        
        ind_r = np.squeeze(np.argwhere(np.asarray(glob_msk.cpu().numpy())))
        if ind_r.size == 1:
            ind_r = np.resize(ind_r,(1,))
            
        #print(f'ind_r: {ind_r}, new_mask: {new_msk}')
        assert len(ind_r) == len(new_msk)
        
        for i in range(len(ind_r)): 
            if new_msk[i] == 0: 
                glob_msk[ind_r[i]] = 0 
    
    for k in range(len(glob_mask_fc)): 
        glob_msk = glob_mask_fc[k] # [120, 400]
        
        if k == 0: 
            last_msk_ch = glob_mask_ch[-1].cpu().numpy()
            for j in range(len(last_msk_ch)): 
                if last_msk_ch[j] == 0: 
                    glob_msk[:,j*ks*ks:j*ks*ks+ks*ks] = 0 

    return glob_mask_ch, glob_mask_fc

def real_prune_ch(model, cfg, cfg_mask, ks, in_ch, device, args):
    '''
    This function applies the derived mask of conv layers. It zeros the weights needed to be pruned 
    based on the updated mask of fc layers
    
    **NOTE: Please note that this function creates a new model based on the new config and returns the new model 
    
    :param model: a pytorch model of client
    :param cfg: updated new config of the model after deriving pruning 
    :return cfg_mask: updated list of pruning mask for each conv layer 
    :param ks: kernel size of the model 
    :param in_ch: number of input channels to the 1st conv layer
    :param device: device to run
    
    :return state_dict: new model 
    '''
    if args.model == 'lenet5' and args.dataset == 'cifar10':
        newmodel = LeNetBN5Cifar(nclasses = 10, cfg=cfg, ks=args.ks).to(args.device)
    elif args.model == 'lenet5' and args.dataset == 'cifar100':
        newmodel = LeNetBN5Cifar(nclasses = 100, cfg=cfg, ks=args.ks).to(args.device)
    elif args.model == 'lenet5' and args.dataset == 'mnist':
        newmodel = LeNetBN5Mnist(cfg=cfg, ks=args.ks).to(args.device)
    
    layer_id_in_cfg = 0
    start_mask = torch.ones(in_ch)
    end_mask = cfg_mask[layer_id_in_cfg]
    fc_count = 0
    remain_ch = []
    for [m0, m1] in zip(model.modules(), newmodel.modules()):
        if isinstance(m0, nn.BatchNorm2d):
            if torch.sum(end_mask) == 0:
                continue
                
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))
                
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                end_mask = cfg_mask[layer_id_in_cfg]
                
        elif isinstance(m0, nn.Conv2d):
            if torch.sum(end_mask) == 0:
                continue
                
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            #print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            #remain_channel.append(idx1)
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
                
            assert len(idx0) == m1.weight.data.shape[1]
            assert len(idx1) == m1.weight.data.shape[0]
            
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone() # [6, 3, 5, 5] [out_channel, in_channel, k, k]
            w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()                         # [3, 3, 5, 5]
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()

        elif isinstance(m0, nn.Linear):
            if fc_count == 0: 
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                    
                #print(idx0)
                #print(m0.weight.shape)
                ind = []
                for ix in idx0: 
                    ii = np.arange(0, ks*ks) + ix*ks*ks
                    ii = ii.tolist()
                    for j in ii:
                        ind.append(j)
                #print(ind)
                assert len(ind) == m1.weight.data.shape[1]

                m1.weight.data = m0.weight.data[:, ind].clone() # [120, 400] [out, in]
                m1.bias.data = m0.bias.data.clone()
            else: 
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()

            fc_count += 1 
            
    return newmodel 

def real_prune_fc(model, mask_ch, mask_fc, ks):
    '''
    This function applies the derived mask of fc layers. It zeros the weights needed to be pruned 
    based on the updated mask of fc layers
    
    :param model: a pytorch model of client
    :param mask_ch: pruning mask of channels 
    :param mask_fc: pruning mask of fully connected layers
    :param ks: kernel size of the model 
    
    :return state_dict: updated (pruned) model state_dict
    '''
    step = 0
    for m in model.modules():
        # We do not prune bias term
        if isinstance(m, nn.Linear):
            tensor = m.weight.data.cpu().numpy()
            weight_dev = m.weight.device
            if step == 0: 
                temp_mask = np.zeros_like(tensor)
                msk = mask_ch[-1].cpu().numpy()
                
                idx0 = np.squeeze(np.argwhere(np.asarray(msk)))
                if idx0.size == 1:
                    idx0 = np.resize(idx0,(1,))
                    
                for i in range(len(idx0)):
                    ix = idx0[i]
                    temp_mask[:, i*ks*ks:i*ks*ks+ks*ks] = mask_fc[step][:, ix*ks*ks:ix*ks*ks+ks*ks]
                    
                m.weight.data = torch.from_numpy(tensor * temp_mask).to(weight_dev)
            else: 
                # Apply new weight and mask
                m.weight.data = torch.from_numpy(tensor * mask_fc[step]).to(weight_dev)
            step += 1
            
    return model.state_dict()

def print_pruning(model, net_glob, is_print=False):
    '''
    This function prints the pruning percentage and status of a given model 
    
    :param model: a pytorch model of the client 
    :param net_glob: original pytorch unpruned model 
    
    :return total pruned percentage: 
    :return total channel pruned percentage: 
    :return total fc pruned percentage: 
    :return total_net_channel_pruning: how much of the network has been pruned from channel 
    :return total_net_fc_pruning: how much of the network has been pruned from fc 
    '''
    net_glob_state_dict = net_glob.state_dict()
    total_param = 0 
    total_channel = 0 
    total_remain = 0 
    total_channel_remain = 0 
    total_fc_weight = 0 
    total_fc_bias = 0 
    total_fc = 0 
    total_remain_fc = 0 
    for name, param in model.named_parameters():
        #print(name, param.size())
        total_param += np.prod(net_glob_state_dict[name].size())
        total_remain += np.prod(param.size())
        if 'conv' in name or 'bn' in name: 
            total_channel += np.prod(net_glob_state_dict[name].size())
            total_channel_remain += np.prod(param.size())
        elif 'fc' in name and 'weight' in name: 
            tensor = param.data.cpu().numpy()
            total_remain_fc += np.count_nonzero(tensor)
            total_fc_weight += np.prod(net_glob_state_dict[name].size())
        elif 'fc' in name and 'bias' in name: 
            total_fc_bias += np.prod(net_glob_state_dict[name].size())
            #total_remain_fc += np.prod(net_glob_state_dict[name].size())
            
    total_fc = total_fc_weight + total_fc_bias
    remain = total_remain_fc + total_channel_remain
    
    if is_print:
        print(f'total pruned: ({100 * (total_param-remain) / total_param:3.2f}% pruned), '
              f'channel pruned: ({100 * (total_channel-total_channel_remain) / total_channel:3.2f}% pruned), ' 
              f'fc pruned: ({100 * (total_fc_weight-total_remain_fc) / total_fc_weight:3.2f}% pruned)')
    
    total_pruning = 100 * (total_param-remain) / total_param
    total_channel_pruning = 100 * (total_channel-total_channel_remain) / total_channel
    total_fc_pruning = 100 * (total_fc_weight-total_remain_fc) / total_fc_weight
    total_net_channel_pruning = 100 * (total_param-total_channel_remain) / total_param
    total_net_fc_pruning = 100 * (total_param-total_remain_fc) / total_param
    
    output = [total_pruning, total_channel_pruning, total_fc_pruning, total_net_channel_pruning, 
              total_net_fc_pruning]
    
    return output

def dist_masks(m1, m2): 
    '''
    Calculates hamming distance of two pruning masks. It averages the hamming distance of all layers and returns it
    
    :param m1: pruning mask 1 
    :param m2: pruning mask 2 
    
    :return average hamming distance of two pruning masks: 
    '''
    temp_dist = []
    for step in range(len(m1)): 
        #1 - float(m1[step].reshape([-1]) == m2[step].reshape([-1])) / len(m2[step].reshape([-1]))
        temp_dist.append(distance.hamming(m1[step].reshape([-1]), m2[step].reshape([-1])))
    dist = np.mean(temp_dist)
    return dist