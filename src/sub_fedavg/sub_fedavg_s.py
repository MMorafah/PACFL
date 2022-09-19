import numpy as np 
import torch 
import torch.nn as nn

def Sub_FedAVG_S(w_server, w_clients, masks_ch, masks_fc, model, ks, in_ch):
    '''
    This function performs Sub-FedAvg-S (structured and unstructured pruning--Hybrid) as stated in the paper. 
    This function updates the server model based on Sub-FedAvg. It is called at the end of each round. 
    
    :param w_server: server model's state_dict 
    :param w_clients: list of clients' model state_dict to be averaged 
    :param masks_ch: list of clients' pruning masks of channels to be averaged 
    :param masks_fc: list of clients' pruning masks of fcs to be averaged
    :param model: the original model model (net_glob)
    :param ks: kernel size of the model 
    :param in_ch: number of input channel to the 1st layer 
    
    :return w_server: updated server model's state_dict
    '''
    step_ch = 0 
    step_fc = 0 
    conv_layer = 1 
    bn_layer = 1 
    fc_layer = 1 
    
    start_masks = [torch.ones(in_ch) for i in range(len(masks_ch))]
    end_masks = [masks_ch[i][step_ch] for i in range(len(masks_ch))]
    for m0 in (model.modules()):
        
        if isinstance(m0, nn.BatchNorm2d):
            #print(m0)
            name_weight = 'main.bn{}.weight'.format(bn_layer)
            name_bias = 'main.bn{}.bias'.format(bn_layer)
            name_running_mean = 'main.bn{}.running_mean'.format(bn_layer)
            name_running_var = 'main.bn{}.running_var'.format(bn_layer)

            names = [name_weight, name_bias, name_running_mean, name_running_var]

            for name in names: 
                #print(name)
                weight_dev = w_server[name].device

                count = np.zeros_like(w_server[name].data.cpu().numpy())
                avg = np.zeros_like(w_server[name].data.cpu().numpy())
                
                for i in range(len(masks_ch)): 
                    count += end_masks[i].cpu().numpy()
                    
                    idx0 = np.squeeze(np.argwhere(np.asarray(end_masks[i].cpu().numpy())))
                    if idx0.size == 1:
                        idx0 = np.resize(idx0,(1,))
                        
                    assert idx0.shape == w_clients[i][name].data.cpu().numpy().shape
                    assert avg[idx0.tolist()].shape == w_clients[i][name].data.cpu().numpy().shape
                    
                    for j in range(len(idx0)):
                        ix0 = idx0[j]
                        avg[ix0] += w_clients[i][name][j].data.cpu().numpy()

                avg_reshape = avg.reshape([-1]) 
                count_reshape = count.reshape([-1])
                final_avg = np.divide(avg_reshape, count_reshape)

                ind = np.isfinite(final_avg)

                server_reshape = w_server[name].data.cpu().numpy().reshape([-1])
                server_reshape[ind] = final_avg[ind]

                shape = w_server[name].data.cpu().numpy().shape
                w_server[name].data = torch.from_numpy(server_reshape.reshape(shape)).to(weight_dev)  

                #print(f'Name: {name}, NAN: {np.mean(np.isnan(server_reshape))}, INF: {np.mean(np.isinf(server_reshape))}')

            ## Updating step 
            step_ch += 1 
            
            start_masks = end_masks
            if step_ch < len(masks_ch[0]):
                end_masks = [masks_ch[i][step_ch] for i in range(len(masks_ch))]
            ## Updating Layer 
            bn_layer += 1 

        elif isinstance(m0, nn.Conv2d):
            #print(m0)
            name_weight = 'main.conv{}.weight'.format(conv_layer)
            name_bias = 'main.conv{}.bias'.format(conv_layer)

            names = [name_weight, name_bias]

            for name in names: 
                #print(name)
                weight_dev = w_server[name].device

                avg = np.zeros_like(w_server[name].data.cpu().numpy()) # [6, 3, 5, 5] [6]
                count = np.zeros_like(w_server[name].data.cpu().numpy()) # [6, 3, 5, 5] [6]
                
                mm = np.zeros_like(w_server[name].data.cpu().numpy()) # [6, 3, 5, 5] [6]
                temp_masks = [np.zeros_like(mm) for _ in range(len(masks_ch))] 
                
                for i in range(len(masks_ch)): 
                    temp_client = np.zeros_like(w_server[name].data.cpu().numpy()) # [6, 3, 5, 5] [6]

                    idx0 = np.squeeze(np.argwhere(np.asarray(start_masks[i].cpu().numpy())))
                    if idx0.size == 1:
                        idx0 = np.resize(idx0,(1,))
                        
                    idx1 = np.squeeze(np.argwhere(np.asarray(end_masks[i].cpu().numpy()))) # IND OUT 
                    if idx1.size == 1:
                        idx1 = np.resize(idx1,(1,))
                         
                    if name == name_weight: 
                        #print(f'Client Shape: {w_clients[i][name].data.cpu().numpy().shape}, Supposed Shape: {[len(idx1),
                        #len(idx0), ks, ks]}')
                        assert w_clients[i][name].data.cpu().numpy().shape == (len(idx1), len(idx0), ks, ks)
                        
                        for j in range(len(idx0)): # [8, 3, 5, 5]
                            ix0 = idx0[j]
                            for k in range(len(idx1)):
                                ix1 = idx1[k]
                                #print(f'Server Shape: {avg[idx0, ix].shape} ')
                                assert temp_client[ix1, ix0].shape == (ks, ks)
                                assert w_clients[i][name][k, j].cpu().numpy().shape == (ks, ks)

                                temp_client[ix1, ix0] = w_clients[i][name][k, j].cpu().numpy() # [out_channel, in_channel, k, k]
                                temp_masks[i][ix1, ix0] = 1 
                                
                        non_zero_ind = np.nonzero(temp_masks[i].reshape([-1]))[0]
                        assert len(non_zero_ind) == len(w_clients[i][name].data.cpu().numpy().reshape([-1]))
                        #temp_masks[i][non_zero_ind.tolist()] = 1 
                        count += temp_masks[i]
                        avg += temp_client
                        
                    elif name == name_bias: 
                        #print(f'Client Shape: {w_clients[i][name].data.cpu().numpy().shape}, Supposed Shape: {[len(idx1)]}')

                        assert w_clients[i][name].data.cpu().numpy().shape == (len(idx1), )
                        
                        for j in range(len(idx1)):
                            ix1 = idx1[j]
                            temp_client[ix1] = w_clients[i][name][j].data.cpu().numpy()
                            temp_masks[i][ix1] = 1 
                            
                        non_zero_ind = np.nonzero(temp_masks[i].reshape([-1]))[0]
                        assert len(non_zero_ind) == len(w_clients[i][name].data.cpu().numpy().reshape([-1]))
                        #temp_masks[i][non_zero_ind.tolist()] = 1 
                        count += temp_masks[i]
                        avg += temp_client
                        
                avg_reshape = avg.reshape([-1])
                count_reshape = count.reshape([-1])
                #print(f'Name: {name}, AVG: {avg_reshape}, \n count_reshape: {count_reshape}')
                
                final_avg = np.divide(avg_reshape, count_reshape)  # 6*3*5*5
                #print(f'Name: {name}, Final AVG: {final_avg}')

                ind = np.isfinite(final_avg)

                server_reshape = w_server[name].data.cpu().numpy().reshape([-1])
                server_reshape[ind] = final_avg[ind]

                shape = w_server[name].data.cpu().numpy().shape
                w_server[name].data = torch.from_numpy(server_reshape.reshape(shape)).to(weight_dev)

                #print(f'Name: {name}, NAN: {np.mean(np.isnan(w_server[name].cpu().numpy()))}, INF: 
                #{np.mean(np.isinf(w_server[name].cpu().numpy()))}')

            ## Updating Layer 
            conv_layer += 1 

        elif isinstance(m0, nn.Linear):
            #print(m0)

            name_weight = 'fc{}.weight'.format(fc_layer)
            name_bias = 'fc{}.bias'.format(fc_layer)

            names = [name_weight, name_bias]
            
            ## Weight 
            name = names[0]
            weight_dev = w_server[name].device

            avg = np.zeros_like(w_server[name].data.cpu().numpy()) # [120, 400]
            count = np.zeros_like(w_server[name].data.cpu().numpy())
                
            for i in range(len(masks_fc)): 
                count += masks_fc[i][step_fc]
                
                if fc_layer == 1: 
                    temp_client = np.zeros_like(w_server[name].data.cpu().numpy())

                    idx0 = np.squeeze(np.argwhere(np.asarray(end_masks[i])))
                    if idx0.size == 1:
                        idx0 = np.resize(idx0,(1,))

                    assert w_clients[i][name].data.cpu().numpy().shape == (120, len(idx0) * ks * ks)

                    for j in range(len(idx0)): 
                        ix0 = idx0[j]
                        for k in range(ks*ks):
                            temp_client[:, ix0*ks*ks + k] = w_clients[i][name][:,j*ks*ks + k].data.cpu().numpy()

                    avg += temp_client
                    
                else: 
                    avg += w_clients[i][name].data.cpu().numpy()
                
            avg_reshape = avg.reshape([-1])
            count_reshape = count.reshape([-1])

            final_avg = np.divide(avg_reshape, count_reshape)

            ind = np.isfinite(final_avg)

            server_reshape = w_server[name].data.cpu().numpy().reshape([-1])
            server_reshape[ind] = final_avg[ind]

            shape = w_server[name].data.cpu().numpy().shape
            w_server[name].data = torch.from_numpy(server_reshape.reshape(shape)).to(weight_dev)

            #print(f'Name: {name}, NAN: {np.mean(np.isnan(w_server[name].cpu().numpy()))}, INF: 
            #{np.mean(np.isinf(w_server[name].cpu().numpy()))}')

            ## Bias 
            name = names[1]
            avg = np.zeros_like(w_server[name].data.cpu().numpy())
            for i in range(len(masks_fc)): 
                avg += w_clients[i][name].data.cpu().numpy()

            final_avg = np.divide(avg, len(masks_fc))

            w_server[name].data = torch.from_numpy(final_avg).to(weight_dev) 
            
            #print(f'Name: {name}, NAN: {np.mean(np.isnan(w_server[name].cpu().numpy()))}, INF: 
            #{np.mean(np.isinf(w_server[name].cpu().numpy()))}')

            ## Updating Layer
            fc_layer += 1 
            step_fc += 1 

    return w_server

def Sub_FedAvg_S_initial(mask_ch, mask_fc, model, w_server, in_ch, ks):  
    '''
    This function initializes each client's subnetwork by the server's model at the begining of each round. 
    It is called at the begining of each round 
    
    :param mask_ch: pruning channel mask of the client receiving the initial from the server 
    :param mask_fc: pruning fc mask of the client receiving the initial from the server 
    :param model: client model 
    :param w_server: server model's state_dict 
    :param ks: kernel size of the model 
    :param in_ch: number of input channel to the 1st layer
    
    :return updated client model's state_dict: 
    '''
    step_ch = 0
    step_fc = 0 
    conv_layer = 1 
    bn_layer = 1 
    fc_layer = 1 
    
    start_mask = torch.ones(in_ch).cpu().numpy()
    end_mask = mask_ch[step_ch].cpu().numpy()
    for name, param in model.named_parameters(): 
        if 'bn' in name and 'weight' in name: 
            idx0 = np.squeeze(np.argwhere(np.asarray(end_mask)))
            if idx0.size == 1:
                idx0 = np.resize(idx0,(1,))
            weight_dev = param.device
            param.data = w_server[name][idx0.tolist()].to(weight_dev)
        elif 'bn' in name and 'bias' in name: 
            idx0 = np.squeeze(np.argwhere(np.asarray(end_mask)))
            if idx0.size == 1:
                idx0 = np.resize(idx0,(1,))
            weight_dev = param.device
            param.data = w_server[name][idx0.tolist()].to(weight_dev)
            
            step_ch += 1 
            start_mask = end_mask
            
            if step_ch < len(mask_ch): 
                end_mask = mask_ch[step_ch].cpu().numpy()
                
            bn_layer += 1
        elif 'conv' in name and 'weight' in name: 
            weight_dev = param.device

            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask)))
            if idx0.size == 1:
                idx0 = np.resize(idx0,(1,))
            
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask)))
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))

            assert param.data.shape == (len(idx1), len(idx0), ks, ks)

            for j in range(len(idx0)):
                for k in range(len(idx1)):
                    ix0 = idx0[j]
                    ix1 = idx1[k]
                    param.data[k,j] = w_server[name][ix1, ix0].to(weight_dev)

        elif 'conv' in name and 'bias' in name: 
            idx0 = np.squeeze(np.argwhere(np.asarray(end_mask)))
            if idx0.size == 1:
                idx0 = np.resize(idx0,(1,))
                
            assert param.data.shape == (len(idx0),)
            weight_dev = param.device
            param.data = w_server[name][idx0.tolist()].to(weight_dev)
            
            conv_layer += 1 
        elif 'fc' in name and 'weight' in name: 
            if fc_layer == 1: 
                idx0 = np.squeeze(np.argwhere(np.asarray(end_mask)))
                if idx0.size == 1:
                    idx0 = np.resize(idx0,(1,))
                    
                weight_dev = param.device
                for j in range(len(idx0)):
                    ix0 = idx0[j]
                    ind = np.arange(0, ks*ks) + ix0*ks*ks
                    temp = mask_fc[step_fc] * w_server[name].data.cpu().numpy()
                    param.data[:, j*ks*ks:j*ks*ks+ks*ks] = torch.from_numpy(temp[:, ind]).to(weight_dev)
            else: 
                weight_dev = param.device
                param.data = torch.from_numpy(mask_fc[step_fc] * w_server[name].data.cpu().numpy()).to(weight_dev)  ####
        elif 'fc' in name and 'bias' in name:
            weight_dev = param.device
            param.data = w_server[name].to(weight_dev)
            
            fc_layer += 1 
            step_fc += 1 
            
    return model.state_dict()