import numpy as np 
import torch 
import torch.nn as nn

def Sub_FedAVG_U(w_server, w_clients, masks):
    '''
    This function performs Sub-FedAvg-U (for unstructured pruning) as stated in the paper. 
    This function updates the server model based on Sub-FedAvg. It is called at the end of each round. 
    
    :param w_server: server model's state_dict 
    :param w_clients: list of clients' model state_dict to be averaged 
    :param masks: list of clients' pruning masks to be averaged 
    
    :return w_server: updated server model's state_dict
    '''
    step = 0
    for name in w_server.keys():
        
        if 'weight' in name:
            
            weight_dev = w_server[name].device
            
            indices = []
            count = np.zeros_like(masks[0][step].reshape([-1]))
            avg = np.zeros_like(w_server[name].data.cpu().numpy().reshape([-1]))
            for i in range(len(masks)): 
                count += masks[i][step].reshape([-1])
                avg += w_clients[i][name].data.cpu().numpy().reshape([-1])
            
            final_avg = np.divide(avg, count)         
            ind = np.isfinite(final_avg)
            
            temp_server = w_server[name].data.cpu().numpy().reshape([-1])
            temp_server[ind] = final_avg[ind]
            
            #print(f'Name: {name}, NAN: {np.mean(np.isnan(temp_server))}, INF: {np.mean(np.isinf(temp_server))}')
            
            shape = w_server[name].data.cpu().numpy().shape
            w_server[name].data = torch.from_numpy(temp_server.reshape(shape)).to(weight_dev)            
            
            step += 1
        else: 
            
            avg = np.zeros_like(w_server[name].data.cpu().numpy().reshape([-1]))
            for i in range(len(masks)): 
                avg += w_clients[i][name].data.cpu().numpy().reshape([-1])
            avg /= len(masks)
            
            #print(f'Name: {name}, NAN: {np.mean(np.isnan(avg))}, INF: {np.mean(np.isinf(avg))}')
            weight_dev = w_server[name].device
            shape = w_server[name].data.cpu().numpy().shape
            w_server[name].data = torch.from_numpy(avg.reshape(shape)).to(weight_dev)            
            
    return w_server


def Sub_FedAvg_U_initial(mask, model, w_server):  
    '''
    This function initializes each client's subnetwork by the server's model at the begining of each round. 
    It is called at the begining of each round 
    
    :param mask: pruning mask of the client receiving the initial from the server 
    :param model: client model 
    :param w_server: server model's state_dict 
    
    :return updated client model's state_dict: 
    '''
    model.load_state_dict(w_server)
    step = 0
    for name, param in model.named_parameters(): 
        if "weight" in name: 
            weight_dev = param.device
            param.data = torch.from_numpy(mask[step] * w_server[name].cpu().numpy()).to(weight_dev)
            step = step + 1
        if "bias" in name:
            param.data = w_server[name]
    return model.state_dict()
