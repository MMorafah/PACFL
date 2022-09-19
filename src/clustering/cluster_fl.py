import numpy as np
import copy
import torch
import torch.nn.functional as F

from scipy.cluster.hierarchy import dendrogram , linkage
from sklearn.cluster import AgglomerativeClustering


def cluster_logits(clients_idxs, clients, shared_data_loader, args, alpha = 0.5, nclasses=10, nsamples=2500):
    #clients_idxs = np.arange(10)
    
    nclients = len(clients_idxs)
    #nclasses = 10
    #nsamples = 2500
    
    clients_correct_pred_per_label = {idx: {i: 0 for i in range(nclasses)} for idx in clients_idxs}
    clients_pred_per_label = {idx: [] for idx in clients_idxs}
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(shared_data_loader):
            data, target = data.to(args.device), target.to(args.device)
            for idx in clients_idxs: 
                #test_loss = 0
                correct = 0
                
                net = copy.deepcopy(clients[idx].get_net())
                net.to(args.device)
                net.eval()

                output = net(data)
                #test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct = pred.eq(target.data.view_as(pred)).long().cpu().sum()

                clients_pred_per_label[idx].append(F.one_hot(pred.view(-1), num_classes=nclasses))
                clients_correct_pred_per_label[idx][batch_idx] = correct.item()

    A = {idx: torch.stack(clients_pred_per_label[idx]).view(nsamples, nclasses) for idx in clients_idxs}
    clients_similarity = {idx: [] for idx in clients_idxs}
    clusters = []
    
    for idx1 in clients_idxs:
        for idx2 in clients_idxs:
            A1_norm = torch.norm(A[idx1].type(torch.cuda.FloatTensor), 'fro')
            A2_norm = torch.norm(A[idx2].type(torch.cuda.FloatTensor), 'fro')
            A1_A2 = A1_norm * A2_norm
            sim = ((A[idx1]*A[idx2]).sum() / A1_A2).item()
            clients_similarity[idx1].append(sim)
            
    mat_sim = np.zeros([nclients,nclients])
    for i in range(nclients):
        mat_sim[i, :] = np.array(clients_similarity[clients_idxs[i]])
        
    for i in range(nclients):
        temp = np.vstack([np.arange(nclients), mat_sim[i]])
        temp = temp[:, temp[1,:].argsort()[::-1]]
        
        sorted_idx = temp[0]
        sorted_sim = temp[1]
        
        #print(f'temp: {temp}')
        #print(f'sorted_idx[1]: {sorted_idx[1]}, type: {type(sorted_idx[1])}')
        index = 0 
        flag = True 
        found_above_th = False
        cc = []
        cc.append(clients_idxs[i])
        while flag: 
            if sorted_sim[index] >= 0.96:
                if i != int(sorted_idx[index]): 
                    #clusters.append(tuple([clients_idxs[i], clients_idxs[int(sorted_idx[index])]]))
                    cc.append(clients_idxs[int(sorted_idx[index])])
                    found_above_th = True
                index += 1
            elif sorted_sim[index] >= alpha: 
                #clusters.append(tuple([clients_idxs[i], clients_idxs[int(sorted_idx[index])]]))
                cc.append(clients_idxs[int(sorted_idx[index])])
                found_above_th = True
                index += 1
            else: 
                #if not found_above_th:
                    #clusters.append((clients_idxs[i],))
                flag = False
                
            if index == nclients:
                flag = False
                
        clusters.append(copy.deepcopy(cc))
    #print(f'clusters before merge: {clusters}')
    clusters_bm = copy.deepcopy(clusters)
    #clusters = merge_clusters(clusters)
    #print(f'clusters after merge: {clusters}')
    
#     count = 0
#     for el in clusters:
#         count += len(el)
        
    #print(f'count: {count}') 
    
#     assert count == nclients
    
    w_locals_clusters = {i: [] for i in range(len(clusters))}
    for i in range(len(clusters)):
        temp = []
        for idx in clusters[i]:
            temp.append(copy.deepcopy(clients[idx].get_state_dict()))
            
        w_locals_clusters[i].extend(temp)
    
    return clusters, clusters_bm, w_locals_clusters, clients_correct_pred_per_label, clients_similarity, mat_sim, A


def hc_cluster_logits(clients_idxs, clients, shared_data_loader, args, alpha = 5, nclasses=10, nsamples=2500):
    #clients_idxs = np.arange(10)
    
    nclients = len(clients_idxs)
    #nclasses = 10
    #nsamples = 2500
    
    clients_correct_pred_per_label = {idx: {i: 0 for i in range(nclasses)} for idx in clients_idxs}
    clients_pred_per_label = {idx: [] for idx in clients_idxs}
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(shared_data_loader):
            data, target = data.to(args.device), target.to(args.device)
            for idx in clients_idxs: 
                #test_loss = 0
                correct = 0
                
                net = copy.deepcopy(clients[idx].get_net())
                net.to(args.device)
                net.eval()

                output = net(data)
                #test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct = pred.eq(target.data.view_as(pred)).long().cpu().sum()

                clients_pred_per_label[idx].append(F.one_hot(pred.view(-1), num_classes=nclasses))
                clients_correct_pred_per_label[idx][batch_idx] = correct.item()

    A = {idx: torch.stack(clients_pred_per_label[idx]).view(nsamples, nclasses) for idx in clients_idxs}
    clients_similarity = {idx: [] for idx in clients_idxs}
    clusters = []
    
    for idx1 in clients_idxs:
        for idx2 in clients_idxs:
            A1_norm = torch.norm(A[idx1].type(torch.cuda.FloatTensor), 'fro')
            A2_norm = torch.norm(A[idx2].type(torch.cuda.FloatTensor), 'fro')
            A1_A2 = A1_norm * A2_norm
            sim = ((A[idx1]*A[idx2]).sum() / A1_A2).item()
            clients_similarity[idx1].append(sim)
            
    sim_mat = np.zeros([nclients,nclients])
    for i in range(nclients):
        sim_mat[i, :] = np.array(clients_similarity[clients_idxs[i]])
        
    num_clusters=alpha
    Z = linkage(sim_mat, method = 'ward')
    agg_clustering = AgglomerativeClustering(n_clusters = num_clusters, affinity = 'euclidean', linkage = 'ward')

    labels = agg_clustering.fit_predict(sim_mat)    

    clusters = []
    for i in range(num_clusters):
        clusters.append(np.where(labels==i)[0].tolist())

    #print(f'clusters before merge: {clusters}')
    clusters_bm = copy.deepcopy(clusters)
    #clusters = merge_clusters(clusters)
    #print(f'clusters after merge: {clusters}')
    
#     count = 0
#     for el in clusters:
#         count += len(el)
        
    #print(f'count: {count}') 
    
#     assert count == nclients
    
    w_locals_clusters = {i: [] for i in range(len(clusters))}
    for i in range(len(clusters)):
        temp = []
        for idx in clusters[i]:
            temp.append(copy.deepcopy(clients[idx].get_state_dict()))
            
        w_locals_clusters[i].extend(temp)
    
    return clusters, clusters_bm, w_locals_clusters, clients_correct_pred_per_label, clients_similarity, sim_mat, A


def merge_clusters(belist):
    res = list({*map(tuple, map(sorted, belist))})
    
    parents = {}
    def find(i):
        j = parents.get(i, i)
        if j == i:
            return i
        k = find(j)
        if k != j:
            parents[i] = k
        return k
    
    for l in filter(None, res):
        parents.update(dict.fromkeys(map(find, l), find(l[0])))
    merged = {}
    for k, v in parents.items():
        merged.setdefault(find(v), []).append(k)
    return list(merged.values())

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

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
            TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
            FP += 1
        if y_actual[i]==y_hat[i]==0:
            TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
            FN += 1

    return(TP, FP, TN, FN)

def error_clustering(clusters_bm, idxs_users, traindata_cls_counts):
    n = len(idxs_users)
    gt = np.zeros([n, n])
    for i in range(n):
        a = list(traindata_cls_counts[idxs_users[i]].keys())
        for j in range(n):
            b = list(traindata_cls_counts[idxs_users[j]].keys())
            overlap = list(set(a) & set(b))
            #print(f'{i}, {j}: {len(overlap)}')
            if len(overlap) >= (int(np.ceil(len(a)/2))):
                gt[i,j] = 1 
    
    pred = np.zeros([n, n])
    for i in range(n):
        for k in clusters_bm[i]:
            ind = np.where(idxs_users == k)[0]
            pred[i,ind] = 1
    
    clust_err = []
    clust_acc = []
    for i in range(n):
        (TP, FP, TN, FN) = perf_measure(gt[i], pred[i])
        error = FP + FN
        acc = (TP+TN)/(TP+TN+FP+FN)
        clust_err.append(error)
        clust_acc.append(acc)
    
    return np.mean(clust_err), np.mean(clust_acc)
