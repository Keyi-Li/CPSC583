#!/usr/bin/env python
# coding: utf-8

# ## 1. Package & Functions

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  
os.environ["OPENBLAS_NUM_THREADS"] = "4" # use this to avoid sklearn crashing the kernel

import sys
import numpy as np
import pandas as pd
import random
import umap
import matplotlib.pyplot as plt
from itertools import product
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata, beta

from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.cm as cm
import seaborn as sns

import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.autograd import Function
import torch.nn as nn
from torch.nn import Linear, ReLU, Dropout
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool, BatchNorm
from collections import OrderedDict

import networkx as nx

from tqdm.notebook import tqdm as tqdm
from loguru import logger

from module.preprocessing import *


def balanced_sample(old_idx_list, young_idx_list):
    '''
    sample equal number of subgraphs from two given index list, the number is determined by the minimum graph in one group
    '''
    old_range = np.arange(0)
    young_range = np.arange(0)
    for idx in old_idx_list:
        temp = np.arange(idx[0],idx[1])
        old_range = np.hstack([old_range,temp])
    for idx in young_idx_list:
        temp = np.arange(idx[0],idx[1])
        young_range = np.hstack([young_range,temp])
    n = min(old_range.shape[0],young_range.shape[0])
    old_idx = np.random.choice(old_range, n, replace=False)
    young_idx = np.random.choice(young_range, n, replace = False)
    return old_idx, young_idx


def test_accu_auroc(my_net, loader, alpha):
    my_net = my_net.eval()
    n_correct = 0
    all_predictions = []
    all_labels = []
    for data in loader:
        class_out, _ = my_net(data=data, alpha=alpha)
        pred = class_out.data.max(1, keepdim=True)[1]
        prob = torch.softmax(class_out, dim = 1)
        n_correct += pred.eq(data.y.data.view_as(pred)).cpu().sum()
        all_predictions.append(prob.detach().cpu().numpy()[:,1])
        all_labels.append(data.y.cpu().numpy())
        
    accu = n_correct.data.numpy() * 1.0 / len(loader.dataset)

    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    auroc = roc_auc_score(all_labels, all_predictions)
    return accu, auroc, all_predictions


def domain_accu(my_net, loader, alpha):
    my_net = my_net.eval()
    n_correct = 0
    all_predictions = []
    all_labels = []
    for data in loader:
        _, domain_out = my_net(data=data, alpha=alpha)
        pred = domain_out.data.max(1, keepdim=True)[1]
        prob = torch.softmax(domain_out, dim = 1)
        n_correct += pred.eq(data.id.data.view_as(pred)).cpu().sum()
        all_predictions.append(prob.detach().cpu().numpy())
        all_labels.append(data.id.cpu().numpy())
        
    accu = n_correct.data.numpy() * 1.0 / len(loader.dataset)
    
    return accu



def loss_acc_auc(result, title):

    loss_main_source = result["loss_main_source"]
    loss_domain_source = result["loss_domain_source"]
    loss_domain_target = result["loss_domain_target"]

    accu_t_list = result["accu_t_list"]
    accu_s_list = result["accu_s_list"]

    auroc_t_list = result["auroc_t_list"]
    auroc_s_list = result["auroc_s_list"]

    accu_t_domain_list = result["accu_t_domain_list"]
    accu_s_domain_list = result["accu_s_domain_list"]
    
    fig, axs = plt.subplots(1,3, figsize=(16, 5))  
    axs[0].plot(range(len(loss_main_source)), loss_main_source, label='loss_main_source')
    axs[0].plot(range(len(loss_domain_source)), loss_domain_source, label='loss_domain_source')
    axs[0].plot(range(len(loss_domain_target)), loss_domain_target, label='loss_domain_target')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Loss')
    axs[0].legend()
    
    axs[1].plot(range(len(accu_t_list)), accu_t_list, label='Main Test Accuracy')
    axs[1].plot(range(len(accu_s_list)), accu_s_list, label='Main Train Accuracy')
    axs[1].plot(range(len(accu_t_domain_list)), accu_t_domain_list, label='Domain Test Accuracy')
    axs[1].plot(range(len(accu_s_domain_list)), accu_s_domain_list, label='Domain Train Accuracy')
    
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_title('Train and Test Accuracy')
    axs[1].legend()    

    
    axs[2].plot(range(len(auroc_s_list)), auroc_s_list, label='Train AUC')
    axs[2].plot(range(len(auroc_t_list)), auroc_t_list, label='Test AUC')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('AUC score')
    axs[2].set_title('Train and Test AUC')
    axs[2].legend()    

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


# ## 2. Dataset and Split


output_dir = "/data/banach2/kl825/stg_gnn_checked"
os.chdir(output_dir)


dataset_dict = {}
dataset_dict["dataset3k"] = torch.load(os.path.join("dataset","dataset3k_id.pt"))
old_dict, young_dict = datasets_info(dataset_dict)

- data overview:
Overview of Dataset dataset3k:
Total: 4314 subgraphs
Old : 2961 subgraphs
Sample LN00837: 	 Index: [2187 ,2294], Num_graphs: 107   "Age": 86
Sample LN13560: 	 Index: [794 ,837], Num_graphs: 43      "Age": 74
Sample LN22921: 	 Index: [3297 ,4305], Num_graphs: 1008  "Age": 55
Sample LN24336: 	 Index: [837 ,1588], Num_graphs: 751    "Age": 65
Sample LN27766: 	 Index: [1929 ,2187], Num_graphs: 258   "Age": 50
Sample LN291: 	 Index: [0 ,794], Num_graphs: 794           "Age": 71
Young : 1353 subgraphs
Sample LN00560: 	 Index: [2294 ,2914], Num_graphs: 620   "Age": 25
Sample LN21333: 	 Index: [1588 ,1929], Num_graphs: 341   "Age": 22
Sample LN21756: 	 Index: [4305 ,4314], Num_graphs: 9     "Age": 22
Sample LN23574: 	 Index: [2914 ,3297], Num_graphs: 383   "Age": 34- make equal split of old and young, e.g 3 old and 2 young in train and the rest in test
- make sure that train and test are both balanced
- in total, we can have 120 splits, we just randomly sample 5-10 to do "cross-validation"
# - fold_splits contains sets of subgraph index for train and test
# - combinations contain information of which old and which young is used as test


from itertools import product, combinations
dataset = dataset_dict["dataset3k"]
old_combinations = list(combinations(old_dict["dataset3k"].keys(),3))
young_combinations = list(combinations(young_dict["dataset3k"].keys(),2))
combinations = list(product(old_combinations,young_combinations))

import random
random.seed(42)
random.shuffle(combinations)
np.random.seed(5)

fold_idx = []
for old_samples, young_samples in combinations[:5]:
    
    old_train= []
    young_train = []
    old_test = []
    young_test = []
    
    for item in old_dict["dataset3k"].keys():
        if item in old_samples:
            old_train.append(old_dict["dataset3k"][item]["index"])
        else:
            old_test.append(old_dict["dataset3k"][item]["index"])
            
    for item in young_dict["dataset3k"].keys():
        if item in young_samples:
            young_train.append(young_dict["dataset3k"][item]["index"])
        else:
            young_test.append(young_dict["dataset3k"][item]["index"])
    fold_idx.append([old_train,young_train,old_test,young_test])


fold_splits = []
for train_old, train_young, test_old, test_young in fold_idx:
    train_old_idx, train_young_idx = balanced_sample(train_old,train_young)
    test_old_idx, test_young_idx = balanced_sample(test_old, test_young)
    train_idx = np.hstack([train_old_idx,train_young_idx])
    test_idx = np.hstack([test_old_idx,test_young_idx])
    fold_splits.append([train_idx,test_idx])


### 3. Adversarial Training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_list = []
result_dict_list = []

for (train_idx, test_idx), test_sample_name in zip(fold_splits, combinations[:5]):
    print(f"\033[1m Use sample {test_sample_name} as test data \033[0m \t")
    # data loader
    train_set = [dataset[idx].to(device) for idx in train_idx]
    test_set = [dataset[idx].to(device) for idx in test_idx]
    batch_train = int(np.ceil(len(train_set)/8))
    batch_test = int(np.ceil(len(test_set)/8))
    train_loader = DataLoader(train_set, batch_size = batch_train, shuffle = True)
    test_loader = DataLoader(test_set, batch_size = batch_test, shuffle = True)

    # model
    my_net = Model(in_channel = 30, hidden_channel_lspin = [64], hidden_channels_rest = [64,64,32,16,8], out_channel = 3, gnn_layer = 2 , lin_layer = 3, dropout_rate = 0.2, num_sample = 10, lam = 0.0001*30, lam_sta = 0.0001/210,sigma = 0.5)
    lr = 5e-4
    optimizer = torch.optim.Adam(my_net.parameters(), lr=lr)
    loss_main = torch.nn.NLLLoss()
    loss_domain = torch.nn.NLLLoss()
    log_softmax = torch.nn.LogSoftmax(dim=1)
    cuda = True
    
    if cuda:
        my_net = my_net.cuda()
        loss_class = loss_main.cuda()
        loss_domain = loss_domain.cuda()

    for p in my_net.parameters():
        p.requires_grad = True

    # training
    best_accu_t = 0.0
    best_accu_s = 0
    n_epoch = 50
    
    loss_main_source = []
    loss_domain_source = []
    loss_domain_target = []
    accu_t_list = []
    accu_s_list = []
    auroc_t_list = []
    auroc_s_list = []

    accu_t_domain_list = []
    accu_s_domain_list = []
    
    
    for epoch in range(n_epoch):
        # here manually made sure that train_loader and test_loader have the same number of batch
        len_dataloader = min(len(train_loader), len(test_loader))
        data_source_iter = iter(train_loader)
        data_target_iter = iter(test_loader)
    
        s_main_loss = 0
        s_domain_loss = 0
        t_domain_loss = 0
        
        for i in range(len_dataloader):
    
            p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
    
            # training model using source data
            data_source = next(data_source_iter)
    
            optimizer.zero_grad()
                data = data.to(device)
            class_out, sample_out, regularization, stab_regularization = my_net(data=data_source, alpha=alpha)
            err_s_main = loss_main(log_softmax(class_out), data_source.y)
            err_s_domain = loss_domain(log_softmax(sample_out), data_source.id)
            s_main_loss += err_s_main.item()
            s_domain_loss += err_s_domain.item()
    
            # training model using target data (for DANN)
            data_target = next(data_target_iter)
    
            _, sample_out = my_net(data=data_target, alpha=alpha)
            err_t_domain= loss_domain(log_softmax(sample_out), data_target.id)
            t_domain_loss += err_t_domain.item()
    
            total_loss = err_s_main + err_s_domain + err_t_domain + regularization + stab_regularization
            
            total_loss.backward()
            optimizer.step()
    
        s_main_loss /= len(train_loader.dataset)
        s_domain_loss /= len(train_loader.dataset)
        t_domain_loss /= len(test_loader.dataset)
        
        loss_main_source.append(s_main_loss)
        loss_domain_source.append(s_domain_loss)
        loss_domain_target.append(t_domain_loss)
        
        print(f'Epoch: {epoch:03d}, source main loss: {s_main_loss:.4f}, source domain loss: {s_domain_loss:.4f}, target domain loss:{t_domain_loss:.4f}')
     
        accu_s, auroc_s, _ = test_accu_auroc(my_net, train_loader, alpha)
        accu_t, auroc_t, _ = test_accu_auroc(my_net, test_loader, alpha)

        accu_t_list.append(accu_t)
        accu_s_list.append(accu_s)
        auroc_t_list.append(auroc_t)
        auroc_s_list.append(auroc_s)
        
        print(f'Epoch: {epoch:03d}, accuracy train: {accu_s:.4f}, AUROC train: {auroc_s:.4f}, accuracy test: {accu_t:.4f}, AUROC test: {auroc_t:.4f}')

        accu_s_domain = domain_accu(my_net, train_loader, alpha)
        accu_t_domain = domain_accu(my_net, test_loader, alpha)

        accu_t_domain_list.append(accu_t_domain)
        accu_s_domain_list.append(accu_s_domain)


    result_dict = {
            "loss_main_source": loss_main_source,
            "loss_domain_source": loss_domain_source,
            "loss_domain_target": loss_domain_target,
            "accu_t_list": accu_t_list,
            "accu_s_list": accu_s_list,
            "auroc_t_list": auroc_t_list,
            "auroc_s_list": auroc_s_list,
            "accu_t_domain_list": accu_t_domain_list,
            "accu_s_domain_list":accu_s_domain_list}
    
    result_dict_list.append(result_dict)
    model_list.append(my_net)

    loss_acc_auc(result_dict, f"Test Data: {test_sample_name}")



for result_dict, test_sample_name in zip(result_dict_list, combinations[:5]):
    loss_acc_auc(result_dict, f"Test Data: {test_sample_name}")


