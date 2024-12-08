#!/usr/bin/env python
# coding: utf-8


get_ipython().run_line_magic('env', 'OPENBLAS_NUM_THREADS=4')

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # check before use
os.environ["OPENBLAS_NUM_THREADS"] = "4" # use this to avoid sklearn crashing the kernel

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
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm

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


class GCN(torch.nn.Module):
  def __init__(self,in_channel:int, hidden_channel_lspin: list, hidden_channels_rest: list, out_channel: int, gnn_layer: int, lin_layer: int, dropout_rate: float, num_sample :int, lam: float, lam_sta: float, sigma: float, mode: str, graph_embed: bool):
        super().__init__()
        torch.manual_seed(12345)
        self.num_sample = num_sample #for one hot encoding
        self.lam = lam
        self.lam_sta = lam_sta
        self.mode = mode
        self.graph_embed = graph_embed
        
        self.Lspin = LspinBlock(in_channel, hidden_channel_lspin, sigma)
        
        self.GNN = ModuleList()
        for i in range(gnn_layer):
            self.GNN.append(GCNBlock(in_channel, hidden_channels_rest[i], dropout_rate))
            in_channel =  hidden_channels_rest[i]

        self.linear_sample = nn.Linear(num_sample,in_channel)
        
        in_channel = 2*in_channel
        
        self.MLP = ModuleList()
        for j in range(gnn_layer,gnn_layer+lin_layer):
            self.MLP.append(LinearBlock(in_channel,hidden_channels_rest[j],dropout_rate))
            in_channel = hidden_channels_rest[j]

        self.predictor = nn.Linear(in_channel, out_channel)

        self.class_classifier = nn.Sequential(
            OrderedDict([
            ("c_fc1",Linear(hidden_channels, 16)),
            ("c_dropout1", Dropout(0.25)),
            ("c_relu1",nn.ReLU()),
            ("c_fc2",Linear(16,4)),
            ("c_dropout2", Dropout(0.25)),
            ("c_relu2",nn.ReLU()),
            ("c_fc3", Linear(4,2)),
            ("c_softmax",nn.LogSoftmax(dim=1))
            ])
        )

        self.sample_classifier = nn.Sequential(
            OrderedDict([
            ("s_fc1",Linear(hidden_channels, 32)),
            ("s_dropout1", Dropout(0.25)),
            ("s_relu1",nn.ReLU()),
            ("s_fc2",Linear(32,16)),
            ("c_dropout2", Dropout(0.25)),
            ("s_relu2",nn.ReLU()),
            ("s_fc3", Linear(16,10)),
            ("s_softmax",nn.LogSoftmax(dim=1))
            ])
        )
        

    def forward(self, data):
raw_x, edge_idx, sample_id, batch = data.x, data.edge_index, data.sample_id, data.batch
        sample_id =sample_id.T
        sample_id = F.one_hot(sample_id, self.num_sample).float()
        sample_id = sample_id.to(self.device)

        if self.graph_embed:
            graph_embedding = self.Lspin.get_graph_embedding(raw_x, batch)

        x, mu, z = self.Lspin(raw_x, batch, graph_embedding)

        regularization = self.Lspin.regularizer(mu)* self.lam

        if self.mode not in ["same", "diff"]:
            stability_regularization = torch.tensor(0)
        else:
            stability_regularization = self.Lspin.stability_regularizer(graph_embedding,z,self.mode) * self.lam_sta
        
        for gcn_block in self.GNN:
            x = gcn_block(x, edge_idx)

        unbatched_x = utils.unbatch(x, batch)
        x = torch.stack([sample[0] for sample in unbatched_x])

        unbatched_y = utils.unbatch(data.y,batch)
        y = torch.stack([sample[0] for sample in unbatched_y])

        x_1 = self.linear_sample(sample_id)
        x = torch.cat((x, x_1), dim=-1)

        for linear_block in self.MLP:
            x = linear_block(x)
            
        n_embed = x
        g_embed = global_max_pool(n_embed, batch)
        r_g_embed = ReverseLayerF.apply(g_embed, alpha)
        class_out = self.class_classifier(g_embed)
        sample_out = self.sample_classifier(r_g_embed)

        return class_out, sample_out


def train_epoch(model, criterion, optimizer,train_loader):
    model.train()
    total_loss = 0
    for data in train_loader:  
         out = model(data)  
         loss = criterion(out, data.y)
         loss.backward()  
         optimizer.step() 
         optimizer.zero_grad()
         total_loss += loss.item() 

    return model, optimizer, total_loss/len(train_loader.dataset)

def test(loader, model, criterion):
     model.eval()
     correct = 0
     total_loss = 0
    
     for data in loader:  
         out = model(data)  
         loss = criterion(out, data.y)
         total_loss += loss.item() 
         pred = out.argmax(dim = 1)  
         correct += int((pred == data.y).sum())  
     return correct / len(loader.dataset), total_loss/len(loader.dataset)

from sklearn.metrics import roc_auc_score
def evaluate_auroc(model, loader):
    model.eval()  
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for data in loader:
            out = model(data)
            prob = torch.softmax(out, dim = 1) 
            all_predictions.append(prob.cpu().numpy()[:,1]) # take the prob of class with larger label in binary classification
            all_labels.append(data.y.cpu().numpy())

    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    auroc = roc_auc_score(all_labels, all_predictions)
    return auroc, all_predictions

def auroc_cal(model, loader):
    model.eval()  
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for data in loader:
            out = model(data)
            prob = torch.softmax(out, dim = 1) 
            all_predictions.append(prob.cpu().numpy()[:,1]) # take the prob of class with larger label in binary classification
            all_labels.append(data.y.cpu().numpy())

    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    auroc = roc_auc_score(all_labels, all_predictions)
    return auroc, all_predictions, all_labels


def train_test_wrapper(model, criterion, optimizer, train_loader, test_loader, n_epoch):
    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []
    train_auc_list = []
    test_auc_list = []
    
    for epoch in range(1, n_epoch+1):
        model, optimizer, train_loss = train_epoch(model, criterion, optimizer, train_loader)
        train_loss_list.append(train_loss) 
        train_acc, _ = test(train_loader, model, criterion)
        train_acc_list.append(train_acc)
        test_acc, test_loss =  test(test_loader, model, criterion)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)
        auc_train,_ = evaluate_auroc(model, train_loader)
        auc_test,_ = evaluate_auroc(model, test_loader)
        train_auc_list.append(auc_train)
        test_auc_list.append(auc_test)


        if epoch % 1 == 0:
            print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train AUC :{auc_train:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test AUC: {auc_test:.4f}')
            
    fig, axs = plt.subplots(3, figsize=(15, 8))  
    axs[0].plot(range(len(train_loss_list)), train_loss_list, label='Train Loss')
    axs[0].plot(range(len(test_loss_list)), test_loss_list, label='Test Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Train and Test Loss')
    axs[0].legend()
    
    axs[1].plot(range(len(train_acc_list)), train_acc_list, label='Train Accuracy')
    axs[1].plot(range(len(test_acc_list)), test_acc_list, label='Test Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_title('Train and Test Accuracy')
    axs[1].legend()    

    
    axs[2].plot(range(len(train_auc_list)), train_auc_list, label='Train AUC')
    axs[2].plot(range(len(test_auc_list)), test_auc_list, label='Test AUC')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('AUC score')
    axs[2].set_title('Train and Test AUC')
    axs[2].legend()    
    plt.tight_layout()
    plt.show()
        

    result_dict = {"train_loss": train_loss_list,
                   "test_loss": test_loss_list,
                   "train_acc": train_acc_list,
                   "test_acc": test_acc_list,
                  "train_auc": train_auc_list,
                  "test_auc": test_auc_list}

    return model, optimizer, result_dict


def loss_acc_auc(result, title):
    train_loss_list = result['train_loss']
    test_loss_list = result['test_loss']
    train_acc_list = result['train_acc']
    test_acc_list = result['test_acc']
    train_auc_list = result['train_auc']
    test_auc_list = result['test_auc']

    fig, axs = plt.subplots(1,3, figsize=(16, 5))  
    axs[0].plot(range(len(train_loss_list)), train_loss_list, label='Train Loss')
    axs[0].plot(range(len(test_loss_list)), test_loss_list, label='Test Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Train and Test Loss')
    axs[0].legend()
    
    axs[1].plot(range(len(train_acc_list)), train_acc_list, label='Train Accuracy')
    axs[1].plot(range(len(test_acc_list)), test_acc_list, label='Test Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_title('Train and Test Accuracy')
    axs[1].legend()    

    
    axs[2].plot(range(len(train_auc_list)), train_auc_list, label='Train AUC')
    axs[2].plot(range(len(test_auc_list)), test_auc_list, label='Test AUC')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('AUC score')
    axs[2].set_title('Train and Test AUC')
    axs[2].legend()    

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def spatial_feature_visual(df, feature, title, show = True, ax = None, vmin = 0, vmax = 1, figsize = (10,8), cmap = "bwr", alpha=1):
    x = df["X"]
    y = df["Y"]
    feature = df[feature]
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        show = True

    norm = Normalize(vmin=vmin, vmax=vmax)
    scatter = ax.scatter(x, y, c=feature, cmap= cmap, s=1, norm=norm, alpha=alpha)
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f"{title}")
    ax.grid(True)

    if show:
        plt.show()
    else:
        return ax

def spatial_categorical_visual(df, feature, title, show = True, ax = None, vmin = 0, vmax = 1, figsize = (10,8), palette="tab20",alpha =1):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        show = True
    scatter = sns.scatterplot(df, x = "X", y = "Y" , hue=feature, s=1, palette= palette, legend=False, edgecolor=None, hue_order=np.sort(df[feature].unique()), ax=ax, alpha=alpha)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f"{title}")
    ax.grid(True)
    
    if show:
        plt.show()
    else:
        return ax


def pred_prob(model, loader, sample_pred_dict, common_feature):
    model.eval()
    with torch.no_grad():
        for data in loader:
            pred = model(data)
            prob = torch.softmax(pred, dim =1)
            old_prob = prob[:,1]
            X = unbatch_x(data)
            for x, y, pred_current, sample, index, spatial in zip(X, data.y.to('cpu'), old_prob.to('cpu'), data.sample, data.graph_index, data.spatial):
                sample_pred = pd.DataFrame(x.to("cpu"), columns = common_feature)
                sample_pred["X"] = spatial[:,0]
                sample_pred["Y"] = spatial[:,1]
                sample_pred["label"] = y.item()
                sample_pred["graph_idx"] = index
                sample_pred["pred_old_prob"] = pred_current.item()
                
                if sample in sample_pred_dict.keys():
                    sample_pred_dict[sample] = pd.concat([sample_pred_dict[sample], sample_pred], axis=0)
                else:
                    sample_pred_dict[sample] = sample_pred
                
    return sample_pred_dict

def unbatch_x(data):
    batch_size = data.batch.max().item() + 1
    unbatched_x = []

    for batch_idx in range(batch_size):
        x = data.x[data.batch == batch_idx]
        unbatched_x.append(x)

    return unbatched_x



def embed(model,loader):
    embedding = np.empty((0, 64))
    label = []
    age = []
    model.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for data in loader:
                latent = model.embed(data).cpu()
                embedding = np.vstack((embedding,latent))
                label += data.sample
                age.append(data.age)
    return embedding, label, torch.hstack(age)
    
def temp_umap(model,loader,title):
    embedding, label, age = embed(model,loader)
    reducer = umap.UMAP()
    scaled_embedding = StandardScaler().fit_transform(embedding)
    embedding = reducer.fit_transform(scaled_embedding)
    color_mapping = {
    'LN291': 'red',
    'LN13560': 'blue',
    'LN24336': 'green',
    'LN21333': 'yellow',
    'LN27766': 'purple',
    'LN00837': 'orange',
    'LN00560': 'cyan',
    'LN23574': 'magenta',
    'LN22921': 'pink',
    'LN21756': 'teal'}
    color_mapping2 = {"Old" : "red", "Young" : "blue"}
    
    fig, ax = plt.subplots(1,2, figsize=(16, 6))
    for sample in color_mapping.keys():
        idx = np.array(label) == sample
        sample_embedding = embedding[idx,:]
        ax[0].scatter(sample_embedding[:, 0], sample_embedding[:, 1], label=f"{sample}", color = color_mapping[sample], s=1)
    ax[0].legend(bbox_to_anchor=(-0.5, 1), loc='upper left')

    old_idx = age >=50
    sample_embedding = embedding[old_idx,:]
    ax[1].scatter(sample_embedding[:, 0], sample_embedding[:, 1], label = "Old", color = "red", s= 1)
    young_idx = age <50
    sample_embedding = embedding[young_idx,:]
    ax[1].scatter(sample_embedding[:, 0], sample_embedding[:, 1], label = "Young", color = "blue", s= 1)
    ax[1].legend(bbox_to_anchor= (1.05, 1), loc = "upper left")
    plt.suptitle(f"{title}")
    plt.show()


# ## 2. Data Split - on subgraphs with roughly 3k cells


output_dir = "/data/banach2/kl825/stg_gnn_checked"
os.chdir(output_dir)

dataset_dict = {}
dataset_dict["dataset3k"] = torch.load(os.path.join("dataset","dataset3k.pt"))
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
Sample LN23574: 	 Index: [2914 ,3297], Num_graphs: 383   "Age": 34
# In[24]:


dataset_dict['dataset3k'][0].feature_name



print(combinations)


# - group old into 4 groups and young into three groups
# - construct test by sampling one old group and one young group, and do a balanced sample
# - construct train by sampling the rest and form a balanced sample
# - in total 3*4 = 12 train-test splits are constructed

old = { "LN00837,LN13560,LN27766" : [[2187 ,2294],[794 ,837], [1929 ,2187]],
         "LN22921" : [[3297 ,4305]],
         "LN24336" : [[837 ,1588]],
         "LN291" : [[0 ,794]]}

young = {"LN00560" : [[2294 ,2914]], 
        "LN21333,LN21756": [[1588 ,1929],[4305 ,4314]],
        "LN23574": [[2914 ,3297]]}


dataset = dataset_dict["dataset3k"]

from itertools import product
fold_splits = []
combinations = list(product(list(old.keys()), list(young.keys())))
np.random.seed(42)
for item in combinations:
    test_old_idx, test_young_idx = balanced_sample(old[item[0]],young[item[1]])
    rest_old_list =[]
    for key in old.keys():
        if key not in item:
            rest_old_list += old[key]
    rest_young_list = []
    for key in young.keys():
        if key not in item:
            rest_young_list += young[key]
    train_old_idx, train_young_idx = balanced_sample(rest_old_list, rest_young_list)
    train_idx = np.hstack([train_old_idx,train_young_idx])
    test_idx = np.hstack([test_old_idx,test_young_idx])
    fold_splits.append([train_idx,test_idx])


# ## 3. Model Training


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_list = []
result_list = []
for (train_idx, test_idx), test_sample_name in zip(fold_splits, combinations):
    print(f"\033[1m Use sample {test_sample_name} as test data \033[0m \t")
    
    train_set = [dataset[idx].to(device) for idx in train_idx]
    test_set = [dataset[idx].to(device) for idx in test_idx]
    train_loader = DataLoader(train_set, batch_size = 128, shuffle = True)
    test_loader = DataLoader(test_set, batch_size = 64, shuffle = False)
    
    model = GCN(hidden_channels=64)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    model, optimizer, result_dict = train_test_wrapper(model, criterion, optimizer, train_loader, test_loader, n_epoch = 100)
    
    model_list.append(model)
    result_list.append(result_dict)


for result, title in zip(result_list, combinations):
    loss_acc_auc(result, f"Test Data: {title}")



sample_meta = {}
for item in os.listdir(os.path.join(output_dir,"sample_level","dataset3k")):
    sample_name = item.split("_")[0]
    sample_meta[sample_name] = pd.read_csv(os.path.join(output_dir,"sample_level","dataset3k",item),index_col = 0)


for i in range(len(dataset)):
   dataset[i] = dataset[i].to(device)
all_loader = DataLoader(dataset, batch_size=64, shuffle = False)
sample_pred_list = []
for model, test_sample_name in zip(model_list, combinations):
    logger.info(f"Processing model which uses {test_sample_name} as test data")
    sample_pred_dict = pred_prob(model, all_loader, {}, dataset[0].feature_name)
    sample_pred_list.append(sample_pred_dict)


fig, axs = plt.subplots(1,2, figsize=(10, 4))
spatial_feature_visual(sample_pred_list[8]["LN23574"], "pred_old_prob", "Prob of Old", cmap = "Blues_r", show=False, ax = axs[0])
spatial_categorical_visual(sample_meta["LN23574"], "cluster", "Spatial Cluster", show = False, ax = axs[1]) 
fig.suptitle("LN23574: Age 34")
plt.savefig('LN23574.png', format='png', bbox_inches = "tight")
plt.show()


fig, axs = plt.subplots(1,2, figsize=(10, 4))
spatial_feature_visual(sample_pred_list[4]["LN21333"], "pred_old_prob", "Prob of Old", cmap = "Blues_r", show=False, ax = axs[0])
spatial_categorical_visual(sample_meta["LN21333"], "cluster", "Spatial Cluster", show = False, ax = axs[1]) 
fig.suptitle("LN21333: Age 22")
plt.savefig('LN21333.png', format='png', bbox_inches = "tight")
plt.show()


fig, axs = plt.subplots(1,2, figsize=(10, 4))
spatial_feature_visual(sample_pred_list[8]["LN24336"], "pred_old_prob", "Prob of Old", cmap = "Reds", show=False, ax = axs[0])
spatial_categorical_visual(sample_meta["LN24336"], "cluster", "Spatial Cluster", show = False, ax = axs[1]) 
fig.suptitle("LN24336: Age 65")
plt.savefig('LN24336.png', format='png', bbox_inches = "tight")
plt.show()

fig, axs = plt.subplots(1,2, figsize=(10, 4))
spatial_feature_visual(sample_pred_list[4]["LN22921"], "pred_old_prob", "Prob of Old", cmap = "Reds", show=False, ax = axs[0])
spatial_categorical_visual(sample_meta["LN22921"], "cluster", "Spatial Cluster", show = False, ax = axs[1]) 
fig.suptitle("LN22921: Age 55")
plt.savefig('LN22921.png', format='png', bbox_inches = "tight")
plt.show()


for sample in list(old_dict["dataset3k"].keys()) + list(young_dict["dataset3k"]):
    logger.info(f"Processing Sample : {sample}")
    result = sample_meta[sample]
    fig, axs = plt.subplots(4,4, figsize=(22, 20))
    axs = axs.flatten()
    i = 0
    for sample_pred_dict, test_sample_name in zip(sample_pred_list,combinations):
        spatial_feature_visual(sample_pred_dict[sample], "pred_old_prob", test_sample_name, show=False, ax = axs[i])
        i+=1
    spatial_categorical_visual(result, "cluster", "Spatial Cluster", show = False, ax = axs[i]) 
    fig.suptitle(f"{sample}")
    plt.show()




# for i in range(len(dataset)):
#    dataset[i] = dataset[i].to(device)
# all_loader = DataLoader(dataset, batch_size=64, shuffle = False)
for model, test_sample_name in zip(model_list, combinations):
    temp_umap(model,all_loader,test_sample_name)


## ROC
fpr_folds = []
tpr_folds = []
auroc_list = []
thre_list = []
for (train_idx, test_idx), test_sample_name, model in zip(fold_splits, combinations,model_list):
    
    test_set = [dataset[idx].to(device) for idx in test_idx]
    test_loader = DataLoader(test_set, batch_size = 64, shuffle = False)
    auroc_score, pred, label = auroc_cal(model, test_loader)
    fpr_fold, tpr_fold, thre = roc_curve(label, pred)
    auroc_list.append(auroc_score)
    fpr_folds.append(fpr_fold)
    tpr_folds.append(tpr_fold)
    thre_list.append(thre)

# Plot the mean ROC curve and error bars
plt.figure(figsize=(8, 6))

# Plot the ROC curves for each fold
for i in range(len(fpr_folds)):
    plt.plot(fpr_folds[i], tpr_folds[i], label=f'Fold {i+1}')

# Plot the random classifier line
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')

# Set plot labels and title
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Receiver Operating Characteristic (ROC) Curve: Mean AUC {np.mean(auroc_list)}')
plt.legend(loc='lower right',bbox_to_anchor=(1.4, 0))
plt.show()


## ROC
fpr_folds = []
tpr_folds = []
auroc_list = []
thre_list = []
for (train_idx, test_idx), test_sample_name, model in zip(fold_splits, combinations,model_list):
    
    test_set = [dataset[idx].to(device) for idx in test_idx]
    test_loader = DataLoader(test_set, batch_size = 64, shuffle = False)
    auroc_score, pred, label = auroc_cal(model, test_loader)
    fpr_fold, tpr_fold, thre = roc_curve(label, pred)
    auroc_list.append(auroc_score)
    fpr_folds.append(fpr_fold)
    tpr_folds.append(tpr_fold)
    thre_list.append(thre)

from scipy.interpolate import interp1d

max_points = max(fpr.shape[0] for fpr in fpr_folds)
common_fprs = np.linspace(0, 1, max_points)

interpolated_tprs = []
for fpr, tpr in zip(fpr_folds, tpr_folds):
    interp_func = interp1d(fpr, tpr, kind='linear', fill_value='extrapolate')
    interpolated_tpr = interp_func(common_fprs)
    interpolated_tprs.append(interpolated_tpr)

custom_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                 '#aec7e8', '#ffbb78']

# Plot the ROC curves for each fold
for i in range(len(fpr_folds)):
    plt.plot(fpr_folds[i], tpr_folds[i], linestyle='--', label=f'Fold {i+1}', color=custom_colors[i])

# Plot the random classifier line
plt.plot([0, 1], [0, 1], 'k', label='Random Classifier')

plt.plot(common_fprs, np.mean(np.vstack(interpolated_tprs), axis = 0), "r", label = "Mean Performance")

plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xticks(np.linspace(0, 1, 6))
plt.yticks(np.linspace(0, 1, 6))
plt.grid(which='major', axis='y')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Receiver Operating Characteristic (ROC) Curve: Mean AUC {np.mean(auroc_list)}')
plt.legend(loc='lower right', bbox_to_anchor=(1.4, 0))
plt.savefig('ROC.png', format='png', bbox_inches = "tight")
plt.show()




# Plot the ROC curves for each fold
for i in range(len(fpr_folds)):
    plt.plot(fpr_folds[i], tpr_folds[i], linestyle='--', label=f'Fold {i+1}', color=custom_colors[i])

# Plot the random classifier line
plt.plot([0, 1], [0, 1], 'k', label='Random Classifier')

plt.plot(common_fprs, np.mean(np.vstack(interpolated_tprs), axis = 0), "r", label = "Mean Performance")

plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xticks(np.linspace(0, 1, 6))
plt.yticks(np.linspace(0, 1, 6))
plt.grid(which='major', axis='y')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Receiver Operating Characteristic (ROC) Curve: Mean AUC {np.mean(auroc_list)}')
plt.legend(loc='lower right', bbox_to_anchor=(1.4, 0))
plt.savefig('ROC.png', format='png', bbox_inches = "tight")
plt.show()




