#!/usr/bin/env python
# coding: utf-8

# ## 0. Packages and Functions

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4" 
import torch
from torch.utils.data import TensorDataset, Dataset, RandomSampler, DataLoader
import torchvision.datasets as datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import random
import scanpy as sc

os.chdir("..")
print(os.getcwd())

#!pip3 install 'scanpy[leiden]'
## to deal with plotting error:
## vim /usr/local/lib/python3.8/dist-packages/scanpy/plotting/_tools/scatterplots.py 
## add import matplotlib.pyplot as plt
## replace line 160 something: cmap = copy(colormaps.get_cmap(cmap)) with  cmap = copy(plt.cm.get_cmap(cmap))

np.random.seed(0)
random.seed(0)
def sample_subset(data, label,target_size = 1000):
    category = np.unique(label)
    selected_data = []
    selected_label = []
    for item in category:
        idxs = np.where(label ==item)[0]
        selected_idxs = np.random.choice(idxs, size = target_size, replace = False)
        selected_data.append(data[selected_idxs,...])
        selected_label.append(label[selected_idxs,...])
    selected_data = np.vstack(selected_data)
    selected_label = np.hstack(selected_label)
    num_samples = selected_label.shape[0]
    permuted_indices = np.random.permutation(num_samples)
    return selected_data[permuted_indices,...], selected_label[permuted_indices,...]


class TransformedDataset(Dataset):
    def __init__(self, images, labels, dataset_id):
        self.data = images
        self.targets = labels
        self.dataset_id = dataset_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]
        sample_id = self.dataset_id[idx]
        return image, label, sample_id


def visual_inspect(dataset,name):
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.flatten()
    imgs, labels = dataset["img"], dataset["label"]
    for i in range(10):
        img, label = imgs[i], labels[i]
        axes[i].imshow(img.squeeze());
        axes[i].set_title(str(label)) 
    plt.suptitle(name);


# ## 1.Preprocessing Dataset
# - process each dataset to have mean 0 and std 1

dataset_dir = "./dataset/mnist_c"
train_name = ['impulse_noise','canny_edges','zigzag','dotted_line','scale']
test_name = ["fog", "stripe"]

data_dict = {}
current_id = 0
for item in train_name:
    data_all = np.load(os.path.join(dataset_dir,item,"train_images.npy")).squeeze()
    label_all = np.load(os.path.join(dataset_dir,item,"train_labels.npy"))
    
    data, label = sample_subset(data_all, label_all,target_size = 1000) # each digit 
    mean = data.mean()/255
    std = data.std()/255
    print(f"Dataset: {item}, mean: {mean:.4f}, std: {std:.4f}")
    
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((mean,), (std,))
                                ])
    
    data = torch.stack([transform(img) for img in data])
    
    data_dict[item] = {"id": np.array([current_id]*data.shape[0]).squeeze(),
                      "mean": mean,
                       "std": std,
                       "img": data,
                       "label": label
                      }

    current_id +=1



test_dict ={}
for item in test_name:

    data_all = np.load(os.path.join(dataset_dir,item,"test_images.npy")).squeeze()
    label_all = np.load(os.path.join(dataset_dir,item,"test_labels.npy"))
    data, label = sample_subset(data_all, label_all,target_size = 500)
    
    mean = data.mean()/255
    std = data.std()/255
    print(f"Dataset: {item}, mean: {mean:.4f}, std: {std:.4f}")
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((mean,), (std,))
                                ])
    data = torch.stack([transform(img) for img in data])
    test_dict[item] = {"img": data,
                       "label": label,
                       "id" : np.array([current_id]*data.shape[0]).squeeze()
                      }



# Train data visualization
for name, dataset in data_dict.items():
    visual_inspect(dataset,name)


# test data visualization
for name, dataset in test_dict.items():
    visual_inspect(dataset,name)


# ## 2. UMAP Visualization
# - randomly select 1000 samples from each dataset and plot as umap

import random
random.seed(42) 
domain_list = []
label_list = []
data_list =[]
train_test_list = []
for name, dataset in data_dict.items():
    random_numbers = random.sample(range(dataset["img"].shape[0]), 2000)
    domain_list += [name]*2000
    train_test_list += ["Train"] * 2000
    for i in random_numbers:
        data, label = dataset["img"][i], dataset["label"][i]
        data_list.append(data.squeeze().reshape(-1))
        label_list.append(label)

for name, dataset in test_dict.items():
    random_numbers = random.sample(range(dataset["img"].shape[0]), 2000)
    domain_list += [name]*2000
    train_test_list += ["Test"] * 2000
    for i in random_numbers:
        data, label = dataset["img"][i], dataset["label"][i]
        data_list.append(data.squeeze().reshape(-1))
        label_list.append(label)

data = np.vstack(data_list)
adata = sc.AnnData(data)
adata.obs["domain"] = domain_list
adata.obs["label"] = label_list
adata.obs["label"] = adata.obs["label"].astype(str)
adata.obs["train_test"] = train_test_list

sc.tl.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)

# very obvious batch effect/distribution shift
with plt.rc_context({"figure.figsize": (8,6)}):
    sc.pl.umap(
        adata,
        color=["domain", "label","train_test"],
        size=30,
        ncols=1
    )


# ## 3. Turn Data into Graph Data

import torch
from torch_geometric.data import Data

def mnist_to_grid_graph(data_dict):

    graphs = []
    
    edge_index = []
    for i in range(28):
        for j in range(28):
            pixel_idx = i * 28 + j

            left = pixel_idx - 1
            right = pixel_idx + 1
            upper = pixel_idx - 28
            lower = pixel_idx + 28

            for nei in [left, right, upper, lower]:
                if nei >= 0 and nei <= 783:
                    edge_index.append([pixel_idx, nei])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    
    # Process each image
    for name, dataset in data_dict.items():
        for i in range(dataset["img"].shape[0]):
            x = dataset["img"][i].reshape(-1).unsqueeze(1) 
            graph = Data(
                x = x,
                edge_index=edge_index,
                sample_id = dataset["label"][i],
                batch_id = dataset["id"][0]
            )
            graphs.append(graph)
    return graphs


train_data_list = mnist_to_grid_graph(data_dict)
test_data_list = mnist_to_grid_graph(test_dict)


# ## 4. Model
# - the model structure here is fundamentally the same as used in the paper
# - modification is only made to accomodate MNIST data property

import torch
import torch_geometric
import math

from torch.nn import Linear , ModuleList
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_add_pool, BatchNorm
from torch_geometric.nn.norm import LayerNorm
import torch_geometric.utils as utils
from torch.autograd import Function

def rbf_kernel(X, gamma=None):
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    
    distances = torch.cdist(X, X, p=2.0)
    K = torch.exp(-gamma * (distances ** 2))
    row_sums = K.sum(dim=1, keepdim=True)
    K_normalized = K / row_sums
    return K_normalized

class LspinBlock(torch.nn.Module):
    def __init__(self, in_channel:int, hidden_channels: list, sigma = 0.5, activation = "tanh", dropout_rate = 0.1):
        super().__init__()
        out_channel = in_channel
        self.MLP = ModuleList()
        for i in hidden_channels:
            self.MLP.append(LinearBlock(in_channel,i, dropout_rate = 0, activation = activation))
            in_channel = i
        self.MLP.append(LinearBlock(in_channel,out_channel,  dropout_rate = dropout_rate, activation = activation))
        self.sigma = sigma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def cal_mu(self,graph_embedding):
        for linear_block in self.MLP:
            graph_embedding = linear_block(graph_embedding)
        return graph_embedding

    def hard_sigmoid(self,z):
        return torch.clamp(z+0.5,0,1)

    def get_graph_embedding(self, x, batch):
        num_graph = int(torch.max(batch)+1)
        if x.shape[-1] == 1:
            x = x.squeeze(-1)
        return x.view(num_graph, 784)
        
    def forward(self, x, batch, graph_embedding):
        mu = self.cal_mu(graph_embedding)
        z_temp = mu + (self.sigma*torch.randn_like(mu)*self.training).to(self.device)
        z = self.hard_sigmoid(z_temp)
        sparse_x = x * z.view(-1,1)
        return sparse_x.view(-1,1), mu, z

    def regularizer(self, mu):
        return torch.mean(torch.sum(0.5*(1+torch.erf((mu+0.5)/self.sigma/math.sqrt(2))),axis = 1))

    def stability_regularizer(self, graph_embedding, z, mode = "same"):
        '''
        choice of mode: 
            "same": encourage similar samples to select similar features
            "diff": encourage different samples to select different features
        '''
        K = rbf_kernel(graph_embedding)
        D = torch.cdist(z, z, p = 2)
        if mode == "same":
            return torch.mean(torch.sum(K*D, axis = 1))
        else:
            return torch.mean(torch.sum((1-K)*(-D),axis = 1))

    def get_gate(self,graph_embedding):
        mu = self.cal_mu(graph_embedding)
        z = self.hard_sigmoid(mu)
        return z

class GCNBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.norm = LayerNorm(out_channels)
        self.dropout = dropout_rate
        self.lin = torch.nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)
        return x


class LinearBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.2, activation='relu'):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = activation

    def forward(self, x):
        x = self.dropout(x)
        x = self.lin(x)
        if self.activation == "relu":
            x = F.relu(x)
        elif self.activation == "tanh":
            x = F.tanh(x)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")
        return x


class ReverseLayerF(Function): # for DANN
    @staticmethod
    def forward(ctx, x, alpha): # Any information that needs to be used during the backward pass must be stored in the ctx object during the forward pass
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class Model(nn.Module):
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
        in_channel = 1
        for i in range(gnn_layer):
            self.GNN.append(GCNBlock(in_channel, hidden_channels_rest[i], dropout_rate))
            in_channel =  hidden_channels_rest[i]

        self.linear_sample = nn.Linear(num_sample,in_channel)
        
        self.MLP = ModuleList()
        for j in range(gnn_layer,gnn_layer+lin_layer):
            self.MLP.append(LinearBlock(in_channel,hidden_channels_rest[j],dropout_rate))
            in_channel = hidden_channels_rest[j]

        self.predictor_label = nn.Linear(in_channel, out_channel)
        self.predictor_pheno = nn.Linear(in_channel, 10) 

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, data, graph_embedding,alpha):
 
        raw_x, edge_idx, sample_id, batch = data.x, data.edge_index, data.sample_id, data.batch

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

        x = global_add_pool(x, batch)

        for linear_block in self.MLP:
            x = linear_block(x)

        label_pheno = self.predictor_pheno(x) # predict for digits label 
        x = ReverseLayerF.apply(x, alpha)
        label_pred = self.predictor_label(x) # predict for domain label

        return label_pred, label_pheno,  regularization, stability_regularization


# ### 5. Training


from torch_geometric.loader import DataLoader

def accu(model, data):
    acc = 0
    model.eval()
    with torch.no_grad():
        for item in data:
            label_pred, label_pheno, regularization, stability_regularization = model(data=item, graph_embedding = None, alpha=alpha)
            pred = torch.argmax(label_pheno, dim=1)
            acc += torch.sum(pred == torch.Tensor(item.sample_id).long().to(device))
    return acc/len(data.dataset)


model = Model(in_channel = 784, hidden_channel_lspin = [16], hidden_channels_rest = [64,64,32,16,8], out_channel = 7, gnn_layer = 2 , lin_layer = 3, dropout_rate = 0.2, num_sample = 10, lam = 0.00001, lam_sta = 0,sigma = 0.5, mode = "diff",graph_embed = True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_set = [train_data_list[idx].to(device) for idx in range(len(train_data_list))]
test_set = [test_data_list[idx].to(device) for idx in range(len(test_data_list))]

train_loader = DataLoader(train_set, batch_size = 128, shuffle = True)
test_loader = DataLoader(test_set, batch_size = 64, shuffle = True)

lr = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_main = torch.nn.NLLLoss(reduction='sum')
loss_domain = torch.nn.NLLLoss(reduction='sum')
log_softmax = nn.LogSoftmax(dim=1)
    
cuda = True
if cuda:
    model = model.cuda()
    loss_class = loss_main.cuda()
    loss_domain = loss_domain.cuda()

for p in model.parameters():
    p.requires_grad = True

n_epoch = 50
gamma = 1/10

loss_main_source = [] # predicting 0-9
loss_domain_source = [] # predicting batch
loss_domain_target = []

accu_t_list = []
accu_s_list = []

for epoch in range(n_epoch):
    # here manually made sure that train_loader and test_loader have the same number of batch
    len_dataloader = min(len(train_loader), len(test_loader))
    data_source_iter = iter(train_loader)
    data_target_iter = iter(test_loader)

    s_main_loss = 0
    s_domain_loss = 0
    t_domain_loss = 0

    model.train()
    for i in range(len_dataloader):

        p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # training model using source data
        data_source = next(data_source_iter)

        optimizer.zero_grad()

        label_pred, label_pheno,  regularization, stability_regularization = model(data=data_source, graph_embedding = None, alpha=alpha)
        label_pred = log_softmax(label_pred)
        label_pheno = log_softmax(label_pheno)
        err_s_main = loss_main(label_pheno, torch.Tensor(data_source.sample_id).long().to(device))
        err_s_domain = loss_domain(label_pred, torch.Tensor(data_source.batch_id).long().to(device))
        s_main_loss += err_s_main.item()
        s_domain_loss += err_s_domain.item()

        # adverarial constrasting model using target data
        data_target = next(data_target_iter)

        label_pred, _ ,_  ,_ = model(data=data_target, graph_embedding = None, alpha=alpha)
        label_pred = log_softmax(label_pred)
        err_t_domain= loss_domain(label_pred, torch.Tensor(data_target.batch_id).long().to(device))
        t_domain_loss += err_t_domain.item()

        total_loss = err_s_main + gamma*(err_s_domain + err_t_domain) + regularization + stability_regularization
        
        total_loss.backward()
        optimizer.step()

    accu_s = accu(model, train_loader)
    accu_s_list.append(accu_s)
    accu_t = accu(model, test_loader)
    accu_t_list.append(accu_t)
    

    s_main_loss /= len(train_loader.dataset)
    s_domain_loss /= len(train_loader.dataset)
    t_domain_loss /= len(test_loader.dataset)
    
    loss_main_source.append(s_main_loss)
    loss_domain_source.append(s_domain_loss)
    loss_domain_target.append(t_domain_loss)
    
    
    print(f'Epoch: {epoch:03d}, source main loss: {s_main_loss:.4f}, source domain loss: {s_domain_loss:.4f}, target domain loss:{t_domain_loss:.4f}, featrue_reg_loss: {regularization:.4f}, stab_reg_loss: {stability_regularization:.4f}, accu_s: {accu_s}, accu_t: {accu_t}')

