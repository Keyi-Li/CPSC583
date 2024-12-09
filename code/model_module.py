import torch
import torch_geometric
import math

from torch.nn import Linear , ModuleList
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
from torch_geometric.nn.norm import LayerNorm
import torch_geometric.utils as utils

def rbf_kernel(X, gamma=None):
    """
    Compute the normalized RBF (Gaussian) kernel matrix for a dataset X.
    This implementation is compatible with GPU tensors.
    
    Args:
    X (torch.Tensor): Input tensor of shape (n_samples, n_features)
    gamma (float, optional): If None, defaults to 1.0 / n_features
    
    Returns:
    torch.Tensor: RBF kernel matrix of shape (n_samples, n_samples)
    """
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
        return global_mean_pool(x,batch)
        
    def forward(self, x, batch, graph_embedding):
        mu = self.cal_mu(graph_embedding)
        z_temp = mu + (self.sigma*torch.randn_like(mu)*self.training).to(self.device)
        z = self.hard_sigmoid(z_temp)
        sparse_x = x * z.index_select(0, batch)
        return sparse_x, mu, z

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
        #self.norm = LayerNorm(out_channels)
        self.dropout = dropout_rate
        self.lin = torch.nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv(x, edge_index, edge_weight)
        x = F.relu(x)
        #x = self.norm(x)
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
        for i in range(gnn_layer):
            self.GNN.append(GCNBlock(in_channel, hidden_channels_rest[i], dropout_rate))
            in_channel =  hidden_channels_rest[i]

        self.linear_sample = nn.Linear(num_sample,in_channel)
        
        in_channel = 2*in_channel
        
        self.MLP = ModuleList()
        for j in range(gnn_layer,gnn_layer+lin_layer):
            self.MLP.append(LinearBlock(in_channel,hidden_channels_rest[j],dropout_rate))
            in_channel = hidden_channels_rest[j]

        self.predictor_label = nn.Linear(in_channel, out_channel)
        self.predictor_pheno = nn.Linear(in_channel, 2) # binary

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, data, graph_embedding):
 
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
        y = torch.stack([sample[0] for sample in unbatched_y]) # sample_id

        x_1 = self.linear_sample(sample_id)
        x = torch.cat((x, x_1), dim=-1)

        for linear_block in self.MLP:
            x = linear_block(x)
            
        label_pred = self.predictor_label(x)
	x = ReverseLayerF.apply(x)
	label_pheno = self.predictor_pheno(x)

        return label_pred, label_pheno, y,  regularization, stability_regularization



