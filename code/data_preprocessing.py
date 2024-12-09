#!/usr/bin/env python
# coding: utf-8

### 1. Packages and Functions

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # check before use

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.stats import rankdata, beta
from scipy.spatial import Delaunay
import scanpy as sc

import torch
import torch_geometric
from torch_geometric.data import Data
import networkx as nx

from tqdm.notebook import tqdm as tqdm
from loguru import logger

import faiss
import itertools
from igraph import Graph
from concurrent.futures import ProcessPoolExecutor as mpi
from itertools import repeat

working_dir = "/data/banach2/sennet/codex/processed"
output_dir = "/data/banach2/kl825/self_embed_delaunay"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
os.chdir(working_dir)


### 2. Data Information and Preprocessing



### samples
raw_sample_file = []
for file in os.listdir():
    if file.endswith(".pq"):
        raw_sample_file.append(file)

### sample_meta data
meta = "/data/banach2/sennet/codex/metadata.xlsx"
meta_data = pd.read_excel(meta)
meta_data.head(2)

sample_meta = {"LN291": {"Age": 71}, 
               "LN13560": {"Age": 74},
               # "LN6243": {"Age": 78}, # potentially exclude
               "LN24336": {"Age": 65},
               "LN21333": {"Age": 22},
               "LN27766": {"Age": 50},
               "LN00837": {"Age": 86},
               "LN00560": {"Age": 25},
               "LN23574": {"Age": 34},
               "LN22921": {"Age": 55},
               "LN21756": {"Age": 22}}

for file in sample_meta.keys():
    file_name = file + ".pq"
    temp_data = pd.read_parquet(file_name)
    raw_columns = ['Centroid X µm', 'Centroid Y µm'] + [col for col in temp_data.columns if "Cell: Mean" in col]
    temp_data = temp_data[raw_columns]
    renamed_columns = ["X","Y"] + [col.split(':')[0] for col in raw_columns[2:]]
    temp_data.columns = renamed_columns
    sample_meta[file]["Profile"] = temp_data.iloc[:,2:]
    sample_meta[file]["Spatial"] = temp_data.iloc[:,:2]

### noticed that samples might have distinct features measured, take the intersection and reorder the dataframe so that the downstream prediction can be same
common_feature = []
for item in sample_meta.keys():
    n = len(sample_meta[item]["Profile"].columns.tolist())
    print(f"{item}: {n} features")
    print(sample_meta[item]["Profile"].columns.tolist())
    if len(common_feature) == 0:
        common_feature = sample_meta[item]["Profile"].columns.tolist()
    else:
        common_feature = list(set(common_feature).intersection(sample_meta[item]["Profile"].columns.tolist()))
        
print("_________________________________________________________________________________________________________________________") 

print(f"common_feature:{' '.join(common_feature)}")
print("_________________________________________________________________________________________________________________________")
for item in sample_meta.keys():
    print(set(sample_meta[item]["Profile"].columns.tolist()).difference(set(common_feature)))

### noticed several markers are named differently in different sample: to rename:
### noticed that if we exclude sample LN6243 we can have several more markers
rename_map = {"CollagenIV" : 'Collagen IV', "CDKN2A/p16": "p16", 'CDKN2A/p16NK4A': "p16", "yH2AX": "rH2AX", "IFNG" : "IFN-G", "HMBG1": "HMGB1"}
for sample in sample_meta.keys():
    columns = sample_meta[sample]["Profile"].columns.tolist()
    for item in rename_map.keys():
        if item in columns:
            item_idx = columns.index(item)
            columns[item_idx] = rename_map[item]
    sample_meta[sample]["Profile"].columns = columns

common_feature = []
for item in sample_meta.keys():
    if len(common_feature) == 0:
        common_feature = sample_meta[item]["Profile"].columns.tolist()
    else:
        common_feature = list(set(common_feature).intersection(sample_meta[item]["Profile"].columns.tolist()))
print(f"common_feature:{' '.join(common_feature)}")

### keep only these common_feature and ensure the same order accorss sample
for sample in sample_meta.keys():
    reordered_profile = sample_meta[sample]["Profile"][common_feature]
    sample_meta[sample]["Profile"] = reordered_profile


# ## 3. Delaunay Graph

# computes a KNN index from faiss
def faiss_index(coords):
    coords = np.ascontiguousarray(coords).astype(np.float32)
    # #"HNSW32,Flat" - This string specifies the type of index to create.
# HNSW32 stands for Hierarchical Navigable Small World, which is a type of approximate nearest neighbor index. The 32 in HNSW32 refers to the number of neighbors to consider at each layer in the graph; you can adjust this number based on your specific use case and computational resources.
# Flat indicates that the index will be built on top of a flat index, which is suitable for smaller datasets. For larger datasets, you might consider more memory-efficient options like IVF indexes.
    index = faiss.index_factory(coords.shape[1], "HNSW32,Flat")
    index.train(coords)
    index.add(coords) # the index is ready to be queried for nearest neighbors.
    return index

# creates an index and searches it
def knn_search(coords, k=256):
    coords = np.ascontiguousarray(coords).astype(np.float32)
    index = faiss.index_factory(coords.shape[1], "HNSW32,Flat")
    index.train(coords)
    index.add(coords)
    return index.search(np.ascontiguousarray(coords).astype(np.float32), k)
# returns a tuple: (squared_distances, nearest_neighbor_idex)
# row i contains the IDs of the neighbors of query vector i, sorted by increasing distance. 
# the corresponding squared distances

# creates a knn_graph from 2D coordinates
def coords_to_knngraph(coords, k):
    loc_index = faiss_index(coords)
    loc_dnn, loc_knn = loc_index.search(np.ascontiguousarray(coords).astype(np.float32), k)    
    return Graph(n=loc_knn.shape[0], edges=to_edge_pairs(loc_knn), directed=False)


# returns the adjacency matrix of the KNN-graph obtained from 2D coordinates
def coords_to_adjacency(coords, k): 
    g = coords_to_knngraph(coords, k)
    return g.get_adjacency_sparse()

# simple minded conversion from a list of neighbors indices to pairs of nodes indicating an edge
def to_edge_pairs(dlnn):
    edge_pairs = []
    for i_row, row in tqdm(enumerate(dlnn), total=len(dlnn), desc='to edge pairs'):
        edge_pairs.extend([[i_row, j_row] for j_row in row if j_row!=i_row])
    return edge_pairs

# creates a list whose i-th entry is another list with the indices of the Delaunay neighbors of the i-th cell 
def delaunay_nn(loc_xy, indices):
        d0 = Delaunay(loc_xy[indices>0,:])
        indices = indices[indices>0]
        simplices = np.array(d0.simplices)
        return indices[np.unique(simplices[np.any(simplices==0, axis=1),:].flatten())]

# this is used to speed up computations
def delaunay_packed(input_data):
    return [delaunay_nn(input_datum[0], input_datum[1]) for input_datum in input_data]
    
# computes both the Delaunay list of neighbors and the corresponding igraph graph using parallel processes
def delaunay_graph(knn, xy):        
    dlnn = []
    
    mpi_data = [
        [
            (xy[knn[j,:16],:], knn[j,:16]) 
            for j in range(i, min([i+5000, knn.shape[0]]))
        ] 
        for i in tqdm(range(0, knn.shape[0], 5000))
    ]
    
    with tqdm(total=len(mpi_data), desc='delaunay graphing') as pbar:
        with mpi(max_workers=50) as executor:
            for res in executor.map(delaunay_packed, mpi_data):
                dlnn.extend(res)
                pbar.update(1)
    edge_pairs = to_edge_pairs(dlnn)
    return dlnn, Graph(n=len(dlnn), edges=edge_pairs, directed=False)

# a class to store all the graph data in one place
class mknn:
    xy = None
    dnn = None
    knn = None
    gdl = None
    g = None
    
    def __init__(self, coords):
        self.xy = coords
    
    def find_nearest_neighbours(self, k=256):
        self.dnn, self.knn = knn_search(self.xy, k)
        
    def find_delaunay_neighbours(self):
        self.gdl, self.g = delaunay_graph(self.knn, self.xy)
        
    @classmethod
    def from_coords(cls, coords):
        nn = mknn(coords)
        nn.find_nearest_neighbours()
        nn.find_delaunay_neighbours()
        return nn


for sample, info in sample_meta.items():
    logger.info(f"Processing {sample}:")
    nn = mknn.from_coords(info["Spatial"].values)
    sample_meta[sample]["nn"] = nn


### 4.Graph Extraction


class embed_sample:
    def __init__(self, Age, Profile, Spatial, nn):
        self.spatial = Spatial
        self.profile = Profile
        self.graph_obj = nn
        self.age = Age
        self.subgraph_node =set()
        self.n = self.spatial.shape[0]

    def get_marker(self, marker_name:list, cell_idx = None):
        if cell_idx is None:
            return self.profile[marker_name]
        else:
            return self.profile[marker_name].iloc[cell_idx,:]
        
    def get_delaunay_neighbor(self,cell_idx,squared_distance_thre=900):
        neis = [i for i in self.graph_obj.gdl[cell_idx] if (self.graph_obj.dnn[cell_idx][self.graph_obj.knn[cell_idx] == i] < squared_distance_thre) and (i != cell_idx)]
        return neis

    def unique_node(self, nodes_taken, potential_nodes):
        return list(set(potential_nodes).difference(nodes_taken))

    def delaunay_subgraph_extract_all(self,node_of_interest, graph_size, no_overlap = False):
        subgraph_list = []
        for node in node_of_interest:
            if node not in self.subgraph_node:
                flag, subgraph = self.make_subgraph(node, graph_size, no_overlap = no_overlap)
                if flag:
                    subgraph_list.append(subgraph)
        return subgraph_list

    def make_subgraph(self, node, n_hop):
        node_list = [node]
        edge_list = []

        current_hop = 1
        k_hop_neighbor = self.get_delaunay_neighbor(node)
        edge_list += [(node, neighbor) for neighbor in k_hop_neighbor]
        node_list += k_hop_neighbor
        
        while current_hop <= n_hop:
            if len(k_hop_neighbor) == 0:
                break
            else:
                current_hop += 1
                next_hop = []
                for nb in k_hop_neighbor:
                    next_hop_neighbor = self.get_delaunay_neighbor(nb)
                    edge_list += [(nb, neighbor) for neighbor in next_hop_neighbor]
                    unique_node = self.unique_node(node_list,next_hop_neighbor)
                    next_hop += unique_node
                    node_list += unique_node
                k_hop_neighbor = next_hop
            
        return (node_list, list(set(edge_list)))


samples = {key : embed_sample(**value) for key, value in sample_meta.items()}


np.random.seed(0)
graph_dict = {}
for name, obj in samples.items():
    logger.info(f"Processing {name}")
    central_node = np.random.choice(np.arange(obj.n), size = 20000, replace = False)
    results = []
    for node in tqdm(central_node, desc = "extract_graph"):
        results.append(obj.make_subgraph(node,2))
    graph_dict[name] = results


### 5. Dataset Creation


def zero_preserved_beta_quantile_normalization(array, a=4, b=4):
    """ see name

    Args:
        array: a feature column of a dataframe
        a, b: parameters for beta distribution

    Returns:
        tansformed: transformed feature column
    """
    
    transformed = np.zeros_like(array)
    nonzero = array > 0
    rank = (rankdata(array[nonzero]) - 0.5)/np.sum(nonzero) # map nonzero values to quantile 
    transformed[nonzero] = beta.ppf(rank,a,b)
    return transformed

def mpi_beta(arg):
    col_name, array, a, b = arg
    return  col_name, zero_preserved_beta_quantile_normalization(array,a, b)


# quantile_normalization for samples
for name, obj in tqdm(samples.items()):
    mpi_data = [(col, obj.profile[col],4,4) for col in obj.profile.columns]
    result  ={}
    with tqdm(total=len(mpi_data), desc='quantile_normalize') as pbar:
        with mpi(max_workers=50) as executor:
            for res in executor.map(mpi_beta, mpi_data):
                result[res[0]] = res[1]
                obj.normalized_profile = pd.DataFrame(result)
                pbar.update(1)


def create_graph_data(node_feature, edge_index, y, sample_name, sample_id, cell_idx, spatial, age, feature_name, prediction_name):
    """
    Creates a single graph Data object.
    
    Args:
        node_features (Tensor): Node feature (quantile_normalized) matrix of shape [num_nodes, num_node_features]. e.g [num_nodes, 33]
        edge_indices (LongTensor): Edge indices of shape [2, num_edges].
        y: Graph-level target or label, 0 for young and 1 for old
        sample_name: name of the sample from which subgraph is extracted
        cell_idx: cell_idx in the original sample
        spatial: corresponding spatial information of nodes
        raw_feature: unnomrlazied feature
        age: age of the sample
        feature_names: name of feature
        
    Returns:
        Data: A graph Data object.
    """
    edge_index_undirected = torch_geometric.utils.to_undirected(edge_index)
    data = Data(x= node_feature, edge_index=edge_index_undirected,y = y)
    data.sample = sample_name
    data.sample_id = sample_id
    data.cell_idx = cell_idx
    data.spatial = spatial
    data.age = age
    data.feature_name = feature_name
    data.prediction_name = prediction_name
    return data

def reindex(node_list, edge_list):
    n = len(node_list)
    mapping = dict(zip(node_list,range(n)))
    re_index_list = []
    for item in edge_list:
        re_index_list.append([mapping[item[0]],mapping[item[1]]])
    return re_index_list


def create_single_dataset(arg):
    subgraph_list, obj, prediction_name, featuren_name, sample_id, sample_name = arg
    data_list = []
    for node_list, edge_list in tqdm(subgraph_list, desc=f"Processing graph groups for sample {sample_name}", leave=False):
        
        profile = obj.normalized_profile[feature_name].iloc[node_list,:]
        node_feature = torch.tensor(profile.values, dtype=torch.float)
    
        y = obj.normalized_profile[prediction_name].iloc[node_list,:]
        y = torch.tensor(y.values, dtype = torch.float)
       
        spatial = obj.spatial.iloc[node_list,:]
         
        sample = sample_name
        reindex_edge_list = reindex(node_list,edge_list)
        edge_idx = torch.tensor(reindex_edge_list,dtype=torch.long).t().contiguous()
        if (len(edge_idx) != 0) and (len(node_list) >=20):
            data = create_graph_data(node_feature, edge_idx, y, sample, sample_id,node_list, spatial.values, obj.age, feature_name, prediction_name)
            data_list.append(data)
    return sample_name, data_list


def create_dataset(subgraph_dict, samples, prediction_name, feature_name):
    data_list = []
    sample_id =0
    for name, obj in samples.items():
        logger.info(f"processing {name}")
        graph_group = subgraph_dict[name]
        for node_list, edge_list in tqdm(graph_group, desc=f"Processing graph groups for sample {name}", leave=False):

            profile = obj.normalized_profile[feature_name].iloc[node_list,:]
            node_feature = torch.tensor(profile.values, dtype=torch.float)

            y = obj.normalized_profile[prediction_name].iloc[node_list,:]
            y = torch.tensor(y.values, dtype = torch.float)
           
            spatial = obj.spatial.iloc[node_list,:]
             
            sample = name
            reindex_edge_list = reindex(node_list,edge_list)
            edge_idx = torch.tensor(reindex_edge_list,dtype=torch.long).t().contiguous()
            if (len(edge_idx) != 0) and (len(node_list) >=20):
                data = create_graph_data(node_feature, edge_idx, y, sample, sample_id,node_list, spatial.values, obj.age, feature_name, prediction_name)
                data_list.append(data)
        sample_id +=1
    return data_list


def save_dataset(dataset, output_dir, size):
    """
    save dataset to disk

    Args:
        dataset: a list of data
        size: str, indicator of the size of each subgraph:
    """
    folder_path = os.path.join(output_dir,"dataset")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    dataset_name = "dataset"+size+".pt"
    torch.save(dataset, os.path.join(folder_path, dataset_name))


id_map = {}
for i in range(len(samples.keys())):
    id_map[list(samples.keys())[i]] = i
id_map


from concurrent.futures import ThreadPoolExecutor
prediction_name = ["p21","p16","Ki67"]
feature_name = list(set(samples["LN291"].profile.columns).difference(prediction_name))

torch_data = {}

mpi_data = [ (graph_dict[name], obj, prediction_name, feature_name, id_map[name], name)
    for name, obj in samples.items()
]

with tqdm(total=len(mpi_data), desc='dataset_creation') as pbar:
    with ThreadPoolExecutor() as executor:
        for res in executor.map(create_single_dataset, mpi_data):
            torch_data[res[0]] = res[1]
            pbar.update(1)


dataset = []
for value in torch_data.values():
    dataset.extend(value)
save_dataset(dataset, output_dir, "2_hop_normalized")


### 6. Visualization

np.random.seed(42)
def plot_spatial(dataset,size = 10, node_size =50):
    selected= np.random.choice(range(len(dataset)), size, replace = False)
    fig, axes = plt.subplots(int(size/5), 5, figsize=(24, 5*size/5),  gridspec_kw={'wspace': 0.4, 'hspace': 0.6})
    axes = axes.flatten()
    for i in range(size):
        idx = selected[i]
        G = torch_geometric.utils.to_networkx(dataset[idx])
        # Assuming you want to highlight node 0
        node_colors = ['r' if node == 0 else 'b' for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos=dataset[idx].spatial, node_size = node_size, node_color=node_colors,ax = axes[i]);
        nx.draw_networkx_edges(G, pos=dataset[idx].spatial,ax = axes[i],arrows=False);
        axes[i].set_title(dataset[idx].sample)
        axes[i].tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        axes[i].set_xlabel('X Axis')
        axes[i].set_ylabel('Y Axis')
    plt.show()
    plt.close(fig)


def random_slice_feature_examine(dataset, n_features, dataset_name,n = 10, mode = "normalized"):
    """
    examine the feature distribution of samples randomly selected from dataset
    Args:
        dataset: a list of torch geometirc data
        n_features: num of features to visualize, data.x.shape[1]
        n: number of sample tp examine:
        mode: visualize normalized or unnormalzied (everything else, usually called raw)
        dataset_name: a string to indicate graph size
    """
    
    np.random.seed(42)
    n_max = len(dataset)
    n = min(n,n_max)
    index_selected = np.random.randint(0, n_max-1, size=n)
    
    for idx in index_selected:
        data = dataset[idx]
        column_names = data.feature_name
        fig, axes = plt.subplots(nrows=6, ncols=6, figsize=(30,25))
        axes = axes.flatten()
        for i in range(n_features):
            if mode == "normalized":
                axes[i].hist(data.x[:,i], bins = 50)
                title = column_names[i]
                axes[i].set_title(f"{title}")
            else:
                axes[i].hist(data.rawX.iloc[:,i], bins = 50)
                title = column_names[i]
                axes[i].set_title(f"{title}")
        fig.suptitle(f"{dataset_name}_{data.sample}_{mode}")
        plt.show()
        plt.close()


from mpl_toolkits.axes_grid1 import make_axes_locatable
def spatial_overlay(subgraph_list, whole_sample, name, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    node_lists = [i[0] for i in subgraph_list]
    all_nodes = list(itertools.chain(*node_lists))
    unique_node, counts = np.unique(all_nodes,return_counts=True)
    background = whole_sample.spatial.values
    ax.scatter(background[:, 0], background[:, 1], color='lightgray', alpha=0.1, label = "Whole_Sample", s= 1)
    if len(unique_node) >0:
        node_pos = background[unique_node,:]
        scatter = ax.scatter(node_pos[:, 0], node_pos[:, 1], c= counts, cmap='Reds', label='Selected Nodes', s = 1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)  
        # Create a new colorbar and assign it to the new axis
        cbar = plt.colorbar(scatter, cax=cax)
        cbar.ax.set_ylabel('Counts', rotation=-90, va="bottom")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(name)

    return ax


plot_spatial(dataset,size = 50)

fig, axes = plt.subplots(2,5, figsize = (40,20))
axes = axes.flatten()
for i in range(len(samples.keys())):
    key = list(samples.keys())[i]
    axes[i] = spatial_overlay(graph_dict[key],samples[key],key, axes[i])
plt.suptitle("Graph : 2-hop", fontsize=20)
plt.show()


def subsample_vs_sample_distribution(subsample, sample, feature_name, sample_name):
    central_node = [example[0][0] for example in subsample]
    sample_feature = sample.normalized_profile[feature_name]
    subsample_feature = sample.normalized_profile[feature_name].iloc[central_node,:]
    
    fig, axes = plt.subplots(nrows=5, ncols=6, figsize=(25,30))
    axes = axes.flatten()
    for i in range(len(feature_name)):
        col = feature_name[i]
        axes[i].hist(sample_feature[col], bins = 50, color='red', alpha=0.5, label='sample', density=True)
        axes[i].hist(subsample_feature[col], bins = 50, color='blue', alpha=0.5, label='subsample', density=True)
        axes[i].set_title(f"{col}")
        axes[i].legend()
    fig.suptitle(f"{sample_name}")
    plt.show()
    plt.close()


for key, obj in samples.items():
    subsample_vs_sample_distribution(graph_dict[key], obj,feature_name,key)

