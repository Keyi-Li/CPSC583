#!/usr/bin/env python
# coding: utf-8

# ## 1. Packages and Functions

# In[4]:


# !pip install igraph
# !pip install pyarrow


# In[1]:


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


# In[2]:


working_dir = "/data/banach2/sennet/codex/processed"
output_dir = "/data/banach2/kl825/stg_gnn_delaunay"
os.chdir(working_dir)


# ## 2. Data Information and Preprocessing

# In[3]:


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

# In[5]:


#!pip install faiss-gpu


# In[4]:


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

## use note:
# nn.xy # coordinates in 2D
# nn.knn[i,:] # for node i gives the indices of the nearest neighbors, if it is -1, then it is like a missing value
# nn.dnn[i,:] # for node i gives the squares of the distances to the nearest neighbors, see nn.knn
# nn.gdl[i,:] # for node i gives the indices of the Delaunay neighbors
# nn.g # igraph Graph from Delaunay trilateration
# nn.g.get_edgelist()# for edge_list
# nn.g.vs.indices # for all indices


# In[5]:


for sample, info in sample_meta.items():
    logger.info(f"Processing {sample}:")
    nn = mknn.from_coords(info["Spatial"].values)
    sample_meta[sample]["nn"] = nn


# ## 4. Graph Extraction

# In[13]:


class whole_sample:
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

    def get_available_neighbor(self, cell_idx):
        neighbor_list = self.get_delaunay_neighbor(cell_idx)
        
        available_neighbor = set(neighbor_list).difference(self.subgraph_node)
        return list(available_neighbor) 
        
    def update_node_avail(self, nodes:list):
        self.subgraph_node = set.union(self.subgraph_node, nodes)

    def unique_node(self, nodes_taken, potential_nodes):
        return list(set(potential_nodes).difference(nodes_taken))

    def edges_k_hop(self,cell_idx,k_hop,max_squared_dis = 900, cur_hop = 0):
        all_edges = []
        if cur_hop >= k_hops:
            return all_edges
        neis = self.get_delaunay_neighbor(cell_idx,max_squared_dis)
        all_edges.extend([[cell_idx, nei] for nei in neis])
        for nei in neis:
            all_edges.extend(self.edges_k_hop(nri, k_hop, cur_hop+1))
        return all_edges

    def delaunay_subgraph_extract_all(self,node_of_interest, graph_size, no_overlap = False):
        subgraph_list = []
        for node in node_of_interest:
            if node not in self.subgraph_node:
                flag, subgraph = self.make_subgraph(node, graph_size, no_overlap = no_overlap)
                if flag:
                    subgraph_list.append(subgraph)
        return subgraph_list

    def make_subgraph(self, node, graph_size, no_overlap = False):
        node_list = [node]
        edge_list = []
        
        k_hop_neighbor = self.get_available_neighbor(node)
        edge_list += [(node, neighbor) for neighbor in k_hop_neighbor]
        #edge_list += [(neighbor, nb) for neighbor in k_hop_neighbor]
        node_list += k_hop_neighbor
        #self.update_node_avail(node_list)
        
        while len(node_list) < graph_size:
            if len(k_hop_neighbor) == 0:
                break
            else:
                next_hop = []
                for nb in k_hop_neighbor:
                    next_hop_neighbor = self.get_available_neighbor(nb)
                    edge_list += [(nb, neighbor) for neighbor in next_hop_neighbor]
                    #edge_list += [(neighbor, nb) for neighbor in next_hop_neighbor]
                    next_hop += self.unique_node(node_list,next_hop_neighbor)
                    node_list += self.unique_node(node_list,next_hop_neighbor)
                   # self.update_node_avail(node_list)
                k_hop_neighbor = next_hop
    
        flag = len(node_list) >= graph_size
        
        if flag and no_overlap:
            self.update_node_avail(node_list)
            
        return flag, (node_list, list(set(edge_list)))


# In[8]:


def filtering(condition_dict, sample):
    marker = list(condition_dict.keys())
    profile = sample.get_marker(marker)
    criterion = []
    for marker, condition in condition_dict.items():
        if condition[0] == "=":
            criterion.append((profile[marker]== condition[1]).values)
        elif condition[0][0] == ">":
            thre = np.quantile(profile[marker].values, condition[1])
            criterion.append((profile[marker] >= thre).values)
        else:
            thre = np.quantile(profile[marker].values, condition[1])
            criterion.append((profile[marker] <= thre).values)     

    passed_row = np.where(np.all(np.vstack(criterion).T, axis = 1))[0]
    return passed_row.tolist()


# In[14]:


samples = {key : whole_sample(**value) for key, value in sample_meta.items()}


# In[16]:


np.random.seed(42)
subgraph_dict_50_allow_overlap ={key : [] for key in samples.keys()}
for key, data in tqdm(samples.items(), desc='Processing samples', total=len(samples)):
    node_idx_1 = filtering({"Ki67": ["=", 0],
                          "p16": [">", 0.99],
                         }, data)
    node_idx_2 = filtering({"Ki67": ["=", 0],
                          "p21": [">", 0.99],
                         }, data)
    node_idx = list(set(node_idx_1).union(node_idx_2))
    thre1 =0
    thre2 = 0.99
    while(len(node_idx) < 1000):
        thre1 +=0.01
        thre2 -=0.01
        node_idx_1 = filtering({"Ki67": ["<", thre1],
                          "p16": [">", thre2],
                         }, data)
        node_idx_2 = filtering({"Ki67": ["<", thre1],
                          "p21": [">", thre2],
                         }, data)
        node_idx = list(set(node_idx_1).union(node_idx_2))

        if len(node_idx) >=1000:
            print(f"Sample {key}, thre1:{thre1:.2f}, thre2:{thre2:.2f}")
    
    if len(node_idx) > 1000:
        node_idx = np.random.choice(node_idx, size=1000, replace=False)
    data.subgraph_node = set()
    subgraph_dict_50_allow_overlap[key] = data.delaunay_subgraph_extract_all(node_idx,50)
    print(f"Sample {key}: extracted {len(subgraph_dict_50_allow_overlap[key])}/{len(node_idx)} graphs")


# In[17]:


np.random.seed(42)
subgraph_dict_100_allow_overlap ={key : [] for key in samples.keys()}
for key, data in tqdm(samples.items(), desc='Processing samples', total=len(samples)):
    node_idx_1 = filtering({"Ki67": ["=", 0],
                          "p16": [">", 0.99],
                         }, data)
    node_idx_2 = filtering({"Ki67": ["=", 0],
                          "p21": [">", 0.99],
                         }, data)
    node_idx = list(set(node_idx_1).union(node_idx_2))
    thre1 =0
    thre2 = 0.99
    while(len(node_idx) < 1000):
        thre1 +=0.01
        thre2 -=0.01
        node_idx_1 = filtering({"Ki67": ["<", thre1],
                          "p16": [">", thre2],
                         }, data)
        node_idx_2 = filtering({"Ki67": ["<", thre1],
                          "p21": [">", thre2],
                         }, data)
        node_idx = list(set(node_idx_1).union(node_idx_2))

        if len(node_idx) >=1000:
            print(f"Sample {key}, thre1:{thre1:.2f}, thre2:{thre2:.2f}")
    
    if len(node_idx) > 1000:
        node_idx = np.random.choice(node_idx, size=1000, replace=False)
    data.subgraph_node = set()
    subgraph_dict_100_allow_overlap[key] = data.delaunay_subgraph_extract_all(node_idx,100)
    print(f"Sample {key}: extracted {len(subgraph_dict_100_allow_overlap[key])}/{len(node_idx)} graphs")


# In[18]:


np.random.seed(42)
subgraph_dict_200_allow_overlap ={key : [] for key in samples.keys()}
for key, data in tqdm(samples.items(), desc='Processing samples', total=len(samples)):
    node_idx_1 = filtering({"Ki67": ["=", 0],
                          "p16": [">", 0.99],
                         }, data)
    node_idx_2 = filtering({"Ki67": ["=", 0],
                          "p21": [">", 0.99],
                         }, data)
    node_idx = list(set(node_idx_1).union(node_idx_2))
    thre1 =0
    thre2 = 0.99
    while(len(node_idx) < 1000):
        thre1 +=0.01
        thre2 -=0.01
        node_idx_1 = filtering({"Ki67": ["<", thre1],
                          "p16": [">", thre2],
                         }, data)
        node_idx_2 = filtering({"Ki67": ["<", thre1],
                          "p21": [">", thre2],
                         }, data)
        node_idx = list(set(node_idx_1).union(node_idx_2))

        if len(node_idx) >=1000:
            print(f"Sample {key}, thre1:{thre1:.2f}, thre2:{thre2:.2f}")
    
    if len(node_idx) > 1000:
        node_idx = np.random.choice(node_idx, size=1000, replace=False)
    data.subgraph_node = set()
    subgraph_dict_200_allow_overlap[key] = data.delaunay_subgraph_extract_all(node_idx,200)
    print(f"Sample {key}: extracted {len(subgraph_dict_200_allow_overlap[key])}/{len(node_idx)} graphs")


# In[19]:


np.random.seed(42)
subgraph_dict_1000_allow_overlap ={key : [] for key in samples.keys()}
for key, data in tqdm(samples.items(), desc='Processing samples', total=len(samples)):
    node_idx_1 = filtering({"Ki67": ["=", 0],
                          "p16": [">", 0.99],
                         }, data)
    node_idx_2 = filtering({"Ki67": ["=", 0],
                          "p21": [">", 0.99],
                         }, data)
    node_idx = list(set(node_idx_1).union(node_idx_2))
    thre1 =0
    thre2 = 0.99
    while(len(node_idx) < 1000):
        thre1 +=0.01
        thre2 -=0.01
        node_idx_1 = filtering({"Ki67": ["<", thre1],
                          "p16": [">", thre2],
                         }, data)
        node_idx_2 = filtering({"Ki67": ["<", thre1],
                          "p21": [">", thre2],
                         }, data)
        node_idx = list(set(node_idx_1).union(node_idx_2))

        if len(node_idx) >=1000:
            print(f"Sample {key}, thre1:{thre1:.2f}, thre2:{thre2:.2f}")
    
    if len(node_idx) > 1000:
        node_idx = np.random.choice(node_idx, size=1000, replace=False)
    data.subgraph_node = set()
    subgraph_dict_1000_allow_overlap[key] = data.delaunay_subgraph_extract_all(node_idx,1000)
    print(f"Sample {key}: extracted {len(subgraph_dict_1000_allow_overlap[key])}/{len(node_idx)} graphs")


# In[20]:


np.random.seed(42)
subgraph_dict_50_no_overlap ={key : [] for key in samples.keys()}
for key, data in tqdm(samples.items(), desc='Processing samples', total=len(samples)):
    node_idx_1 = filtering({"Ki67": ["=", 0],
                          "p16": [">", 0.99],
                         }, data)
    node_idx_2 = filtering({"Ki67": ["=", 0],
                          "p21": [">", 0.99],
                         }, data)
    node_idx = list(set(node_idx_1).union(node_idx_2))
    thre1 =0
    thre2 = 0.99
    while(len(node_idx) < 1000):
        thre1 +=0.01
        thre2 -=0.01
        node_idx_1 = filtering({"Ki67": ["<", thre1],
                          "p16": [">", thre2],
                         }, data)
        node_idx_2 = filtering({"Ki67": ["<", thre1],
                          "p21": [">", thre2],
                         }, data)
        node_idx = list(set(node_idx_1).union(node_idx_2))

        if len(node_idx) >=1000:
            print(f"Sample {key}, thre1:{thre1:.2f}, thre2:{thre2:.2f}")
    
    if len(node_idx) > 1000:
        node_idx = np.random.choice(node_idx, size=1000, replace=False)
    data.subgraph_node = set()
    subgraph_dict_50_no_overlap[key] = data.delaunay_subgraph_extract_all(node_idx,50,no_overlap = True)
    print(f"Sample {key}: extracted {len(subgraph_dict_50_no_overlap[key])}/{len(node_idx)} graphs")


# In[21]:


np.random.seed(42)
subgraph_dict_100_no_overlap ={key : [] for key in samples.keys()}
for key, data in tqdm(samples.items(), desc='Processing samples', total=len(samples)):
    node_idx_1 = filtering({"Ki67": ["=", 0],
                          "p16": [">", 0.99],
                         }, data)
    node_idx_2 = filtering({"Ki67": ["=", 0],
                          "p21": [">", 0.99],
                         }, data)
    node_idx = list(set(node_idx_1).union(node_idx_2))
    thre1 =0
    thre2 = 0.99
    while(len(node_idx) < 1000):
        thre1 +=0.01
        thre2 -=0.01
        node_idx_1 = filtering({"Ki67": ["<", thre1],
                          "p16": [">", thre2],
                         }, data)
        node_idx_2 = filtering({"Ki67": ["<", thre1],
                          "p21": [">", thre2],
                         }, data)
        node_idx = list(set(node_idx_1).union(node_idx_2))

        if len(node_idx) >=1000:
            print(f"Sample {key}, thre1:{thre1:.2f}, thre2:{thre2:.2f}")
    
    if len(node_idx) > 1000:
        node_idx = np.random.choice(node_idx, size=1000, replace=False)
    data.subgraph_node = set()
    subgraph_dict_100_no_overlap[key] = data.delaunay_subgraph_extract_all(node_idx,100,no_overlap = True)
    print(f"Sample {key}: extracted {len(subgraph_dict_100_no_overlap[key])}/{len(node_idx)} graphs")


# In[67]:


np.random.seed(42)
subgraph_dict_200_no_overlap ={key : [] for key in samples.keys()}
for key, data in tqdm(samples.items(), desc='Processing samples', total=len(samples)):
    node_idx_1 = filtering({"Ki67": ["=", 0],
                          "p16": [">", 0.99],
                         }, data)
    node_idx_2 = filtering({"Ki67": ["=", 0],
                          "p21": [">", 0.99],
                         }, data)
    node_idx = list(set(node_idx_1).union(node_idx_2))
    thre1 =0
    thre2 = 0.99
    while(len(node_idx) < 1000):
        thre1 +=0.01
        thre2 -=0.01
        node_idx_1 = filtering({"Ki67": ["<", thre1],
                          "p16": [">", thre2],
                         }, data)
        node_idx_2 = filtering({"Ki67": ["<", thre1],
                          "p21": [">", thre2],
                         }, data)
        node_idx = list(set(node_idx_1).union(node_idx_2))

        if len(node_idx) >=1000:
            print(f"Sample {key}, thre1:{thre1:.2f}, thre2:{thre2:.2f}")
    
    if len(node_idx) > 1000:
        node_idx = np.random.choice(node_idx, size=1000, replace=False)
    data.subgraph_node = set()
    subgraph_dict_200_no_overlap[key] = data.delaunay_subgraph_extract_all(node_idx,200,no_overlap = True)
    print(f"Sample {key}: extracted {len(subgraph_dict_200_no_overlap[key])}/{len(node_idx)} graphs")


# In[68]:


np.random.seed(42)
subgraph_dict_1000_no_overlap ={key : [] for key in samples.keys()}
for key, data in tqdm(samples.items(), desc='Processing samples', total=len(samples)):
    node_idx_1 = filtering({"Ki67": ["=", 0],
                          "p16": [">", 0.99],
                         }, data)
    node_idx_2 = filtering({"Ki67": ["=", 0],
                          "p21": [">", 0.99],
                         }, data)
    node_idx = list(set(node_idx_1).union(node_idx_2))
    thre1 =0
    thre2 = 0.99
    while(len(node_idx) < 1000):
        thre1 +=0.01
        thre2 -=0.01
        node_idx_1 = filtering({"Ki67": ["<", thre1],
                          "p16": [">", thre2],
                         }, data)
        node_idx_2 = filtering({"Ki67": ["<", thre1],
                          "p21": [">", thre2],
                         }, data)
        node_idx = list(set(node_idx_1).union(node_idx_2))

        if len(node_idx) >=1000:
            print(f"Sample {key}, thre1:{thre1:.2f}, thre2:{thre2:.2f}")
    
    if len(node_idx) > 1000:
        node_idx = np.random.choice(node_idx, size=1000, replace=False)
    data.subgraph_node = set()
    subgraph_dict_1000_no_overlap[key] = data.delaunay_subgraph_extract_all(node_idx,1000,no_overlap = True)
    print(f"Sample {key}: extracted {len(subgraph_dict_1000_no_overlap[key])}/{len(node_idx)} graphs")


# ## 5. Torch Geometric Data

# In[24]:


def zero_preserved_beta_quantile_normalization(array,a=4, b=4):
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


# In[46]:


def create_graph_data(node_feature, edge_index, y, sample_name, sample_id, cell_idx, spatial, age, feature_name):
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
    data = Data(x= node_feature, edge_index=edge_index_undirected,y=y)
    data.sample = sample_name
    data.sample_id = sample_id
    data.cell_idx = cell_idx
    data.spatial = spatial
    data.age = age
    data.feature_name = feature_name
    return data


# In[47]:


def reindex(node_list, edge_list):
    n = len(node_list)
    mapping = dict(zip(node_list,range(n)))
    re_index_list = []
    for item in edge_list:
        re_index_list.append([mapping[item[0]],mapping[item[1]]])
    return re_index_list


# In[48]:


def create_dataset(subgraph_dict, samples):
    data_list = []
    sample_id =0
    for name, obj in tqdm(samples.items(), desc="Processing samples"):
        graph_group = subgraph_dict[name]
        for node_list, edge_list in tqdm(graph_group, desc=f"Processing graph groups for sample {name}", leave=False):
            profile = obj.profile.iloc[node_list,:]
            node_feature = torch.tensor(profile.values, dtype=torch.float)
            feature_name = list(obj.profile.columns)
    
            spatial = obj.spatial.iloc[node_list,:]
            
            y = 1 if obj.age >=50 else 0
            y = torch.tensor([y])
    
            sample = name
            reindex_edge_list = reindex(node_list,edge_list)
            edge_idx = torch.tensor(reindex_edge_list,dtype=torch.long).t().contiguous()
    
            data = create_graph_data(node_feature, edge_idx, y, sample, sample_id,node_list, spatial.values, obj.age, feature_name)
            data_list.append(data)
        sample_id +=1
    return data_list


# In[171]:


# def create_dataset(subgraph_dict, samples, normalization = [True,4,4]):
#     data_list = []
#     sample_id =0
#     for name, obj in samples.items():
#         graph_group = subgraph_dict[name]
#         for node_list, edge_list in graph_group:
#             profile = obj.profile.iloc[node_list,:]
#             if normalization[0]:
#                 normalized_profile = profile.apply(lambda col: zero_preserved_beta_quantile_normalization(col,normalization[1],normalization[2]), axis=0)
#             node_feature = torch.tensor(normalized_profile.values, dtype=torch.float)
#             feature_name = list(obj.profile.columns)
    
#             spatial = obj.spatial.iloc[node_list,:]
            
#             y = 1 if obj.age >=50 else 0
#             y = torch.tensor([y])
    
#             sample = name
#             reindex_edge_list = reindex(node_list,edge_list)
#             edge_idx = torch.tensor(reindex_edge_list,dtype=torch.long).t().contiguous()
    
#             data = create_graph_data(node_feature, edge_idx, y, sample, sample_id,node_list, spatial.values, obj.age, feature_name)
#             data_list.append(data)
#         sample_id +=1
#     return data_list


# In[29]:


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


# In[44]:


# quantile_normalization for samples
samples = {key : whole_sample(**value) for key, value in sample_meta.items()}
for name, obj in tqdm(samples.items()):
    obj.profile = obj.profile.apply(lambda col: zero_preserved_beta_quantile_normalization(col,4,4), axis=0)


# In[69]:


# dataset_50_allow_overlap = create_dataset(subgraph_dict_50_allow_overlap, samples)
# dataset_100_allow_overlap = create_dataset(subgraph_dict_100_allow_overlap, samples)
# dataset_200_allow_overlap = create_dataset(subgraph_dict_200_allow_overlap, samples)
# dataset_1000_allow_overlap = create_dataset(subgraph_dict_1000_allow_overlap, samples)

# dataset_50_no_overlap = create_dataset(subgraph_dict_50_no_overlap, samples)
# dataset_100_no_overlap = create_dataset(subgraph_dict_100_no_overlap, samples)
dataset_200_no_overlap = create_dataset(subgraph_dict_200_no_overlap, samples)
dataset_1000_no_overlap = create_dataset(subgraph_dict_1000_no_overlap, samples)


# In[70]:


# save_dataset(dataset_50_allow_overlap, output_dir, "50_allow_overlap")
# save_dataset(dataset_100_allow_overlap, output_dir, "100_allow_overlap")
# save_dataset(dataset_200_allow_overlap, output_dir, "200_allow_overlap")
# save_dataset(dataset_1000_allow_overlap, output_dir, "1000_allow_overlap")

# save_dataset(dataset_50_no_overlap, output_dir, "50_no_overlap")
# save_dataset(dataset_100_no_overlap, output_dir, "100_no_overlap")
save_dataset(dataset_200_no_overlap, output_dir, "200_no_overlap")
save_dataset(dataset_1000_no_overlap, output_dir, "1000_no_overlap")


# ## 6. Visualization

# In[52]:


np.random.seed(42)
def plot_spatial(dataset,size = 10, node_size =50):
    selected= np.random.choice(range(len(dataset)), size, replace = False)
    fig, axes = plt.subplots(int(size/5), 5, figsize=(24, 5*size/5),  gridspec_kw={'wspace': 0.4, 'hspace': 0.6})
    axes = axes.flatten()
    for i in range(size):
        idx = selected[i]
        G = torch_geometric.utils.to_networkx(dataset[idx])
        nx.draw_networkx_nodes(G, pos=dataset[idx].spatial, node_size = node_size, ax = axes[i]);
        nx.draw_networkx_edges(G, pos=dataset[idx].spatial,ax = axes[i],arrows=False);
        axes[i].set_title(dataset[idx].sample)
        axes[i].tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        axes[i].set_xlabel('X Axis')
        axes[i].set_ylabel('Y Axis')
    plt.show()
    plt.close(fig)


# In[53]:


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


# In[55]:


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


# In[54]:


plot_spatial(dataset_50_allow_overlap,size = 10)


# In[56]:


plot_spatial(dataset_100_allow_overlap,size = 10)


# In[59]:


plot_spatial(dataset_200_allow_overlap,size = 10)


# In[60]:


plot_spatial(dataset_1000_allow_overlap,size = 10, node_size = 5)


# In[61]:


plot_spatial(dataset_50_no_overlap,size = 10)


# In[63]:


plot_spatial(dataset_100_no_overlap,size = 10)


# In[71]:


plot_spatial(dataset_200_no_overlap,size = 10)


# In[72]:


plot_spatial(dataset_1000_no_overlap,size = 10, node_size = 5)


# In[75]:


random_slice_feature_examine(dataset_50_allow_overlap, 33, "dataset_50_allow_overlap",n = 2, mode = "normalized")


# In[76]:


random_slice_feature_examine(dataset_100_allow_overlap, 33, "dataset_100_allow_overlap",n = 2, mode = "normalized")


# In[77]:


random_slice_feature_examine(dataset_200_allow_overlap, 33, "dataset_200_allow_overlap",n = 2, mode = "normalized")


# In[78]:


random_slice_feature_examine(dataset_1000_allow_overlap, 33, "dataset_1000_allow_overlap",n = 2, mode = "normalized")


# In[80]:


fig, axes = plt.subplots(2,5, figsize = (40,20))
axes = axes.flatten()
for i in range(len(samples.keys())):
    key = list(samples.keys())[i]
    axes[i] = spatial_overlay(subgraph_dict_50_allow_overlap[key],samples[key],key, axes[i])
plt.suptitle("Graph Size Above 50: allow overlap", fontsize=20)
plt.show()


# In[81]:


fig, axes = plt.subplots(2,5, figsize = (40,20))
axes = axes.flatten()
for i in range(len(samples.keys())):
    key = list(samples.keys())[i]
    axes[i] = spatial_overlay(subgraph_dict_100_allow_overlap[key],samples[key],key, axes[i])
plt.suptitle("Graph Size Above 100: allow overlap", fontsize=20)
plt.show()


# In[82]:


fig, axes = plt.subplots(2,5, figsize = (40,20))
axes = axes.flatten()
for i in range(len(samples.keys())):
    key = list(samples.keys())[i]
    axes[i] = spatial_overlay(subgraph_dict_200_allow_overlap[key],samples[key],key, axes[i])
plt.suptitle("Graph Size Above 200: allow overlap", fontsize=20)
plt.show()


# In[83]:


fig, axes = plt.subplots(2,5, figsize = (40,20))
axes = axes.flatten()
for i in range(len(samples.keys())):
    key = list(samples.keys())[i]
    axes[i] = spatial_overlay(subgraph_dict_1000_allow_overlap[key],samples[key],key, axes[i])
plt.suptitle("Graph Size Above 1000: allow overlap", fontsize=20)
plt.show()


# In[84]:


fig, axes = plt.subplots(2,5, figsize = (40,20))
axes = axes.flatten()
for i in range(len(samples.keys())):
    key = list(samples.keys())[i]
    axes[i] = spatial_overlay(subgraph_dict_50_no_overlap[key],samples[key],key, axes[i])
plt.suptitle("Graph Size Above 50: no overlap", fontsize=20)
plt.show()


# In[85]:


fig, axes = plt.subplots(2,5, figsize = (40,20))
axes = axes.flatten()
for i in range(len(samples.keys())):
    key = list(samples.keys())[i]
    axes[i] = spatial_overlay(subgraph_dict_100_no_overlap[key],samples[key],key, axes[i])
plt.suptitle("Graph Size Above 100: no overlap", fontsize=20)
plt.show()


# In[86]:


fig, axes = plt.subplots(2,5, figsize = (40,20))
axes = axes.flatten()
for i in range(len(samples.keys())):
    key = list(samples.keys())[i]
    axes[i] = spatial_overlay(subgraph_dict_200_no_overlap[key],samples[key],key, axes[i])
plt.suptitle("Graph Size Above 200: no overlap", fontsize=20)
plt.show()


# In[87]:


fig, axes = plt.subplots(2,5, figsize = (40,20))
axes = axes.flatten()
for i in range(len(samples.keys())):
    key = list(samples.keys())[i]
    axes[i] = spatial_overlay(subgraph_dict_1000_no_overlap[key],samples[key],key, axes[i])
plt.suptitle("Graph Size Above 1000: no overlap", fontsize=20)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[482]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[132]:


def create_graph_data(node_feature, edge_index, y, sample_name, sample_id, cell_idx, spatial, age, feature_name):
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
    
    data = Data(x= node_feature, edge_index=edge_index,y=y)
    data.sample = sample_name
    data.sample_id = sample_id
    data.cell_idx = cell_idx
    data.spatial = spatial
    data.age = age
    data.feature_name = feature_name
    return data


# In[133]:


def reindex(node_list, edge_list):
    n = len(node_list)
    mapping = dict(zip(node_list,range(n)))
    re_index_list = []
    for item in edge_list:
        re_index_list.append([mapping[item[0]],mapping[item[1]]])
    return re_index_list


# In[134]:


def create_dataset(subgraph_dict, samples, normalization = [True,4,4]):
    data_list = []
    sample_id =0
    for name, obj in samples.items():
        graph_group = subgraph_dict[name]
        for node_list, edge_list in graph_group:
            profile = obj.profile.iloc[node_list,:]
            if normalization[0]:
                profile = profile.apply(lambda col: zero_preserved_beta_quantile_normalization(col,normalization[1],normalization[2]), axis=0)
            node_feature = torch.tensor(profile.values, dtype=torch.float)
            feature_name = list(obj.profile.columns)
    
            spatial = obj.spatial.iloc[node_list,:]
            
            y = 1 if obj.age >=50 else 0
            y = torch.tensor([y])
    
            sample = name
            reindex_edge_list = reindex(node_list,edge_list)
            edge_idx = torch.tensor(reindex_edge_list,dtype=torch.long).t().contiguous()
    
            data = create_graph_data(profile.values, edge_idx, y, sample, sample_id,node_list, spatial.values, obj.age, feature_name)
            data_list.append(data)
        sample_id +=1
    return data_list


# In[512]:


# def create_dataset(subgraph_dict, samples, normalization = [True,4,4]):
#     data_list = []
#     sample_id =0
#     for name, obj in samples.items():
#         graph_group = subgraph_dict[name]
#         for (node_list, edge_list) in graph_group:
#             profile = obj.profile.iloc[node_list,:]
#             if normalization[0]:
#                 profile = profile.apply(lambda col: zero_preserved_beta_quantile_normalization(col,normalization[1],normalization[2]), axis=0)
#             node_feature = torch.tensor(normalized_profile.values, dtype=torch.float)
#             feature_name = list(obj.profile.columns)
    
#             spatial = obj.spatial.iloc[node_list,:]
            
#             y = 1 if obj.age >=50 else 0
#             y = torch.tensor([y])
    
#             sample = name
#             #reindex_edge_list = reindex(node_list,edge_list)
#             #edge_idx = torch.tensor(reindex_edge_list,dtype=torch.long).t().contiguous()
#             edge_idx = torch.tensor(edge_list,dtype=torch.long).t().contiguous()
    
#             data = create_graph_data(profile.values, edge_idx, y, sample, sample_id,node_list, spatial.values, obj.age, feature_name)
#             data_list.append(data)
#         sample_id +=1
#     return data_list


# In[135]:


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


# In[136]:


temp = create_dataset(subgraph_dict_50, samples)


# In[138]:


temp[0]


# In[137]:


plot_spatial(temp,30)


# In[ ]:





# In[495]:


subgraph_dict


# In[496]:


plot_spatial(temp,30)


# In[ ]:





# In[390]:


data_list_50 = create_dataset(subgraph_dict_50, samples)


# In[391]:


data_list_500 = create_dataset(subgraph_dict_500, samples)


# In[398]:


data_list_100 = create_dataset(subgraph_dict_100, samples)


# In[396]:


save_dataset(data_list_50, output_dir, "50")
save_dataset(data_list_500, output_dir, "500")


# In[399]:


save_dataset(data_list_100, output_dir, "100")


# In[477]:


temp[0]


# In[476]:


plot_spatial(temp,30)


# In[450]:


np.random.seed(42)
def plot_spatial(dataset,size = 10):
    selected= np.random.choice(range(len(dataset)), size, replace = False)
    fig, axes = plt.subplots(int(size/5), 5, figsize=(24, 5*size/5),  gridspec_kw={'wspace': 0.4, 'hspace': 0.6})
    axes = axes.flatten()
    for i in range(size):
        idx = selected[i]
        G = torch_geometric.utils.to_networkx(dataset[idx])
        nx.draw_networkx_nodes(G, pos=dataset[idx].spatial, node_size = 50, ax = axes[i]);
        nx.draw_networkx_edges(G, pos=dataset[idx].spatial,ax = axes[i]);
        axes[i].scatter(dataset[idx].spatial[:,0], dataset[idx].spatial[:,1])
        axes[i].set_title(dataset[idx].sample)
        axes[i].tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        axes[i].set_xlabel('X Axis')
        axes[i].set_ylabel('Y Axis')
    plt.show()
    plt.close(fig)


# In[428]:


data_list_50[0]


# In[438]:


dataset = data_list_50
size = 10
selected= np.random.choice(range(len(dataset)), size, replace = False)
fig, axes = plt.subplots(2, 5, figsize=(24, 10))
axes = axes.flatten()
for i in range(size):
    idx = selected[i]
    G = torch_geometric.utils.to_networkx(dataset[idx])
    nx.draw_networkx_nodes(G, pos=dataset[idx].spatial, node_size = 25, ax = axes[i]);
    nx.draw_networkx_edges(G, pos=dataset[idx].spatial,ax = axes[i]);
    axes[i].scatter(dataset[idx].spatial[:,0], dataset[idx].spatial[:,1])
    axes[i].set_title(dataset[idx].sample)
    axes[i].set_xlabel('X Axis')
    axes[i].set_ylabel('Y Axis')
plt.show()


# In[451]:


plot_spatial(data_list_50, 20)


# In[405]:


G = torch_geometric.utils.to_networkx(data_list_50[0])


# In[412]:


nx.draw_networkx_nodes(G, pos=data_list_50[0].spatial, node_size = 50);
nx.draw_networkx_edges(G, pos=data_list_50[0].spatial);
plt.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

# Set labels for x and y axes if needed
plt.xlabel('X Axis')
plt.ylabel('Y Axis')

# Show the plot
plt.show()


# In[ ]:





# In[ ]:





# In[324]:


for key, data in tqdm(samples.items(), desc='Processing samples', total=len(samples)):
    node_idx = filtering({"Ki67": ["<", 0.05],
                          "p16": [">", 0.95],
                          "p21": [">", 0.95],
                         }, data)
    print(f"{key}: {len(node_idx)}")


# In[321]:


for key, data in tqdm(samples.items(), desc='Processing samples', total=len(samples)):
    node_idx = filtering({"Ki67": ["<", 0.05],
                          "p16": [">", 0.95]
                         }, data)
    print(f"{key}: {len(node_idx)}")


# In[318]:


samples["LN13560"].age


# In[326]:


df = ranking(samples["LN291"],["Ki67","p16","p21"],[True, False, False])


# In[327]:


df["rank"] = np.sum(df[["rank_Ki67","rank_p16","rank_p21"]].values,axis = 1)


# In[328]:


df.sort_values("rank")


# In[297]:


data[data["rank_p21"] == 1]


# In[283]:


np.min(data["Ki67"])


# In[264]:


for key, data in tqdm(samples.items(), desc='Processing samples', total=len(samples)):
    node_idx = filtering({"Ki67": ["=", 0],
                          "p16": [">", 0.95],
                          "p21": [">", 0.95],
                         }, data)
    print(f"{key}: {len(node_idx)} : {data.age}")


# In[262]:


for key, data in tqdm(samples.items(), desc='Processing samples', total=len(samples)):
    node_idx = filtering({"Ki67": ["<", 0.05],
                          "p16": [">", 0.95],
                          "p21": [">", 0.95],
                         }, data)
    print(f"{key}: {len(node_idx)}")


# In[258]:


for key, data in tqdm(samples.items(), desc='Processing samples', total=len(samples)):
    node_idx = filtering({"Ki67": ["<", 0.1],
                          "p16": [">", 0.9],
                          "p21": [">", 0.9],
                         }, data)
    print(f"{key}: {len(node_idx)}")


# In[252]:


for key, data in samples.items():
    print(f"{key} : {data.n}")


# In[ ]:





# In[261]:


fig, axes = plt.subplots(nrows=1, ncols=3)
axes = axes.flatten()
markers = ["Ki67", "p16", "p21"]
marker_combin = list(itertools.combinations(markers, 2))
for i in range(3):
    axes[i].scatter(samples["LN13560"].profile[marker_combin[i][0]],samples["LN13560"].profile[marker_combin[i][1]], s=1)
    axes[i].set_xlabel(marker_combin[i][0])
    axes[i].set_ylabel(marker_combin[i][1])
plt.show()


# In[234]:


for key, graph_list in subgraph_dict_500.items():
    print(f"{key}: {len(graph_list)}")


# In[220]:


samples["LN291"].profile[np.all(np.vstack(cri).T, axis = 1)]


# In[169]:


subgraph_list = temp.delaunay_subgraph_extract_all([553535,2205199], 500)


# In[171]:


node_ls, edge_ls = subgraph_list[0]


# In[177]:


plt.scatter(temp.spatial.iloc[node_ls,:].values[:,0],temp.spatial.iloc[node_ls,:].values[:,1])


# In[226]:


new = samples["LN291"].profile
new.head()


# In[230]:


np.sum(new["Ki67"]==0)


# In[229]:


for key, data in samples.items():
    num = data.profile[data.profile["Ki67"] ==0].shape[0]
    print(f"{key}:{num}")


# In[153]:


a = set(temp.get_delaunay_neighbor(2205199))
b = set()
b.difference(a)


# In[ ]:


# Trial

temp = whole_sample(sample_me)
    def __init__(self, age, profile, spatial, graph_obj):


# In[ ]:



            
            edge_list = []
            node_list = [node]
            
            available_node = self.get_available_neighbor(node)
            edge_list += [(node, neighbor) for neighbor in available_node]
            self.subgraph_node = set.union(self.subgraph_node,node_list)
            while len(node_list) < graph_size:
                next_hop = []
                if len(available_node) ==0:
                    break
                else:
                    node_list += available_node
                    for vx in availble_node:
                        vx_neighbor += self.get_available_neighbor(vx)
                        edge_list += [(vx, neighbor) for neighbor in vx_neighbor]
                        next_hop += vx_neighbor
                        
                        self.subgrpah_node += vx_neighbor
            
                    
                    
                next_hop = [**self.get_available_neighbor(vertex) for vertex in available_node]
                node_list += next_hop
                self.subgraph += next_hop
                available_node = next_hop

            subgraph_list.append(subgraph_idx)
                
            neighbor = get_delaunay_neighbor(node)
            available_node = list(set(neighbor).diff(set(self.subgraph_node)))
            next_
            for current_node in available_node:
                next_hop_neighbor = []
            while len(node_list) < graph_size:
                neighbor = get_delaunay_neighbor(current_node)
                available_node = list(set(get))
            subgraph_list.append()
            current_neighbor = adj_mat[node].nonzero()[1].to_list():
            for neighbor in current_neighbor:
                if ! self.is_node_taken(neighbor):


# In[67]:


adj_matrix = sample_meta["LN291"]["nn"].g.get_adjacency_sparse()


# In[66]:


edge = sample_meta["LN291"]["nn"].g.get_edgelist()


# In[78]:


adj_matrix[1].nonzero()[1].tolist()


# In[76]:


print(adj_matrix[1,:])


# In[88]:


a = set([1,2])
b = [3]
set.union(a,b)


# In[ ]:




