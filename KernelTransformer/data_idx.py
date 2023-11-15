from typing import Any
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
import torch_geometric.utils as utils
from torch_geometric.data import Data
import numpy as np
import os

def my_inc(self, key, value, *args, **kwargs):
    if key == 'subgraph_edge_index':
        return self.num_subgraph_nodes
    if key == 'subgraph_node_index':
        return self.num_nodes
    if key == 'subgraph_indicator':
        return self.num_nodes
    if 'index' in key:
        return self.num_nodes
    else:
        return 0


class GraphDataset(object):
    def __init__(self, dataset, k_hop=2, use_subgraph_edge_attr=False):
        
        self.dataset = dataset
        self.n_features = dataset[0].x.shape[-1]
        self.k_hop = k_hop
        self.use_subgraph_edge_attr = use_subgraph_edge_attr
        Data.__inc__ = my_inc
        self.extract_subgraph()
        
    def extract_subgraph(self):
        print(f"extract {self.k_hop} hops subgraph")

        self.subgraph_node_index = []
        self.subgraph_edge_index = []
        self.subgraph_indicator_index = []
        if self.use_subgraph_edge_attr :
            self.subgraph_edge_attr = []

        for i in range(len(self.dataset)):
            graph = self.dataset[i]
            node_indices = []
            edge_indices = []
            edge_attributes = []
            indicator = []
            edge_start = 0
            for node_idx in range(graph.num_nodes):
                sub_nodes, sub_edge_index, _, edge_mask = utils.k_hop_subgraph(node_idx, self.k_hop, graph.edge_index, True)
                node_indices.append(sub_nodes)
                edge_indices.append(sub_edge_index + edge_start)
                indicator.append(torch.zeros(sub_nodes.shape[0]).fill_(node_idx))
                if self.use_subgraph_edge_attr and graph.edge_attr is not None:
                    edge_attributes.append(graph.edge_attr[edge_mask])
                edge_start += len(sub_nodes)
        
            self.subgraph_node_index.append(torch.cat(node_indices))
            self.subgraph_edge_index.append(torch.cat(edge_indices, dim=1))
            self.subgraph_indicator_index.append(torch.cat(indicator))
            if self.use_subgraph_edge_attr and graph.edge_attr is not None:
                self.subgraph_edge_attr.append(torch.cat(edge_attributes))
        print("End")

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        
        data = self.dataset[index]

        # if self.n_features == 1:
        #     data.x = data.x.squeeze(-1)

        # if not isinstance(data.y, list):
        #     data.y = data.y.view(-1)
        #         # data.y = data.y.view(data.y.shape[0], -1)
        # n = data.num_nodes
        data.idx = []
        data.idx.append(index)
        data.subgraph_node_index = self.subgraph_node_index[index]
        data.subgraph_edge_index = self.subgraph_edge_index[index]
        data.num_subgraph_nodes = len(self.subgraph_node_index[index])
        if self.use_subgraph_edge_attr and data.edge_attr is not None:
            data.subgraph_edge_attr = self.subgraph_edge_attr[index]
        data.subgraph_indicator = self.subgraph_indicator_index[index].type(torch.LongTensor)

        return data
