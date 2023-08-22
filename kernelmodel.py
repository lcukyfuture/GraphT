import torch
from torch import nn
import torch_geometric.nn as gnn
from kernellayer import TransformerEncoderLayer
from einops import repeat


class GraphTransformerEncoder(nn.TransformerEncoder):
    def forward(self, x, num_nodes, edge_index, original_x=None, edge_attr=None,
            ptr=None, num_graphs=None, batch=None):
        output = x

        for mod in self.layers:
            output = mod(output, 
                num_nodes,
                edge_index, 
                original_x=original_x,
                edge_attr=edge_attr,
                ptr=ptr,
                num_graphs = num_graphs,
                batch=batch
            )
        if self.norm is not None:
            output = self.norm(output)
        return output


class GraphTransformer(nn.Module):
    def __init__(self, in_size, num_class, d_model, num_heads=8,
                 dim_feedforward=512, dropout=0.0, num_layers=4,
                 batch_norm=True,  use_global_pool=True,
                 global_pool='mean', **kwargs):
        super().__init__()


        self.embedding = nn.Linear(in_features=in_size,
                                    out_features=d_model,
                                    bias=False)
        
        encoder_layer = TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward, dropout, batch_norm=batch_norm, **kwargs)
        self.encoder = GraphTransformerEncoder(encoder_layer, num_layers)
        self.global_pool = global_pool
        if global_pool == 'mean':
            self.pooling = gnn.global_mean_pool
        elif global_pool == 'add':
            self.pooling = gnn.global_add_pool
        self.use_global_pool = use_global_pool

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(True),
            nn.Linear(d_model, num_class)
        )

    def forward(self, data, return_attn=False):
        x, edge_index, edge_attr, num_nodes, num_graphs, batch= data.x, data.edge_index, data.edge_attr, data.num_nodes, data.num_graphs, data.batch

        node_depth = data.node_depth if hasattr(data, "node_depth") else None
        output = self.embedding(x.float()) if node_depth is None else self.embedding(x.float(), node_depth.view(-1,))

        output = self.encoder(
            output, 
            num_nodes,
            edge_index, 
            original_x=x,
            edge_attr=edge_attr, 
            ptr=data.ptr,
            num_graphs = num_graphs,
            batch = batch
        )

        # readout step
        output = gnn.global_mean_pool(output, data.batch)
        return self.classifier(output)
