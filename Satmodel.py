import torch
from torch import nn
import torch_geometric.nn as gnn
from satlayer import TransformerEncoderLayer
from einops import repeat


class GraphTransformerEncoder(nn.TransformerEncoder):
    def forward(self, x, edge_index,edge_attr=None,
            ptr=None, return_attn=False):
        output = x

        for mod in self.layers:
            output = mod(output, edge_index, 
                edge_attr=edge_attr,
                ptr=ptr,
                return_attn=return_attn
            )
        if self.norm is not None:
            output = self.norm(output)
        return output


class GraphTransformer(nn.Module):
    def __init__(self, in_size, num_class, d_model, num_heads=8,
                 dim_feedforward=512, dropout=0.0, num_layers=4,
                 batch_norm=False,  use_global_pool=True,
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
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        node_depth = data.node_depth if hasattr(data, "node_depth") else None

        output = self.embedding(x) if node_depth is None else self.embedding(x, node_depth.view(-1,))

        output = self.encoder(
            output, 
            edge_index, 
            edge_attr=edge_attr, 
            ptr=data.ptr,
            return_attn=return_attn
        )
        # readout step
        output = gnn.global_mean_pool(output, data.batch)
        return self.classifier(output)