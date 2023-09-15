import torch
from torch import nn
import torch_geometric.nn as gnn
from argskernellayer import TransformerEncoderLayer
from einops import repeat
from scipy.cluster.vq import kmeans2
#k-means clustering to extract features from vectors each layer with more layers. 

# from time import time as gettime
class GraphTransformerEncoder(nn.TransformerEncoder):
    def forward(self, x, num_nodes, edge_index, original_x=None, edge_attr=None,
            ptr=None, num_graphs=None, batch=None, same_attn=True, kernel_type='SP', num_hops=2):
        output = x
        original_x = torch.argmax(original_x, dim=1)
        # print(f"original label: {original_x}")
        attn_weight = None
        for idx, mod in enumerate(self.layers):
            if idx == 0 or not same_attn:
                output, attn_weight = mod(output,
                num_nodes,
                edge_index, 
                original_x=original_x,
                edge_attr=edge_attr,
                ptr=ptr,
                num_graphs = num_graphs,
                batch=batch,
                kernel_type=kernel_type,
                num_hops=num_hops
            )
            else:
                output, _ = mod(output,
                num_nodes,
                edge_index, 
                original_x=original_x,
                edge_attr=edge_attr,
                ptr=ptr,
                num_graphs = num_graphs,
                batch=batch,
                kernel_type=kernel_type,
                num_hops=num_hops,
                attn_weight = attn_weight
            ) 
        if self.norm is not None:
            output = self.norm(output)
        return output


class GraphTransformer(nn.Module):
    def __init__(self, in_size, num_class, d_model, num_heads=8,
                 dim_feedforward=512, dropout=0.0, num_layers=4,
                 batch_norm=True,  use_global_pool=True, same_attn=True,
                 kernel='SP', hop=2, global_pool='mean', **kwargs):
        super().__init__()
        self.kernel_type = kernel
        self.num_hops = hop
        self.same_attn=same_attn
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

        # new_time = gettime()
        node_depth = data.node_depth if hasattr(data, "node_depth") else None
        output = self.embedding(x.float()) if node_depth is None else self.embedding(x.float(), node_depth.view(-1,))
        
        output = self.encoder(
            output, 
            num_nodes,
            edge_index, 
            original_x=x,
            edge_attr=edge_attr, 
            ptr=data.ptr,
            num_graphs=num_graphs,
            batch=batch,
            same_attn=self.same_attn,
            kernel_type=self.kernel_type,
            num_hops=self.num_hops
        )
        # end_time = gettime()
        # time = end_time - new_time
        # print("transformer_time", time)
        # readout step
        output = gnn.global_mean_pool(output, data.batch)
        return self.classifier(output)


#quantinized the node features output from the last layer. Using the k-means clustering.
class SoftAss(nn.Module):
    def __init__(self, num_words, features_dim, softmax=True):
        super(SoftAss, self).__init__()

        if softmax:
            self.normalize = nn.Softmax(-1)
        else:
            self.normalize = nn.Normalize(dim=-1)

        #create a k-means dict which can be leanred. size(num_words x features_dim)
        self.dict = nn.Parameter(torch.rand(num_words, features_dim).float(), requires_grad=True)
        self.codebook_init = False

    def reset_codebook(self, x):
        if self.codebook_init:
            centroid, label = kmeans2(data = x.detach().cpu().numpy(), k = self.dict.detach().cpu().numpy(), minit='matrix')
        else:
            centroid, label = kmeans2(data = (x + torch.randn_like(x) * 1e-4).detach().cpu().numpy(), k = self.dict.shape[0])
        self.dict.data = torch.from_numpy(centroid).float().to(x.device)
        self.codebook_init = True

    def forward(self, x):
        if self.training:
            self.reset_codebook(x)

        _x = self.normalize(x)

        res = torch.tensordot(_x, self.dict, dims=([1], [1]))

        return res