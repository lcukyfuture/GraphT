import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
from torch_geometric.utils import add_self_loops, softmax, to_undirected
from torch_geometric.nn import GCNConv, global_mean_pool, MessagePassing
from torch_scatter import scatter_mean
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, dataset, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, dataset.num_classes)
    
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)

        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x
class GraphMultiHeadAttention(MessagePassing):
    def __init__(self, embed_dim, num_heads=8, dropout=0., bias=False, symmetric=False, **kwargs):
        super().__init__(node_dim=0, aggr='add')
        self.embed_dim = embed_dim
        self.bias = bias
        self.num_heads = num_heads
        self.scale = embed_dim ** -0.5

        self.symmetric = symmetric
        if symmetric:
            self.to_qk = nn.Linear(embed_dim, embed_dim, bias=bias)
        else:
            self.to_qk = nn.Linear(embed_dim, embed_dim * 2, bias=bias)
        self.to_v = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.to_qk.weight)
        nn.init.xavier_uniform_(self.to_v.weight)

        if self.bias:
            nn.init.constant_(self.to_qk.bias, 0.)
            nn.init.constant_(self.to_v.bias, 0.)

    def forward(self, x, edge_index, return_attn=False):
        v = self.to_v(x)
        if self.symmetric:
            qk = self.to_qk(x)
            qk = (qk, qk)
        else:
            qk = self.to_qk(x).chunk(2, dim=-1)
        
        print(qk)
        print("vsize", v.size())
        out, attn = self.propagate(edge_index, v=v, qk=qk, size=None, return_attn=return_attn)
        out = out.view(out.shape[0], -1)
        return self.out_proj(out), attn

    def message(self, v_j, qk_j, qk_i, return_attn):
        qk_i = qk_i.view(-1, self.num_heads, self.embed_dim // self.num_heads)
        qk_j = qk_j.view(-1, self.num_heads, self.embed_dim // self.num_heads)
        v_j = v_j.view(-1, self.num_heads, self.embed_dim // self.num_heads)
        attn = (qk_i * qk_j).sum(-1) * self.scale
        attn = attn.softmax(dim=-1)
        if return_attn:
            self._attn = attn
        attn = self.attn_dropout(attn)
        return v_j * attn.unsqueeze(-1)

# class GraphMultiHeadAttention(MessagePassing):
#     def __init__(self, embed_dim, num_heads=8, dropout=0, bias=False, symmetric=False, **kwargs):
#         super().__init__(node_dim=0, aggr='add')
    
#         self.embed_dim = embed_dim
#         self.bias = bias
#         head_dim = embed_dim // num_heads
#         assert head_dim * num_heads == embed_dim

#         self.scale = head_dim ** -0.5
#         self.symmetric = symmetric
#         if symmetric:
#             self.to_qk = nn.Linear(embed_dim, embed_dim, bias=bias)
#         else:
#             self.to_qk = nn.Linear(embed_dim, embed_dim * 2, bias=bias)
#         self.to_v = nn.Linear(embed_dim, embed_dim, bias=bias)

#         self.attn_dropout = dropout
#         self.out_proj = nn.Linear(embed_dim, embed_dim)
#         self._rest_parameters()
#         self.attn_sum = None
    
#     def _reset_parameters(self):
#         nn.init.xavier_uniform_(self.to_qk.weight)
#         nn.init.xavier_uniform_(self.to_v.weight)

#         if self.bias:
#             nn.init.constant_(self.to_qk.bias, 0.)
#             nn.init.constant_(self.to_v.bias, 0.)
    
#     def forward(self, x, edge_index):
#         edge_index, _ = add_self_loops(edge_index, num)
#         if self.symmetric:
#             qk = self.to_qk(x)
#             qk = (qk, qk)
#             print("qk:", qk)
#         else:
#             qk = self.to_qk(x).chunk(2, dim=-1)
#         v = self.to_v(x)
#         out = self.propagate(edge_index, v=v, qk=qk, size=None)
#         out = out.view(out.shape[0], -1)
#         return self.out_proj(out)
    
#     def message(self, v_j, qk_j, qk_i) -> Tensor:
#         return super().message()
# class MultiHeadAttentionLayer(MessagePassing):
#     def __init__(self, in_dim, out_dim, num_heads):
#         super(MultiHeadAttentionLayer, self).__init__(aggr='add')  # "Add" aggregation
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.num_heads = num_heads
#         self.depth = out_dim // num_heads

#         self.lin_Q = nn.Linear(in_dim, out_dim)  # Note: the total dimensionality is out_dim
#         self.lin_K = nn.Linear(in_dim, out_dim)
#         self.lin_V = nn.Linear(in_dim, out_dim)

#     def forward(self, x, edge_index):
#         print(x.size())
#         Q = self.lin_Q(x).view(-1, self.num_heads, self.depth).permute(1, 0, 2)
#         K = self.lin_K(x).view(-1, self.num_heads, self.depth).permute(1, 0, 2)
#         V = self.lin_V(x).view(-1, self.num_heads, self.depth).permute(1, 0, 2)
#         print("Q size:", Q.size())
#         print("K size:", K.size())
#         print("V size:", V.size())
#         return self.propagate(edge_index, Q=Q, K=K, V=V, size=(x.size(0), x.size(0)))


#     def message(self, Q_j, K_i, V_j, edge_index, size):
#         print("Q_j size:", Q_j.size())
#         print("K_i size:", K_i.size())
#         print("V_j size:", V_j.size())
#         Q_j = Q_j.permute(1, 0, 2)
#         K_i = K_i.permute(1, 0, 2)
#         V_j = V_j.permute(1, 0, 2)
#         alpha = (Q_j * K_i).sum(dim=-1, keepdim = True) / np.sqrt(self.depth)  # compute attention scores
#         print("alpha_size:", alpha.size())
#         alpha = F.softmax(alpha, dim=1)

#         print("V_j:", V_j.size())  # softmax normalization
#         alpha = alpha.view(V_j.size(0), V_j.size(1), -1)
#         output = V_j * alpha
#         print("attn_size", output.size())
#         return output

#     def update(self, aggr_out):
#         return aggr_out.view(-1, self.out_dim)  # aggregate the heads


# class GraphTransformerLayer(nn.Module):
#     def __init__(self, in_dim, out_dim, num_heads, dropout=0.0, residual=True):
#         super(GraphTransformerLayer, self).__init__()
#         self.attention = GraphMultiHeadAttention(in_dim, out_dim, num_heads)

#         self.lin1 = nn.Linear(out_dim, out_dim)
#         self.lin2 = nn.Linear(out_dim, out_dim)
#         self.ln = nn.LayerNorm(out_dim)
#         self.dropout = nn.Dropout(dropout)
#         self.residual = residual

#     def forward(self, x, edge_index):
#         h = self.attention(x, edge_index)

#         # Residual connection
#         if self.residual:
#             h = h + x
#         h = self.dropout(h)
#         h = self.ln(h)

#         h = self.lin1(h)
#         h = F.relu(h)
#         h = self.lin2(h)

#         h = self.dropout(h)
#         h = self.ln(h)
#         h = F.relu(h)

#         return h

# class GraphTransformerEncoder(nn.Module):
#     def __init__(self, in_dim, out_dim, num_heads, num_layers):
#         super(GraphTransformerEncoder, self).__init__()
#         self.layers = nn.ModuleList()
#         for _ in range(num_layers):
#             self.layers.append(GraphTransformerLayer(in_dim if _==0 else out_dim, out_dim, num_heads))
    
#     def forward(self, x, edge_index):
#         print("Input shape:", x.size())
#         for layer in self.layers:
#             x = layer(x, edge_index)
#         return x

class GraphTransformerLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0, residual=True):
        super(GraphTransformerLayer, self).__init__()
        self.attention = GraphMultiHeadAttention(in_dim, num_heads, dropout=dropout)

        self.lin1 = nn.Linear(in_dim, out_dim)
        self.lin2 = nn.Linear(out_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.residual = residual

    def forward(self, x, edge_index):
        h = self.attention(x, edge_index)

        # Residual connection
        if self.residual:
            h = h + x
        h = self.dropout(h)
        h = self.ln(h)

        h = self.lin1(h)
        h = F.relu(h)
        h = self.lin2(h)

        h = self.dropout(h)
        h = self.ln(h)
        h = F.relu(h)

        return h

class GraphTransformerEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, num_layers, dropout=0.0):
        super(GraphTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GraphTransformerLayer(in_dim if _==0 else out_dim, out_dim, num_heads, dropout=dropout))
    
    def forward(self, x, edge_index):
        print("Input shape:", x.size())
        for layer in self.layers:
            x = layer(x, edge_index)
        return x


class ClassificationModel(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, num_layers, num_classes):
        super(ClassificationModel, self).__init__()

        self.encoder = GraphTransformerEncoder(in_dim, out_dim, num_heads, num_layers)
        self.classifier = nn.Linear(out_dim, num_classes)

    def forward(self, x, edge_index, batch):
        print("edge_index", edge_index.size())
        x = self.encoder(x, edge_index)
        x = scatter_mean(x, batch, dim=0)
        x = self.classifier(x)

        return x


