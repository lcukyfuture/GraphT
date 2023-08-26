
import os
os.environ["PATH"] = os.environ["PATH"]+":/usr/local/cuda/bin/"
import torch
import numpy as np
from torch import nn
from torch_geometric.nn.aggr import Aggregation
from torch_scatter import scatter_add, scatter_mean, scatter_max
import torch_geometric.nn as gnn
import torch_geometric.utils as utils
from torch_geometric.data import Data
from einops import rearrange
import torch.nn.functional as F
from grakel.kernels import RandomWalk, ShortestPath, CoreFramework, WeisfeilerLehman, GraphletSampling, VertexHistogram
# from graphdot import Graph
# from graphdot.kernel.marginalized import MarginalizedGraphKernel
# from graphdot.microkernel import (
#     TensorProduct,
#     SquareExponential,
#     KroneckerDelta,
#     Constant
# )
from torch_geometric.utils import to_networkx
from grakel.utils import graph_from_networkx

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Attention(gnn.MessagePassing):
    """Multi-head Structure-Aware attention using PyG interface
    accept Batch data given by PyG

    Args:
    ----------
    embed_dim (int):        the embeding dimension
    num_heads (int):        number of attention heads (default: 8)
    dropout (float):        dropout value (default: 0.0)
    bias (bool):            whether layers have an additive bias (default: False)
    symmetric (bool):       whether K=Q in dot-product attention (default: False)
    """

    def __init__(self, embed_dim, num_heads=8, dropout=0., bias=False,
        symmetric=False, **kwargs):

        super().__init__(node_dim=0, aggr='add')
        self.embed_dim = embed_dim
        self.bias = bias
        head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.scale = head_dim ** -0.5

        self.attend = nn.Softmax(dim=-1)

        self.symmetric = symmetric
        if symmetric:
            self.to_qk = nn.Linear(embed_dim, embed_dim, bias=bias)
        else:
            self.to_qk = nn.Linear(embed_dim, embed_dim * 2, bias=bias)
        self.to_v = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

        self.attn_sum = None

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.to_qk.weight)
        nn.init.xavier_uniform_(self.to_v.weight)

        if self.bias:
            nn.init.constant_(self.to_qk.bias, 0.)
            nn.init.constant_(self.to_v.bias, 0.)

    def forward(self,
            x,
            edge_index,
            ptr=None,
            return_attn=False):
        """
        Compute attention layer. 

        Args:
        ----------
        x:                          input node features
        edge_index:                 edge index from the graph
        edge_attr:                  edge attributes
        return_attn:                return attention (default: False)

        """
        # Compute value matrix
        v = self.to_v(x)

        # Compute query and key matrices
        if self.symmetric:
            qk = self.to_qk(x)
            qk = (qk, qk)
        else:
            qk = self.to_qk(x).chunk(2, dim=-1)
        attn = None
        out, attn = self.self_attn(qk, v, ptr, return_attn=return_attn)
        return self.out_proj(out), attn

    # def message(self, v_j, qk_j, qk_i, edge_attr, index, ptr, size_i, return_attn):
    #     """Self-attention operation compute the dot-product attention """

    #     qk_i = rearrange(qk_i, 'n (h d) -> n h d', h=self.num_heads)
    #     qk_j = rearrange(qk_j, 'n (h d) -> n h d', h=self.num_heads)
    #     v_j = rearrange(v_j, 'n (h d) -> n h d', h=self.num_heads)
    #     attn = (qk_i * qk_j).sum(-1) * self.scale
    #     if edge_attr is not None:
    #         attn = attn + edge_attr
    #     attn = utils.softmax(attn, index, ptr, size_i)
    #     if return_attn:
    #         self._attn = attn
    #     attn = self.attn_dropout(attn)

    #     return v_j * attn.unsqueeze(-1)

    def self_attn(self, qk, v, ptr, return_attn=False):
        """ Self attention which can return the attn """ 
        # print("attn_q", qk[0].size())
        # print("attn_k", qk[1].size())
        qk, mask = pad_batch(qk, ptr, return_mask=True)
        # print("attn_q_pad", qk[0].size())
        # print("attn_k_pad", qk[1].size())
        k, q = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qk)
        # print("q:", q.size())
        # print("k:", k.size())
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        dots = dots.masked_fill(
            mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )

        dots = self.attend(dots)
        dots = self.attn_dropout(dots)

        v = pad_batch(v, ptr)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        out = torch.matmul(dots, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = unpad_batch(out, ptr)

        if return_attn:
            return out, dots
        return out, None

class KernelAttention(gnn.MessagePassing):
    """Multi-head Structure-Aware attention using PyG interface
    accept Batch data given by PyG

    Args:
    ----------
    embed_dim (int):        the embeding dimension
    """

    def __init__(self, embed_dim, bias=False, **kwargs):

        super().__init__(node_dim=0, aggr='add')
        self.embed_dim = embed_dim
        self.bias = bias

        self.to_v = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self._reset_parameters()
        self.attn_sum = None

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.to_v.weight)

        if self.bias:
            nn.init.constant_(self.to_v.bias, 0.)

    def forward(self,
            x,
            edge_index,
            original_x = None,
            edge_attr=None,
            ptr=None,
            num_graphs=None,
            batch = None
            ):
        total_nodes = 0
        row, col = edge_index
        batch_edge = batch[row]
        output = []
        max_length = 0
        for graph_index in range(num_graphs):
            ###not using the original graph node label.
            # single_x = x[batch == graph_index]
            ###use the original graph node label.
            single_x = original_x[batch == graph_index]
            sinlge_row = row[batch_edge == graph_index]-total_nodes
            single_col = col[batch_edge == graph_index]-total_nodes
            single_edge_index = torch.stack((sinlge_row, single_col))
            if edge_attr is not None:
                single_edge_attr = edge_attr[batch_edge == graph_index]
                single_data = Data(single_x, single_edge_index, single_edge_attr)
            else:
                single_data = Data(single_x, single_edge_index)
            total_nodes = total_nodes + single_data.num_nodes
            output.extend(extract_kernel_features(single_data))
        # print(output)
        max_length = max(max_length, max(len(x) for x in output))
        output = pad_sequence(output, max_length)
        output = torch.stack(output)
        new_output = pad_batch(output, ptr)
        v = self.to_v(x)
        v = pad_batch(v, ptr)
        out = torch.matmul(new_output, v)
        out = unpad_batch(out, ptr)
        out = self.out_proj(out)
        return out

class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    r"""Structure-Aware Transformer layer, made up of structure-aware self-attention and feed-forward network.

    Args:
    ----------
        d_model (int):      the number of expected features in the input (required).
        nhead (int):        the number of heads in the multiheadattention models (default=8).
        dim_feedforward (int): the dimension of the feedforward network model (default=512).
        dropout:            the dropout value (default=0.1).
        activation:         the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable (default: relu).
        batch_norm:         use batch normalization instead of layer normalization (default: True).
        pre_norm:           pre-normalization or post-normalization (default=False).
        gnn_type:           base GNN model to extract subgraph representations.
                            One can implememnt customized GNN in gnn_layers.py (default: gcn).
        se:                 structure extractor to use, either gnn or khopgnn (default: gnn).
        k_hop:              the number of base GNN layers or the K hop size for khopgnn structure extractor (default=2).
    """
    def __init__(self, d_model, dim_feedforward=512, dropout=0.1,
                activation="relu", batch_norm=True, pre_norm=False, **kwargs):
        super().__init__(d_model, dim_feedforward, dropout, activation)

        # self.self_attn = Attention(embed_dim=d_model, dropout=dropout, **kwargs)
        self.self_attn = KernelAttention(embed_dim=d_model, **kwargs)
        self.batch_norm = batch_norm
        self.pre_norm = pre_norm
        if batch_norm:
            self.norm1 = nn.BatchNorm1d(d_model)
            self.norm2 = nn.BatchNorm1d(d_model)
        self.soft = nn.Softmax(dim=-1)

    def forward(self, x, 
            num_nodes,
            edge_index,
            original_x=None,
            edge_attr=None,
            ptr=None,
            num_graphs = None,
            batch=None
        ):

        if self.pre_norm:
            x = self.norm1(x)

        x2 = self.self_attn(
            x,
            edge_index,
            original_x=original_x,
            edge_attr = edge_attr,
            ptr=ptr,
            num_graphs=num_graphs,
            batch=batch
        )

        x = x + self.dropout1(x2)
        if self.pre_norm:
            x = self.norm2(x)
        else:
            x = self.norm1(x)
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(x2)

        if not self.pre_norm:
            x = self.norm2(x)
        # label = self.soft(x)
        # original_x = torch.argmax(label, dim=-1)
        temperature = 1
        gumbel_noise = torch.rand_like(x)
        gumbel_noise = -torch.log(-torch.log(gumbel_noise + 1e-20)+1e-20)
        gumbel_logits = (x + gumbel_noise) / temperature
        original_x = torch.argmax(self.soft(gumbel_logits),dim=-1)
        return x, original_x


def pad_batch(x, ptr, return_mask=False):
    bsz = len(ptr) - 1
    # num_nodes = torch.diff(ptr)
    max_num_nodes = torch.diff(ptr).max().item()

    all_num_nodes = ptr[-1].item()
    cls_tokens = False
    x_size = len(x[0]) if isinstance(x, (list, tuple)) else len(x)
    if x_size > all_num_nodes:
        cls_tokens = True
        max_num_nodes += 1
    if isinstance(x, (list, tuple)):
        new_x = [xi.new_zeros(bsz, max_num_nodes, xi.shape[-1]) for xi in x]
        if return_mask:
            padding_mask = x[0].new_zeros(bsz, max_num_nodes).bool()
    else:
        new_x = x.new_zeros(bsz, max_num_nodes, x.shape[-1])
        if return_mask:
            padding_mask = x.new_zeros(bsz, max_num_nodes).bool()

    for i in range(bsz):
        num_node = ptr[i + 1] - ptr[i]
        if isinstance(x, (list, tuple)):
            for j in range(len(x)):
                new_x[j][i][:num_node] = x[j][ptr[i]:ptr[i + 1]]
                if cls_tokens:
                    new_x[j][i][-1] = x[j][all_num_nodes + i]
        else:
            new_x[i][:num_node] = x[ptr[i]:ptr[i + 1]]
            if cls_tokens:
                new_x[i][-1] = x[all_num_nodes + i]
        if return_mask:
            padding_mask[i][num_node:] = True
            if cls_tokens:
                padding_mask[i][-1] = False
    if return_mask:
        return new_x, padding_mask
    return new_x

def unpad_batch(x, ptr):
    bsz, n, d = x.shape
    max_num_nodes = torch.diff(ptr).max().item()
    num_nodes = ptr[-1].item()
    all_num_nodes = num_nodes
    cls_tokens = False
    if n > max_num_nodes:
        cls_tokens = True
        all_num_nodes += bsz
    new_x = x.new_zeros(all_num_nodes, d)
    for i in range(bsz):
        new_x[ptr[i]:ptr[i + 1]] = x[i][:ptr[i + 1] - ptr[i]]
        if cls_tokens:
            new_x[num_nodes + i] = x[i][-1]
    return new_x

def extract_kernel_features(data):
    subgraph_networkx = []
    # gk = RandomWalk(normalize=True, method_type="fast", kernel_type="geometric")
    # gk = ShortestPath(n_jobs = 8, normalize=True, with_labels = True)
    gk = WeisfeilerLehman(n_jobs = 8, normalize=True)
    # gk = GraphletSampling(n_jobs=8, normalize=True, k=3)
    ###GPU accelerated kernel
    # knode = TensorProduct(radius=SquareExponential(0.5),
    #                       catergory=KroneckerDelta(0.5))
    # kedge = Constant(1.0)
    # mlgk = MarginalizedGraphKernel(knode, kedge, q=0.05)
    nodes_indices = 0
    for node_index in range(nodes_indices + data.num_nodes):
        sub_node, sub_edge_index, _, edge_mask = utils.k_hop_subgraph(node_idx=node_index, 
                                                                    num_hops=2, 
                                                                    edge_index=data.edge_index, 
                                                                    relabel_nodes=True)
        x = data.x[sub_node]
        decoded_labels = torch.argmax(x, dim=1).tolist()
        subdata = Data(x, edge_index=sub_edge_index)
        subdata['label'] = decoded_labels
        subdata_networkx = to_networkx(subdata, node_attrs=['label'])
        # nx.draw(subdata_networkx)
        # plt.show()
        subgraph_networkx.append(subdata_networkx)

    nodes_indices = nodes_indices + data.num_nodes
    ###GPU accelerated kernel data transform
    # R = mlgk([Graph.from_networkx(g) for g in subgraph_networkx])
    # d = np.diag(R)**-0.5
    # K = np.diag(d).dot(R).dot(np.diag(d))
    # kernel_out = K
    grakel_data = graph_from_networkx(subgraph_networkx, node_labels_tag='label')
    kernel_out = gk.fit_transform(grakel_data)
    return torch.Tensor(kernel_out).to(device)

def pad_sequence(seq, max_length):
    max_length = int(max_length)
    padded_seq = []
    for vector in seq:
        zero_padding = torch.zeros(max_length - vector.size(0), device=vector.device)
        padded_vector = torch.cat([vector, zero_padding])
        padded_seq.append(padded_vector)

    return padded_seq
