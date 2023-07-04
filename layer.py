from torch_geometric.nn import MessagePassing
from torch import nn
import torch.nn.functional as F
import torch
import math
import numpy as np


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn = (torch.matmul(q, k.transpose(-2, -1)))/math.sqrt(d_k)
    if mask is not None:
        attn = attn.masked_fill(mask==0, -9e15)
    attn = F.Softmax(attn, dim=-1)
    values = torch.matmul(attn, v)

    return values, attn

class multiheadattention(nn.module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj  = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_normal_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_normal_(self.o_proj)
        self.o_proj.bias.data.fill(0)


    def foward(self, x, mask=None, return_attention = False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.num_heads)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=1)

        values, attn = scaled_dot_product(q, k, v, mask = mask)
        values = values.permute(0, 2, 1, 3)
        values = values.reshape(batch_size, seq_length, embed_dim)

        o = self.o_proj(values)

        if return_attention:
            return o, attn
        else:
            return o
class GCNmultiheadattention(nn.modules):
    def __init__(self, input_dim, embed_dim, num_heads):
        super.__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_haeds = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = 

class EncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        super().__init__()

        #atttntion
        self.self_attn = multiheadattention(input_dim, input_dim, num_heads)

        #Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim),
        )
        #layer norm dropout
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def feedforward(self, x, mask):
        attn_out = self.self_attn(x, mask=mask)

        x = x + self.dropout(attn_out)
        x = self.nor1(x)

        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2

        return x


class TransformerEncoder(nn.module):
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])

        def forward(self, x, mask=None):
            for layer in self.layers:
                x = layer(x, mask=mask)
            return x
        

class GCNConv(MessagePassing):
    def __init__(self, in_channel, out_channel):
        super().__init__(aggr='add')
        
        self.mlp = nn.Sequential(
            nn.Linear(2*in_channel, out_channel),
            nn.ReLU(),
            nn.Linear(out_channel, out_channel)
        )

    def feedforward(self, x, edge_index):
        output = self.propagate(edge_index, x=x)
        return output

    def message(self, x_i, x_j):
        tmp = torch.cat([x_i, x_j - x_i], dim=1)
        out = self.mlp(tmp)
        return out

class GCN(GCNConv):
    def __init__(self, hiddenchanel):
        super(GCN, self).__init__()
        torch.manual_seed(12345)        