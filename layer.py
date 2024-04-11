import os
os.environ["PATH"] = os.environ["PATH"]+":/usr/local/cuda/bin/"
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import sys
sys.path.append('./src')

import torch
import torch.nn as nn

class SimplifiedAttention(nn.Module):
    def __init__(self, embed_dim, dropout_p=0.1):
        super(SimplifiedAttention, self).__init__()
        self.embed_dim = embed_dim
        self.dropout_p = dropout_p
        
        self.in_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        # self.in_proj_bias = nn.Parameter(torch.Tensor(embed_dim))
        self.out_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        # self.out_proj_bias = nn.Parameter(torch.Tensor(embed_dim))
        # self.in_proj = nn.Linear(in_features=embed_dim, out_features=embed_dim,bias=False)
        # self.out_proj = nn.Linear(in_features=embed_dim, out_features=embed_dim,bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        # nn.init.constant_(self.in_proj_bias, 0)
        nn.init.xavier_uniform_(self.out_proj_weight)
        # nn.init.constant_(self.out_proj_bias, 0)

    def forward(self, value, attn_output_weights, key_padding_mask=None, need_weights=None):
        tgt_len, bsz, embed_dim = value.size()
        assert embed_dim == self.embed_dim, "Embedding dimension mismatch."
        
        v_proj = F.linear(value, self.in_proj_weight)
        v_proj = v_proj.transpose(0, 1)  # Change shape to (bsz, tgt_len, embed_dim)
        
        # # Apply key padding mask
        # if key_padding_mask is not None:
        #     attn_output_weights = attn_output_weights.view(bsz, 1, tgt_len,
        #                                                tgt_len)
        #     attn_output_weights = attn_output_weights.masked_fill(
        #     key_padding_mask.unsqueeze(1).unsqueeze(2),
        #     float('-inf'),
        #     )
        #     attn_output_weights = attn_output_weights.view(bsz, tgt_len, tgt_len)
        
        # print(attn_output_weights)
            
        # attn_output_weights = torch.exp(attn_output_weights)
        # attn_output_weights = attn_output_weights / attn_output_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        attn_output = torch.bmm(attn_output_weights, v_proj)

        attn_output = attn_output.transpose(0, 1)  # Change back to (tgt_len, bsz, embed_dim)
        attn_output = F.linear(attn_output, self.out_proj_weight)
        # attn_output = self.out_proj(attn_output)
        
        if need_weights:
            # Optionally return the attention weights in addition to the output
            return attn_output, attn_output_weights
        else:
            return attn_output, None

class SimpleTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1,
                 activation="relu", batch_norm=False):
        super().__init__(d_model, nhead=1,  # nhead is set to 1 as it's unused in SimplifiedAttention
                         dim_feedforward=dim_feedforward, dropout=dropout, activation=activation)
        self.self_attn = SimplifiedAttention(d_model)
        self.batch_norm = batch_norm
        if batch_norm:
            self.norm1 = nn.BatchNorm1d(d_model)
            self.norm2 = nn.BatchNorm1d(d_model)
        self.scaling = None

    def forward(self, src, pe, src_mask=None, src_key_padding_mask=None):
        src2, attn = self.self_attn(src, pe, key_padding_mask = src_key_padding_mask, need_weights=False)
        
        # print(src_key_padding_mask)
        src = src + self.dropout1(src2)
        
        if self.batch_norm:
            bsz = src.shape[1]
            src = src.view(-1, src.shape[-1])
        src = self.norm1(src)
        # print(self.norm1)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        if self.batch_norm:
            src = src.view(-1, bsz, src.shape[-1])    
        return src
