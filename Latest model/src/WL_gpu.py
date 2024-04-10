import argparse
import os.path as osp
import warnings

import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

from torch_geometric.data import Batch
from torch_geometric.datasets import TUDataset
from typing import Optional

import torch
from torch import Tensor

from torch_geometric.typing import Adj, SparseTensor
from torch_scatter import scatter
import pickle 

# from pykeops.torch import LazyTensor

# del hash
class WLConv():
    def __init__(self):
        super().__init__()
        self.dictn = None

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.hashmap = {}
    
    def fit(self, x: Tensor, edge_index: Adj, batch=None) -> Tensor:
        self.xm = x.max()+1 #number of unique words in input
        X = torch.nn.functional.one_hot(x,self.xm)
                
        S = scatter(X[edge_index[0,:],:], edge_index[1,:], dim=0, reduce='sum')# per node histogram
        S = torch.cat([x[:,None]+1,S],dim=-1) 

        #Keops implementation (slower, saves memeory)
#         self.dictn = S
#         lD = LazyTensor(self.dictn[None,:,None,:].float())
#         lS = LazyTensor(S[None,None,:,:].float())
#         v =(lD-lS).abs().sum(3).min(1)[0,:,0]
#         H =(lD-lS).abs().sum(3).argmin(1)[0,:,0] 
        
        #Torch implementation        
        Sn = torch.sqrt((S**2).sum(-1,keepdim=True).float())
        S = S/Sn
        S =  torch.cat([Sn,S],-1) #
        #relabeling
        self.dictn = torch.unique(S,dim=0)
        N = 1-(S[:,:1].t()-self.dictn[:,:1]).abs()
        C = (self.dictn[:,1:]@S[:,1:].t())*N
        v,H = C.max(0)
        
        return H
    
    def transform(self, x: Tensor, edge_index: Adj, batch=None) -> Tensor:

        xm = max(x.max()+1,self.dictn.shape[-1]-1)
        X = torch.nn.functional.one_hot(x,xm)
        S = scatter(X[edge_index[0,:],:], edge_index[1,:], dim=0, reduce='sum')
        Sa = torch.cat([x[:,None]+1,S],dim=-1)
        S=Sa
        
        #Keops implementation
#         D = self.dictn
#         if D.shape[-1]!=S.shape[-1]:
#             m = D.shape[-1]
#             D = torch.nn.functional.pad(D, (0,0,0,1), mode='constant', value=0)
#             S[:,m] = S[:,m:].sum(-1)*1e8
#             S = S[:,:m]
#         lD = LazyTensor(D[None,:,None,:].float())
#         lS = LazyTensor(S[None,None,:,:].float())
#         v =(lD-lS).abs().sum(3).min(1)[0,:,0]
#         H =(lD-lS).abs().sum(3).argmin(1)[0,:,0]
#         H[v>1e-3] = self.dictn.shape[0]+1        
        
        #Torch implementation       
        Sn = torch.sqrt((S**2).sum(-1,keepdim=True).float())
        S = S/Sn
        D = self.dictn
        if D.shape[-1]!=S.shape[-1]+1:
            m = D.shape[-1]-1
            D = torch.nn.functional.pad(D, (0,0,0,1), mode='constant', value=0)
            S[:,m] = S[:,m:].sum(-1)*1e8
            S = S[:,:m]
        N = 1-(Sn.t()-D[:,:1]).abs()
        C = (D[:,1:]@S.t())*N
        v,H = C.max(0)
        
#         H[v<1-1e-6] = torch.arange(H[v<1-1e-6].shape[0],device=H.device) + self.dictn.shape[0]
        H[v<1-1e-6] = self.dictn.shape[0] #all previously unseen (during fitting) words are assigned to a 
            #new dummy lablel. This will prevent us to compute the exact magnitude of the histogram. We will
            #approximate it by considering all previously unseen words as unique (abs norm of the dummy label count)
        
        ### without approximation: requires an unique operation ###
#         unq,inv = torch.unique(Sa[v<1-1e-6],return_inverse=True,dim=0)
#         H[v<1-1e-6] = inv+D.shape[0]
        
        return H
        
    def histogram(self, x: Tensor, batch: Optional[Tensor] = None,
                  norm: bool = False) -> Tensor:
        r"""Given a node coloring :obj:`x`, computes the color histograms of
        the respective graphs (separated by :obj:`batch`)."""

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
                
        num_colors = x.max()+1
        batch_size = int(batch.max()) + 1

        index = batch * num_colors + x
        out = scatter(torch.ones_like(index), index, dim=0,
                      dim_size=num_colors * batch_size, reduce='sum')
        out = out.view(batch_size, num_colors).float()
        

        ## since we are not computing the full histogram, we approximate its norm
        if self.dictn is not None:
            nc_fit = self.dictn.shape[0]-1
        else:
            nc_fit = num_colors

        #exact norm, requires changes in relabeling
#         return out, out[:,:].pow(2).sum(-1) 

        #approx norm: the last one is the dummy label representing words outside of the dictionay
        return out, out[:,:nc_fit].pow(2).sum(-1) + out[:,nc_fit:].abs().sum(-1)

        
                    
    
        
class WL():
    def __init__(self, num_layers, norm=False):
        super().__init__()
        self.convs = [WLConv() for _ in range(num_layers)]
        self.conv0 = WLConv()
        self.norm=norm
        
        self.hist0 = None
        self.hists = [None] * num_layers
        
    def fit(self, graph):
        x, edge_index, batch = graph
        
        self.hist0, l0 = self.conv0.histogram(x, batch, norm=self.norm)
        self.L = l0
        for i,conv in enumerate(self.convs):
            x = conv.fit(x, edge_index, batch)    
            self.hists[i], l = conv.histogram(x, batch, norm=self.norm)
            self.L+=l

    def transform(self, graph):
        x, edge_index, batch = graph
        
        hist, L = self.conv0.histogram(x, batch, norm=self.norm)
        
        H = self.hist0[:,:hist.shape[1]]@hist[:,:self.hist0.shape[1]].T
        for h,conv in zip(self.hists,self.convs):
            x = conv.transform(x, edge_index, batch)
            hist, l = conv.histogram(x, batch, norm=self.norm)
            L += l
            H += (h[:,:hist.shape[1]]@hist[:,:h.shape[1]].T)
            
        H = (H/self.L.sqrt()[:,None])/L.sqrt()[None,:]
        return H



    
# import time
# device='cuda'
# wl = WL(num_layers=3)
# X = data.x.argmax(-1).to(device)
# E = data.edge_index.to(device)
# B =  data.batch.to(device)

# t=time.time()
# for i in range(100):
#     H = wl(X, E, B)
# print(time.time()-t)

