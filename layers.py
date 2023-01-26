# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 20:54:58 2023

@author: Haiyang Jiang
"""

import torch
from torch_scatter import scatter_add
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool as gsp
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import add_remaining_self_loops, softmax
from torch_sparse import SparseTensor, matmul
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from torch_sparse import sum as sparsesum
from torch_sparse import fill_diag, mul
from torch_geometric.utils.num_nodes import maybe_num_nodes
import torch.nn as nn

def gcn_norm(r, edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    
    fill_value = 2. if improved else 1.
    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sparsesum(adj_t, dim=1)
        deg_inv_sqrt_left = deg.pow_(-r)
        deg_inv_sqrt_left.masked_fill_(deg_inv_sqrt_left == float('inf'), 0.)
        deg_inv_sqrt_right = deg.pow_(-r)
        deg_inv_sqrt_right.masked_fill_(deg_inv_sqrt_right == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt_left.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt_right.view(1, -1))
        return adj_t

    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt_left = deg.pow_(1-r)
        deg_inv_sqrt_left.masked_fill_(deg_inv_sqrt_left == float('inf'), 0)
        deg_inv_sqrt_right = deg.pow_(-r)
        deg_inv_sqrt_right.masked_fill_(deg_inv_sqrt_right == float('inf'), 0)
        return edge_index, deg_inv_sqrt_left[row] * edge_weight * deg_inv_sqrt_right[col]


class SGC_DAT2(MessagePassing):

    def __init__(self, in_channels: int, out_channels: int, K: int = 1, 
                 add_self_loops: bool = True, 
                 bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.dropout = nn.Dropout(0.)
        self.pre_dropout = False
        self.add_self_loops = add_self_loops
        self.lin1 = Linear(in_channels, out_channels, bias=bias)
        self.lin2 = Linear(out_channels*3, out_channels, bias=bias)
        self.att = Linear(out_channels, 1, bias=bias)  #Linear(in_channels, 1, bias=bias)
        self.act = nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.att.reset_parameters()
    #DAT2
    def forward(self, x: Tensor, edge_index: Adj,batch, r,
                edge_weight: OptTensor = None) -> Tensor:
        #r=0.5
        if isinstance(edge_index, Tensor):
            edge_index, edge_weight = gcn_norm(  # yapf: disable
                r,edge_index, edge_weight, x.size(self.node_dim), False,
                self.add_self_loops, dtype=x.dtype)
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(  # yapf: disable
                r,edge_index, edge_weight, x.size(self.node_dim), False,
                self.add_self_loops, dtype=x.dtype)
        #print("x",x.shape)
        X = [self.lin1(x)]
        att = self.att(X[-1])
        w = softmax(self.act(att), batch).reshape(-1,1)  #做成sparse tensor
        W = [w]
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight,size=None)
            X.append(self.lin1(x))
            att = self.att(X[-1])
            W.append(softmax(self.act(att), batch).reshape(-1,1))
        W = torch.cat(W, dim=1)
        X = torch.stack(X, dim=2)
        W = W.unsqueeze(dim=1)
        
        x1 = [torch.mul(X,W).sum(dim=2), X.mean(dim=2), X.max(dim=2).values]
        
        #x1 = [torch.mul(X,W).sum(dim=2), X.max(dim=2).values]
        #x1 = [X.mean(dim=2), X.max(dim=2).values]
        x1 = torch.cat(x1,dim=1)
        
        return self.lin2(x1)
        

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={self.K})')