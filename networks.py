# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 20:56:06 2023

@author: Haiyang Jiang
"""

import torch
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool as gsp
import torch.nn.functional as F
from layers import SGC_DAT2
from torch_geometric.utils import softmax
from torch.nn import Linear
 


class new_readout3(torch.nn.Module):
    def __init__(self,in_channels: int):
        super(new_readout3,self).__init__()
        self.in_channels = in_channels
        self.lin = Linear(self.in_channels, 1, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        
    def forward(self, x, batch):
        v = F.sigmoid(self.lin(x))
        v = softmax(v, batch)
        sv = torch.mul(v,x)
        return torch.cat([gmp(x, batch), gsp(sv, batch)], dim=1)

class Scalble_DAT_Net(torch.nn.Module):
    def __init__(self,args):
        super(Scalble_DAT_Net, self).__init__()
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio1 = args.dropout_ratio1
        self.dropout_ratio2 = args.dropout_ratio2
        self.K = args.K
        
        self.conv1 = SGC_DAT2(self.num_features, self.nhid, self.K)
        
        self.readout1 = new_readout3(self.nhid)
        
        self.lin1 = torch.nn.Linear(self.nhid*2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid//2)
        self.lin3 = torch.nn.Linear(self.nhid//2, self. num_classes)
    
    def reset_parameters(self):
        
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()
        
        self.readout1.reset_parameters()
        
        self.conv1.reset_parameters()
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index, batch, 0.5)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_ratio1, training=self.training)
        
        x3 = self.readout1(x,batch)
        #x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        
        x = F.relu(self.lin1(x3))
        x = F.dropout(x, p=self.dropout_ratio2, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.dropout_ratio2, training=self.training)
        x = F.log_softmax(self.lin3(x), dim=-1)
        return x












