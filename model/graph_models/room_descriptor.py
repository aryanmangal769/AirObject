#!/usr/bin/env python3
import torch
import torch.nn as nn

class RoomDescriptor(nn.Module):
  def __init__(self, config):
    super(RoomDescriptor, self).__init__()

    nfeat = config['descriptor_dim']
    nhid = config['hidden_dim']
    dropout = config['dropout']
    nheads = config['nheads']
    nout = config['nout']
    self.gcn = GCN(nfeat, nhid, nout, dropout)

  def forward(self, batch_descs):
    '''
    inputs:
      batch_points: List[Tensor], normalized points, each tensor belonging to an object
      batch_descs: List[Tensor], local feature descriptors, each tensor belonging to an object
      batch_adj: List[Tensor], adjacency matrix corresponding to the triangulation based object points graph
      return_features: bool, return node-wise graph features
    '''


    batch_features = []
    #This is a temporary function, need to replace this with GCN descriptor similar to NetVlad Descriptor for batched processing.
    for obj in batch_descs:
        nodes = torch.tensor(obj.shape[0])
        adj = torch.ones(nodes,nodes,device = obj.get_device())
        output= self.gcn(obj, adj)
        batch_features.append(output.squeeze())
    
    batch_features = torch.stack(batch_features)
    batch_features = nn.functional.normalize(batch_features, p=2, dim=-1)

    return batch_features


class GCN(nn.Module):
    '''
    GCN: Graph Convolutional Network, ICLR 2017
    https://arxiv.org/pdf/1609.02907.pdf
    '''
    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        super().__init__()
        self.gcn1 = GraphConv(nfeat, nhid)
        self.gcn2 = GraphConv(nhid, nclass)
        self.acvt = nn.Sequential(nn.ReLU(), nn.Dropout(dropout))
        self.linear = nn.Linear(nclass, nclass)
        
    def forward(self, x, adj):
        x = self.gcn1(x, adj)
        x = self.acvt(x)
        x = self.gcn2(x, adj)
        x = torch.sum(x, 0)
        x = self.linear(x)
        
        return x


class GraphConv(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

    def forward(self, x, adj):
        return adj @ self.linear(x)
