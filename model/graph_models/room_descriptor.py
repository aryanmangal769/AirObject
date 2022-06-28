#!/usr/bin/env python3
import torch
import torch.nn as nn
import math

class RoomDescriptor(nn.Module):
  def __init__(self, config):
    super(RoomDescriptor, self).__init__()

    graph_model = config['graph_model']
    nfeat = config['descriptor_dim']
    nhid = config['hidden_dim']
    dropout = config['dropout']
    nheads = config['nheads']
    nout = config['nout']

    if graph_model == "gat":
      self.gnn = GAT(nfeat, nhid, nout, dropout)
    elif graph_model == "gcn":
      self.gnn = GCN(nfeat, nhid, nout, dropout)

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
        output= self.gnn(obj, adj)
        batch_features.append(output.squeeze())
    
    batch_features = torch.stack(batch_features)  
    batch_features = nn.functional.normalize(batch_features, p=2, dim=-1)

    return batch_features


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, alpha=0.2, nheads=8):
        '''
        GAT: Graph Attention Network, ICLR 2018
        https://arxiv.org/pdf/1710.10903.pdf
        '''
        super().__init__()
        self.attns = [GraphAttn(nfeat, nhid, dropout, alpha) for _ in range(nheads)]
        for i, attention in enumerate(self.attns):
            self.add_module('attention_{}'.format(i), attention)

        self.attn = GraphAttn(nhid * nheads, nclass, dropout, alpha)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.acvt = nn.LeakyReLU()  #ELU earlier
        self.linear = nn.Linear(nclass, nclass)


    def forward(self, x, adj):
        x = torch.cat([attn(self.dropout1(x), adj) for attn in self.attns], dim=1)
        x =  self.attn(self.dropout2(self.acvt(x)), adj)
        x = torch.mean(x, 0)  #sum
        x = self.linear(x)
        #Just for test
        # x = nn.functional.normalize(x, p=2, dim=-1)
        return x


class GraphAttn(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha):
        super().__init__()
        self.tran = nn.Linear(in_features, out_features, bias=False)
        self.att1 = nn.Linear(out_features, 1, bias=False)
        self.att2 = nn.Linear(out_features, 1, bias=False)
        self.norm = nn.Sequential(nn.Softmax(dim=1), nn.Dropout(dropout))
        self.leakyrelu = nn.LeakyReLU(alpha)  #ELU
        
    def forward(self, x, adj):
        h = self.tran(x)
        e = self.att1(h).unsqueeze(0) + self.att2(h).unsqueeze(1)
        e = self.leakyrelu(e.squeeze())
        # e[adj.to_strided()<=0] = -math.inf # only neighbors
        return self.norm(e) @ h



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
