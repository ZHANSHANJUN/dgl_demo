import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn
import dgl.function as fn
from torch.utils.data import IterableDataset, DataLoader
import random
import numpy as np


class NodePairBatchSampler(IterableDataset):
    def __init__(self, g, batch_size, test_data=None):
        self.g = g
        self.batch_size = batch_size
        self.test_data = test_data
        if self.test_data:
            self.test_user, self.test_item = test_data
            self.test_num = len(self.test_user)

    def __iter__(self):
        while True:
            if self.test_data:
                random_num = torch.randint(0, self.test_num, (self.batch_size,))
                heads, tails = self.test_user[random_num], self.test_item[random_num]
                neg_tails = torch.randint(0, self.g.number_of_nodes(), (self.batch_size,))
                yield heads, tails, neg_tails
            else:
                heads = torch.randint(0, self.g.number_of_nodes(), (self.batch_size,))
                tails = dgl.sampling.random_walk(self.g, heads, length=1)[0][:, 1]
                neg_tails = torch.randint(0, self.g.number_of_nodes(), (self.batch_size,))
#                 neg_tails = torch.multinomial(self.neg_weights, self.batch_size, replacement=True)
                mask = (tails != -1)
                yield heads[mask], tails[mask], neg_tails[mask]
        
        
class NeighborSampler:
    def __init__(self, g, fanouts):
        self.g = g
        self.fanouts = fanouts
        
    def sample_blocks(self, seeds, heads=None, tails=None, neg_tails=None, use_all=False):
        blocks = []
        for fanout in self.fanouts: 
            if use_all:
                fanout = -1
            frontier = dgl.sampling.sample_neighbors(self.g, seeds, fanout, replace=True)
            if heads is not None:
                eids = frontier.edge_ids(torch.cat([heads, heads]), torch.cat([tails, neg_tails]), return_uv=True)[2]
                if len(eids) > 0:
                    old_frontier = frontier
                    frontier = dgl.remove_edges(old_frontier, eids)
            
            block = dgl.to_block(frontier, seeds)
            seeds = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        return blocks
    
    def sample_from_pairs(self, heads, tails, neg_tails):
        pos_graph = dgl.graph((heads, tails), num_nodes=self.g.number_of_nodes())
        neg_graph = dgl.graph((heads, neg_tails), num_nodes=self.g.number_of_nodes())
        pos_graph, neg_graph = dgl.compact_graphs([pos_graph, neg_graph])
        seeds = pos_graph.ndata[dgl.NID]
        blocks = self.sample_blocks(seeds, heads=heads, tails=tails, neg_tails=neg_tails)
        return pos_graph, neg_graph, blocks
    
    def collate_fn(self, batches):
        heads, tails, neg_tails = batches[0]
        pos_graph, neg_graph, blocks = self.sample_from_pairs(heads, tails, neg_tails)
        return pos_graph, neg_graph, blocks
    
    def collate_fn_export(self, seeds):
        blocks = self.sample_blocks(seeds, use_all=True)
        return blocks
    
    
class SAGEModel(nn.Module):
    def __init__(self, g, feat_dim_dict):
        super().__init__()
        self.proj = LinearProjector(g, feat_dim_dict)
        self.sage = SAGENet(feat_dim_dict)
        self.scorer = PairScorer()
        
    def forward(self, pos_graph, neg_graph, blocks):
        h_item = self.get_repr(blocks)
        pos_score = self.scorer(pos_graph, h_item)
        neg_score = self.scorer(neg_graph, h_item)
        return pos_score, neg_score
    
    def get_repr(self, blocks):
        h_item = self.proj(blocks[0].srcdata)
        h_item_dst = self.proj(blocks[-1].dstdata)
        return self.sage(blocks, h_item, h_item_dst)        
    
    
class LinearProjector(nn.Module):
    def __init__(self, g, feat_dim_dict):
        super().__init__()
        self.feat_dim_dict = feat_dim_dict
        self.inputs = _init_input_modules(g, feat_dim_dict)

    def forward(self, ndata):
        projections = []
        for col, data in ndata.items():
            if col not in self.feat_dim_dict:
                continue
            
            module = self.inputs[col]
            result = module(data)
            projections.append(result)

#         return torch.stack(projections, 1).sum(1)
        return torch.cat(projections, dim=-1)
    

def _init_input_modules(g, feat_dim_dict):
    module_dict = nn.ModuleDict()
    for column, data in g.ndata.items():
        if column not in feat_dim_dict:
            print("node feat", column, "not exist")
            continue
            
        feat_dim = feat_dim_dict[column]
        if data.dtype == torch.float32 or data.dtype == torch.float64:
            assert data.ndim == 2
            m = nn.Linear(data.shape[1], feat_dim)
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
            module_dict[column] = m
        elif data.dtype == torch.int32 or data.dtype == torch.int64:
            assert data.ndim == 1
            m = nn.Embedding(data.max() + 2, feat_dim, padding_idx=-1)
            nn.init.xavier_uniform_(m.weight)
            module_dict[column] = m

    return module_dict


class SAGENet(nn.Module):
    def __init__(self, feat_dim_dict):
        super().__init__()
        feat_dim_sum = sum(feat_dim_dict.values())
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(feat_dim_sum, feat_dim_sum*2, 'pool'))
        self.layers.append(dglnn.SAGEConv(feat_dim_sum*2, feat_dim_sum, 'pool'))
        self.linear = nn.Linear(feat_dim_sum*2, feat_dim_sum)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, blocks, h_item, h_item_dst):
        h = h_item
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h_item_dst+h
    
      
class PairScorer(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pair_graph, h):
        with pair_graph.local_scope():
            pair_graph.ndata['h'] = h
            pair_graph.apply_edges(fn.u_dot_v('h', 'h', 's'))
            pair_score = pair_graph.edata['s'].view(-1)
            
        return pair_score
    