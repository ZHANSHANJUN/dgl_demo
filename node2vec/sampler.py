import numpy as np
import dgl
import torch
import random
from torch.utils.data import IterableDataset, DataLoader


class PinSAGECollator(object):
    def __init__(self, sampler, g, ntype):
        self.sampler = sampler
        self.ntype = ntype
        self.g = g
        self.neg_weights = g.nodes[ntype].data['neg_weight']

    def collate_train(self, batches):
        heads, tails, neg_tails = batches[0]
        pos_graph, neg_graph, blocks = self.sampler.sample_from_item_pairs(heads, tails, neg_tails, False)
        assign_features_to_blocks(blocks, self.g, self.ntype)
        return pos_graph, neg_graph, blocks

    def collate_test(self, samples):
        batch = torch.LongTensor(samples)
        blocks = self.sampler.sample_blocks_interface(batch)
        assign_features_to_blocks(blocks, self.g, self.ntype)
        return blocks
    
    def collate_test_auc(self, test_user, test_item, user_to_item_etype):
        heads = test_item
        tails = dgl.sampling.random_walk(self.g, test_user, metapath=[user_to_item_etype])[0][:, 1]
        neg_tails = torch.multinomial(self.neg_weights, len(test_item), replacement=True)
        mask = (heads != tails) & (tails != -1)
        pos_graph, neg_graph, blocks = self.sampler.sample_from_item_pairs(heads[mask], tails[mask], neg_tails[mask], True)
        assign_features_to_blocks(blocks, self.g, self.ntype)
        return pos_graph, neg_graph, blocks

def assign_features_to_blocks(blocks, g, ntype):
    for col in g.nodes[ntype].data.keys():
        if col == dgl.NID:
            continue
            
        induced_nodes = blocks[0].srcdata[dgl.NID]
        blocks[0].srcdata[col] = g.nodes[ntype].data[col][induced_nodes]
        
        induced_nodes = blocks[-1].dstdata[dgl.NID]
        blocks[-1].dstdata[col] = g.nodes[ntype].data[col][induced_nodes]
    

class ItemToItemBatchSampler(IterableDataset):
    def __init__(self, g, user_type, item_type, batch_size, hard_neg_ratio):
        self.g = g
        self.user_type = user_type
        self.item_type = item_type
        self.user_to_item_etype = 'clicked'
        self.item_to_user_etype = 'clicked-by'
        self.batch_size = batch_size
        self.neg_weights = g.nodes[item_type].data['neg_weight']
        self.neg_etype = 'neg'
        self.hard_neg_ratio = hard_neg_ratio

    def __iter__(self):
        while True:
            heads = torch.randint(0, self.g.number_of_nodes(self.item_type), (self.batch_size,))
            tails = dgl.sampling.random_walk(self.g, heads, metapath=[self.item_to_user_etype, self.user_to_item_etype])[0][:, 2]
            if random.random() < self.hard_neg_ratio:
                neg_tails = dgl.sampling.random_walk(self.g, heads, metapath=[self.item_to_user_etype, self.neg_etype])[0][:, 2]
            else:
                neg_tails = torch.multinomial(self.neg_weights, self.batch_size, replacement=True)
            mask = (heads != tails) & (tails != -1) & (heads != neg_tails) & (neg_tails != -1)
            yield heads[mask], tails[mask], neg_tails[mask]

            
class NeighborSampler(object):
    def __init__(self, g, user_type, item_type, random_walk_length, random_walk_restart_prob,
                 num_random_walks, num_neighbors, num_layers):
        self.g = g
        self.user_type = user_type
        self.item_type = item_type
        self.user_to_item_etype = 'clicked'
        self.item_to_user_etype = 'clicked-by'
        self.samplers = [
            dgl.sampling.PinSAGESampler(g, item_type, user_type, random_walk_length,
                random_walk_restart_prob, num_random_walks, num_neighbors) for _ in range(num_layers)]
        self.samplers_interface = [
            dgl.sampling.PinSAGESampler(g, item_type, user_type, random_walk_length,
                random_walk_restart_prob, 500, 10) for _ in range(num_layers)]

    def sample_blocks(self, seeds, heads=None, tails=None, neg_tails=None):
        blocks = []
        for sampler in self.samplers:
            frontier = sampler(seeds)
            if heads is not None:
                eids = frontier.edge_ids(torch.cat([heads, heads]), torch.cat([tails, neg_tails]), return_uv=True)[2]
                if len(eids) > 0:
                    old_frontier = frontier
                    frontier = dgl.remove_edges(old_frontier, eids)
#                     print(old_frontier)
#                     print(frontier)
#                     print(frontier.edata['weights'])
#                     frontier.edata['weights'] = old_frontier.edata['weights'][frontier.edata[dgl.EID]]
                    
            block = dgl.to_block(frontier, seeds)
#             for col, data in frontier.edata.items():
#                 if col == dgl.EID:
#                     continue
#                 block.edata[col] = data[block.edata[dgl.EID]]
                
            seeds = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        return blocks
    
    def sample_blocks_interface(self, seeds):
        blocks = []
        for sampler in self.samplers_interface:
            frontier = sampler(seeds)    
            block = dgl.to_block(frontier, seeds)
            seeds = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        return blocks

    def sample_from_item_pairs(self, heads, tails, neg_tails, interface):
        pos_graph = dgl.graph((heads, tails), num_nodes=self.g.number_of_nodes(self.item_type))
        neg_graph = dgl.graph((heads, neg_tails), num_nodes=self.g.number_of_nodes(self.item_type))
        pos_graph, neg_graph = dgl.compact_graphs([pos_graph, neg_graph])
        seeds = pos_graph.ndata[dgl.NID]

        if interface:
            blocks = self.sample_blocks_interface(seeds)
        else:
            blocks = self.sample_blocks(seeds, heads, tails, neg_tails)
        return pos_graph, neg_graph, blocks
    
