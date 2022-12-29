import time
import numpy as np
import pandas as pd
import pickle
import dgl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# import scipy.sparse as ssp
from sklearn.metrics import roc_auc_score

from graph import *
from model import *
from sampler import *
from evaluation import *
from conf import *


def train_test_split(g):
    eids = np.arange(g.number_of_edges('clicked'))
    np.random.shuffle(eids)
    test_size = int(len(eids) * 0.01)
#     test_size = 10000
    train_indices, test_indices = eids[test_size:], eids[:test_size]
    train_g = g.edge_subgraph({'clicked': train_indices, 'clicked-by': train_indices}, preserve_nodes=True, store_ids=False)

#     n_users = g.number_of_nodes('user')
#     n_items = g.number_of_nodes('item')
#     test_src, test_dst = g.find_edges(test_indices, etype='clicked')
#     test_src, test_dst = test_src.numpy(), test_dst.numpy()
#     test_matrix = ssp.coo_matrix((np.ones_like(test_src), (test_src, test_dst)), (n_users, n_items))
#     return train_g, test_matrix

    test_user, test_item = g.find_edges(test_indices, etype='clicked')
    print("test cnt", len(test_item))
    return train_g, (test_user, test_item)

def train_test_split_by_item(g):
    item2user_g = g.edge_type_subgraph(['clicked-by'])
    item2user_g.edata['random_weight'] = torch.rand(item2user_g.number_of_edges())
    item2user_test_g = dgl.sampling.select_topk(item2user_g, 1, 'random_weight', edge_dir='out')
    test_eids = item2user_test_g.edata['_ID']
    test_item, test_user = g.find_edges(test_eids, etype='clicked-by')
    print("test cnt", len(test_item))
    
    g = dgl.remove_edges(g, test_eids, etype='clicked')
    g = dgl.remove_edges(g, test_eids, etype='clicked-by')
    return g, (test_user, test_item)

def split_parameters(module):
    params_decay = []
    params_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            params_decay.append(m.weight)
            if m.bias is not None:
                params_no_decay.append(m.bias)
        elif isinstance(m, nn.Embedding):
            params_decay.append(m.weight)
            if m.bias is not None:
                params_no_decay.append(m.bias)
        elif len(list(m.children())) == 0:
            params_no_decay.extend([*m.parameters()])
    assert len(list(module.parameters())) == len(params_decay) + len(params_no_decay)
    return params_decay, params_no_decay

def compute_auc(pos_score, neg_score):
    labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
    scores = torch.cat([pos_score, neg_score])
    auc = roc_auc_score(labels.numpy(), scores.numpy())
    return auc

def train_model(g):
#     g, (test_user, test_item) = train_test_split(g)
    g, (test_user, test_item) = train_test_split_by_item(g)

    batch_sampler = ItemToItemBatchSampler(g, 'user', 'item', batch_size, hard_neg_ratio)
    neighbor_sampler = NeighborSampler(g, 'user', 'item', random_walk_length, random_walk_restart_prob, num_random_walks, num_neighbors, num_layers)

    collator = PinSAGECollator(neighbor_sampler, g, 'item')
    dataloader = DataLoader(batch_sampler, collate_fn=collator.collate_train, num_workers=num_workers)
    dataloader_test = DataLoader(torch.arange(g.number_of_nodes('item')), batch_size=batch_size, collate_fn=collator.collate_test, num_workers=num_workers)
    dataloader_it = iter(dataloader)

    model = PinSAGEModel(g, 'item', hidden_dims_dict, num_layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    #opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_weight)

    for epoch_id in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_id in range(batches_per_epoch):
            pos_graph, neg_graph, blocks = next(dataloader_it)
            pos_score, neg_score = model(pos_graph, neg_graph, blocks)
            
#             scores = torch.cat([pos_score, neg_score])
#             labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
#             loss = F.binary_cross_entropy_with_logits(scores, labels)
            
            loss = (neg_score - pos_score + 1).clamp(min=0).mean()
            
            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item()
            if batch_id % 100 == 99:
                with torch.no_grad():
                    train_auc = compute_auc(pos_score, neg_score)
                    print("epoch", epoch_id+1, "batch", batch_id+1, "loss", running_loss/100, "train_auc", train_auc)
                    
                running_loss = 0.0

        model.eval()
        with torch.no_grad():
            pos_graph, neg_graph, blocks = collator.collate_test_auc(test_user, test_item, 'clicked')
            pos_score, neg_score = model(pos_graph, neg_graph, blocks)
            test_auc = compute_auc(pos_score, neg_score)
            print("[eval]                            epoch", epoch_id+1, "test_auc ", test_auc)
              
        if epoch_id % 5 == 4:
            print("reduce lr")
            for p in opt.param_groups:
                p['lr'] *= 0.5
       
    # get item embeddings
    model.eval()
    with torch.no_grad():
        item_batches = torch.arange(g.number_of_nodes('item')).split(batch_size)
        h_item_batches = []
        for blocks in dataloader_test:
            h_item_batches.append(model.get_repr(blocks))
        h_item = torch.cat(h_item_batches, 0)
            
#             hit_rate = cal_hr_k(g, h_item, test_user, test_item)
#             print("[eval]                            hit_rate@", top_k, hit_rate)
            
    return h_item.numpy()

def main():
#     g, iid_map_reverse = build_graph(file_path)
    g, iid_map_reverse = build_graph_with_neg(file_path)
    print("click graph", g)
    item_embeddings = train_model(g)
    print("item_embeddings", item_embeddings.shape, len(iid_map_reverse))

    print("output_path", output_path)
    with open(output_path, 'wb') as f:
        pickle.dump([item_embeddings, iid_map_reverse], f)
    
if __name__ == "__main__":
    main()
