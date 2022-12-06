import dgl
import torch
import math
import numpy as np
import redis
import pandas as pd
import pickle
import faiss
from torch.utils.data import IterableDataset, DataLoader
from model import NodePairBatchSampler, NeighborSampler, SAGEModel
from sklearn.metrics import roc_auc_score


class Config:
    def __init__(self):
        self.input_path = "/data/zsj/sage/input/friend.txt"
        self.output_path = "/data/zsj/sage/output/user.pkl"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.test_ratio = 0.1
        self.batch_size = 256
        self.batch_size_test = 1024
        self.batch_size_export = 4096
        
        self.fanouts = [10, 25]
        self.num_workers = 0
        self.feat_dim_dict = {"id":64}
        
        self.lr = 1e-3
        self.num_epochs = 1
        self.batches_per_epoch = 100
        self.batches_per_epoch_test = 100
        
        self.top_k = 50
        self.redisClient = redis.StrictRedis(host="sg-prod-research-worker-5", port=6379)
        self.redisPrefix = "friend:sage:"
        self.redisTTL = 86400*3
        
    def parse(self, args):
        pass

    
def build_graph(path):
#     path = "./input/friend.txt"
    friend = pd.read_csv(path, header=None, sep=" ")
    friend.columns = ['from', 'to']
    
    user_ids = pd.concat([friend['from'], friend['to']]).unique()
    id_map = dict(zip(user_ids, range(0, len(user_ids))))
    friend['from'] = friend['from'].map(id_map)
    friend['to'] = friend['to'].map(id_map)
    
    g = dgl.graph((friend['from'].values, friend['to'].values))
    g.ndata['id'] = torch.LongTensor(list(range(len(user_ids))))
    print(g)
    return g, user_ids


def train_test_split(g, test_ratio):
    eids = np.arange(g.number_of_edges())
    np.random.shuffle(eids)
    
    test_size = int(len(eids) * test_ratio)
    train_indices, test_indices = eids[test_size:], eids[:test_size]
    train_g = g.edge_subgraph(train_indices, relabel_nodes=False, store_ids=False)
    
    test_user, test_item = g.find_edges(test_indices)
    print("test cnt", len(test_user))
    return train_g, (test_user, test_item)


def compute_auc(pos_score, neg_score):
    labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
    scores = torch.cat([pos_score, neg_score])
    auc = roc_auc_score(labels.cpu().numpy(), scores.cpu().numpy())
    return auc
        
    
def save_to_redis(h_item, user_ids, conf):
    emb_size = h_item.shape[1]
    index = faiss.IndexFlatIP(emb_size)
    index_with_id = faiss.IndexIDMap(index)
    index_with_id.add_with_ids(h_item, user_ids)
    
    pipeline = conf.redisClient.pipeline(transaction=False)
    D, I = index_with_id.search(h_item, conf.top_k)
    print("diversity", len(Counter(I[:1000].reshape(-1))))
    print(I[1], I[100], I[1000])
    
    for i in range(len(user_ids)):
        uid = user_ids[i]       
        redisKey = conf.redisPrefix + str(uid)
        sim_res = dict(zip(I[i].tolist(), D[i].tolist()))
        redisVal = json.dumps(sim_res)
#         print(redisKey, redisVal)
        pipeline.set(redisKey, redisVal)
        pipeline.expire(redisKey, conf.redisTTL)
    pipeline.execute()
    
    
def main():
    conf = Config()
    print(conf.__dict__)
    
    g, user_ids = build_graph(conf.input_path)
    g, test_data = train_test_split(g, conf.test_ratio)
    print(g)
    
    batch_sampler = NodePairBatchSampler(g, conf.batch_size)
    batch_sampler_test = NodePairBatchSampler(g, conf.batch_size_test, test_data)
    sampler = NeighborSampler(g, conf.fanouts)
    
    dataloader = DataLoader(batch_sampler, collate_fn=sampler.collate_fn, num_workers=conf.num_workers)
    dataloader_test = DataLoader(batch_sampler_test, collate_fn=sampler.collate_fn, num_workers=conf.num_workers)
    dataloader_it = iter(dataloader)
    dataloader_it_test = iter(dataloader_test)

    model = SAGEModel(g, conf.feat_dim_dict).to(conf.device)
    opt = torch.optim.Adam(model.parameters(), lr=conf.lr)
    
    for epoch_id in range(conf.num_epochs):
        model.train()
        running_loss = 0.0
        for batch_id in range(conf.batches_per_epoch):
            pos_graph, neg_graph, blocks = next(dataloader_it)
            for i in range(len(blocks)):
                blocks[i] = blocks[i].to(conf.device)
            pos_graph = pos_graph.to(conf.device)
            neg_graph = neg_graph.to(conf.device)
            
            pos_score, neg_score = model(pos_graph, neg_graph, blocks)
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
        test_auc_sum = 0.0
        with torch.no_grad():
            for _ in range(conf.batches_per_epoch_test):
                pos_graph, neg_graph, blocks = next(dataloader_it_test)
                for i in range(len(blocks)):
                    blocks[i] = blocks[i].to(conf.device)
                pos_graph = pos_graph.to(conf.device)
                neg_graph = neg_graph.to(conf.device)
                
                pos_score, neg_score = model(pos_graph, neg_graph, blocks)
                test_auc = compute_auc(pos_score, neg_score)
                test_auc_sum += test_auc
                
            print("epoch", epoch_id+1, "test_auc", test_auc_sum/conf.batches_per_epoch_test)
            
            
    # export
    model.eval()
    dataloader_export = DataLoader(torch.arange(g.number_of_nodes()), batch_size=conf.batch_size_export, collate_fn=sampler.collate_fn_export, num_workers=conf.num_workers)
    h_item_batches = []
    model = model.cpu()
    for blocks in dataloader_export:
#         for i in range(len(blocks)):
#             blocks[i] = blocks[i].to(conf.device)
        h_item_batches.append(model.get_repr(blocks))
    h_item = torch.cat(h_item_batches, 0).numpy()
    
    assert h_item.shape[0] == len(user_ids)
    with open(conf.output_path, 'wb') as f:
        pickle.dump([h_item, user_ids], f)
        
    # u2i
    save_to_redis(h_item, user_ids, conf)

if __name__ == "__main__":
    main()
