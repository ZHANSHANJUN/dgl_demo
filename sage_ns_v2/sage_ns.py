import dgl
import torch
import math
import numpy as np
import redis
import pandas as pd
import pickle
import faiss
import json
from tqdm import tqdm
from torch.utils.data import IterableDataset, DataLoader
from model import BatchSampler, SAGEModel
from sklearn.metrics import roc_auc_score


class Config:
    def __init__(self):
        self.input_path = "/data/zsj/sage/input/friend.txt"
        self.output_path = "/data/zsj/sage/output/user.pkl"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.test_ratio = 0.05
        self.hard_neg_ratio = 0.1
        self.batch_size = 1024
        self.batch_size_test = 4096
        self.batch_size_export = 256
        
        self.fanouts = [10, 20]
        self.num_workers = 2
        self.feat_dim_dict = {"id":64}
        
        self.lr = 1e-3
        self.num_epochs = 5
        
        self.top_k = 200
        self.redisPrefix = "friend:friend_sage_ns:"
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
    g.ndata['neg_weight'] = torch.FloatTensor(pd.concat([friend['from'], friend['to']]).value_counts().sort_index().map(lambda x:pow(x, 0.75)).values)
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
    resource = faiss.StandardGpuResources()
    index = faiss.IndexFlatIP(emb_size)
    index_with_id = faiss.IndexIDMap(index)
    index_with_id_gpu = faiss.index_cpu_to_gpu(resource, 0, index_with_id)
    index_with_id_gpu.add_with_ids(h_item, user_ids)
    
    redisClient = redis.StrictRedis(host="sg-proxy01.starmaker.co", port=22122)
    pipeline = redisClient.pipeline(transaction=False)
    batch_size = 100000
    for i in range(len(user_ids)//batch_size + 1):
        print("batch", i, "done...")
        h_item_batch = h_item[batch_size*i:batch_size*(i+1)]
        user_ids_batch = user_ids[batch_size*i:batch_size*(i+1)]
        D, I = index_with_id_gpu.search(h_item_batch, conf.top_k)
        for j in range(len(user_ids_batch)):
            uid = user_ids_batch[j]       
            redisKey = conf.redisPrefix + str(uid)
            sim_res = dict(zip(I[j].tolist(), D[j].tolist()))
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
    test_data = torch.stack(test_data, dim=-1)
    print(g)
    
    batch_sampler = BatchSampler(g, conf.fanouts, conf.hard_neg_ratio)    
    dataloader = DataLoader(torch.arange(g.number_of_nodes()), batch_size=conf.batch_size, shuffle=True, collate_fn=batch_sampler.collate_fn_train, num_workers=conf.num_workers)
    dataloader_test = DataLoader(test_data, batch_size=conf.batch_size_test, shuffle=True, collate_fn=batch_sampler.collate_fn_test, num_workers=conf.num_workers)
    dataloader_export = DataLoader(torch.arange(g.number_of_nodes()), batch_size=conf.batch_size_export, collate_fn=batch_sampler.collate_fn_export, num_workers=conf.num_workers)
    
    model = SAGEModel(g, conf.feat_dim_dict).to(conf.device)
    opt = torch.optim.Adam(model.parameters(), lr=conf.lr)
#     opt = torch.optim.Adam(model.parameters(), lr=conf.lr, weight_decay=5e-5)
    
    print("model training...")
    for epoch_id in range(conf.num_epochs):
        model.train()
        running_loss = 0.0
        total_pos_score = torch.Tensor().to(conf.device)
        total_neg_score = torch.Tensor().to(conf.device)
        batch_sampler.hard_neg_ratio += 0.05
        print("hard_neg_ratio", batch_sampler.hard_neg_ratio)
        for batch_id, (pos_graph, neg_graph, blocks) in enumerate(dataloader):
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
            total_pos_score = torch.cat([total_pos_score, pos_score])
            total_neg_score = torch.cat([total_neg_score, neg_score])
            if batch_id % 100 == 99:
                with torch.no_grad():
                    train_auc = compute_auc(total_pos_score, total_neg_score)
                    print("epoch", epoch_id+1, "batch", batch_id+1, "loss", running_loss/100, "train_cnt", len(total_pos_score), "train_auc", train_auc)
                    
                running_loss = 0.0
                total_pos_score = torch.Tensor().to(conf.device)
                total_neg_score = torch.Tensor().to(conf.device)
    
        model.eval()
        with torch.no_grad():
            total_pos_score = torch.Tensor().to(conf.device)
            total_neg_score = torch.Tensor().to(conf.device)
            for pos_graph, neg_graph, blocks in dataloader_test:
                for i in range(len(blocks)):
                    blocks[i] = blocks[i].to(conf.device)
                pos_graph = pos_graph.to(conf.device)
                neg_graph = neg_graph.to(conf.device)
                
                pos_score, neg_score = model(pos_graph, neg_graph, blocks)
                total_pos_score = torch.cat([total_pos_score, pos_score])
                total_neg_score = torch.cat([total_neg_score, neg_score])
                                
            print("epoch", epoch_id+1, "test_cnt", len(total_pos_score), "test_auc", compute_auc(total_pos_score, total_neg_score))
            
            
    # export
    print("export user embedding...")
    model.eval()
    h_item_batches = []
    for blocks in tqdm(dataloader_export):
        for i in range(len(blocks)):
            blocks[i] = blocks[i].to(conf.device)
        with torch.no_grad():
            h_item_batches.append(model.get_node_emb(blocks).cpu())

    h_item = torch.cat(h_item_batches, 0).numpy()
    
    assert h_item.shape[0] == len(user_ids)
    with open(conf.output_path, 'wb') as f:
        pickle.dump([h_item, user_ids], f)
        
    # u2i
    print("save to redis...")
    save_to_redis(h_item, user_ids, conf)

if __name__ == "__main__":
    main()
