import torch
import pickle
import dgl
from conf import *
from train import *

def cal_hr_k(g, item_embeddings, test_user, test_item):
    batches = torch.arange(len(test_item)).split(batch_size)

    hit = 0
    for batch in batches:
        users = test_user[batch]
        items = test_item[batch]
        item_emb = item_embeddings[items]

        dist = item_emb @ item_embeddings.t()
        top_sim_ids = dist.topk(top_k, 1)[1]
        for i in range(len(batch)):
            user = users[i]
            sim_ids = top_sim_ids[i]
            clicked_ids = g.successors(user, etype='clicked')
            hit_ids_cnt = len(set(sim_ids.numpy()) & set(clicked_ids.numpy()))
            if len(clicked_ids) != 0:
                hit += hit_ids_cnt
                #hit += hit_ids_cnt/len(clicked_ids)

    hit_rate = hit/len(test_item)
#     print("hr@", top_k, hit_rate)
    return hit_rate

def cal_hr_k_real(g, item_embeddings, test_user, test_item):
    batches = torch.arange(len(test_item)).split(batch_size)

    hit = 0
    for batch in batches:
        users = test_user[batch]
        items = test_item[batch]
        
        for i in range(len(batch)):
            user = users[i]
            item = items[i]
            clicked_ids = g.successors(user, etype='clicked')
            item_emb = item_embeddings[clicked_ids]
            dist = item_emb @ item_embeddings.t()
            top_sim_ids = dist.topk(top_k, 1)[1]

            item_hit = 0
            for sim_ids in top_sim_ids:
                if item in sim_ids:
                    item_hit+=1

            hit += item_hit/len(clicked_ids)

    hit_rate = hit/len(test_item)
#     print("hr@", top_k, hit_rate)
    return hit_rate

if __name__ == "__main__":
    g, iid_map_reverse = build_graph(file_path)
    #g, (test_user, test_item) = train_test_split(g)
    g, (test_user, test_item) = train_test_split_by_item(g)

    with open(output_path, 'rb') as f:
        item_embeddings, iid_map_reverse = pickle.load(f)
        print("item:", item_embeddings.shape, len(iid_map_reverse))
 
    hit_rate = cal_hr_k(g, torch.tensor(item_embeddings), test_user, test_item)
    print("hr", hit_rate)

