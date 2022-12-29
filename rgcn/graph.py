import dgl
import torch
import math
import numpy as np
import pandas as pd
from conf import *


def read_txt():
    live = pd.read_csv(live_path, header=None, sep=" ")
    live.columns = ["country", "uid", "rid", "duration"]
    live = live[live['duration']>=duration_min]
    
    friend = pd.read_csv(friend_path, header=None, sep=" ")
    friend.columns = ["uid_from", "uid_to"]
    return live, friend

def build_graph():
    print("params:", "duration_min", duration_min)
    live, friend = read_txt()
#     print("origin data shape", data.shape[0], "uid cnt", data.nunique()['uid'], "iid cnt", data.nunique()['iid'])
    
    #filter
    user_ids = live['uid'].unique()
    room_ids = live['rid'].unique()
    countrys = live['country'].unique()
    user_id_exist = dict(zip(user_ids, [1]*len(user_ids)))

    friend['label_from'] = friend['uid_from'].map(user_id_exist)
    friend['label_to'] = friend['uid_to'].map(user_id_exist)
    friend = friend[~friend['label_from'].isnull() & ~friend['label_to'].isnull()]
    print("uid cnt", len(user_ids), "rid cnt", len(room_ids), "country cnt", len(countrys)) 
    print("live cnt", live.shape[0], "friend cnt", friend.shape[0])
     
    #trans nodes
    user_id_map = dict(zip(user_ids, range(0, len(user_ids))))
    room_id_map = dict(zip(room_ids, range(0, len(room_ids))))
    country_map = dict(zip(countrys, range(0, len(countrys))))

    live['uid_node'] = live['uid'].map(user_id_map)
    live['rid_node'] = live['rid'].map(room_id_map)
    friend['uid_from_node'] = friend['uid_from'].map(user_id_map)
    friend['uid_to_node'] = friend['uid_to'].map(user_id_map)
    live['country_feat'] = live['country'].map(country_map)

    #build graph
    graph_data = {
        ('user', 'clicked', 'room'): (live['uid_node'].values, live['rid_node'].values),
        ('room', 'clicked-by', 'user'): (live['rid_node'].values, live['uid_node'].values),
        ('user', 'friend', 'user'): (friend['uid_from_node'].values, friend['uid_to_node'].values),
    }
    g = dgl.heterograph(graph_data, idtype=torch.int64)
    g.nodes['user'].data['uid'] = torch.LongTensor(list(range(len(user_ids))))
    user_country = live.groupby('uid_node')['country_feat'].apply(set).map(list).map(lambda x:x[0]).values
    g.nodes['user'].data['country'] = torch.LongTensor(user_country)
    neg_weights = live['uid_node'].value_counts().sort_index().map(lambda x:pow(x, 0.75)).values
    g.nodes['user'].data['neg_weight'] = torch.FloatTensor(neg_weights)
#     print(g)
    return g, user_ids

if __name__ == '__main__':
    g, user_ids = build_graph()
    print(g)
