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
    
    sing = pd.read_csv(sing_path, header=None, sep=" ")
    sing.columns = ["country", "uid", "sid", "click_pv"]

    friend = pd.read_csv(friend_path, header=None, sep=" ")
    friend.columns = ["uid_from", "uid_to"]
    print("txt cnt:", live.shape[0], sing.shape[0], friend.shape[0])
    return live, sing, friend

def build_graph():
    print("params:", "duration_min", duration_min)
    live, sing, friend = read_txt()
#     print("origin data shape", data.shape[0], "uid cnt", data.nunique()['uid'], "iid cnt", data.nunique()['iid'])
    
    #filter
    user_ids = list(set(live['uid'].unique()) & set(sing['uid'].unique()))
    user_id_map = dict(zip(user_ids, [1]*len(user_ids)))

    live['label'] = live['uid'].map(user_id_map)
    live = live[~live['label'].isnull()]

    sing['label'] = sing['uid'].map(user_id_map)
    sing = sing[~sing['label'].isnull()]

    friend['label_from'] = friend['uid_from'].map(user_id_map)
    friend['label_to'] = friend['uid_to'].map(user_id_map)
    friend = friend[~friend['label_from'].isnull() & ~friend['label_to'].isnull()]
    print("uid cnt", len(user_ids), "rid cnt", live.nunique()['rid'], "sid cnt", sing.nunique()['sid'])  
     
    #trans nodes
    unique_val = pd.concat([live['uid'], live['rid'], sing['uid'], sing['sid']]).unique()
    id_map = dict(zip(unique_val, range(0, len(unique_val))))

    live['uid_node'] = live['uid'].map(id_map)
    live['rid_node'] = live['rid'].map(id_map)

    sing['uid_node'] = sing['uid'].map(id_map)
    sing['sid_node'] = sing['sid'].map(id_map)

    friend['uid_from_node'] = friend['uid_from'].map(id_map)
    friend['uid_to_node'] = friend['uid_to'].map(id_map)

    uids = np.array(list(set(live['uid_node'].values) | set(sing['uid_node'].values)))
    rids = np.array(list(set(live['rid_node'].values)))
    sids = np.array(list(set(sing['sid_node'].values)))
    all_id_map = {"all_ids":unique_val, "uids":uids, "rids":rids, "sids":sids}
    
    #build graph
    src_ids = pd.concat([live['uid_node'], live['rid_node'], sing['uid_node'], sing['sid_node'], friend['uid_from_node']]).values
    dst_ids = pd.concat([live['rid_node'], live['uid_node'], sing['sid_node'], sing['uid_node'], friend['uid_to_node']]).values
    g = dgl.graph((src_ids, dst_ids))
    return g, all_id_map

if __name__ == '__main__':
    g, all_id_map = build_graph()
    print(g)
