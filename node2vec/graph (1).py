import dgl
import torch
import math
import pandas as pd
from conf import *


def label_encoder(series, return_dict=False):
    unique_val = series.unique()
    fea_dict = dict(zip(unique_val, range(0, len(unique_val))))
    fea_dict_reverse = dict(zip(range(0, len(unique_val)), unique_val))
    new_series = series.map(fea_dict)
    if return_dict:
        return new_series, fea_dict_reverse
    return new_series

def build_graph(file_path):
    print("params:", "file_path", file_path, "uid_min_cnt", uid_min_cnt, "duration_min", duration_min)
    data = pd.read_csv(file_path, header=None, sep=" ")
    data.columns = ["uid", "iid", "duration"]
    print("origin data shape", data.shape[0], "uid cnt", data.nunique()['uid'], "iid cnt", data.nunique()['iid'])
    
    #filter
    data = data[data['uid']!=data['iid']]
    data = data[data['duration']>=duration_min]
    data['uid_cnt'] = data['uid'].map(dict(data['uid'].value_counts()))
    data = data[data['uid_cnt']>=uid_min_cnt]    
    print("data shape", data.shape[0], "uid cnt", data.nunique()['uid'], "iid cnt", data.nunique()['iid'])  

    #gen weight
    item_cnt_df = data['iid'].value_counts().reset_index()
    item_cnt_df.columns = ['iid', 'cnt']
    item_cnt_df['walk_weight'] = 1/item_cnt_df['cnt'].map(lambda x:pow(x, 0.75))
    walk_weight_map = dict(zip(item_cnt_df['iid'].values, item_cnt_df['walk_weight'].values))
    data['walk_weight'] = data['iid'].map(walk_weight_map)
    data["pos_weight"] = data["duration"].map(math.log1p)
    
    #trans feat
    data['uid'] = label_encoder(data['uid'])
    data['iid'], iid_map_reverse = label_encoder(data['iid'], return_dict=True)

    #build graph
    graph_data = {
       ('user', 'clicked', 'item'): (data['uid'].values, data['iid'].values),
       ('item', 'clicked-by', 'user'): (data['iid'].values, data['uid'].values),
    }
    g = dgl.heterograph(graph_data, idtype=torch.int64)
    
    g.nodes['item'].data['iid'] = torch.LongTensor(list(iid_map_reverse.keys()))
    g.nodes['item'].data['neg_weight'] = torch.FloatTensor(data['iid'].value_counts().sort_index().map(lambda x:pow(x, 0.75)).values)
    g.edges['clicked'].data['pos_weight'] = torch.FloatTensor(data['pos_weight'].values)
    g.edges['clicked-by'].data['pos_weight'] = torch.FloatTensor(data['pos_weight'].values)
    g.edges['clicked'].data['walk_weight'] = torch.FloatTensor(data['walk_weight'].values)
    g.edges['clicked-by'].data['walk_weight'] = torch.FloatTensor(data['pos_weight'].values)
    return g, iid_map_reverse

def build_graph_with_neg(file_path):
    print("params:", "file_path", file_path, "uid_min_cnt", uid_min_cnt, "duration_min", duration_min)
    data = pd.read_csv(file_path, header=None, sep=" ")
    data.columns = ["uid", "iid", "duration"]
    print("origin data shape", data.shape[0], "uid cnt", data.nunique()['uid'], "iid cnt", data.nunique()['iid'])
    
    #filter
    data = data[data['uid']!=data['iid']]
    data['label'] = data['duration'].map(lambda x:"neg" if x < duration_min else "pos")
    data['uid_pos_cnt'] = data['uid'].map(dict(data[data['label']=='pos']['uid'].value_counts()))
    data = data[data['uid_pos_cnt']>=5]
    print("data shape", data.shape[0], "uid cnt", data.nunique()['uid'], "iid cnt", data.nunique()['iid'])  

    #gen weight
#     item_cnt_df = data['iid'].value_counts().reset_index()
#     item_cnt_df.columns = ['iid', 'cnt']
#     item_cnt_df['walk_weight'] = 1/item_cnt_df['cnt'].map(lambda x:pow(x, 0.75))
#     walk_weight_map = dict(zip(item_cnt_df['iid'].values, item_cnt_df['walk_weight'].values))
#     data['walk_weight'] = data['iid'].map(walk_weight_map)
#     data["pos_weight"] = data["duration"].map(math.log1p)
    
    #trans feat
    data['uid'] = label_encoder(data['uid'])
    data['iid'], iid_map_reverse = label_encoder(data['iid'], return_dict=True)
    pos_data, neg_data = data[data['label']=='pos'], data[data['label']=='neg']

    #build graph
    graph_data = {
        ('user', 'clicked', 'item'): (pos_data['uid'].values, pos_data['iid'].values),
        ('item', 'clicked-by', 'user'): (pos_data['iid'].values, pos_data['uid'].values),
        ('user', 'neg', 'item'): (neg_data['uid'].values, neg_data['iid'].values),
    }
    g = dgl.heterograph(graph_data, idtype=torch.int64)
    
    g.nodes['item'].data['iid'] = torch.LongTensor(list(iid_map_reverse.keys()))
    g.nodes['item'].data['neg_weight'] = torch.FloatTensor(data['iid'].value_counts().sort_index().map(lambda x:pow(x, 0.75)).values)
#     g.edges['clicked'].data['pos_weight'] = torch.FloatTensor(data['pos_weight'].values)
#     g.edges['clicked-by'].data['pos_weight'] = torch.FloatTensor(data['pos_weight'].values)
#     g.edges['clicked'].data['walk_weight'] = torch.FloatTensor(data['walk_weight'].values)
#     g.edges['clicked-by'].data['walk_weight'] = torch.FloatTensor(data['pos_weight'].values)
    return g, iid_map_reverse

if __name__ == '__main__':
    g, iid_map_reverse = build_graph_with_neg(file_path)
    print(g)
