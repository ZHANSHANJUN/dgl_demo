import dgl
import torch
import numpy as np
import pickle
from gensim.models import Word2Vec
from graph import build_graph
from conf import *


def gen_walks(g):
    all_walks = []
    for i in range(num_walks):
        walks = dgl.sampling.node2vec_random_walk(g, np.arange(g.number_of_nodes()), p, q, walk_length).tolist()
        all_walks += walks

    np.random.shuffle(all_walks)
    print("all_walks cnt:", len(all_walks))
    return all_walks

def save_vec(w2v, all_id_map):
    rids = all_id_map['rids']
    sids = all_id_map['sids']
    all_ids = all_id_map['all_ids']
    
    rid_keys = all_ids[rids]
    rid_values = w2v.wv.vectors[[w2v.wv.key_to_index[i] for i in rids]]
    with open(rid_output_path, 'wb') as f:
        pickle.dump([rid_keys, rid_values], f)
        
    sid_keys = all_ids[sids]
    sid_values = w2v.wv.vectors[[w2v.wv.key_to_index[i] for i in sids]]
    with open(sid_output_path, 'wb') as f:
        pickle.dump([sid_keys, sid_values], f)

def main():
    g, all_id_map = build_graph()
    print(g)
    walks = gen_walks(g)
    w2v = Word2Vec(walks, vector_size=emb_size, min_count=1, sg=1, workers=32, window=window_size, epochs=3)
    save_vec(w2v, all_id_map)

if __name__ == '__main__':
    main()
    
