import pickle
import faiss
import numpy as np
import redis
import json
from tqdm import tqdm
from conf import *
from collections import Counter

work10redis = redis.StrictRedis(host="sg-prod-research-worker-10", port=6379)
redisPrefix = "live_node2vec_"
redisTTL = 86400*7


def main():
    print("params:", "emb_size", emb_size, "top_k", top_k)
    with open(rid_output_path, 'rb') as f:
        rid_keys, rid_values = pickle.load(f)
        print("rid:", len(rid_keys))
        
    with open(sid_output_path, 'rb') as f:
        sid_keys, sid_values = pickle.load(f)
        print("sid:", len(sid_keys))
        
    index = faiss.IndexFlatIP(emb_size)
    index_with_id = faiss.IndexIDMap(index)
    index_with_id.add_with_ids(rid_values, rid_keys)
    
    pipeline = work10redis.pipeline(transaction=False)
    D, I = index_with_id.search(rid_values, top_k)
    print("rid diversity", len(Counter(I.reshape(-1))))
    print(I[1], I[100], I[1000], I[10000])
    for i in range(len(rid_keys)):
        rid = rid_keys[i]       
        redisKey = redisPrefix + "r2r_" + str(rid)
        sim_res = dict(zip(I[i].tolist(), D[i].tolist()))
        redisVal = json.dumps(sim_res)
#         print(redisKey, redisVal)
        pipeline.set(redisKey, redisVal)
        pipeline.expire(redisKey, redisTTL)
        
    print("----------")
    D, I = index_with_id.search(sid_values, top_k)
    print("sid diversity", len(Counter(I.reshape(-1))))
    print(I[1], I[100], I[1000], I[10000])
    for i in range(len(sid_keys)):
        sid = sid_keys[i]       
        redisKey = redisPrefix + "s2r_" + str(sid)
        sim_res = dict(zip(I[i].tolist(), D[i].tolist()))
        redisVal = json.dumps(sim_res)
#         print(redisKey, redisVal)
        pipeline.set(redisKey, redisVal)
        pipeline.expire(redisKey, redisTTL)
        
    pipeline.execute()
    
if __name__ == "__main__":
    main()
    