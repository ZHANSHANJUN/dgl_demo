import pickle
import faiss
import numpy as np
import redis
import json
from tqdm import tqdm
from conf import *
from collections import Counter

work5redis = redis.StrictRedis(host="sg-prod-research-worker-5", port=6379)
#redisPrefix = "live_i2i_recall_"
redisPrefix = "live_pinsage_"
redisTTL = 86400*7


def main():
    print("params:", "emb_size", emb_size, "top_k", top_k)
    with open(output_path, 'rb') as f:
        item_embeddings, iid_map_reverse = pickle.load(f)
        print("item:", item_embeddings.shape, len(iid_map_reverse))
        
    iids = np.array(list(iid_map_reverse.values()))
    index = faiss.IndexFlatIP(emb_size)
    index_with_id = faiss.IndexIDMap(index)
    index_with_id.add_with_ids(item_embeddings, iids)
    
    pipeline = work5redis.pipeline(transaction=False)
    D, I = index_with_id.search(item_embeddings, top_k)
    print("diversity", len(Counter(I.reshape(-1))))
    print(I[1], I[100], I[1000], I[10000])
    #return 
    for i in range(len(iids)):
        iid = iids[i]       
        redisKey = redisPrefix + str(iid)
        redisField = "pinsage"
        sim_res = dict(zip(I[i].tolist(), D[i].tolist()))
        redisVal = json.dumps(sim_res)
#         print(redisKey, redisField, redisVal)
        pipeline.set(redisKey, redisVal, redisTTL)
        #pipeline.hset(redisKey, redisField, redisVal)
        #pipeline.expire(redisKey, redisTTL)
    pipeline.execute()
    
if __name__ == "__main__":
    main()
    
