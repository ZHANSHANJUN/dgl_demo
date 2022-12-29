import os
import requests
import redis
from gensim.models import Word2Vec
import json

work5redis = redis.StrictRedis(host="sg-prod-research-worker-5", port=6379)
#redisPrefix = "live_i2i_recall_"
redisPrefix = "live_item2vec_"
redisTTL = 86400*7
topk=50

def send_warning():
    data = '{"type": "Card", "touser": ["shanjun.zhan"], "id": 9, "data": {"title": "live recall", "msg": "item2vec error..."}}'
    requests.post(url="https://devops.ushow.media/devops-goserver-v1/notify/message/", data=data)

def read_data():
    local_file = "/data/zsj/item2vec/input/item_session.txt"
    fp = open(local_file)
    lines = fp.readlines()
    fp.close()

    sequence = []
    for line in lines:
        sps = line.split(":")
        if len(sps) != 2:
            continue
        item_list = sps[1].split()
        sequence.append(item_list)

    if len(sequence) < 100:
        send_warning()

    print("sequence cnt", len(sequence))
    return sequence

def run():
    sequence = read_data()
    w2v = Word2Vec(sequence, vector_size=128, min_count=5, sg=1, workers=32, window=5, epochs=5)
    vocabulary = list(w2v.wv.key_to_index.keys())
    print("vocabulary cnt", len(vocabulary))
    pipeline = work5redis.pipeline(transaction=False)
    for uid in vocabulary:
        redisKey = redisPrefix + uid
        redisField = "item2vec"
        sim_res = w2v.wv.similar_by_word(uid, topn=topk)
        redisVal = json.dumps(dict(sim_res))    
        #print(redisKey, redisField, redisVal)
        pipeline.set(redisKey, redisVal, redisTTL)
        #pipeline.hset(redisKey, redisField, redisVal)
        #pipeline.expire(redisKey, redisTTL)
    pipeline.execute()	

if __name__ == '__main__':
    try:
        run()
    except Exception as e:
        print('error ' + str(e))
        send_warning()
