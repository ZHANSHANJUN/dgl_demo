#input params
duration_min = 10
live_path = '/data/zsj/node2vec/input/live.txt'
sing_path = '/data/zsj/node2vec/input/sing.txt'
friend_path = '/data/zsj/node2vec/input/friend.txt'


#random walk params
num_walks = 10
walk_length = 50
emb_size = 128
window_size = 10
p = 1
q = 0.5


#output params
rid_output_path = '/data/zsj/node2vec/output/rid.pkl'
sid_output_path = '/data/zsj/node2vec/output/sid.pkl'


#i2i params
top_k = 100
