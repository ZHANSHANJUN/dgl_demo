## graph params
file_path = "/data/zsj/pinsage/input/graph.txt"
# file_path = "/data/zsj/pinsage/input/sample.txt"
uid_min_cnt = 5
duration_min = 10

output_path = "/data/zsj/pinsage/output/item.pkl"

## sampler params
random_walk_length = 2
random_walk_restart_prob = 0.5
num_random_walks = 10
num_neighbors = 3
hard_neg_ratio = 0.1

## train params
batch_size = 1024
num_layers = 2
num_workers = 0

lr = 5e-3
l2_weight = 5e-4
num_epochs = 15
batches_per_epoch = 1000

hidden_dims_dict = {
    "iid":64
}


## i2i params
emb_size = sum(hidden_dims_dict.values())
top_k = 50

