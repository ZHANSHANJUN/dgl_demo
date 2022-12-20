import pickle
import redis
import faiss
from sage_ns import Config, save_to_redis


def main():
    conf = Config()
    print(conf.__dict__)
    
    with open(conf.output_path, 'rb') as f:
        h_item, user_ids = pickle.load(f)
        
    save_to_redis(h_item, user_ids, conf)


if __name__ == "__main__":
    main()
