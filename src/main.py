import os
import random
import numpy as np
from CascadeXML import CascadeXML
from torch.utils.data import DataLoader
from transformers import AdamW
import torch
import torch.distributed as dist
import torch.nn as nn
import argparse
from dataset import *
from data_utils import load_data
from Runner import Runner
from Runner_sparse import SparseRunner
# from Runner_accelerate import Runner
from dist_eval_sampler import DistributedEvalSampler

NUM_LABELS = {'Amazon-670K': 670091, 'Amazon-3M': 2812281, 'Wiki-500K' : 501070, 'AmazonCat-13K': 13330, 'Wiki10-31K': 30938, 'Eurlex': 3993, 'AT670': 670091, 'WT500': 501070, 'WSAT350': 352072}
NUM_CLUSTERS = {'Amazon-670K': 8192, 'Amazon-3M': 131072, 'Wiki-500K' : 65536, 'AmazonCat-13K': 128, 'Wiki10-31K': 256, 'Eurlex': 64, 'AT670': 8192, 'WT500':8192, 'WSAT350': 8192}
EVAL_SCHEME = {'Amazon-670K': 'weighted', 'Amazon-3M': 'weighted', 'Wiki-500K' : 'level', 'AmazonCat-13K': 'level', 'Wiki10-31K': 'weighted'}

def get_exp_name():
    name = [params.dataset, params.mn, '' if 'bert-base' in params.bert else params.bert]
    return '_'.join([i for i in name if i != ''])

def multi_collate(batch):
    batch = list(zip(*batch))
    input_ids = torch.stack(batch[0])
    attention_mask = torch.stack(batch[1])
    labels = [list(b) for b in batch[2:]]
    return input_ids, attention_mask, labels

def init_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def main(params):
    assert not (params.distributed and params.sparse), "DDP doesn't work with sparse gradients. NVCC backend doesn't support sparse and gloo is too slow to train"
    
    if params.distributed:  # parameters to initialize the process group
        env_dict = {
            key: os.environ[key]
            for key in ("MASTER_ADDR", "MASTER_PORT", "RANK",
                        "LOCAL_RANK", "WORLD_SIZE")
        }
        print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
        if params.sparse:
            dist.init_process_group(backend="gloo")
        else:
            dist.init_process_group(backend="nccl")
        print(
            f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
            + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
        )

        params.rank = int(os.environ["RANK"])
        params.local_rank = int(os.environ["LOCAL_RANK"])
        n = torch.cuda.device_count() // params.local_world_size
        device_ids = list(
            range(params.local_rank * n, (params.local_rank + 1) * n)
        )

        print(
            f"[{os.getpid()}] rank = {dist.get_rank()} ({params.rank}), "
            + f"world_size = {dist.get_world_size()}, n = {n}, device_ids = {device_ids}"
        )
        device = torch.device("cuda", params.local_rank)
        # params.seed = params.local_rank + params.seed
    else:
        params.local_rank = 0
        device = torch.device('cuda:0')

    init_seed(params.seed)
    params.num_labels = NUM_LABELS[params.dataset]
    params.num_clusters = NUM_CLUSTERS[params.dataset]
    params.model_name = get_exp_name()
    if params.local_rank == 0:
        print(get_exp_name())
        os.makedirs(params.model_name, exist_ok=True)
        print(f'load {params.dataset} dataset...')
    
    if len(params.load_model):
        params.load_model = os.path.join(params.model_name, params.load_model)

    params.data_path = os.path.join('./data/', params.dataset)

    X_train, X_test, Y_train, Y_test, X_tfidf, inv_prop = load_data(params.data_path, params.bert, params.num_labels, params.train_W)
    
    train_dataset = MultiXMLGeneral(X_train, Y_train, params, X_tfidf, mode='train')
    test_dataset = MultiXMLGeneral(X_test, Y_test, params, mode='test')

    if params.distributed:
        sampler = torch.utils.data.DistributedSampler(train_dataset, shuffle=True, seed=params.seed, 
            num_replicas=int(env_dict['WORLD_SIZE']))
        train_dl = DataLoader(train_dataset, batch_size=params.batch_size, num_workers=4, 
            collate_fn=multi_collate, pin_memory=True, sampler=sampler, shuffle=False)
        
        # Either distributed evaluation or single gpu evaluation
        if params.dist_eval:
            sampler = DistributedEvalSampler(test_dataset, shuffle=False)
            test_dl = DataLoader(test_dataset, batch_size=params.batch_size, num_workers=4, 
                collate_fn=multi_collate, pin_memory=True, sampler=sampler)
        else:
            test_dl = DataLoader(test_dataset, batch_size=params.batch_size, num_workers=4, 
                collate_fn=multi_collate, pin_memory=True, shuffle=False)

    else:
        shuf_train = not (params.return_embeddings or params.return_shortlist)
        print("Shuffle Train dataloader : ", shuf_train)
        train_dl = DataLoader(train_dataset, batch_size=params.batch_size, num_workers=4, 
                                    shuffle=shuf_train, collate_fn=multi_collate, pin_memory=True)
        test_dl = DataLoader(test_dataset, batch_size=params.batch_size, num_workers=4, 
                                    shuffle=False, collate_fn=multi_collate, pin_memory=True)

    model = CascadeXML(params = params, train_ds = train_dataset, device = device).to(device)

    if params.sparse:  # only for A3M and larger datasets
        runner = SparseRunner(params, train_dl, test_dl, inv_prop)
    else:
        runner = Runner(params, train_dl, test_dl, inv_prop)

    if params.ensemble_files:
        print("Files to ensemble: ", params.ensemble_files)
        runner.test_ensemble(params)

    runner.train(model, params, device)

    if params.distributed:
        dist.destroy_process_group()  # tear down the process group


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, required=False, default=16)
    parser.add_argument('--update_count', type=int, required=False, default=4)
    parser.add_argument('--lr', type=float, required=False, default=1e-4)
    parser.add_argument('--seed', type=int, required=False, default=29)

    parser.add_argument('--mn', type=str, required=True)
    parser.add_argument('--lm', dest='load_model', type=str, default="", help='model to load')
    parser.add_argument('--test', action='store_true', help='Testing mode or training mode')

    parser.add_argument('--num_epochs', type=int, required=False, default=15)
    parser.add_argument('--dataset', type=str, required=False, default='Amazon-670K')
    parser.add_argument('--bert', type=str, required=False, default='bert-base')
    parser.add_argument('--max_len', type=int, required=False, default=128)

    parser.add_argument('--topk', required=False, type=int, default=10, nargs='+')
    parser.add_argument('--freeze_layer_count', type=int, default=6)

    # DDP settings (remove later)
    parser.add_argument('--distributed', action='store_true', help='distributed training')
    parser.add_argument('--local_rank', type=int, help='node rank for distributed training', default=None)
    parser.add_argument('--local_world_size', type=int, default=2,
                            help='number of GPUs each process')
    parser.add_argument('--dist_eval', action='store_true', help='use ddp for eval as well')

    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--embed_drops', type=float, default=[0.2, 0.25, 0.4, 0.5], nargs='+')
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--use_multi_cls', action='store_true')

    parser.add_argument('--eval_scheme', type=str, choices=['weighted, level'], default='weighted')
    parser.add_argument('--sparse', action='store_true')
    parser.add_argument('--no_space', action='store_true')
    parser.add_argument('--return_embeddings', action='store_true')
    parser.add_argument('--return_shortlist', action='store_true')
    parser.add_argument('--train_W', action='store_true')
    parser.add_argument('--gradient_checkpointing', action='store_true')

    parser.add_argument('--ensemble_files', nargs='+', default=[], type=str)

    #Parabel Cluster params
    parser.add_argument('--cluster_name', default='clusters_eclare_4.pkl')
    parser.add_argument('--tree_depth', type=int, nargs='+', default=[10, 13, 16])  #b_factor
    parser.add_argument('--cluster_method', default='AugParabel')
    parser.add_argument('--verbose_lbs', type=int, default=0)

    #Graph params
    parser.add_argument('--graph_name', default='graph.npz')
    parser.add_argument('--prune_max_dist', type=float, default=1.0)
    parser.add_argument('--p_reset', type=float, default=0.8)
    parser.add_argument('--walk_len', type=int, default=400)
    parser.add_argument('--top_k', type=int, default=10)

    params = parser.parse_args()
    main(params)
