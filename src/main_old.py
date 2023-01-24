import os
import random
import numpy as np
from LightXML import LightXML
# from detached_model import Detached_LightXML
# from LightIncept import LightIncXML
from torch.utils.data import DataLoader
from transformers import AdamW
import torch
import torch.distributed as dist
import torch.nn as nn

# from dataset import XMLData, MultiXMLData
from dataset import *
from data_utils import load_data, load_group, load_cluster_tree
from Runner_Plus import Runner as LightXMLRunner
# from Runner_detached import Runner as DetachedRunner
from dist_eval_sampler import DistributedEvalSampler

NUM_LABELS = {'AT131': 131073, 'amazon670k': 670091, 'amazon3M': 2812281, 'wiki500K' : 501070, 'amazoncat13K': 13330, 'wiki31k': 30938, 'eurlex': 3993, 'AT670': 670091, 'WT500': 501070, 'WSAT350': 352072}
NUM_CLUSTERS = {'AT131': 8192, 'amazon670k': 8192, 'amazon3M': 131072, 'wiki500K' : 65536, 'amazoncat13K': 128, 'wiki31k': 256, 'eurlex': 64, 'AT670': 8192, 'WT500':8192, 'WSAT350': 8192}
NAME_MAP = {'amazon670k': 'Amazon-670K', 'AT131': 'LF-AmazonTitles-131K'}

def get_exp_name():
    name = [params.dataset, params.mn, '' if params.bert == 'bert-base' else params.bert]
    if params.dataset in ['wiki500k', 'amazon670k', 'WSAT', 'WT500', 'AT131']:
        name.append('t'+str(params.tree_id))

    return '_'.join([i for i in name if i != ''])

def collate_func(batch):
    collated = []
    for i, b in enumerate(zip(*batch)):
        if i != 2:
            b = torch.stack(b)
        collated.append(b)
    return collated

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


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, required=False, default=16)
parser.add_argument('--update_count', type=int, required=False, default=1)
parser.add_argument('--lr', type=float, required=False, default=1e-4)
parser.add_argument('--seed', type=int, required=False, default=29)

parser.add_argument('--mn', type=str, required=True)
parser.add_argument('--lm', dest='load_model', type=str, default="", help='model to load')
parser.add_argument('--test', action='store_true', help='Testing mode or training mode')

parser.add_argument('--num_epochs', type=int, required=False, default=15)
parser.add_argument('--dataset', type=str, required=False, default='eurlex4k')
parser.add_argument('--bert', type=str, required=False, default='bert-base')
parser.add_argument('--max_len', type=int, required=False, default=128)

parser.add_argument('--swa', action='store_true')
parser.add_argument('--swa_warmup', type=int, required=False, default=30)
parser.add_argument('--swa_step', type=int, required=False, default=3000)

parser.add_argument('--tree_id', type=int, default=0)
parser.add_argument('--topk', required=False, type=int, default=10, nargs='+')
# parser.add_argument('--freeze_epoch', type=int, default=50)
parser.add_argument('--freeze_layer_count', type=int, default=6)

parser.add_argument('--eval_step', type=int, required=False, default=20000)
parser.add_argument('--hidden_dim', type=int, required=False, default=300)
parser.add_argument('--load_model', type=str, default='', required=False)

parser.add_argument('--use_detach', action='store_true')
parser.add_argument('--use_incept', action='store_true')
parser.add_argument('--adahess', action='store_true')

parser.add_argument('--distributed', action='store_true', help='distributed training')
parser.add_argument('--local_rank', type=int, help='node rank for distributed training', default=None)
parser.add_argument('--local_world_size', type=int, default=2,
                        help='number of GPUs each process')
parser.add_argument('--dist_eval', action='store_true', help='use ddp for eval as well')
parser.add_argument('--embed_drops', type=float, default=[0.3, 0.3, 0.3, 0.4], nargs='+')
parser.add_argument('--weight_decay', type=float, default=0.05)

#Parabel Cluster params
parser.add_argument('--recluster', action='store_true')
parser.add_argument('--reinit', action='store_true')
parser.add_argument('--cluster_name', default='clusters_eclare.pkl') 
# parser.add_argument('--cluster_name', default='recluster_eclare_768CM.pkl')
parser.add_argument('--b_factors', type=list, default=[13, 16])
parser.add_argument('--cluster_method', default='AugParabel')
parser.add_argument('--verbose_lbs', type=int, default=500)

#Graph params
parser.add_argument('--graph_name', default='graph.npz')
parser.add_argument('--prune_max_dist', type=float, default=1.0)
parser.add_argument('--p_reset', type=float, default=0.8)
parser.add_argument('--walk_len', type=int, default=400)
parser.add_argument('--top_k', type=int, default=10)

params = parser.parse_args()

if __name__ == '__main__':
    if params.distributed:  # parameters to initialize the process group
        env_dict = {
            key: os.environ[key]
            for key in ("MASTER_ADDR", "MASTER_PORT", "RANK",
                        "LOCAL_RANK", "WORLD_SIZE")
        }
        print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
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

        torch.cuda.set_device(params.local_rank)
        device = torch.device("cuda", params.local_rank)

        print(
            f"[{os.getpid()}] rank = {dist.get_rank()} ({params.rank}), "
            + f"world_size = {dist.get_world_size()}, n = {n}, device_ids = {device_ids}"
        )

        params.seed = params.local_rank + params.seed
    else:
        params.local_rank = 0
        device = torch.device('cuda:0')

    init_seed(params.seed)
    if params.local_rank == 0:
        print(get_exp_name())
        print(f'load {params.dataset} dataset...')
    
    params.num_labels = NUM_LABELS[params.dataset]
    params.num_clusters = NUM_CLUSTERS[params.dataset]
    params.model_name = get_exp_name()

    if not os.path.exists(params.model_name) and params.local_rank == 0:
        os.makedirs(params.model_name)
    
    if len(params.load_model):
        params.load_model = os.path.join(params.model_name, params.load_model)

    params.data_path = os.path.join('../Datasets/', NAME_MAP[params.dataset])

    X_train, X_test, lbl, Y_train, Y_test, X_tfidf, inv_prop = load_data(params.data_path, params.bert, params.num_labels, None)
    
    if not params.use_detach:
        group_y = load_group(NAME_MAP[params.dataset], params.num_clusters) if params.dataset in ['wiki500k', 'amazon670k','AT131', 'AT670'] else None
        collate_fn = collate_func if params.dataset in ['wiki500k', 'AT131', 'amazon670k', 'AT670'] else None

        train_dataset = XMLData(X_train, Y_train, params.num_labels, params.max_len, params, group_y, model_name = params.bert, mode='train')
        test_dataset = XMLData(X_test, Y_test, params.num_labels, params.max_len, params, group_y, model_name = params.bert, mode='test')
        train_dl = DataLoader(train_dataset, batch_size=params.batch_size, num_workers=0, shuffle=True, collate_fn=collate_fn, pin_memory=True)
        test_dl = DataLoader(test_dataset, batch_size=params.batch_size, num_workers=0, shuffle=False, collate_fn=collate_fn, pin_memory=True)

        if params.use_incept:
            if params.dataset in ['wiki500k', 'amazon670k']:
                model = LightIncXML(params = params, group_y=group_y)
            else:
                model = LightIncXML(params = params)
        
        else:
            if params.dataset in ['wiki500k', 'AT131', 'amazon670k', 'AT670']:
                model = LightXML(params = params, group_y=group_y)
            else:
                model = LightXML(params = params)            
        
        runner = LightXMLRunner(params, train_dl, test_dl)
        runner.train(model, params)
    
    else:
        collate_fn = multi_collate
        
        # clusters = load_cluster_tree(params.dataset) if params.dataset in ['wiki500k', 'amazon670k'] else None
        # train_dataset = MultiXMLData(X_train, Y_train, params.num_labels, params.max_len, clusters, model_name = params.bert, mode='train')
        # test_dataset = MultiXMLData(X_test, Y_test, params.num_labels, params.max_len, clusters, model_name = params.bert, mode='test')
        
        train_dataset = MultiXMLGeneral(X_train, Y_train, params, X_tfidf, mode='train')
        test_dataset = MultiXMLGeneral(X_test, Y_test, params, mode='test')

        # clusters = np.load('/scratch/project_2001083/devaansh/pecos/pecos/xmc/xtransformer/clustering_xr.npy', allow_pickle=True)
        # train_dataset = PecosDataset(X_train, Y_train, params.num_labels, params.max_len, clusters, model_name = params.bert, mode='train')
        # test_dataset = PecosDataset(X_test, Y_test, params.num_labels, params.max_len, clusters, model_name = params.bert, mode='test', cluster_levels=[12, 15])
        
        if params.distributed:
            sampler = torch.utils.data.DistributedSampler(train_dataset)
            train_dl = DataLoader(train_dataset, batch_size=params.batch_size, num_workers=4, 
                collate_fn=collate_fn, pin_memory=True, sampler=sampler)
        else:
            train_dl = DataLoader(train_dataset, batch_size=params.batch_size, num_workers=4, shuffle=True, collate_fn=collate_fn, pin_memory=True)
        
        if params.distributed and params.dist_eval:
            sampler = DistributedEvalSampler(test_dataset)
            test_dl = DataLoader(test_dataset, batch_size=params.batch_size, num_workers=4, 
                collate_fn=collate_fn, pin_memory=True, sampler=sampler)
        else:
            test_dl = DataLoader(test_dataset, batch_size=params.batch_size, num_workers=4, shuffle=False, collate_fn=collate_fn, pin_memory=True)
        
        if params.dataset in ['wiki500k', 'amazon670k']:
            model = Detached_LightXML(params = params, train_ds = train_dataset).to(device)
        else:
            model = Detached_LightXML(params = params).to(device)

        if params.distributed:
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[params.local_rank], output_device=params.local_rank,
                find_unused_parameters=True
            )
        
        runner = DetachedRunner(params, train_dl, test_dl)
        runner.train(model, params, device)

    if params.distributed:
        di.destroy_process_group()  # tear down the process group
