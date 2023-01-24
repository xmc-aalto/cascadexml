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
from accelerate import Accelerator, DistributedDataParallelKwargs

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
    
    if params.distributed:
        ddp_handler = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(kwargs_handlers=[ddp_handler])
    else:
        accelerator = Accelerator(gradient_accumulation_steps=4)
    
    device = accelerator.device

    init_seed(params.seed)
    if accelerator.is_main_process:
        print(get_exp_name())
        os.makedirs(params.model_name, exist_ok=True)
        print(f'load {params.dataset} dataset...')
    
    params.num_labels = NUM_LABELS[params.dataset]
    params.num_clusters = NUM_CLUSTERS[params.dataset]
    params.model_name = get_exp_name()
    
    if len(params.load_model):
        params.load_model = os.path.join(params.model_name, params.load_model)

    params.data_path = os.path.join('./data/', params.dataset)

    X_train, X_test, Y_train, Y_test, X_tfidf, inv_prop = load_data(params.data_path, params.bert, params.num_labels, params.train_W)
    
    train_dataset = MultiXMLGeneral(X_train, Y_train, params, X_tfidf, mode='train')
    test_dataset = MultiXMLGeneral(X_test, Y_test, params, mode='test')

    train_dl = DataLoader(train_dataset, batch_size=params.batch_size, num_workers=4, 
                                shuffle=True, collate_fn=multi_collate, pin_memory=True)
    test_dl = DataLoader(test_dataset, batch_size=params.batch_size, num_workers=4, 
                                shuffle=False, collate_fn=multi_collate, pin_memory=True)

    model = CascadeXML(params = params, train_ds = train_dataset)

    if params.sparse:
        runner = SparseRunner(params, train_dl, test_dl, inv_prop)
    else:
        runner = Runner(params, train_dl, test_dl, inv_prop)

    if params.ensemble_files and accelerator.is_main_process:
        print("Files to ensemble: ", params.ensemble_files)
        runner.test_ensemble(params)

    runner.train(model, params, device, accelerator)

    # if params.distributed:
    #     dist.destroy_process_group()  # tear down the process group


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

    parser.add_argument('--swa', action='store_true')
    parser.add_argument('--swa_warmup', type=int, required=False, default=30)
    parser.add_argument('--swa_step', type=int, required=False, default=3000)
    parser.add_argument('--swa_update_step', type=int, required=False, default=3000)

    parser.add_argument('--topk', required=False, type=int, default=10, nargs='+')
    parser.add_argument('--freeze_layer_count', type=int, default=6)

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
    parser.add_argument('--rw_loss', type=int, nargs='+', default=[1, 1, 1, 1])

    parser.add_argument('--ensemble_files', nargs='+', default=[], type=str)

    #Parabel Cluster params
    parser.add_argument('--cluster_name', default='clusters_eclare_4.pkl')
    parser.add_argument('--b_factors', type=int, nargs='+', default=[10, 13, 16])
    parser.add_argument('--cluster_method', default='AugParabel')
    parser.add_argument('--verbose_lbs', type=int, default=0)

    #Graph params
    parser.add_argument('--graph_name', default='graph.npz')
    parser.add_argument('--prune_max_dist', type=float, default=1.0)
    parser.add_argument('--p_reset', type=float, default=0.8)
    parser.add_argument('--walk_len', type=int, default=400)
    parser.add_argument('--top_k', type=int, default=10)

    parser.add_argument('--use_r', action='store_true')
    parser.add_argument('--word_pool', action='store_true')

    params = parser.parse_args()
    main(params)
