import os
import random
import numpy as np
from model import LightXML
from detached_model import Detached_LightXML
from LightIncept import LightIncXML
from torch.utils.data import DataLoader
from transformers import AdamW
import torch
from dataset import XMLData, MultiXMLData
from data_utils import load_data, load_group, load_cluster_tree
from Runner_Plus import Runner as LightXMLRunner
from Runner_detached import Runner as DetachedRunner

NUM_LABELS = {'amazon670k': 670091, 'amazon3M': 2812281, 'wiki500K' : 501070, 'amazoncat13K': 13330, 'wiki31k': 30938, 'eurlex': 3993, 'AT670': 670091, 'WT500': 501070, 'WSAT350': 352072}
NUM_CLUSTERS = {'amazon670k': 8192, 'amazon3M': 131072, 'wiki500K' : 65536, 'amazoncat13K': 128, 'wiki31k': 256, 'eurlex': 64, 'AT670': 8192, 'WT500':8192, 'WSAT350': 8192}


def get_exp_name():
    name = [params.dataset, params.mn, '' if params.bert == 'bert-base' else params.bert]
    if params.dataset in ['wiki500k', 'amazon670k', 'WSAT', 'WT500']:
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
    cluster_binarized = torch.stack(batch[2])
    cluster_ids_1 = batch[3]
    labels = batch[4]

    return input_ids, attention_mask, (cluster_binarized, cluster_ids_1, labels)

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

parser.add_argument('--num_epochs', type=int, required=False, default=20)
parser.add_argument('--dataset', type=str, required=False, default='eurlex4k')
parser.add_argument('--bert', type=str, required=False, default='bert-base')
parser.add_argument('--max_len', type=int, required=False, default=512)

parser.add_argument('--swa', action='store_true')
parser.add_argument('--swa_warmup', type=int, required=False, default=30)
parser.add_argument('--swa_step', type=int, required=False, default=3000)

parser.add_argument('--tree_id', type=int, default=0)
parser.add_argument('--topk', required=False, type=int, default=10, nargs='+')

parser.add_argument('--eval_step', type=int, required=False, default=20000)
parser.add_argument('--hidden_dim', type=int, required=False, default=300)
parser.add_argument('--load_model', type=str, default='', required=False)

parser.add_argument('--use_detach', action='store_true')
parser.add_argument('--use_incept', action='store_true')
parser.add_argument('--adahess', action='store_true')

params = parser.parse_args()

if __name__ == '__main__':
    init_seed(params.seed)
    
    print(get_exp_name())
    print(f'load {params.dataset} dataset...')
    
    params.num_labels = NUM_LABELS[params.dataset]
    params.num_clusters = NUM_CLUSTERS[params.dataset]
    params.model_name = get_exp_name()

    if not os.path.exists(params.model_name):
        os.makedirs(params.model_name)
    
    if len(params.load_model):
        params.load_model = os.path.join(params.model_name, params.load_model)

    X_train, X_test, Y_train, Y_test = load_data(params.dataset, params.bert)
    
    if not params.use_detach:
        group_y = load_group(params.dataset, params.num_clusters) if params.dataset in ['wiki500k', 'amazon670k', 'AT670'] else None
        collate_fn = collate_func if params.dataset in ['wiki500k', 'amazon670k', 'AT670'] else None

        train_dataset = XMLData(X_train, Y_train, params.num_labels, params.max_len, group_y, model_name = params.bert, mode='train')
        test_dataset = XMLData(X_test, Y_test, params.num_labels, params.max_len, group_y, model_name = params.bert, mode='test')
        train_dl = DataLoader(train_dataset, batch_size=params.batch_size, num_workers=4, shuffle=True, collate_fn=collate_fn, pin_memory=True)
        test_dl = DataLoader(test_dataset, batch_size=params.batch_size, num_workers=4, shuffle=False, collate_fn=collate_fn, pin_memory=True)

        if params.use_incept:
            if params.dataset in ['wiki500k', 'amazon670k']:
                model = LightIncXML(params = params, group_y=group_y)
            else:
                model = LightIncXML(params = params)
        
        else:
            if params.dataset in ['wiki500k', 'amazon670k', 'AT670']:
                model = LightXML(params = params, group_y=group_y)
            else:
                model = LightXML(params = params)            
        
        runner = LightXMLRunner(params, train_dl, test_dl)
        runner.train(model, params)
    
    else:
        clusters = load_cluster_tree(params.dataset) if params.dataset in ['wiki500k', 'amazon670k'] else None
        collate_fn = multi_collate

        train_dataset = MultiXMLData(X_train, Y_train, params.num_labels, params.max_len, clusters, model_name = params.bert, mode='train')
        test_dataset = MultiXMLData(X_test, Y_test, params.num_labels, params.max_len, clusters, model_name = params.bert, mode='test')
        train_dl = DataLoader(train_dataset, batch_size=params.batch_size, num_workers=0, shuffle=True, collate_fn=collate_fn, pin_memory=True)
        test_dl = DataLoader(test_dataset, batch_size=params.batch_size, num_workers=0, shuffle=False, collate_fn=collate_fn, pin_memory=True)

        if params.dataset in ['wiki500k', 'amazon670k']:
            model = Detached_LightXML(params = params, train_ds = train_dataset)
        else:
            model = Detached_LightXML(params = params)
        
        runner = DetachedRunner(params, train_dl, test_dl)
        runner.train(model, params)
