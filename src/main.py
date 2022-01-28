import sys
import os
import random
import numpy as np
from apex import amp
from model import LightXML
from torch.utils.data import DataLoader
from transformers import AdamW
import torch
from torch.utils.data import DataLoader
from dataset import XMLData
from data_utils import load_data, load_group
from log import Logger

NUM_LABELS = {'amazon670k': 670091, 'amazon3M': 2812281, 'wiki500K' : 501070, 'amazoncat13K': 13330, 'wiki31k': 30938, 'eurlex': 3993}
NUM_CLUSTERS = {'amazon670k': 8192, 'amazon3M': 131072, 'wiki500K' : 65536, 'amazoncat13K': 128, 'wiki31k': 256, 'eurlex': 64}

def train(model, train_dl, test_dl, group_y = None):
    
    model.cuda()
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=params.lr)
    
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    max_only_p5 = 0
    
    for epoch in range(params.epoch+5):
        train_loss = model.one_epoch(epoch, train_dl, optimizer, mode='train',
                                     eval_loader=test_dl, eval_step=params.eval_step, log=LOG)

        if epoch >= 20: #skip
            ev_result = model.one_epoch(epoch, test_dl, optimizer, mode='eval')
            g_p1, g_p3, g_p5, p1, p3, p5 = ev_result
            log_str = f'{epoch:>2}: {p1:.4f}, {p3:.4f}, {p5:.4f}, train_loss:{train_loss}'
            if params.dataset in ['wiki500k', 'amazon670k', 'AT670', 'WSAT', 'WT500']:
                log_str += f' {g_p1:.4f}, {g_p3:.4f}, {g_p5:.4f}'
            LOG.log(log_str)

            if max_only_p5 < p5:
                max_only_p5 = p5
                model.save_model(f'models/model-{get_exp_name()}.bin')

            if epoch >= params.epoch + 5 and max_only_p5 != p5:
                break


def get_exp_name():
    name = [params.dataset, '' if params.bert == 'bert-base' else params.bert]
    if params.dataset in ['wiki500k', 'amazon670k', 'WSAT', 'WT500']:
        name.append('t'+str(params.group_y))

    return '_'.join([i for i in name if i != ''])

def collate_func(batch):
    collated = []
    for i, b in enumerate(zip(*batch)):
        if i != 2:
            b = torch.stack(b)
        collated.append(b)
    return collated

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

parser.add_argument('--epoch', type=int, required=False, default=20)
parser.add_argument('--dataset', type=str, required=False, default='eurlex4k')
parser.add_argument('--bert', type=str, required=False, default='bert-base')
parser.add_argument('--max_len', type=int, required=False, default=512)

parser.add_argument('--valid', action='store_true')

parser.add_argument('--swa', action='store_true')
parser.add_argument('--swa_warmup', type=int, required=False, default=10)
parser.add_argument('--swa_step', type=int, required=False, default=100)

parser.add_argument('--group_y', type=int, default=0)
parser.add_argument('--topk', type=int, required=False, default=10)

parser.add_argument('--eval_step', type=int, required=False, default=20000)
parser.add_argument('--hidden_dim', type=int, required=False, default=300)
parser.add_argument('--eval_model', action='store_true')

params = parser.parse_args()

if __name__ == '__main__':
    init_seed(params.seed)
    
    print(get_exp_name())
    LOG = Logger('log_'+get_exp_name())
    print(f'load {params.dataset} dataset...')
    
    params.num_labels = NUM_LABELS[params.dataset]
    params.num_clusters = NUM_CLUSTERS[params.dataset]

    X_train, X_test, Y_train, Y_test = load_data(params.dataset, params.bert)
    
    group_y = load_group(params.dataset, params.num_clusters) if params.dataset in ['wiki500k', 'amazon670k'] else None
    collate_fn = collate_func if params.dataset in ['wiki500k', 'amazon670k'] else None

    train_dataset = XMLData(X_train, Y_train, params.num_labels, params.max_len, group_y, model_name = params.bert, mode='train')
    test_dataset = XMLData(X_test, Y_test, params.num_labels, params.max_len, group_y, model_name = params.bert, mode='train')
    train_dl = DataLoader(train_dataset, batch_size=params.batch_size, num_workers=4, shuffle=True, collate_fn=collate_fn, pin_memory=True)
    test_dl = DataLoader(test_dataset, batch_size=params.batch_size, num_workers=4, shuffle=False, collate_fn=collate_fn, pin_memory=True)

    if params.dataset in ['wiki500k', 'amazon670k']:
        model = LightXML(params = params, group_y=group_y)
    else:
        model = LightXML(params = params)

    if params.eval_model and params.dataset in ['wiki500k', 'amazon670k']:
        print(f'load models/model-{get_exp_name()}.bin')
        model.load_state_dict(torch.load(f'models/model-{get_exp_name()}.bin'))
        model = model.cuda()

        pred_scores, pred_labels = model.one_epoch(0, test_dl, None, mode='test')
        np.save(f'results/{get_exp_name()}-labels.npy', np.array(pred_labels))
        np.save(f'results/{get_exp_name()}-scores.npy', np.array(pred_scores))
        sys.exit(0)

    train(model, train_dl, test_dl, group_y)
