import os
import json
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
# from Runner import Runner
from Runner_sparse import SparseRunner
from Runner_accelerate import Runner
from dist_eval_sampler import DistributedEvalSampler

NUM_LABELS = {'Amazon-670K': 670091, 'Amazon-3M': 2812281, 'Wiki-500K' : 501070, 'AmazonCat-13K': 13330, 'Wiki10-31K': 30938, 'Eurlex': 3993, 'AT670': 670091, 'WT500': 501070, 'WSAT350': 352072}
NUM_CLUSTERS = {'Amazon-670K': 8192, 'Amazon-3M': 131072, 'Wiki-500K' : 65536, 'AmazonCat-13K': 128, 'Wiki10-31K': 256, 'Eurlex': 64, 'AT670': 8192, 'WT500':8192, 'WSAT350': 8192}
EVAL_SCHEME = {'Amazon-670K': 'weighted', 'Amazon-3M': 'weighted', 'Wiki-500K' : 'level', 'AmazonCat-13K': 'level', 'Wiki10-31K': 'weighted'}


def init_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def load_model(model, name):
    checkpoint = torch.load(name, map_location=torch.device('cpu'))
    if 'state_dict' in checkpoint:
        # checkpoint has entire training state
        try:
            model.load_state_dict(checkpoint['state_dict'], strict=True)
        except RuntimeError as e:
            print(traceback.format_exc())
            raise e
    else:
        # checkpoint only has model
        try:
            model.load_state_dict(checkpoint, strict=True)
        except RuntimeError as e:
            print(traceback.format_exc())
            raise e
    return model


@torch.no_grad()
def main(params):
    init_seed(params.seed)
    device = torch.device('cuda:0')

    params.num_labels = NUM_LABELS[params.dataset]
    params.num_clusters = NUM_CLUSTERS[params.dataset]
    params.return_shortlist = False
    params.rw_loss = False
    params.no_space = False
    params.sparse = False
    params.embed_drops = [0,0,0,0]

    # import pdb; pdb.set_trace()
    if not os.path.exists(params.model_name):
        raise ValueError("Model path doesn't exist")

    params.data_path = os.path.join('./data/', params.dataset)
    inference_groups = InferenceDataset(params)
    # import pdb; pdb.set_trace()

    model = CascadeXML(params, inference_groups).to(device)
    model.eval()
    print("Loading model from ", params.model_name)
    model = load_model(model, params.model_name)

    print('loading bert-base-uncased tokenizer')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    if not os.path.exists(params.input):
        print('Input is not a valid path, assuming string input')
        text = params.input
        text = text.replace('\n', ' ')
    else:
        with open(params.input, 'r') as f:
            text = f.read()
        text = text.replace('\n', ' ')
    
    if os.path.exists(params.label_map):
        label_map = json.load(open(params.label_map, 'r'))
    text_tokens = torch.tensor(tokenizer.encode(text)[:params.max_len])
    attn_mask = torch.ones_like(text_tokens)

    all_probs, all_candidates, all_probs_weighted = model(text_tokens.unsqueeze(0).to(device), attn_mask.unsqueeze(0).to(device))
    if params.eval_scheme == 'level':
        all_preds = [torch.topk(probs, 10)[1].cpu() for probs in all_probs]
    else:
        all_preds = [torch.topk(probs, 10)[1].cpu() for probs in all_probs_weighted]

    all_preds = [candidates[np.arange(preds.shape[0]).reshape(-1, 1), preds].cpu()
                for candidates, preds in zip(all_candidates, all_preds)]
    
    # Meta labels discarde for inference
    actual_labels = all_preds[-1][0]

    # turn label idx to text -> missing label remapping + label text querying
    print([label_map[i]['title'] for i in actual_labels])


        







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=False, default=29)
    parser.add_argument('--mn', dest='model_name', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=False, default='Wiki-500K')
    parser.add_argument('--bert', type=str, required=False, default='bert-base')
    parser.add_argument('--max_len', type=int, required=False, default=128)
    parser.add_argument('--topk', required=False, type=int, default=[128, 256, 512], nargs='+')
    parser.add_argument('--eval_scheme', type=str, choices=['weighted, level'], default='level')
    #Parabel Cluster params
    parser.add_argument('--cluster_name', default='Eclusters_1865.pkl')
    parser.add_argument('--b_factors', type=int, nargs='+', default=[10, 13, 16])
    parser.add_argument('--cluster_method', default='AugParabel')
    parser.add_argument('--verbose_lbs', type=int, default=0)

    parser.add_argument('--input', required=True, type=str, 
    help='input to run model on. If text, will run on text, if csv, will run on first column entries')
    parser.add_argument('--label_map', required=False, help='label id to value map as json')

    params = parser.parse_args()
    # import pdb; pdb.set_trace()
    main(params)
