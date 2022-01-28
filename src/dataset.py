import os
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizer

def get_tokenizer(model_name):
    if 'roberta' in model_name:
        print('loading roberta-base tokenizer')
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
    elif 'xlnet' in model_name:
        print('loading xlnet-base-cased tokenizer')
        tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    else:
        print('loading bert-base-uncased tokenizer')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    return tokenizer

class XMLData(Dataset):
    def __init__(self, x, y, num_labels, max_length, group_y=None, 
                    model_name = 'bert-base', mode='train'):
        super(XMLData).__init__()
        assert mode in ["train", "test"]
        self.mode = mode
        self.x = x
        self.y = list(list(map(int, x)) for x in y)
        self.n_labels = num_labels
        self.tokenizer = get_tokenizer(model_name)
        self.max_len = max_length
        self.label_to_cluster_ids = None
        self.label_space = torch.zeros(self.n_labels)
        self.cls_token_id = [self.tokenizer.cls_token_id]
        self.sep_token_id = [self.tokenizer.sep_token_id]

        if group_y is not None:
            # group y mode
            self.num_clusters = group_y.shape[0]
            self.cluster_space = torch.zeros(self.num_clusters)
            self.label_to_cluster_ids = np.zeros(self.n_labels, dtype=np.int64) - 1
            for idx, labels in enumerate(group_y):
                self.label_to_cluster_ids[[int(l) for l in labels]] = idx
            
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        labels = torch.LongTensor(self.y[idx])
        input_ids = self.x[idx]

        if len(input_ids) > self.max_len-2:
            input_ids = input_ids[:self.max_len-2]
        
        input_ids = self.cls_token_id + input_ids + self.sep_token_id
        padding_length = self.max_len - len(input_ids)
        attention_mask = torch.tensor([1] * len(input_ids) + [0] * padding_length)
        input_ids = torch.tensor(input_ids + ([0] * padding_length))
        
        if self.label_to_cluster_ids is not None and self.mode == 'train':
            cluster_ids = np.unique(self.label_to_cluster_ids[labels])
            if cluster_ids[0] == -1:
                cluster_ids = cluster_ids[1:]
            cluster_binarized = self.cluster_space.scatter(0, torch.tensor(cluster_ids), 1.0)
        
        if self.label_to_cluster_ids is not None: 
            if self.mode=='train':            
                return input_ids, attention_mask, labels, cluster_binarized
            else:
                return input_ids, attention_mask, labels
        else:
            label_binarized = self.label_space.scatter(0, labels, 1.0)
            return input_ids, attention_mask, label_binarized
