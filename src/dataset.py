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
        self.y = list(list(map(int, l)) for l in y)
        self.n_labels = num_labels
        # self.tokenizer = get_tokenizer(model_name)
        self.max_len = max_length
        self.label_to_cluster_ids = None
        self.label_space = torch.zeros(self.n_labels)
        self.cls_token_id = [101] #[self.tokenizer.cls_token_id]
        self.sep_token_id = [102] #[self.tokenizer.sep_token_id]


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
            return input_ids, attention_mask, labels
            

def word_pool(words, tfidfs, out_shape):
    assert out_shape <= words.shape[0]
    bs = words.shape[0]
    
    chunks = torch.tensor_split(torch.arange(words.shape[0]), out_shape, dim=0)
    new_words = [(words[:, c] * tfidf[:, c].reshape(bs, -1, 1)).sum(1) for c in chunks]

    return torch.stack(new_words, dim=1)  # b, out_shape, 300


class MultiXMLData(Dataset):
    def __init__(self, x, y, num_labels, max_length, groups=None, 
                    model_name = 'bert-base', mode='train'):
        super().__init__()
        assert mode in ["train", "test"]
        self.mode = mode
        self.x = x
        self.y = list(list(map(int, l)) for l in y)
        self.n_labels = num_labels
        # self.tokenizer = get_tokenizer(model_name)
        self.max_len = max_length
        self.label_to_cluster_ids = None
        self.label_space = torch.zeros(self.n_labels)
        self.cls_token_id = [101] #[self.tokenizer.cls_token_id]
        self.sep_token_id = [102] #[self.tokenizer.sep_token_id]

        self.groups = groups.copy()
        # self.groups = [self.groups[0], self.groups[2]]

        if self.groups is not None:
            self.num_clusters = [len(g) for g in self.groups]
            # self.cluster_space = torch.zeros(self.num_clusters)

            self.label_to_cluster_ids_1 = np.zeros(self.n_labels, dtype=np.int64) - 1
            for idx, labels in enumerate(self.groups[1]):
                self.label_to_cluster_ids_1[[int(l) for l in labels]] = idx

            self.groups[0] = [np.unique(self.label_to_cluster_ids_1[l]) for l in self.groups[0]]

            self.label_to_cluster_ids_2 = np.zeros(self.num_clusters[1], dtype=np.int64) - 1
            for idx, labels in enumerate(self.groups[0]):
                self.label_to_cluster_ids_2[[int(l) for l in labels]] = idx
            
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

        cluster_ids_1 = torch.LongTensor(np.unique(self.label_to_cluster_ids_1[labels]))
        if cluster_ids_1[0] == -1:
            cluster_ids_1 = cluster_ids_1[1:]
        # cluster_binarized = torch.zeros(self.num_clusters[-1]).scatter(0, torch.tensor(cluster_ids_1), 1.0)
        
        cluster_ids_2 = torch.LongTensor(np.unique(self.label_to_cluster_ids_2[cluster_ids_1]))
        if cluster_ids_2[0] == -1:
            cluster_ids_2 = cluster_ids_2[1:]
        cluster_binarized = torch.zeros(self.num_clusters[0]).scatter(0, cluster_ids_2, 1.0)

        return input_ids, attention_mask, cluster_binarized, cluster_ids_1, labels