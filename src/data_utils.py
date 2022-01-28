import os
import torch
import pickle as pkl
import numpy as np
from torch.utils.data import Dataset
import tqdm
from transformers import BertTokenizer

def get_fast_tokenizer(self):
    if 'roberta' in self.bert_name:
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', do_lower_case=True)
    elif 'xlnet' in self.bert_name:
        tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased') 
    else:
        tokenizer = BertWordPieceTokenizer(
            "data/.bert-base-uncased-vocab.txt",
            lowercase=True)
    return tokenizer

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

def create_data(dataset, model):
    print(f"Creating new data for {model} model")
    tokenizer = get_tokenizer(model)

    fext = '_texts.txt' if dataset == 'Eurlex-4K' else '_raw_texts.txt'
    train_texts, test_texts = [], []
    with open(f'./data/{dataset}/train{fext}') as f:
        for point in tqdm.tqdm(f):
            text = tokenizer.encode(point.replace('\n', ''), add_special_tokens=False)
            train_texts.append(text)

    with open(f'./data/{dataset}/test{fext}') as f:
        for point in tqdm.tqdm(f):
            text = tokenizer.encode(point.replace('\n', ''), add_special_tokens=False)
            test_texts.append(text)

    os.makedirs(f'./data/{dataset}/{model}')

    with open(f'./data/{dataset}/{model}/train_encoded.pkl', 'wb') as f:
        pkl.dump(train_texts, f)

    with open(f'./data/{dataset}/{model}/test_encoded.pkl','wb') as f:
        pkl.dump(test_texts, f)

def load_data(dataset, model): 
    train_labels, test_labels = [], []
    train_texts, test_texts = [], []

    name_map = {'wiki31k': 'Wiki10-31K',
                'wiki500k': 'Wiki-500K',
                'amazoncat13k': 'AmazonCat-13K',
                'amazon670k': 'Amazon-670K',
                'eurlex4k': 'Eurlex-4K'}
    
    assert dataset in name_map
    dataset = name_map[dataset]
    
    # The following code ensures that data is not re-created when different bert/roberta/xlnet are used, since they use the same tokenizer. 
    if 'roberta' in model:
        model = 'roberta'
        if not os.path.exists(f'./data/{dataset}/roberta'):
            create_data(dataset, 'roberta')
    elif 'bert' in model:
        model = 'bert'
        if not os.path.exists(f'./data/{dataset}/bert'):
            create_data(dataset, 'bert')
    elif 'xlnet' in model:
        model = 'xlnet'
        if not os.path.exists(f'./data/{dataset}/xlnet'):
            create_data(dataset, 'xlnet')
    else:
        raise ValueError(f'Tokenizer for {model} not implemented. Add it src/data_utils.py and rerun')
    
    with open(f'./data/{dataset}/{model}/train_encoded.pkl', 'rb') as f:
        train_texts = pkl.load(f)

    with open(f'./data/{dataset}/{model}/test_encoded.pkl', 'rb') as f:
        test_texts = pkl.load(f)

    with open(f'./data/{dataset}/train_labels.txt') as f:
        for lab in tqdm.tqdm(f):
            train_labels.append(lab.replace('\n', '').split())

    with open(f'./data/{dataset}/test_labels.txt') as f:
        for lab in tqdm.tqdm(f):
            test_labels.append(lab.replace('\n', '').split())

    return train_texts, test_texts, train_labels, test_labels

def load_group(dataset, num_clusters):
    if dataset == 'wiki500k':
        return np.load(f'./data/Wiki-500K/label_group_{num_clusters}.npy', allow_pickle=True)
    if dataset == 'amazon670k':
        return np.load(f'./data/Amazon-670K/label_group_{num_clusters}.npy', allow_pickle=True)
    if dataset == 'AT670':
        return np.load(f'./data/AmazonTitles-670K/label_group_{num_clusters}.npy', allow_pickle=True)
    if dataset == 'WSAT':
        return np.load(f'./data/WikiSeeAlsoTitles-350K/label_group_{num_clusters}.npy', allow_pickle=True)  
    if dataset == 'WT500':
        return np.load(f'./data/WikiTitles-500K/label_group_{num_clusters}.npy', allow_pickle=True)            
