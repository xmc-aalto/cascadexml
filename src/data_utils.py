import os
import torch
import pickle as pkl
import numpy as np
from torch.utils.data import Dataset
import scipy.sparse as sp
from tqdm import tqdm
from transformers import BertTokenizer
import re
from nltk.corpus import stopwords
from multiprocessing import Pool, cpu_count
from tqdm.contrib.concurrent import process_map

cachedStopWords = stopwords.words("english")

def clean_str(string):
    string = string.replace('\n', ' ')
    string = re.sub(r"_", " ", string)
    string = re.sub("\[\d+\]", "", string)
    string = re.sub(r"[^A-Za-z0-9!?\.\'\`]", " ", string)
    string = ' '.join([word for word in string.split() if word not in cachedStopWords])
    # string = re.sub('(?<=[A-Za-z]),', ' ', string)
    # string = re.sub(r"(),!?", " ", string)
    # string = re.sub(r"[^A-Za-z0-9!?\.\'\`]", " ", string)
    # string = re.sub('(?<=[A-Za-z])\.', '', string)
    # string = re.sub(r'([\d]+)([A-Za-z]+)', '\g<1> \g<2>', string)
    # string = re.sub(r"\'s ", " ", string)
    # string = re.sub(r"s\' ", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = string.strip().lower()
    return string


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

def get_inv_prop(dataset, Y):
    if os.path.exists(os.path.join(dataset, 'inv_prop.npy')):
        inv_prop = np.load(os.path.join(dataset, 'inv_prop.npy'))
        return inv_prop

    print("Creating inv_prop file")
    
    A = {'Amazon-670K': 0.6, 'Amazon-3M': 0.6, 'AmazonCat-13K': 0.6, 'Wiki-500K' : 0.5, 'Wiki10-31K' : 0.5}
    B = {'Amazon-670K': 2.6, 'Amazon-3M': 2.6, 'AmazonCat-13K': 2.6, 'Wiki-500K': 0.4, 'Wiki10-31K': 0.4}

    d = dataset.split('/')[-1]
    a, b = A[d], B[d]
    
    num_samples = Y.shape[0]
    inv_prop = np.array(Y.sum(axis=0)).ravel()
    
    c = (np.log(num_samples) - 1) * np.power(b+1, a)
    inv_prop = 1 + c * np.power(inv_prop + b, -a)
    
    np.save(os.path.join(dataset, 'inv_prop.npy'), inv_prop)
    return inv_prop

def make_csr_tfidf(dataset):
    file_name = f'{dataset}/tfidf.npz'
    if os.path.exists(file_name):
        tfidf_mat = sp.load_npz(file_name)
    else:
        with open(f'{dataset}/train.txt') as fil:
            row_idx, col_idx, val_idx = [], [], []
            for i, data in enumerate(fil.readlines()):
                data = data.split()[1:]
                for tfidf in data:
                    token, weight = tfidf.split(':')
                    row_idx.append(i)
                    col_idx.append(int(token))
                    val_idx.append(float(weight))
            m = max(row_idx) + 1
            n = max(col_idx) + 1
            tfidf_mat = sp.csr_matrix((val_idx, (row_idx, col_idx)), shape=(m, n))
            sp.save_npz(file_name, tfidf_mat)
    return tfidf_mat

def make_csr_labels(num_labels, file_name):
    if os.path.exists(file_name):
        Y = sp.load_npz(file_name)
    else:
        with open(os.path.splitext(file_name)[0]+'.txt') as fil:
            row_idx, col_idx, val_idx = [], [], []
            for i, lab in enumerate(fil.readlines()):
                l_list = lab.replace('\n', '').split()
                for y in l_list:
                    row_idx.append(i)
                    col_idx.append(int(y))
                    val_idx.append(1)
            m = max(row_idx) + 1
            n = num_labels
            Y = sp.csr_matrix((val_idx, (row_idx, col_idx)), shape=(m, n))
            sp.save_npz(file_name, Y)
    return Y

def encode(text):
    return sp_token.encode(text, add_special_tokens=False)

# tokenizer = get_tokenizer(model)
def create_data(dataset, model):
    print(f"Creating new data for {model} model")
    tokenizer = get_tokenizer(model)
    global sp_token 
    sp_token = tokenizer

    fext = '_texts.txt' if dataset == 'Eurlex-4K' else '_raw_texts.txt'
    train_texts, test_texts = [], []
    with open(f'{dataset}/train{fext}') as f:
        for point in tqdm(f.readlines()):
            train_texts.append(point.replace('\n', ''))
            # text = tokenizer.encode(point.replace('\n', ''), add_special_tokens=False)
            # text = tokenizer.encode(clean_str(point), add_special_tokens=False)
            # train_texts.append(text)

    with open(f'{dataset}/test{fext}') as f:
        for point in tqdm(f.readlines()):
            test_texts.append(point.replace('\n', ''))
            # text = tokenizer.encode(point.replace('\n', ''), add_special_tokens=False)
            # text = tokenizer.encode(clean_str(point), add_special_tokens=False)
            # test_texts.append(text)
    
    with Pool(cpu_count() - 1) as p:
        encoded_train = process_map(encode, train_texts, max_workers=cpu_count()-1)

    with Pool(cpu_count() - 1) as p:
        encoded_test = process_map(encode, test_texts, max_workers=cpu_count()-1)

    os.makedirs(f'{dataset}/{model}')

    with open(f'{dataset}/{model}/train_encoded.pkl', 'wb') as f:
        pkl.dump(encoded_train, f)

    with open(f'{dataset}/{model}/test_encoded.pkl','wb') as f:
        pkl.dump(encoded_test, f)

def load_data(dataset, model, num_labels, load_precomputed): 
    train_labels, test_labels = [], []
    train_texts, test_texts = [], []

    name_map = {'wiki31k': 'Wiki10-31K',
                'wiki500k': 'Wiki-500K',
                'amazoncat13k': 'AmazonCat-13K',
                'amazon670k': 'Amazon-670K',
                'eurlex4k': 'Eurlex-4K',
                'AT670': 'AmazonTitles-670K',
                'WT500': 'WikiTitles-500K',
                'WSAT350': 'WikiSeeAlsoTitles-350K',
                }
    
    # assert dataset in name_map
    # dataset = name_map[dataset]
    
    # The following code ensures that data is not re-created when different bert/roberta/xlnet are used, since they use the same tokenizer. 
    if load_precomputed:
        train_texts = np.load('./bert_train_768.npy')
        test_texts = np.load('./bert_test_768.npy')
    else:
        assert any([x in model for x in ['roberta', 'bert', 'xlnet']]), f'Tokenizer for {model} not implemented. Add it src/data_utils.py and rerun'

        if not os.path.exists(f'{dataset}/{model}'):
            create_data(dataset, model)
        
        with open(f'{dataset}/{model}/train_encoded.pkl', 'rb') as f:
            train_texts = pkl.load(f)
        
        # print("truncating train texts")
        # for i, text in enumerate(train_texts):
        #     train_texts[i] = text[:1024]

        # with open(f'{dataset}/{model}/train_encoded_short.pkl', 'wb') as f:
        #     pkl.dump(train_texts, f)

        with open(f'{dataset}/{model}/test_encoded.pkl', 'rb') as f:
            test_texts = pkl.load(f)
        
        # print("truncating test texts")
        # for i, text in enumerate(test_texts):
        #     test_texts[i] = text[:1024]

        # with open(f'{dataset}/{model}/test_encoded_short.pkl','wb') as f:
        #     pkl.dump(test_texts, f)

    train_labels = make_csr_labels(num_labels, f'{dataset}/Y.trn.npz')
    test_labels = make_csr_labels(num_labels, f'{dataset}/Y.tst.npz')
    tfidf = make_csr_tfidf(dataset)
    inv_prop = get_inv_prop(dataset, train_labels)

    return train_texts, test_texts, train_labels, test_labels, tfidf, inv_prop

def load_group(dataset, num_clusters):
    if dataset == 'wiki500k':
        return np.load(f'Wiki-500K/label_group_{num_clusters}.npy', allow_pickle=True)
    if dataset == 'Amazon-670K':
        return np.load(f'Amazon-670K/label_group_{num_clusters}.npy', allow_pickle=True) 
        # return np.load(f'Amazon-670K/label_group_tree-Level-1.npy', allow_pickle=True)
    if dataset == 'AT670':
        # return np.load(f'AmazonTitles-670K/label_group_{num_clusters}.npy', allow_pickle=True)
        return np.load(f'AmazonTitles-670K/label_group8192.npy', allow_pickle=True)
    if dataset == 'WSAT':
        return np.load(f'WikiSeeAlsoTitles-350K/label_group_{num_clusters}.npy', allow_pickle=True)  
    if dataset == 'WT500':
        return np.load(f'WikiTitles-500K/label_group_{num_clusters}.npy', allow_pickle=True)            

def load_cluster_tree(dataset, levels=2):
    if dataset == 'Amazon-670K':
        return [np.load(f'./data/Amazon-670K/label_group_tree-Level-{i}.npy', allow_pickle=True) for i in range(levels)]
        # return [np.load(f'Amazon-670K/label_group_stable-Level-{i}.npy', allow_pickle=True) for i in [1, 2]]
