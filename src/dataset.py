import os
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizer
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from tree import build_tree as Tree
from xclib.data import data_utils as du
from xclib.utils.sparse import retain_topk
from xclib.utils.graph import normalize_graph
from random_walks import PrunedWalk
import scipy.sparse as sp

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

class InferenceDataset():
    'Not an actual dataset, just a container for clusters'
    def __init__(self, params):
        tree = Tree(b_factors=params.tree_depth, leaf_size=params.num_labels, force_shallow=True)
        self.build(tree, params)

        self.groups.append(np.arange(params.num_labels).reshape(-1, 1))
        self.num_clusters = [len(g) for g in self.groups]
        
        self.label_to_cluster_maps = []
        for i in range(len(self.groups)-2, -1, -1):
            self.label_to_cluster_maps.append(np.zeros(self.num_clusters[i + 1] , dtype=np.int64) - 1)
            for idx, labels in enumerate(self.groups[i]):
                self.label_to_cluster_maps[-1][[int(l) for l in labels]] = idx
            
            for j in range(i):
                self.groups[j] = [np.unique(self.label_to_cluster_maps[-1][l]) for l in self.groups[j]]
        
        self.groups.pop(-1)
    
    def build(self, tree, params, lbl_dense=None, word_embeds=None):
        print(f"Loading clusters from {params.cluster_name}")
        cluster_name = os.path.join(params.data_path, params.cluster_name)
        if not os.path.exists(cluster_name):
            raise ValueError('Clusters should be precomputed for inference')
        else:
            tree.load(cluster_name)
        
        if len(params.topk) == 2 and len(tree.b_factors) == 3:
            self.groups = [tree._get_cluster_depth(1), tree._get_cluster_depth(2)]
        else:
            self.groups = [tree._get_cluster_depth(d) for d, b in enumerate(tree.b_factors)]    

class MultiXMLGeneral(Dataset):
    def __init__(self, x, Y, params, X_tfidf = None, mode='train'):
        super().__init__()
        assert mode in ["train", "test"]
        self.mode = mode
        self.x = x
        self.Y = Y
        
        if mode == 'train':
            self.tf_X = normalize(X_tfidf, norm='l2')
            print(f"Loading clusters from {params.cluster_name}")

        elif mode == 'test' and os.path.exists(os.path.join(params.data_path, 'filter_labels_test.txt')):
            print("Loading filter_labels_test.txt")
            filter_test = np.loadtxt(os.path.join(params.data_path, 'filter_labels_test.txt'))
            self.filter_test = torch.from_numpy(filter_test.astype(np.int64)) 
            
        self.n_labels = params.num_labels
        # self.tokenizer = get_tokenizer(params.bert)
        self.max_len = params.max_len

        self.label_space = torch.zeros(self.n_labels)
        self.cls_token_id = [101]  # [self.tokenizer.cls_token_id]
        self.sep_token_id = [102]  # [self.tokenizer.sep_token_id]

        self.label_graph = self.load_graph(params)
        self.tree = Tree(b_factors=params.tree_depth, leaf_size=params.num_labels, force_shallow=True)
        self.build(params)
        self.groups.append(np.arange(self.n_labels).reshape(-1, 1))
        
        if self.groups is not None:
            self.num_clusters = [len(g) for g in self.groups]
            
            self.label_to_cluster_maps = []
            for i in range(len(self.groups)-2, -1, -1):
                self.label_to_cluster_maps.append(np.zeros(self.num_clusters[i + 1] , dtype=np.int64) - 1)
                for idx, labels in enumerate(self.groups[i]):
                    self.label_to_cluster_maps[-1][[int(l) for l in labels]] = idx
                
                for j in range(i):
                    self.groups[j] = [np.unique(self.label_to_cluster_maps[-1][l]) for l in self.groups[j]]
            
            self.groups.pop(-1)
            
    def __len__(self):
        return len(self.x)

    def load_graph(self, params, word_embeds=None):
        
        if not os.path.exists(os.path.join(
                params.data_path, params.graph_name)):
            
            trn_y = self.Y
            n_lbs = self.Y.shape[1]
            diag = np.ones(n_lbs, dtype=np.int64)

            if params.verbose_lbs > 0:
                verbose_labels = np.where(
                    np.ravel(trn_y.sum(axis=0) > params.verbose_lbs))[0]
                print("Verbose_labels:", verbose_labels.size)
                diag[verbose_labels] = 0
            else:
                verbose_labels = np.asarray([])
            diag = sp.diags(diag, shape=(n_lbs, n_lbs))
            print("Avg: labels", trn_y.nnz/trn_y.shape[0])
            trn_y = trn_y.dot(diag).tocsr()
            trn_y.eliminate_zeros()
            yf = None
            if word_embeds is not None:
                print("Using label features for PrunedWalk")
                emb = word_embeds.detach().cpu().numpy()
                yf = normalize(self.Yf.dot(emb)[:-1])
            graph = PrunedWalk(trn_y, yf=yf).simulate(
                params.walk_len, params.p_reset,
                params.top_k, max_dist=params.prune_max_dist)
            if verbose_labels.size > 0:
                graph = graph.tolil()
                graph[verbose_labels, verbose_labels] = 1
                graph = graph.tocsr()
            sp.save_npz(os.path.join(
                params.data_path, params.graph_name), graph)
        else:
            graph = sp.load_npz(os.path.join(
                params.data_path, params.graph_name))
        return graph

    def build(self, params, lbl_dense=None, word_embeds=None):
        cluster_name = os.path.join(params.data_path, params.cluster_name)
        if not os.path.exists(cluster_name):
            freq_y = np.ravel(self.Y.sum(axis=0))
            verb_lbs, norm_lbs = np.asarray([]), np.arange(freq_y.size)
            if params.verbose_lbs > 0:
                verb_lbs = np.where(freq_y > params.verbose_lbs)[0]
                norm_lbs = np.where(freq_y <= params.verbose_lbs)[0]

            lbl_sparse = self.create_label_fts_vec()

            print("Augmenting graphs")
            self.label_graph = self.load_graph(params, word_embeds)
            n_gph = normalize_graph(self.label_graph) # catching exceptions here
            
            print("Using Sparse Features")
            print("Avg features", lbl_sparse.nnz / lbl_sparse.shape[0])
            lbl_sparse = n_gph.dot(lbl_sparse).tocsr()
            lbl_sparse = retain_topk(lbl_sparse.tocsr(), k=1000).tocsr() #changed: original:1000
            print("Avg features", lbl_sparse.nnz / lbl_sparse.shape[0])

            if lbl_dense is not None:
                print("Using Dense Features")
                lbl_dense = n_gph.dot(normalize(lbl_dense))
                lbl_sparse = [lbl_sparse, lbl_dense]
                
            print('start fitting the tree...')
                
            self.tree.fit(norm_lbs, verb_lbs, lbl_sparse)
            self.tree.save(cluster_name)
        else:
            self.tree.load(cluster_name)
        
        if len(params.topk) == 2 and len(self.tree.b_factors) == 3:
            self.groups = [self.tree._get_cluster_depth(1), self.tree._get_cluster_depth(2)]
        else:
            self.groups = [self.tree._get_cluster_depth(d) for d, b in enumerate(self.tree.b_factors)]
        

    def create_label_fts_vec(self):
        _labels = self.Y.T
        _features = self.tf_X
        lbl_sparse = _labels.dot(_features).tocsr()
        lbl_sparse = retain_topk(lbl_sparse, k=1000)
        return lbl_sparse   
    
    def __getitem__(self, idx):
        cluster_ids = [torch.LongTensor(self.Y[idx].indices)]
        for map in self.label_to_cluster_maps:
            cluster_ids.append(torch.LongTensor(np.unique(map[cluster_ids[-1]])))
            assert cluster_ids[-1][0] != -1
        
        cluster_ids = cluster_ids[::-1]

        input_ids = self.x[idx]
        
        if len(input_ids) > self.max_len-2:
            input_ids = input_ids[:self.max_len-2]
        input_ids = self.cls_token_id + input_ids + self.sep_token_id

        padding_length = self.max_len - len(input_ids)
        attention_mask = torch.tensor([1] * len(input_ids) + [0] * padding_length)
        input_ids = torch.tensor(input_ids + ([0] * padding_length))

        return input_ids, attention_mask, *cluster_ids
