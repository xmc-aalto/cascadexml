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
from make_r import make_r

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
    def __init__(self, x, Y, num_labels, max_length, groups=None, 
                    model_name = 'bert-base', mode='train'):
        super().__init__()
        assert mode in ["train", "test"]
        self.mode = mode
        self.x = x
        self.Y = Y
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
        labels = torch.LongTensor(self.Y[idx].indices)
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
        
        cluster_ids_2 = torch.LongTensor(np.unique(self.label_to_cluster_ids_2[cluster_ids_1]))
        if cluster_ids_2[0] == -1:
            cluster_ids_2 = cluster_ids_2[1:]

        return input_ids, attention_mask, cluster_ids_2, cluster_ids_1, labels

class MultiXMLGeneral(Dataset):
    def __init__(self, x, Y, params, X_tfidf = None, mode='train'):
        super().__init__()
        assert mode in ["train", "test"]
        self.mode = mode
        self.x = x
        self.train_W = params.train_W
        self.Y = Y
        if mode == 'train':
            self.tf_X = normalize(X_tfidf, norm='l2')
        self.n_labels = params.num_labels
        # self.tokenizer = get_tokenizer(params.bert)
        self.max_len = params.max_len

        self.label_space = torch.zeros(self.n_labels)
        self.cls_token_id = [101]  # [self.tokenizer.cls_token_id]
        self.sep_token_id = [102]  # [self.tokenizer.sep_token_id]

        self.label_graph = self.load_graph(params)
        self.tree = Tree(b_factors=params.b_factors, method=params.cluster_method, 
                            leaf_size=params.num_labels, force_shallow=True)
        self.build(params)
        if self.mode == 'train' and params.use_r:
            self.rs = make_r(self.groups, Y)
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
        print(os.path.join(params.data_path, params.graph_name))
        if not os.path.exists(os.path.join(
                params.data_path, params.graph_name)):
            
            trn_y = self.Y
            n_lbs = self.Y.shape[1]
            diag = np.ones(n_lbs, dtype=np.int)

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
        print(f"Loading clusters from {params.cluster_name}")
        cluster_name = os.path.join(params.data_path, params.cluster_name)
        if not os.path.exists(cluster_name):
            freq_y = np.ravel(self.Y.sum(axis=0))
            
            # _doc_repr = np.load('./doc_features_768C.npy')
            # _doc_repr = normalize(_doc_repr)
            # lbl_dense = normalize(self.Y.T.dot(_doc_repr))
            
            verb_lbs, norm_lbs = np.asarray([]), np.arange(freq_y.size)
            if params.verbose_lbs > 0:
                verb_lbs = np.where(freq_y > params.verbose_lbs)[0]
                norm_lbs = np.where(freq_y <= params.verbose_lbs)[0]

            lbl_sparse = self.create_label_fts_vec()

            if params.cluster_method == 'AugParabel':
                print("Augmenting graphs")
                self.label_graph = self.load_graph(params, word_embeds)
                n_gph = normalize_graph(self.label_graph)
                
                print("Using Sparse Features")
                print("Avg features", lbl_sparse.nnz / lbl_sparse.shape[0])
                lbl_sparse = n_gph.dot(lbl_sparse).tocsr()
                lbl_sparse = retain_topk(lbl_sparse.tocsr(), k=1000).tocsr()
                print("Avg features", lbl_sparse.nnz / lbl_sparse.shape[0])

                if lbl_dense is not None:
                    print("Using Dense Features")
                    lbl_dense = n_gph.dot(normalize(lbl_dense))
                    lbl_sparse = [lbl_sparse, lbl_dense]
                
            self.tree.fit(norm_lbs, verb_lbs, lbl_sparse)
            self.tree.save(cluster_name)
            # exit()
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
            # if cluster_ids[-1][0] == -1:
            #     cluster_ids[-1] = cluster_ids[-1][1:]
        if hasattr(self, 'rs'):
            rs = []
            for r in self.rs:
                rs.append((r[idx].data.astype(np.float32), r[idx].indices.astype(np.int64)))
            cluster_ids = cluster_ids[::-1] + rs
        else:
            cluster_ids = cluster_ids[::-1]

        if self.train_W:
            return torch.FloatTensor(self.x[idx]), torch.ones(128), *cluster_ids[::-1]

        input_ids = self.x[idx]

        if len(input_ids) > self.max_len-2:
            input_ids = input_ids[:self.max_len-2]
        
        input_ids = self.cls_token_id + input_ids + self.sep_token_id
        padding_length = self.max_len - len(input_ids)
        attention_mask = torch.tensor([1] * len(input_ids) + [0] * padding_length)
        input_ids = torch.tensor(input_ids + ([0] * padding_length))

        return input_ids, attention_mask, *cluster_ids


class PoolableMultiXMLGeneral(MultiXMLGeneral):
    def __init__(self, x, Y, params, X_tfidf = None, mode='train', bert_tfidf=None):
        super().__init__(x, Y, params, X_tfidf, mode)
        if mode=='train':
            self.bert_tfidf = sp.load_npz('data/Wiki10-31K/bert-base/bert_unnorm_tfidf_train.npz')
        else:
            self.bert_tfidf = sp.load_npz('data/Wiki10-31K/bert-base/bert_unnorm_tfidf_test.npz')
        print('using word pooling')
        # self.bert_tfidf[0] = 0
        # self.bert_tfidf[self.cls_token_id[0]] = 1/8
        # self.bert_tfidf[self.sep_token_id[0]] = 1/8

    def word_pool(self, words, tfidfs, bin_size=8):
        if len(words) > 8 * (self.max_len-2):
            words = words[:8*(self.max_len-2)]
        words = np.array(words)
        if self.max_len-2 <= words.shape[0]:
            chunks = np.array_split(np.arange(words.shape[0]), self.max_len-2)
            new_words = [words[c] for c in chunks]
            new_words = [np.concatenate([nw, np.zeros((bin_size - nw.shape[0]))]) for nw in new_words]

            new_tfidfs = [tfidfs[nw.astype(int)] for nw in new_words]

            new_words = [np.array(bin_size*self.cls_token_id)] + new_words + [np.array(bin_size*self.sep_token_id)]
            new_tfidfs = [np.array(bin_size*[1/bin_size])] + new_tfidfs + [np.array(bin_size*[1/bin_size])]

            attention_mask = torch.tensor([1] * self.max_len)

            return torch.from_numpy(np.stack([np.stack(new_words), np.stack(new_tfidfs)])), attention_mask
        else:
            new_words = [np.concatenate([nw.reshape(1), np.zeros((bin_size - 1))]) for nw in words]
            new_tfidfs = [tfidfs[nw.astype(int)] for nw in new_words]
            
            new_words = [np.array(bin_size*self.cls_token_id)] + new_words + [np.array(bin_size*self.sep_token_id)]
            new_tfidfs = [np.array(bin_size*[1/bin_size])] + new_tfidfs + [np.array(bin_size*[1/bin_size])]

            attention_mask = torch.tensor([1] * len(new_words) + [0] * (self.max_len - len(new_words)))
            new_words = new_words + [np.zeros(bin_size)] * (self.max_len - len(new_words))
            new_tfidfs = new_tfidfs + [np.zeros(bin_size)] * (self.max_len - len(new_tfidfs))

            return torch.from_numpy(np.stack([np.stack(new_words), np.stack(new_tfidfs)])), attention_mask

    
    def __getitem__(self, idx):
        cluster_ids = [torch.LongTensor(self.Y[idx].indices)]
        for map in self.label_to_cluster_maps:
            cluster_ids.append(torch.LongTensor(np.unique(map[cluster_ids[-1]])))
            assert cluster_ids[-1][0] != -1
            # if cluster_ids[-1][0] == -1:
            #     cluster_ids[-1] = cluster_ids[-1][1:]
        if hasattr(self, 'rs'):
            rs = []
            for r in self.rs:
                rs.append((r[idx].data.astype(np.float32), r[idx].indices.astype(np.int64)))
            cluster_ids = cluster_ids[::-1] + rs
        else:
            cluster_ids = cluster_ids[::-1]

        if self.train_W:
            return torch.FloatTensor(self.x[idx]), torch.ones(128), *cluster_ids[::-1]

        input_ids = self.x[idx]
        tfidfs = np.array(self.bert_tfidf[idx].todense())[0] #[input_ids]

        # if len(input_ids) > 8 * (self.max_len-2):
        #     input_ids = input_ids[:8*(self.max_len-2)]
        # if len(input_ids) % 8 != 0:
        #     input_ids = input_ids + [0] * (8  - len(input_ids)%8)
        
        # input_ids = self.cls_token_id * 8 + input_ids + self.sep_token_id * 8
        # padding_length = 8 * self.max_len - len(input_ids)
        # attention_mask = torch.tensor([1] * (len(input_ids)//8) + [0] * (padding_length//8))
        
        # input_ids = torch.tensor(input_ids + ([0] * padding_length))
        # input_ids = input_ids.reshape(-1, 8)
        # tfidfs = torch.from_numpy(tfidfs.reshape(-1, 8))

        # input_ids = torch.stack([input_ids, tfidfs], dim=0)  # 2, 256, 8

        input_ids, attention_mask = self.word_pool(input_ids, tfidfs, 8)

        return input_ids, attention_mask, *cluster_ids
        

class PecosDataset(Dataset):
    def __init__(self, x, y, num_labels, max_length, groups=None, 
                 cluster_levels=[12, 15], model_name = 'bert-base', mode='train', alpha=1e-2):
        super().__init__()
        assert mode in ["train", "test"]
        self.mode = mode
        self.x = x
        self.y = list(list(map(int, l)) for l in y)
        y_indices = [(i, int(j)) for i in range(len(y)) for j in y[i]]
        y_rows, y_cols = zip(*y_indices)
        sparse_y = csr_matrix((np.ones(len(y_rows)), (y_rows, y_cols)), shape=[len(y), num_labels])
        self.n_labels = num_labels
        # self.tokenizer = get_tokenizer(model_name)
        self.max_len = max_length
        # self.label_to_cluster_ids = None
        self.label_space = torch.zeros(self.n_labels)
        self.cls_token_id = [101] #[self.tokenizer.cls_token_id]
        self.sep_token_id = [102] #[self.tokenizer.sep_token_id]

        # self.groups = groups.copy()  # list
        self.num_clusters = []
        # self.groups.append(np.arange(self.n_labels).reshape(-1, 1))
        # import pdb; pdb.set_trace()

        if groups is not None:
            self.label_to_cluster_maps = []
            self.rs = []
            self.alpha=alpha
            self.groups = []
            self.cluster_levels = cluster_levels

            assert len(groups)-1 >= self.cluster_levels[-1]
            group_mat = None
            r_mat = sparse_y
            for idx in range(len(groups)-1, -1 ,-1):
                if group_mat is None:
                    group_mat = groups[idx].T
                else:
                    group_mat = groups[idx].T @ group_mat
                r_mat = r_mat @ groups[idx]
                
                if group_mat.shape[0] == 2**self.cluster_levels[-1]:
                    group_mat_tlil = group_mat.copy().T.tolil().rows
                    self.label_to_cluster_maps.append(np.array([g[0] for g in group_mat_tlil]))
                    group_mat_lil = group_mat.copy().tolil().rows
                    self.groups.append([list(g) for g in group_mat_lil])
                    r_mat_curr = r_mat.copy()
                    r_mat_curr = normalize(r_mat_curr, norm='l1', axis=1)
                    self.rs.append(r_mat_curr)
                    self.num_clusters.append(r_mat_curr.shape[-1])
                    self.cluster_levels.pop(-1)
                    if len(self.cluster_levels) == 0:
                        break
                    group_mat = None
                # import pdb; pdb.set_trace()
            # self.label_to_cluster_maps = self.label_to_cluster_maps
            # self.rs = self.rs
            self.groups = self.groups[::-1]
            # import pdb; pdb.set_trace()
            
    def __len__(self):
        return len(self.x)

    # def __getitem__(self, idx):
    #     # import pdb; pdb.set_trace()
    #     cluster_ids, rs = [torch.LongTensor(self.y[idx])], [None]
    #     input_ids = self.x[idx]

    #     if len(input_ids) > self.max_len-2:
    #         input_ids = input_ids[:self.max_len-2]
        
    #     input_ids = self.cls_token_id + input_ids + self.sep_token_id
    #     padding_length = self.max_len - len(input_ids)
    #     attention_mask = torch.tensor([1] * len(input_ids) + [0] * padding_length)
    #     input_ids = torch.tensor(input_ids + ([0] * padding_length))

    #     for r, lmap in zip(self.rs, self.label_to_cluster_maps):
    #         cluster_ids.append(torch.LongTensor(np.unique(lmap[cluster_ids[-1]])))
    #         if cluster_ids[-1][0] == -1:
    #             cluster_ids[-1] == cluster_ids[-1][1:]
    #         rs.append((r[idx].data.astype(np.float32), r[idx].indices.astype(np.int64)))
        
    #     # binarize first cluster        
    #     cluster_ids[-1] = torch.zeros(self.num_clusters[0]).scatter(0, cluster_ids[-1], 1.0)
    #     # rs[-1] = torch.zeros(self.num_clusters[0]).scatter(0, torch.from_numpy(rs[-1][1]), torch.from_numpy(rs[-1][0]))
    #     if self.mode == 'train':
    #         return input_ids, attention_mask, *zip(cluster_ids[::-1], rs[::-1])
    #     else:
    #         return input_ids, attention_mask, *cluster_ids[::-1]

    def __getitem__(self, idx):
        cluster_ids = [torch.LongTensor(self.y[idx])]
        input_ids = self.x[idx]

        if len(input_ids) > self.max_len-2:
            input_ids = input_ids[:self.max_len-2]
        
        input_ids = self.cls_token_id + input_ids + self.sep_token_id
        padding_length = self.max_len - len(input_ids)
        attention_mask = torch.tensor([1] * len(input_ids) + [0] * padding_length)
        input_ids = torch.tensor(input_ids + ([0] * padding_length))

        for lmap in self.label_to_cluster_maps:
            cluster_ids.append(torch.LongTensor(np.unique(lmap[cluster_ids[-1]])))
            if cluster_ids[-1][0] == -1:
                cluster_ids[-1] == cluster_ids[-1][1:]

        return input_ids, attention_mask, *cluster_ids[::-1]
