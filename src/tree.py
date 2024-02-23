import numpy as np
import torch
import copy
import time
import scipy.sparse as sp
from sklearn.preprocessing import normalize as scale
from functools import partial
import operator
import functools
import _pickle as pik
import tqdm
from multiprocessing import Pool, cpu_count


# def _normalize(X, norm='l2'):
#     print('X:',X)
#     X = scale(X, norm='l2')
#     return X

def _normalize(X, norm='l2'):
    if isinstance(X, np.matrix):
        X = np.asarray(X)
    X = scale(X, norm='l2')
    return X


def b_kmeans_dense_multi(fts_lbl, index, metric='cosine', tol=1e-4, leakage=None):
    lbl_cent = _normalize(np.squeeze(fts_lbl[:, 0, :]))
    lbl_fts = _normalize(np.squeeze(fts_lbl[:, 1, :]))
    if lbl_cent.shape[0] == 1:
        return [index]
    cluster = np.random.randint(low=0, high=lbl_cent.shape[0], size=(2))
    while cluster[0] == cluster[1]:
        cluster = np.random.randint(low=0, high=lbl_cent.shape[0], size=(2))
    _centeroids = lbl_cent[cluster]
    _sim = np.dot(lbl_cent, _centeroids.T)
    old_sim, new_sim = -1000000, -2
    while new_sim - old_sim >= tol:
        c_lbs = np.array_split(np.argsort(_sim[:, 1]-_sim[:, 0]), 2)
        _centeroids = _normalize(np.vstack([
            np.mean(lbl_cent[x, :], axis=0) for x in c_lbs
        ]))
        _sim_1 = np.dot(lbl_cent, _centeroids.T)
        _centeroids = _normalize(np.vstack([
            np.mean(lbl_fts[x, :], axis=0) for x in c_lbs
        ]))
        _sim_2 = np.dot(lbl_fts, _centeroids.T)
        _sim = _sim_1 + _sim_2
        old_sim, new_sim = new_sim, np.sum([np.sum(_sim[c_lbs[0], 0]),
                                            np.sum(_sim[c_lbs[1], 1])])
    return list(map(lambda x: index[x], c_lbs))


def b_kmeans_dense(labels_features, index, metric='cosine', tol=1e-4, leakage=None):
    labels_features = _normalize(labels_features)
    if labels_features.shape[0] == 1:
        return [index]
    cluster = np.random.randint(low=0, high=labels_features.shape[0], size=(2))
    while cluster[0] == cluster[1]:
        cluster = np.random.randint(low=0, high=labels_features.shape[0], size=(2))
    
    _centeroids = labels_features[cluster]
    _similarity = np.dot(labels_features, _centeroids.T)
    old_sim, new_sim = -1000000, -2
    while new_sim - old_sim >= tol:
        clustered_lbs = np.array_split(np.argsort(_similarity[:, 1]-_similarity[:, 0]), 2)

        _centeroids = _normalize(np.vstack([
            np.mean(labels_features[x, :], axis=0) for x in clustered_lbs
        ]))
        _similarity = np.dot(labels_features, _centeroids.T)
        old_sim, new_sim = new_sim, np.sum(
            [np.sum(
                _similarity[indx, i]
            ) for i, indx in enumerate(clustered_lbs)])

    return list(map(lambda x: index[x], clustered_lbs))

def b_kmeans_sparse_dense(lf_sparse, lf_dense, index, metric='cosine', tol=1e-4, leakage=None):
    lf_sparse = _normalize(lf_sparse)
    lf_dense = _normalize(lf_dense)
    if lf_sparse.shape[0] == 1:
        return [index]
    cluster = np.random.randint(low=0, high=lf_sparse.shape[0], size=(2))

    while cluster[0] == cluster[1]:
        cluster = np.random.randint(low=0, high=lf_sparse.shape[0], size=(2))

    cent_sparse = _normalize(lf_sparse[cluster].todense())
    sim_sparse = _sdist(lf_sparse, cent_sparse)

    cent_dense = lf_dense[cluster]
    sim_dense = np.dot(lf_dense, cent_dense.T)

    _similarity = sim_sparse + sim_dense

    old_sim, new_sim = -1000000, -2
    while new_sim - old_sim >= tol:
        c_lbs = np.array_split(np.argsort(_similarity[:, 1]-_similarity[:, 0]), 2)

        cent_sparse = _normalize(np.vstack([lf_sparse[x, :].mean(axis=0) for x in c_lbs]))
        sim_sparse = _sdist(lf_sparse, cent_sparse)

        cent_dense = _normalize(np.vstack([np.mean(lf_dense[x, :], axis=0) for x in c_lbs]))
        sim_dense = np.dot(lf_dense, cent_dense.T)

        _similarity = sim_sparse + sim_dense

        old_sim, new_sim = new_sim, np.sum([np.sum(_similarity[c_lbs[0], 0]), np.sum(_similarity[c_lbs[1], 1])])

    return list(map(lambda x: index[x], c_lbs))


def b_kmeans_sparse(labels_features, index, metric='cosine', tol=1e-4, leakage=None):
    labels_features = _normalize(labels_features)
    if labels_features.shape[0] == 1:
        return [index]
    cluster = np.random.randint(low=0, high=labels_features.shape[0], size=(2))
    while cluster[0] == cluster[1]:
        cluster = np.random.randint(
            low=0, high=labels_features.shape[0], size=(2))
    _centeroids = _normalize(np.asarray(labels_features[cluster].todense())) #changed
    _sim = _sdist(labels_features, _centeroids)
    old_sim, new_sim = -1000000, -2
    while new_sim - old_sim >= tol:
        c_lbs = np.array_split(np.argsort(_sim[:, 1]-_sim[:, 0]), 2)
        _centeroids = _normalize(np.vstack([
            labels_features[x, :].mean(axis=0) for x in c_lbs]))
        _sim = _sdist(labels_features, _centeroids)
        old_sim, new_sim = new_sim, np.sum([
            np.sum(_sim[c_lbs[0], 0]), np.sum(_sim[c_lbs[1], 1])])
    return list(map(lambda x: index[x], c_lbs))


def _sdist(XA, XB, norm=None):
    return XA.dot(XB.transpose())


def _merge_tree(cluster, verbose_label_index, first_split = -1, avg_size=0, force=False):
    if cluster[0].size < verbose_label_index[0].size:
        print("Merging trees", np.log2(len(cluster)))
        return cluster + verbose_label_index, [np.asarray([])]
    elif verbose_label_index[0].size > 0 and force:
        if verbose_label_index.shape[0] > 0:
            print("Force Merging trees")
            return cluster + verbose_label_index, [np.asarray([])]
        else:
            print("Nothing else to do")
            return cluster, [np.asarray([])]
    else:
        return cluster, verbose_label_index


def cluster_labels(labels, clusters, verbose_label_index, num_nodes, splitter, first_split = -1):
    start = time.time()
    clusters, verbose_label_index = _merge_tree(clusters, verbose_label_index, first_split)
    with Pool(cpu_count()-1) as p:
        while len(clusters) < num_nodes:
            if isinstance(labels, list):
                temp_cluster_list = functools.reduce(
                    operator.iconcat,
                    p.starmap(splitter, map(lambda x: (labels[0][x], labels[1][x], x), clusters)), [])
            else:    
                temp_cluster_list = functools.reduce(
                    operator.iconcat,
                    p.starmap(splitter, map(lambda x: (labels[x], x), clusters)), [])

            end = time.time()
            print("Total clusters {}".format(len(temp_cluster_list)),
                  "Avg. Cluster size {}".format(
                      np.mean(list(map(len, temp_cluster_list+verbose_label_index)))),
                  "Total time {} sec".format(end-start))
            
            clusters = temp_cluster_list
            clusters, verbose_label_index = _merge_tree(clusters, verbose_label_index, first_split)
            del temp_cluster_list
    return clusters, verbose_label_index


def representative(lbl_fts):
    scores = np.ravel(np.sum(np.dot(lbl_fts, lbl_fts.T), axis=1))
    return lbl_fts[np.argmax(scores)]


class hash_map_index:
    def __init__(self, clusters, label_to_idx, total_elements, total_valid_nodes, padding_idx=None):
        self.clusters = clusters
        self.padding_idx = padding_idx
        self.total_elements = total_elements
        self.size = total_valid_nodes
        self.weights = None
        if padding_idx is not None:
            self.weights = np.zeros((self.total_elements), np.float)
            self.weights[label_to_idx == padding_idx] = -np.inf

        self.hash_map = label_to_idx

    def _get_hash(self):
        return self.hash_map

    def _get_weights(self):
        return self.weights


class build_tree:
    def __init__(self, b_factors=[2], M=1, leaf_size=0, force_shallow=True):
        self.b_factors = b_factors
        self.C = []
        self.leaf_size = leaf_size
        self.force_shallow = force_shallow
        self.height = 2

    def fit(self, label_index=[], verbose_label_index=[], lbl_repr=None):
        clusters = [label_index]
        self.hash_map_array = []
        print("Total verbose labels", verbose_label_index.size)

        # if len(lbl_repr.shape) > 2:
        #     print("Using multi objective kmeans++")
        #     b_kmeans = b_kmeans_dense_multi

        if isinstance(lbl_repr, list):
            print("Using sparse & dense kmeans++")
            b_kmeans = b_kmeans_sparse_dense
            self.num_labels = lbl_repr[0].shape[0]

        elif isinstance(lbl_repr, np.ndarray):
            print("Using dense kmeans++")
            b_kmeans = b_kmeans_dense
            self.num_labels = lbl_repr.shape[0]

        else:
            print("Using sparse kmeans++")
            lbl_repr = lbl_repr.tocsr()
            b_kmeans = b_kmeans_sparse
            self.num_labels = lbl_repr.shape[0]

        self._parabel(lbl_repr, clusters, [verbose_label_index],
                      b_kmeans, self.force_shallow)

    def _parabel(self, labels, clusters, verbose_label_index,
                 splitter=None, force_shallow=True):
        depth = 0
        T_verb_lbl = verbose_label_index[0].size
        while True:
            orignal_num_nodes = 2**self.b_factors[depth]
            n_child_nodes = orignal_num_nodes
            if self.num_labels/n_child_nodes < T_verb_lbl or len(self.b_factors) == 1:
                if T_verb_lbl > 0:
                    add_at = np.floor(np.log2(self.num_labels/T_verb_lbl))+1
                    addition = 2**(self.b_factors[depth]-add_at)
                    n_child_nodes += addition
            depth += 1
            print("Building tree at height %d with nodes: %d" %
                  (depth, n_child_nodes))
            if n_child_nodes >= self.num_labels:
                print("No need to do clustering")
                clusters = list(np.arange(self.num_labels).reshape(-1, 1))
            else:
                clusters, verbose_label_index = cluster_labels(
                    labels, clusters, verbose_label_index,
                    orignal_num_nodes, splitter)#, self.b_factors[0])
                if depth == len(self.b_factors):
                    clusters, verbose_label_index = _merge_tree(
                        clusters, verbose_label_index)#, self.b_factors[0])
            self.hash_map_array.append(
                hash_map_index(
                    clusters,
                    np.arange(n_child_nodes),
                    n_child_nodes,
                    n_child_nodes
                )
            )
            self.C.append(max(list(map(lambda x: x.size, clusters))))
            if depth == len(self.b_factors):
                print("Preparing Leaf")
                break

    def _get_cluster_depth(self, depth):
        return self.hash_map_array[depth].clusters

    def load(self, fname):
        self.__dict__ = pik.load(open(fname, 'rb'))

    def save(self, fname):
        pik.dump(self.__dict__, open(fname, 'wb'))