# cluster from AttentionXML
import os
import tqdm
import joblib
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from sklearn.preprocessing import normalize
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import MultiLabelBinarizer

def get_sparse_feature(feature_file, label_file):
    sparse_x, _ = load_svmlight_file(feature_file, multilabel=True)
    sparse_labels = [i.replace('\n', '').split() for i in open(label_file)]
    sparse_labels = list(list(map(int, l)) for l in sparse_labels)
    return normalize(sparse_x), np.array(sparse_labels)

# def build_tree_by_level(sparse_data_x, sparse_data_y, eps: float, max_leaf: int, levels: list, groups_path):
#     print('Clustering')
#     sparse_x, sparse_labels = get_sparse_feature(sparse_data_x, sparse_data_y)
#     mlb = MultiLabelBinarizer(sparse_output=True)
#     sparse_y = mlb.fit_transform(sparse_labels)
#     joblib.dump(mlb, groups_path+'mlb')
#     print('Getting Labels Feature')
#     labels_f = normalize(sparse_y.T @ csc_matrix(sparse_x))
#     print(F'Start Clustering {levels}')
#     levels, q = [2**x for x in levels], None
#     if levels[-1] > sparse_y.shape[0]:
#         levels[-1] = sparse_y.shape[0]  # clamp max level's count to num labels
#     for i in range(len(levels)-1, -1, -1):
#         if os.path.exists(F'{groups_path}-Level-{i}.npy'):
#             print(F'{groups_path}-Level-{i}.npy')
#             labels_list = np.load(F'{groups_path}-Level-{i}.npy', allow_pickle=True)
#             q = [(labels_i, labels_f[labels_i]) for labels_i in labels_list]
#             break
#     if q is None:
#         q = [(np.arange(labels_f.shape[0]), labels_f)]
#     while q:
#         labels_list = np.asarray([x[0] for x in q])
#         assert sum(len(labels) for labels in labels_list) == labels_f.shape[0]
#         if len(labels_list) in levels:
#             level = levels.index(len(labels_list))
#             print(F'Finish Clustering Level-{level}')
#             np.save(F'{groups_path}-Level-{level}.npy', np.asarray(labels_list))
#         else:
#             print(F'Finish Clustering {len(labels_list)}')
#         next_q = []
#         max_size = max([len(node_i) for node_i, _ in q])
#         print(f'Maxinum size of node is {max_size}')
#         for node_i, node_f in q:
#             if len(node_i) > max_leaf:
#                 next_q += list(split_node(node_i, node_f, eps))
#             else:
#                 np.save(F'{groups_path}-last.npy', np.asarray(labels_list))
#         q = next_q
#     print('Finish Clustering')
#     return mlb

def make_sparse_mat(labels):
    row_idx, col_idx, val_idx = [], [], []
    for i, l_list in enumerate(labels):
        for y in l_list:
            row_idx.append(i)
            col_idx.append(y)
            val_idx.append(1)
    m = max(row_idx) + 1
    n = max(col_idx) + 1
    Y = csr_matrix((val_idx, (row_idx, col_idx)), shape=(m, n))
    return Y

def build_tree_by_level(sparse_data_x, sparse_data_y, eps: float, max_leaf: int, levels: list, groups_path):
    os.makedirs(os.path.split(groups_path)[0], exist_ok=True)
    print('Clustering')
    sparse_x, sparse_labels = get_sparse_feature(sparse_data_x, sparse_data_y)
    # mlb = MultiLabelBinarizer(sparse_output=True)
    # sparse_y = mlb.fit_transform(sparse_labels)
    sparse_y = make_sparse_mat(sparse_labels)
    print('Getting Labels Feature')
    # labels_f = normalize(csr_matrix(sparse_y.T) @ csc_matrix(sparse_x))
    labels_f = normalize(sparse_y.T @ csc_matrix(sparse_x))
    print(F'Start Clustering {levels}')
    levels, q = [2**x for x in levels], None
    if levels[-1] > sparse_y.shape[1] or levels[-1] == 0.5:
        levels[-1] = sparse_y.shape[1]  # clamp max level's count to num labels

    for i in range(len(levels)-1, -1, -1):  # continue from last done level
        if os.path.exists(F'{groups_path}-Level-{i}.npy'):
            labels_list = np.load(F'{groups_path}-Level-{i}.npy', allow_pickle=True)
            q = [(labels_i, labels_f[labels_i]) for labels_i in labels_list]
            break

    if q is None:
        q = [(np.arange(labels_f.shape[0]), labels_f)]
    
    while q:
        # labels_list = np.asarray([x[0] for x in q])
        # if levels[-1] == sparse_y.shape[1] and levels[-1] >= len(labels_list) >= levels[-2]:
        #     # early exit; 2nd last level completed
        #     # break
        #     level = len(levels) - 1 
        #     print(F'Finish Clustering Level-{level}')
        #     np.save(F'{groups_path}-Level-{level}.npy', np.arange(levels[-1]).reshape(-1, 1))
        #     break
        
        labels_list = np.asarray([x[0] for x in q])
        assert sum(len(labels) for labels in labels_list) == labels_f.shape[0]
        if len(labels_list) in levels:
            level = levels.index(len(labels_list))
            print(F'Finish Clustering Level-{level}')
            np.save(F'{groups_path}-Level-{level}.npy', np.asarray(labels_list))
        else:
            print(F'Finish Clustering {len(labels_list)}')
        next_q = []
        for node_i, node_f in q:
            if len(node_i) > max_leaf:
                next_q += list(split_node(node_i, node_f, eps))
        q = next_q
    print('Finish Clustering')


def split_node(labels_i: np.ndarray, labels_f: csr_matrix, eps: float):
    n = len(labels_i)
    c1, c2 = np.random.choice(np.arange(n), 2, replace=False)
    centers, old_dis, new_dis = labels_f[[c1, c2]].toarray(), -10000.0, -1.0
    l_labels_i, r_labels_i = None, None
    while new_dis - old_dis >= eps:
        dis = labels_f @ centers.T  # N, 2
        partition = np.argsort(dis[:, 1] - dis[:, 0])
        l_labels_i, r_labels_i = partition[:n//2], partition[n//2:]
        old_dis, new_dis = new_dis, (dis[l_labels_i, 0].sum() + dis[r_labels_i, 1].sum()) / n
        centers = normalize(np.asarray([np.squeeze(np.asarray(labels_f[l_labels_i].sum(axis=0))),
                                        np.squeeze(np.asarray(labels_f[r_labels_i].sum(axis=0)))]))
    return (labels_i[l_labels_i], labels_f[l_labels_i]), (labels_i[r_labels_i], labels_f[r_labels_i])

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, required=False, default='670k')
parser.add_argument('--tree', action='store_true')
parser.add_argument('--id', type=str, required=False, default='_stable')

args = parser.parse_args()

if __name__ == '__main__':
    dataset = args.dataset
    datapath = os.path.join('./data/', dataset)
    
    if dataset == '670k':
        mlb = build_tree_by_level('../data/Amazon-670K/train_v1.txt', 
                                  '../data/Amazon-670K/train_labels.txt',
                                  eps=1e-4, max_leaf=1, levels=[7, 11, 15], groups_path='../data/Amazon-670K/label_group'+args.id)
        groups = np.load(f'../data/Amazon-670K/label_group{args.id}-last.npy', allow_pickle=True)
        new_group = []
        for group in groups:
            new_group.append([mlb.classes_[i] for i in group])
        np.save(f'../../Datasets/Amazon-670K/label_group{args.id}.npy', np.array(new_group))

    elif dataset == '500k':
        mlb = build_tree_by_level('./data/Wiki-500K/train.txt', 
                                  './data/Wiki-500K/train_labels.txt',
                                  1e-4, 8, [11, 14, 17], './data/Wiki-500K/groups')
        groups = np.load(f'./data/Wiki-500K/groups-last{args.id}.npy', allow_pickle=True)
        new_group = []
        for group in groups:
            new_group.append([mlb.classes_[i] for i in group])
        np.save(f'./data/Wiki-500K/label_group{args.id}.npy', np.array(new_group))

    elif dataset == 'WikiSeeAlsoTitles-350K' or dataset == 'AmazonTitles-670K':
        final_name = f'label_group{args.id}'
        final_name = os.path.join(datapath, final_name)
        
        train_file = os.path.join(datapath, 'bow-train.txt')
        labels_file = os.path.join(datapath, 'bow-labels.txt')
        
        mlb = build_tree_by_level(train_file, labels_file, 1e-4, 120, [], f'{final_name}') #4K
        # mlb = build_tree_by_level(train_file, labels_file, 1e-4, 85, [], f'{final_name}') #8K
        # mlb = build_tree_by_level(train_file, labels_file, 1e-4, 30, [], f'{final_name}') #16K

        groups = np.load(f'{final_name}-last.npy', allow_pickle=True)
        new_group = []
        for group in groups:
            new_group.append([mlb.classes_[i] for i in group])
        np.save(f'{final_name}.npy', np.array(new_group))
