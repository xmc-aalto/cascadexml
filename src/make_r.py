from scipy.sparse import csr_matrix, lil_matrix
from sklearn.preprocessing import normalize
from data_utils import load_data, load_group, load_cluster_tree
import numpy as np
from tqdm import tqdm


def make_r(clusters, y):
    # make label to cluster maps
    # label_to_cluster_maps = train_ds.label_to_cluster_maps 
    label_to_cluster_maps = []
    for i in range(len(clusters)):
        label_to_cluster_maps.append(np.zeros(y.shape[1], dtype=np.int64) - 1)
        for idx, labels in enumerate(clusters[i]):
            label_to_cluster_maps[-1][[int(l) for l in labels]] = idx
    
    # import pdb; pdb.set_trace()
    rs = [lil_matrix((y.shape[0], len(cluster)), dtype=np.float32) for cluster in clusters]
    for idx in range(y.shape[0]):
        labels = y[idx].indices
        for map, r in zip(label_to_cluster_maps, rs):
            cluster_ids = map[labels]
            for cid in cluster_ids:
                r[idx, cid] += 1
    
    for rid in range(len(rs)):
        rs[rid] = normalize(rs[rid], norm='l1', axis=1)
        rs[rid] = rs[rid].tocsr()

        # iterate over datapoints
        # pick indices of non zero y
        # increment cluster of each y with label to cluster map
    # import pdb; pdb.set_trace()
    return rs



# import argparse
# parser = argparse.ArgumentParser()

# parser.add_argument('--clusters', type=str, required=False, default='670k')
# parser.add_argument('--dataset', type=str, required=False, default='670k')
# parser.add_argument('--tree', action='store_true')
# parser.add_argument('--id', type=str, required=False, default='_stable')

# args = parser.parse_args()


# if __name__ == '__main__':
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--batch_size', type=int, required=False, default=16)
#     parser.add_argument('--update_count', type=int, required=False, default=1)
#     parser.add_argument('--lr', type=float, required=False, default=1e-4)
#     parser.add_argument('--seed', type=int, required=False, default=29)

#     parser.add_argument('--mn', type=str, required=True)
#     parser.add_argument('--lm', dest='load_model', type=str, default="", help='model to load')
#     parser.add_argument('--test', action='store_true', help='Testing mode or training mode')

#     parser.add_argument('--num_epochs', type=int, required=False, default=15)
#     parser.add_argument('--dataset', type=str, required=False, default='eurlex4k')
#     parser.add_argument('--bert', type=str, required=False, default='bert-base')
#     parser.add_argument('--max_len', type=int, required=False, default=128)

#     parser.add_argument('--swa', action='store_true')
#     parser.add_argument('--swa_warmup', type=int, required=False, default=30)
#     parser.add_argument('--swa_step', type=int, required=False, default=3000)

#     parser.add_argument('--tree_id', type=int, default=0)
#     parser.add_argument('--topk', required=False, type=int, default=10, nargs='+')
#     parser.add_argument('--freeze_layer_count', type=int, default=6)
    
#     parser.add_argument('--hidden_dim', type=int, required=False, default=300)
#     parser.add_argument('--load_model', type=str, default='', required=False)

#     parser.add_argument('--adahess', action='store_true')

#     parser.add_argument('--distributed', action='store_true', help='distributed training')
#     parser.add_argument('--local_rank', type=int, help='node rank for distributed training', default=None)
#     parser.add_argument('--local_world_size', type=int, default=2,
#                             help='number of GPUs each process')
#     parser.add_argument('--dist_eval', action='store_true', help='use ddp for eval as well')

#     #Parabel Cluster params
#     parser.add_argument('--train_W', action='store_true')
#     parser.add_argument('--rw_loss', type=int, nargs='+', default=[1, 1, 1, 1])

#     parser.add_argument('--cluster_name', default='clusters_eclare_4.pkl') 
#     # parser.add_argument('--cluster_name', default='recluster_eclare_768C.pkl')
#     parser.add_argument('--b_factors', type=int, nargs='+', default=[9, 12])
#     parser.add_argument('--cluster_method', default='AugParabel')
#     parser.add_argument('--verbose_lbs', type=int, default=0)

#     #Graph params
#     parser.add_argument('--graph_name', default='graph.npz')
#     parser.add_argument('--prune_max_dist', type=float, default=1.0)
#     parser.add_argument('--p_reset', type=float, default=0.8)
#     parser.add_argument('--walk_len', type=int, default=400)
#     parser.add_argument('--top_k', type=int, default=10)

#     params = parser.parse_args()


    # X_train, X_test, Y_train, Y_test, X_tfidf, inv_prop = load_data('./data/Wiki10-31K/', 'bert-base', 30938, False)
    # train_ds = MultiXMLGeneral(X_train, Y_train, params, X_tfidf, mode='train')

    # make_r(train_ds, Y_train)
