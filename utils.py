"""
program: utilities needed by affinity matrix computation script
author: indranil ojha

"""
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.random import set_seed as tf_seed
from torch_geometric.utils import remove_isolated_nodes, homophily 
from numpy.random import seed as np_seed
from random import seed as random_seed
from sklearn.metrics import roc_auc_score
from ogb.linkproppred import Evaluator
import torch
import numpy as np
import os

# set all seeds for reproducibility
def set_seeds(seed=13):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random_seed(seed)
    tf_seed(seed)
    np_seed(seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if device == 'cuda:0':
      torch.cuda.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False

# retrieve best parameters for dataset
def get_params(dataset_name, root):
    params_fname = f"{root}params/params.csv"
    params_df = pd.read_csv(params_fname)
    params_df = params_df[params_df['dataset']==dataset_name]
    params_df = params_df[['emb_features', 'n_heads', 'max_nodes', 'pca_preserve', 'init_lr', 'epochs']]
    emb_features, n_heads, max_nodes, pca_preserve, init_lr, epochs = params_df.iloc[0].values
    emb_features = int(emb_features)
    n_heads = int(n_heads)
    max_nodes = int(max_nodes)
    epochs = int(epochs)
    return(emb_features, n_heads, max_nodes, pca_preserve, init_lr, epochs)
    
# retrieve homophily for dataset
def get_edge_h(dataset_name, root):
    edge_h_fname = f"{root}params/edge_h.csv"
    edge_h_df = pd.read_csv(edge_h_fname)
    edge_h_df = edge_h_df[edge_h_df['dataset']==dataset_name]
    edge_h_df = edge_h_df['edge_h']
    return(edge_h_df.iloc[0])

# compute node level edge homophily
def get_node_h(adj, y):
    eps = 1e-8
    n_nodes = len(y)
    y_np = y.numpy()
    # adj can be sparse or dense
    if isinstance(adj, tf.Tensor): # dense
        adj_np = adj.numpy()
    else: # sparse
        adj_np = tf.sparse.to_dense(adj).numpy()
    yy = np.tile(y_np, (n_nodes,1))
    yy_t = np.transpose(yy)
    match = (yy==yy_t).astype(int) 
    match = np.multiply(adj_np, match)
    nodewise_match = np.sum(match, axis=1)
    degree = np.sum(adj_np, axis=1)
    node_h = np.divide(nodewise_match+eps, degree+eps)
    return(node_h)    

def get_self_affinity(aff_matrix):
    n_nodes = aff_matrix.shape[0]
    diag = tf.linalg.diag_part(aff_matrix).numpy()
    all_sum = tf.reduce_sum(aff_matrix).numpy()
    diag_sum = tf.reduce_sum(diag).numpy()
    non_diag_sum = all_sum - diag_sum
    non_diag_mean = non_diag_sum/(n_nodes * (n_nodes - 1))
    self_affinity = diag / non_diag_mean
    return(self_affinity)

def get_degree_1D(edge_index, n_nodes):
    deg = np.zeros(n_nodes)
    for i in range(n_nodes):
        deg[i] = np.sum(edge_index.numpy()[0,:]==i)
    return(deg)

def remove_isolated(data):
    n_nodes_before =data.x.shape[0]
    _, _, mask = remove_isolated_nodes(data.edge_index, num_nodes=n_nodes_before)
    data = data.subgraph(mask)
    n_nodes_after =data.x.shape[0]
    nodes_removed = n_nodes_before - n_nodes_after
    return(data, nodes_removed)

def get_homophily(data):
    h = homophily(data.edge_index, data.y, method='edge')
    return(h)

def bin_it(a, n_bins=5):
    min_a, max_a = np.min(a), np.max(a)
    width = (max_a-min_a)/n_bins
    b = np.floor(((a-min_a)/width))
    return(b)

def plot_hist(losses, dataset_name, results_folder, ylabel, plot_flag=True, save_flag=False):
    epochs = len(losses[0])
    title = dataset_name
    legends = ["single-headed w/o sep", "multi-headed w/o sep", 
               "single-headed with sep", "multi-headed with sep"]
    linestyles = ['dotted', 'dashed','solid', 'solid' ]
    linewidths = [2, 2, 2, 3]
    plt.figure(figsize=(6,6), dpi=600)
    for i, loss in enumerate(losses):
        plt.plot(loss, label=legends[i], linestyle=linestyles[i], linewidth=linewidths[i])
    plt.xlabel("epochs", fontsize=32)
    plt.ylabel(ylabel, fontsize=32)
    plt.title(title, fontsize=36)
    plt.xticks([0, epochs//2, epochs], [0, epochs//2, epochs], fontsize=28)
    plt.yticks([0, 0.2, 0.4], [0, 0.2, 0.4], fontsize=28)
    #plt.yticks(fontsize=28)
    plt.tight_layout()
    if save_flag:
        savefile=f"{results_folder}hist/{dataset_name} {ylabel}.png"
        plt.savefig(savefile)
    if plot_flag:
        plt.show()

def plot_affinity(aff_matrices, dataset_name, results_folder, plot_flag=True, save_flag=False, n_points=100):
    adj = tf.sparse.to_dense(aff_matrices[0])[:n_points,:n_points]
    aff_matrices = [adj, aff_matrices[-1]]

    # plot affinity matrix against adjacency
    c00, c01, c1 = 'ivory', 'navajowhite', 'darkgoldenrod'
    plt.figure(figsize=(6,3), dpi=600)   
    n_plots = len(aff_matrices)
    for k in range(n_plots):
        if k==0:
            c0 = c00
        else:
            c0 = c01
        aff_mat = aff_matrices[k][:n_points,:n_points]
        aff_mat = aff_mat.numpy()
        zeros = np.where(aff_mat<0.5)
        ones = np.where(aff_mat>=0.5)
        plt.subplot(1, n_plots, k+1)
        plt.scatter(zeros[0], n_points-zeros[1], color=c0, s=5)
        plt.scatter(ones[0], n_points-ones[1], color=c1, s=5)
        plt.xticks([])
        plt.yticks([])
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_color('black')  # Set boundary color to blue
            #spine.set_linewidth(2)   # Optional: Set the boundary thickness
    plt.tight_layout()
    if save_flag:
        savefile=f"{results_folder}aff_plots/{dataset_name}.png"
        plt.savefig(savefile)
    if plot_flag:
        plt.show()

# compute hit@k between true and pred 
def compute_hit_rate(true, pred, k=50):
    # hit rate for edges or positive samples
    sort_idx = np.argsort(pred)[::-1]
    t = true[sort_idx][:k]
    hit = np.sum(t)
    hit_at_k = hit / k
   
    return(hit_at_k)

# compute MRR between Adj (true) and Aff (pred) using mask  
def compute_mrr(adj, aff, mask):
    adj = adj.astype(int)
    mask = mask.astype(int)
    n = adj.shape[0]
    ranks = []
    for i in range(n):
        a, f, m = adj[i], aff[i], mask[i]
        valid_f = f[m]
        sorted_f = np.argsort(-valid_f) + 1
        edge_ranks = sorted_f[a]
        ranks.extend(list(edge_ranks))
    reciprocal_ranks = 1 / np.array(ranks)
    mrr = np.mean(reciprocal_ranks)
    return(mrr)

def evaluate_ogb(dataset_name, aff, pos_edge_index, neg_edge_index):

    evaluator = Evaluator(name=dataset_name)

    aff = aff.numpy()
    
    dense_pos_edges = tf.sparse.to_dense(pos_edge_index)
    true_pos_edge_indices = tf.where(dense_pos_edges != 0).numpy()
    aff_pos = aff[true_pos_edge_indices[:, 0], true_pos_edge_indices[:, 1]]
    
    dense_neg_edges = tf.sparse.to_dense(neg_edge_index)
    true_neg_edge_indices = tf.where(dense_neg_edges != 0).numpy()
    if dataset_name == "ogbl-citation2":
        aff_neg = aff[true_neg_edge_indices[:, 0]]
    else:
        aff_neg = aff[true_neg_edge_indices[:, 0], true_neg_edge_indices[:, 1]]
    
    input_dict = {
        "y_pred_pos": torch.from_numpy(aff_pos),
        "y_pred_neg": torch.from_numpy(aff_neg),
    }
    
    results = evaluator.eval(input_dict)
    if dataset_name == "ogbl-ppa":
        metric_name = "Hits@100"
        metric = results["hits@100"]
    elif dataset_name == "ogbl-collab":
        metric_name = "Hits@50"
        metric = results["hits@50"]
    elif dataset_name == "ogbl-citation2":
        metric_name = "MRR"
        metric = np.mean(results["mrr_list"].numpy())
    return(metric_name, metric)

def eval_link_pred(dataset_name, aff, pos_edge_index, neg_edge_index):
    if dataset_name[:4]=='ogbl':
        metric_name, metric = evaluate_ogb(dataset_name, aff, pos_edge_index, neg_edge_index)
    else:
        # aff is dense (n x n) tensor, two edge_index tensors are in sparse form
        pos_adj = tf.sparse.to_dense(pos_edge_index).numpy()
        neg_adj = tf.sparse.to_dense(neg_edge_index).numpy()
        mask = pos_adj + neg_adj
        aff = aff.numpy()
        aff = np.multiply(aff, mask)
        adj = pos_adj
        
        # compute AUROC
        metric_name = 'AUC'
        metric = roc_auc_score(adj[mask==1], aff[mask==1])
    return(metric_name, metric)    

