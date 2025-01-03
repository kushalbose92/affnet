"""
program: AffNet 
version: first version
   
author: indranil ojha

"""

# setup environment
env = 'windows' 
if env=='windows':
    root = 'D:/Indranil/JRF/Submission/IEEE_multiheaded/AffNet/'
    data_folder = root+"Datasets/"

# import libraries, including utils
import os, sys
sys.path.append(root)

import pandas as pd
import numpy as np
import time
from utils import set_seeds, get_params, get_edge_h, eval_link_pred
from utils import plot_affinity, plot_hist
from models import compute_affinty
from load import load_dataset, split_data_on_edges, get_subgraphs, tf_GData, apply_pca
import gc

###
### main program ###
###

seed = 13
set_seeds(seed)

results_folder = f"{root}results/"
if not os.path.exists(results_folder):
    os.makedirs(results_folder)
if not os.path.exists(results_folder+'aff_plots'):
    os.makedirs(results_folder+'aff_plots')

datasets = ['Texas', 'Cornell', 'Wisconsin']
datasets = ['PubMed', 'Cora', 'CiteSeer']
datasets = ['Actor', 'Photo']
datasets = ["Questions", "Amazon-ratings", "Roman-empire"]
datasets = ["Questions"]
datasets = ["ogbl-ppa", "ogbl-collab", "ogbl-citation2"]
datasets = ["ogbl-citation2"]

result_cols = ['dataset', 'nodes', 'features', 'classes', 'directed', 
               'emb_features', 'n_heads', 'max_nodes', 'init_lr', 'train_frac', 'epochs', 
               'edge_h', 'aff_h', 'min_beta', 'max_beta', 'metric_name', 'metric_value', 'elapsed', 'seed']

try:
   results_df = pd.read_csv(results_folder+'results.csv') 
except:
    results_df = pd.DataFrame(columns = result_cols)

print()
for dataset_name in datasets:

    set_seeds(seed)

        # read edge homophily value for dataset, for reporting
    edge_h = get_edge_h(dataset_name, root)    

    # get best parameters
    emb_features, n_heads, max_nodes, pca_preserve, init_lr, epochs = get_params(dataset_name, root)
    min_lr, decay_steps = 0.0001, 25
    lr_decay = np.power((min_lr/init_lr), decay_steps/epochs)
    heads, sep_learning = n_heads, True
    train_frac=0.8
    max_parts = 10

    # load dataset
    data, n_classes = load_dataset(data_folder, dataset_name)
    #data.x = apply_pca(data.x, pca_preserve)
    data.num_features = data.x.shape[1]
    n_nodes, n_features, n_edges, is_directed = data.num_nodes, data.num_features, data.num_edges, False
    n_orig_nodes = n_nodes
    print(f'\n{dataset_name:<12} #nodes: {n_nodes:>5} #features: {n_features:>5} #classes: {n_classes:>2} #edges: {n_edges:>5} Directed: {is_directed}')

    subgraphs = get_subgraphs(dataset_name, data, max_nodes, max_parts)
    del data
    n_chunks = len(subgraphs)
    aff_matrices, aff_values, beta_values, elapsed_times = [], [], [], []
    metrics = []
    i = 0
    for sg in subgraphs:

        i += 1
        print(f'chunk {i}/{n_chunks}')
        data_train, data_test = split_data_on_edges(sg, train_frac=train_frac) 
        data_train = tf_GData(data_train)
        data_test = tf_GData(data_test)
        n_nodes, n_edges = data_train.num_nodes, data_train.num_edges    
    
        start = time.time()
    
        aff_matrix, aff_h, beta, hist_loss, hist_aff, hist_metric = compute_affinty(dataset_name, data_train, data_test, emb_features, heads, is_directed, sep_learning, init_lr, epochs)    
        metric_name, metric_value = eval_link_pred(dataset_name, aff_matrix, data_test.pos_edge_index, data_test.neg_edge_index)
        print(f"edge-homophily: {edge_h:.4f}, aff-h: {aff_h:.4f}, beta={beta:.2f}, {metric_name}: {metric_value:.4f}")
        
        aff_matrices.append(aff_matrix)
        beta_values.append(beta)
        aff_values.append(aff_h)
        metrics.append(metric_value)
        stop = time.time()
        elapsed = int(stop-start)
        elapsed_times.append(elapsed)
        gc.collect()

    aff_matrices = [data_train.pos_edge_index] + aff_matrices
    aff_h = np.mean(aff_values)
    min_beta = np.min(beta_values)
    max_beta = np.max(beta_values)
    metric_value = np.mean(np.array(metrics), axis=0)
    elapsed = np.mean(elapsed_times)

    # save results
    if env=="windows":
        plot_flag, save_flag = True, True
    else:
        plot_flag, save_flag = False, True
        
    plot_flag, save_flag = True, False
    plot_affinity(aff_matrices, dataset_name, results_folder, plot_flag, save_flag)

    print(f"edge-homophily: {edge_h:.4f}, aff-h: {aff_h:.4f}, min_beta={min_beta:.2f}, max_beta={max_beta:.2f}, {metric_name}: {metric_value:.4f}")
    results_df.loc[len(results_df)] = [dataset_name, n_orig_nodes, n_features, n_classes, 
            is_directed, emb_features, n_heads, max_nodes, init_lr, train_frac,
            epochs, edge_h, aff_h, min_beta, max_beta, metric_name, metric_value, elapsed, seed]
    results_df.to_csv(results_folder+'results.csv', index=False, float_format="%.4f")



