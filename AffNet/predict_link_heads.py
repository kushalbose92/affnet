"""
program: AffNet : ablation on number of heads
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
from utils import set_seeds, get_params, get_edge_h, eval_link_pred
from models import compute_affinty
from load import load_dataset, split_data_on_edges, get_subgraphs, tf_GData
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

datasets = ['Texas', 'Cornell']
datasets = ['Chameleon']
datasets = ['Cora']

result_cols = ['dataset', 'heads', 'metric_name', 'metric_mean', 'metric_std']

results_fname = "heads.csv"
try:
    results_df = pd.read_csv(results_folder+results_fname) 
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
    sep_learning = True
    train_frac=0.8
    max_parts = 10

    print(f"Dataset: {dataset_name}")
    for heads in [16, 8, 4, 2, 1]:
        print(f'   head: {heads}')

        metrics = []
        for i in range(5):
            print(f"      {i+1}. ", end='')
            # load dataset
            data, n_classes = load_dataset(data_folder, dataset_name)
            #data.x = apply_pca(data.x, pca_preserve)
            data.num_features = data.x.shape[1]
            n_nodes, n_features, n_edges, is_directed = data.num_nodes, data.num_features, data.num_edges, False
            n_orig_nodes = n_nodes
        
            sg = get_subgraphs(dataset_name, data, max_nodes, max_parts)[0]
            del data
       
            data_train, data_test = split_data_on_edges(sg, train_frac=train_frac) 
            data_train = tf_GData(data_train)
            data_test = tf_GData(data_test)
            n_nodes, n_edges = data_train.num_nodes, data_train.num_edges    
        
            aff_matrix, _, _, _, _, _ = compute_affinty(dataset_name, data_train, data_test, emb_features, heads, is_directed, sep_learning, init_lr, epochs)    
            metric_name, metric_value = eval_link_pred(dataset_name, aff_matrix, data_test.pos_edge_index, data_test.neg_edge_index)
            metrics.append(metric_value)
            gc.collect()
    
        metrics = np.array(metrics)
        metric_mean = np.mean(metrics, axis=0)
        metric_std = np.std(metrics, axis=0)
        
        print(f"   {metric_name}: {metric_value:.4f}\n")

        results_df.loc[len(results_df)] = [dataset_name, heads, metric_name, metric_mean, metric_std]
        results_df.to_csv(results_folder+results_fname, index=False, float_format="%.4f")



