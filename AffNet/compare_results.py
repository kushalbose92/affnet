"""
program:
    retrieves homophily (edge & node) scores of different datasets
    compares and creates reports
"""

# setup environment
env = 'windows' 
if env=='windows':
    root = 'D:/Indranil/JRF/Submission/IEEE_multiheaded/AffNet/'
    data_folder = root+"Datasets/"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt    
###
### main program ###
###

# datasets
datasets = ['Texas', 'Cornell', 'Wisconsin', 'Cora', 'CiteSeer', 'PubMed', 'Actor', 'Photo']

results_df = pd.read_csv(root+'results/results.csv')[['dataset', 'aff_h']]
homophily_df = pd.read_csv(root+'params/homophily.csv')[['dataset', 'node_h', 'edge_h']]
df = pd.merge(results_df, homophily_df, on='dataset', how='inner')
df.sort_values(by='aff_h', axis=0, ascending=False, inplace=True)
df = df[df['dataset'].isin(datasets)]

X_axis = np.arange(len(df))
if len(df)>1:
    # plot results
    plt.figure(figsize=(12,9), dpi=600)
    plt.bar(X_axis - 0.2, df['node_h'].values, 0.2, label = 'Node-homophily', hatch='\\\\')
    plt.bar(X_axis, df['edge_h'].values, 0.2, label = 'Edge-homophily', hatch='.')
    plt.bar(X_axis + 0.2, df['aff_h'].values, 0.2, label = 'Affinity-homophily', hatch='//')
    plt.xticks(X_axis, df['dataset'].values, rotation=45, fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylabel("Homophily / Affinity", fontsize=20)
    plt.ylim(0,1)
    xmin, xmax = plt.xlim()
    plt.hlines(0.5, xmin, xmax, 'b','dotted')
    plt.text(4.5, 0.52, '$\mathit{middle\ line\: (0.5)}$', fontsize = 20, color='k') 
    plt.text(0.15, 0.90, 'Homophily', fontsize = 20, color='b') 
    plt.text(5.25, 0.36, 'Heterophily', fontsize = 20, color='b') 
    plt.legend(fontsize=20)
    plt.savefig(f'{root}results/comparison.png')
    plt.show()
