"""
program: plots beta values of each of 10 rounds per dataset

"""

# setup environment
env = 'windows' 
if env=='windows':
    path = 'D:/Indranil/JRF/Submission/IEEE_multiheaded/AffNet/results/'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt    
#from labellines import labelLines
###
### main program ###
###

# small datasets
fname = path+'beta_values.csv'
results_df = pd.read_csv(fname)[['dataset', 'run_no', 'beta']]
datasets = np.unique(results_df['dataset'].values)
n_datasets = len(datasets)
max_beta = np.round(results_df.beta.max())+0.5
markers = ['s', 'D', '.', 'o', '^', 'v', 'p', '8', '*']
#results_df.sort_values(by='run_no', axis=0, inplace=True)

# plot results
plt.figure(figsize=(12,8), dpi=600)
X_axis = np.arange(10)+1
plt.xlim(1, 10)
plt.ylim(0.5, max_beta)
for i in range(n_datasets):
    dataset_name = datasets[i]
    marker = markers[i]
    beta_values = results_df[results_df['dataset']==dataset_name]['beta'].values
    plt.plot(X_axis, beta_values, label = dataset_name, marker=marker, markersize=10)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel("Runs", fontsize=24)
plt.ylabel(r'$\beta$   ', rotation=0, fontsize=30)
plt.legend(loc='lower center', ncol=n_datasets//2, fontsize=20, 
           edgecolor='black', facecolor='lightcyan', fancybox=True)
#xvals = 4+np.arange(n_datasets)/2
xvals = [4.5 , 4.5, 4.5 , 7.2, 4.5 , 4.5]
lines = plt.gca().get_lines()
#labelLines(lines, align=False, xvals=xvals, fontsize=30)
#plt.title(r"Learned $\beta$-values across different runs", fontsize=30)
plt.tight_layout()
plt.savefig(f'{path}beta.png')
plt.show()

