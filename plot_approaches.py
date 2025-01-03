"""
program: Compare link prediction accuracy from results obtained in 4 approaches
author: indranil ojha

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

env = 'windows' 
if env=='windows':
    root = 'D:/Indranil/JRF/Submission/IEEE_multiheaded/AffNet/'
results_fname = root + "results/approaches.csv"

# Define the mapping for combinations
mapping = {
    (False, 1): 'Single headed, no adaptive threshold',
    (False, 4): 'Multi-headed, no adaptive threshold',
    (True, 1): 'Single headed, adaptive threshold',
    (True, 4): 'Multi-headed, adaptive threshold'
}

df = pd.read_csv(results_fname)
df = df[['dataset', 'sep_learning', 'n_heads', 'metric_value']]
df['approach'] = df.apply(lambda row: mapping[(row['sep_learning'], row['n_heads'])], axis=1)
df.sort_values(by=['dataset', 'approach'], ascending=False, inplace=True)
df.drop(columns=['sep_learning', 'n_heads'], inplace=True)
#df.drop(df[df['approach'] == 'Multi-headed, no adaptive threshold'].index, inplace=True)
#df.drop(df[df['approach'] == 'Single headed, adaptive threshold'].index, inplace=True)
df = df[df['dataset'].isin(['Cora', 'CiteSeer', 'Photo', "Amazon-ratings", "Questions", 'ogbl-ppa', 'ogbl-collab', 'ogbl-citation2'])] 
datasets = df['dataset'].unique()



pivot_df = df.pivot(index=['dataset'], columns='approach', values='metric_value')
datasets = pivot_df.index
approaches = pivot_df.columns[::-1]

bar_width = 0.15
x = np.arange(len(datasets))
plt.figure(figsize=(12, 5))

# Plot each dataset's metrics
hatches = ['\\\\', '.', '//', None]
for i, approach in enumerate(approaches):
    plt.bar(x + i * bar_width, pivot_df[approach], width=bar_width, label=approach, hatch=hatches[i])

# Customizing the plot
#plt.xlabel('Dataset')
plt.ylabel('AUC / Hits / MRR', fontsize=16)
plt.ylim(0.5, 1.0)
plt.xticks(x+0.21, datasets, fontsize=14)  # Center the tick labels
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=12)
#plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.savefig(f'{root}results/approaches.png')
plt.show()
