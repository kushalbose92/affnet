import matplotlib.pyplot as plt
import numpy as np

path = 'D:/Indranil/JRF/Submission/IEEE_multiheaded/AffNet/results'
datasets = ['Cora', 'CiteSeer', 'Cornell', 'Texas', 'Wisconsin', 'Actor']

# Define the formats
labels = ["single-headed w/o sep", "multi-headed w/o sep", 
           "single-headed with sep", "multi-headed with sep"]
#linestyles = ['dotted', 'dashed','solid', 'solid' ]
linestyles = ['--',':', '-.', '-']
linewidths = [2, 2, 2, 2]
markers = ['s', 'D', 'o', '^']

# Create subplots
fig, axes = plt.subplots(2, 3, figsize=(12, 9))
axes = axes.ravel()  # Flatten the 2D array of axes for easier iteration

for j in range(6):
    dataset_name = datasets[j]
    hist_fname = f"{path}/hist_{dataset_name}.csv"
    hist_data = np.loadtxt(hist_fname, delimiter=",")
    n_points = hist_data.shape[1]
    x = np.arange(n_points) + 1
    epochs = max(x)
    for i in range(4):
        axes[j].plot(x[::200], hist_data[i][::200], label=labels[i], 
                linestyle=linestyles[i], linewidth=linewidths[i])
        axes[j].set_xlabel("epochs", fontsize=20)
        axes[j].set_ylabel("loss", fontsize=20)
        axes[j].set_xticks([0, epochs//2, epochs], [0, epochs//2, epochs], fontsize=16)
        axes[j].set_yticks([0, 0.2, 0.4], [0, 0.2, 0.4], fontsize=16)
    axes[j].set_title(dataset_name, fontsize=24)

# Create a common legend at the bottom
fig.legend(labels, loc='lower center', ncol=2, fontsize=20, frameon=False)

# Adjust layout
plt.tight_layout()
fig.subplots_adjust(hspace=0.6, bottom=0.18)  # Increase the vertical space
plt.show()
