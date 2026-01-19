"""
program: utility routines for affinity matrix computation script
author: indranil ojha
"""

# import libraries
from torch_geometric.datasets import Planetoid, WebKB, Actor, WikipediaNetwork, Amazon, HeterophilousGraphDataset
from ogb.linkproppred import PygLinkPropPredDataset
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, to_undirected
from scipy.io import loadmat
from scipy.sparse import csc_matrix
import tensorflow as tf
import torch
import numpy as np
from sklearn.decomposition import PCA

def convert_edge_index_to_tf(edge_index, n_nodes, n_edges):
    edge_index = tf.cast(edge_index, tf.int64)
    edge_index = tf.sparse.SparseTensor(indices=tf.transpose(edge_index), 
                     values=tf.ones(n_edges), dense_shape=[n_nodes, n_nodes])
    edge_index = tf.sparse.reorder(edge_index)
    print("aaaaa")
    print("bbbb")
    return(edge_index)

class tf_GData:
  def __init__(self, torch_GData):
    self.x = tf.cast(torch_GData.x, tf.float32)
    if hasattr(torch_GData, 'y'):
        if torch_GData.y is not None:
            self.y = tf.cast(torch_GData.y, tf.float32)
        else:
            self.y = None
    else:
        self.y = None
    if torch_GData.edge_index is not None:
        self.edge_index = convert_edge_index_to_tf(torch_GData.edge_index, torch_GData.num_nodes, torch_GData.num_edges)
    if hasattr(torch_GData, 'pos_edge_index'):
        self.pos_edge_index = convert_edge_index_to_tf(torch_GData.pos_edge_index, torch_GData.num_nodes, torch_GData.n_pos)
    if hasattr(torch_GData, 'neg_edge_index'):
        self.neg_edge_index = convert_edge_index_to_tf(torch_GData.neg_edge_index, torch_GData.num_nodes, torch_GData.n_neg)
    self.num_nodes, self.num_features = torch_GData.x.shape
    self.num_edges = torch_GData.num_edges
    self.num_features = torch_GData.num_features
    self.n_pos = torch_GData.n_pos
    self.n_neg = torch_GData.n_neg

# convert a graph dataset from torch to tf
def convert_data_to_tf(data):
    def convert_edge_index_to_tf(edge_index, n_nodes, n_edges):
        edge_index = tf.cast(edge_index, tf.int64)
        edge_index = tf.sparse.SparseTensor(indices=tf.transpose(edge_index), 
                         values=tf.ones(n_edges), dense_shape=[n_nodes, n_nodes])
        edge_index = tf.sparse.reorder(edge_index)
        return(edge_index)
    
    data.x = tf.cast(data.x, tf.float32)
    if hasattr(data, 'y'):
        data.y = tf.cast(data.y, tf.float32)
    if data.edge_index is not None:
        data.edge_index = convert_edge_index_to_tf(data.edge_index, data.num_nodes, data.num_edges)
    if hasattr(data, 'pos_edge_index'):
        data.pos_edge_index = convert_edge_index_to_tf(data.pos_edge_index, data.num_nodes, data.n_pos)
    if hasattr(data, 'neg_edge_index'):
        data.neg_edge_index = convert_edge_index_to_tf(data.neg_edge_index, data.num_nodes, data.n_neg)
    data.num_features = data.x.shape[1]
    return(data)


# create trtain and test masks where not present
def get_mask(n_nodes):
    mask = np.array([1]*n_nodes, dtype=bool)
    step =  10 # 10% for test, uniform & not random to ensure reproducability 
    for i in range(round(n_nodes/step)):
        mask[step*i] = False
    train_mask = tf.reshape(tf.convert_to_tensor(mask), (-1,1))
    test_mask = tf.math.logical_not(train_mask)
    return(train_mask, test_mask)

def normalize_adj(edge_index, n_nodes):
    def d_half_inv(e, n_nodes):
        deg = [e.count(i) for i in range(n_nodes)]
        deg_inv_sqrt = [1/np.sqrt(d) if d!=0 else 0 for d in deg]
        return(deg_inv_sqrt)
    edges = edge_index.indices.numpy()
    vals = list(edge_index.values.numpy())
    n_edges = len(edges)
    deg_inv_sqrt_0 = d_half_inv(list(edges[:,0]), n_nodes)
    deg_inv_sqrt_1 = d_half_inv(list(edges[:,1]), n_nodes)
    v_norm = [vals[i]*deg_inv_sqrt_0[edges[i,0]]*deg_inv_sqrt_1[edges[i,1]] 
              for i in range(n_edges)]
    edge_index = tf.sparse.SparseTensor(indices=edge_index.indices, 
                    values=v_norm, dense_shape=[n_nodes, n_nodes])
    return(edge_index)

def load_snap_patents(data_folder):
    fname = f"{data_folder}snap_patents/snap_patents.mat"
    patents_dict = loadmat(fname)
    X = torch.from_numpy(csc_matrix.toarray(patents_dict['node_feat']))
    y = torch.from_numpy(patents_dict['years'][0])
    edge_index = torch.from_numpy(patents_dict['edge_index'])
    n_classes = len(np.unique(y))

    data = Data(x=X, y=y, edge_index=edge_index, is_directed=True)
    return(data, n_classes)

def load_ogbl(data_folder, dataset_name):
    # load ogb data
    dataset = PygLinkPropPredDataset(name=dataset_name, root=data_folder+'ogb')
    ogb_data = dataset[0]
    
    """
    # build ogb_data.x for ppa and ddi, as per paper implementation 
    if dataset_name == "ogbl-ppa":
            ogb_data.x = torch.argmax(ogb_data.x, dim=-1)
            ogb_data.max_x = torch.max(ogb_data.x).item()
    elif dataset_name == "ddi":
            ogb_data.x = torch.arange(ogb_data.num_nodes)
            ogb_data.max_x = ogb_data.num_nodes
    """

    ogb_data.x = ogb_data.x.to(torch.float)

    # take care of normalize_features
    x_norm = ogb_data.x.norm(p=2, dim=1, keepdim=True)   
    ogb_data.x = ogb_data.x / x_norm

    n_classes = dataset.num_classes 
    # convert to undirected graph
    ogb_data.edge_index = to_undirected(ogb_data.edge_index)
    # remove self loops 
    selfloops = ogb_data.edge_index[0,:]==ogb_data.edge_index[1,:]
    ogb_data.edge_index = ogb_data.edge_index[:,torch.logical_not(selfloops)]

    return(ogb_data, n_classes)

def load_ogbn(data_folder, dataset_name):
    # load ogb data
    dataset = PygNodePropPredDataset(name=dataset_name, root=data_folder+'ogb')
    ogb_data = dataset[0]
    
    """
    # build ogb_data.x for ppa and ddi, as per paper implementation 
    if dataset_name == "ogbl-ppa":
            ogb_data.x = torch.argmax(ogb_data.x, dim=-1)
            ogb_data.max_x = torch.max(ogb_data.x).item()
    elif dataset_name == "ddi":
            ogb_data.x = torch.arange(ogb_data.num_nodes)
            ogb_data.max_x = ogb_data.num_nodes
    """

    ogb_data.x = ogb_data.x.to(torch.float)

    # take care of normalize_features
    x_norm = ogb_data.x.norm(p=2, dim=1, keepdim=True)   
    ogb_data.x = ogb_data.x / x_norm

    n_classes = dataset.num_classes 
    # convert to undirected graph
    ogb_data.edge_index = to_undirected(ogb_data.edge_index)
    # remove self loops 
    selfloops = ogb_data.edge_index[0,:]==ogb_data.edge_index[1,:]
    ogb_data.edge_index = ogb_data.edge_index[:,torch.logical_not(selfloops)]

    return(ogb_data, n_classes)

def load_arxiv(folder):
    data_folder = f"{folder}arxiv-year/ogbn_arxiv/raw/"
    mask_fname = f"{folder}arxiv-year/splits/arxiv-year-splits.npy"

    X = np.loadtxt(f'{data_folder}node-feat.csv', delimiter=',')
    y = np.loadtxt(f'{data_folder}node_year.csv', delimiter=',')
    edge_index = np.loadtxt(f'{data_folder}edge.csv', delimiter=',', dtype=int)

    X = torch.from_numpy(X)
    y = torch.from_numpy(y)
    edge_index = torch.from_numpy(np.transpose(edge_index))
    n_classes = len(np.unique(y))
    n_nodes = len(y)
    
    mask_dict = np.load(mask_fname, allow_pickle=True)
    n_split = len(mask_dict)
    train_mask = []
    test_mask = []
    for i in range(n_split):
        train_mask_idx = mask_dict[i]['train']
        train_mask_bool = np.zeros(n_nodes, dtype=bool)
        train_mask_bool[train_mask_idx] = True
        train_mask.append(train_mask_bool)
        test_mask_idx = mask_dict[i]['test']
        test_mask_bool = np.zeros(n_nodes, dtype=bool)
        test_mask_bool[test_mask_idx] = True
        test_mask.append(test_mask_bool)
    train_mask = np.transpose(np.array(train_mask))
    test_mask = np.transpose(np.array(test_mask))
    train_mask = torch.from_numpy(train_mask)
    test_mask = torch.from_numpy(test_mask)

    data = Data(x=X, y=y, edge_index=edge_index, 
            train_mask=train_mask, test_mask=test_mask, is_directed=True)
    return(data, n_classes)

# load dataset
def load_dataset(data_folder, dataset_name, rand_train=False):
        
    if dataset_name == 'SnapPatents':
        data, n_classes = load_snap_patents(data_folder)
    elif dataset_name[:4] == 'ogbn':
        data, n_classes = load_ogbn(data_folder, dataset_name)
    elif dataset_name[:4] == 'ogbl':
        data, n_classes = load_ogbl(data_folder, dataset_name)
    else:
        if dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
            dataset = Planetoid(root=f'{data_folder}pyG', name=dataset_name, 
                        transform=NormalizeFeatures())
        if dataset_name in ['Cornell', 'Wisconsin', 'Texas']:
            dataset = WebKB(root=f'{data_folder}pyG', name=dataset_name, 
                        transform=NormalizeFeatures())
        if dataset_name in ['Squirrel', 'Chameleon']:
            dataset = WikipediaNetwork(root=f'{data_folder}pyG', name=dataset_name, 
                        transform=NormalizeFeatures())
        if dataset_name in ['Actor']:
            dataset = Actor(root=f'{data_folder}pyG/Actor', 
                        transform=NormalizeFeatures())
        if dataset_name in ["Computers", "Photo"]:
            dataset = Amazon(root=f'{data_folder}pyG', name=dataset_name, 
                        transform=NormalizeFeatures())
        if dataset_name in ["Roman-empire", "Amazon-ratings", "Minesweeper", "Tolokers", "Questions"]:
            dataset = HeterophilousGraphDataset(root=f'{data_folder}pyG', name=dataset_name, 
                        transform=NormalizeFeatures())
    
        n_classes = dataset.num_classes 
        data = dataset[0]  # Get the first graph object.
        # convert to undirected graph
        data.edge_index = to_undirected(data.edge_index)
        # remove self loops 
        selfloops = data.edge_index[0,:]==data.edge_index[1,:]
        data.edge_index = data.edge_index[:,tf.logical_not(selfloops).numpy()]

    return(data, n_classes)


# partition graph in subgraphs of a max number of nodes
def get_subgraphs(dataset_name, data, max_nodes, max_parts=10):
    
    # we need a minimum 100 edges for hits@100 and 50 edges for hits@50.
    if dataset_name == "ogbl-ppa":
        min_edges = 100
    else:
        min_edges = 50

    n_nodes = data.num_nodes
    nodes = np.arange(n_nodes)
    np.random.shuffle(nodes)
    n_chunks = int(np.ceil(n_nodes/max_nodes))
    chunks = np.array_split(nodes, n_chunks)
    # remove last subgraph if too small
    if len(chunks) > 1 and len(chunks[-1]) < 100:
        chunks = chunks[:-1]
    if len(chunks) > max_parts:
        chunks = chunks[:max_parts]

    subgraphs = []
    chunk_count = 0 # may go upto max_parts
    for selected_nodes in chunks:
    
        selected_nodes = torch.tensor(sorted(selected_nodes), dtype=torch.long)
        
        # Extract the subgraph
        subgraph_edge_index = subgraph(selected_nodes, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)[0]
        n_edges = subgraph_edge_index.shape[1]
        if n_edges < min_edges:
            continue
        subgraph_data = Data(x=data.x[selected_nodes], edge_index=subgraph_edge_index, num_nodes=len(selected_nodes))
        if hasattr(data, 'y'):
            if data.y is not None:
                subgraph_data = Data(x=data.x[selected_nodes], y=data.y[selected_nodes], edge_index=subgraph_edge_index, num_nodes=len(selected_nodes))
        subgraphs.append(subgraph_data)
        chunk_count += 1
        if chunk_count>= max_parts:
            break
    del data
    return(subgraphs)
        
# returns single subgraph with randomly chosen max nodes 
# also includes y if exists - not the case yet for get_subgraphs
def get_random_subgraph(data, max_nodes, selected_nodes=None):
    if selected_nodes is None:
        # Randomly select node indices
        selected_nodes = sorted(np.random.choice(range(data.num_nodes), max_nodes, replace=False))
        
    selected_nodes = torch.tensor(selected_nodes, dtype=torch.long)
            
    # Extract the subgraph
    subgraph_edge_index = subgraph(selected_nodes, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)[0]
    if hasattr(data, 'y'):
        if data.y is not None:
            subgraph_data = Data(x=data.x[selected_nodes], y=data.y[selected_nodes], edge_index=subgraph_edge_index, num_nodes=max_nodes)
            return(subgraph_data)
    subgraph_data = Data(x=data.x[selected_nodes], edge_index=subgraph_edge_index, num_nodes=max_nodes)
    return(subgraph_data)
    
def split_data_on_edges(data, train_frac):

    def get_non_edge_index(n_nodes, edge_index):
        # generate random points of non-edges, same number of edges
        # initially taken double count to address removal due to:
            # self-loops & duplicates & overlap with edges
        n_edges = edge_index.shape[1]
        non_edge_index = np.random.randint(1, n_nodes, size=(2, int(n_edges*2.0)))
        valid_flags = non_edge_index[0]!=non_edge_index[1] # remove self-loops
        non_edge_index = non_edge_index[:, valid_flags]
        non_edge_index_set = set([tuple(t) for t in np.transpose(non_edge_index)])
        edge_index_set = set([tuple(t) for t in np.transpose(edge_index)])
        non_edge_index = list(non_edge_index_set - edge_index_set)
        non_edge_index = np.transpose(non_edge_index[:n_edges])
        #n_edges = non_edge_index.shape[0]
        # sort the index    
        lexsorted_indices = np.lexsort((non_edge_index[1, :], non_edge_index[0, :]))
        non_edge_index = non_edge_index[:, lexsorted_indices]
        return(torch.from_numpy(non_edge_index))
    
    def split_edges(edge_index, train_frac):
        n_edges = edge_index.shape[1]
        train_idx = np.random.choice(n_edges, int(train_frac*n_edges), replace=False)
    
        train_mask = np.zeros(n_edges, dtype=int)
        train_mask[train_idx] = 1
        train_mask = train_mask.astype(bool)
        test_mask = np.logical_not(train_mask)
    
        train_edge_index = edge_index[:, train_mask]
        test_edge_index = edge_index[:, test_mask]
        return(train_edge_index, test_edge_index)
    
    edge_index = data.edge_index
    n_nodes, n_features = data.num_nodes, data.num_features

    # generate non_edge_index using edge_index
    non_edge_index = get_non_edge_index(n_nodes, edge_index)

    if train_frac is None: # no split, but create data_train in right format
        
        n_edges = edge_index.shape[1]
        n_pos_edges = n_edges
        n_neg_edges = non_edge_index.shape[1]
        if hasattr(data, 'y'):
            data_train = Data(x=data.x, y=data.y, edge_index = edge_index,
                              pos_edge_index=edge_index, 
                              neg_edge_index=non_edge_index, num_nodes=n_nodes, 
                              num_features=n_features, num_edges=n_edges, 
                              n_pos=n_pos_edges, n_neg=n_neg_edges)
        else:
            data_train = Data(x=data.x, edge_index = edge_index,
                              pos_edge_index=edge_index, 
                              neg_edge_index=non_edge_index, num_nodes=n_nodes, 
                              num_features=n_features, num_edges=n_edges, 
                              n_pos=n_pos_edges, n_neg=n_neg_edges)
        data_test = None

    else:
        
        # split both edge and non-edge into train and test
        train_pos_edge_index, test_pos_edge_index = split_edges(data.edge_index, train_frac=train_frac)
        train_neg_edge_index, test_neg_edge_index = split_edges(non_edge_index, train_frac=train_frac)
        n_train_edges = train_pos_edge_index.shape[1]
        n_test_edges = test_pos_edge_index.shape[1]
        
        if hasattr(data, 'y'):
            data_train = Data(x=data.x, y=data.y, edge_index = train_pos_edge_index,
                              pos_edge_index=train_pos_edge_index, 
                              neg_edge_index=train_neg_edge_index, num_nodes=n_nodes, 
                              num_features=n_features, num_edges=n_train_edges, 
                              n_pos=n_train_edges, n_neg=n_train_edges)
            data_test = Data(x=data.x, y=data.y, edge_index = test_pos_edge_index,
                              pos_edge_index=test_pos_edge_index, 
                              neg_edge_index=test_neg_edge_index, num_nodes=n_nodes, 
                              num_features=n_features, num_edges=n_test_edges, 
                              n_pos=n_test_edges, n_neg=n_test_edges)
        else:
            data_train = Data(x=data.x, y=data.y, edge_index = train_pos_edge_index,
                              pos_edge_index=train_pos_edge_index, 
                              neg_edge_index=train_neg_edge_index, num_nodes=n_nodes, 
                              num_features=n_features, num_edges=n_train_edges, 
                              n_pos=n_train_edges, n_neg=n_train_edges)
            data_test = Data(x=data.x, y=data.y, edge_index = test_pos_edge_index,
                              pos_edge_index=test_pos_edge_index, 
                              neg_edge_index=test_neg_edge_index, num_nodes=n_nodes, 
                              num_features=n_features, num_edges=n_test_edges, 
                              n_pos=n_test_edges, n_neg=n_test_edges)
            
    return(data_train, data_test)

def apply_pca(x, pca_preserve):
    if pca_preserve<1.0:
        pca = PCA(n_components=pca_preserve)  # Preserve specific amount of variance
        x = pca.fit_transform(x).astype('float32')
    return(x)
    