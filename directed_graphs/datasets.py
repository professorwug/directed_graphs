# AUTOGENERATED! DO NOT EDIT! File to edit: 03b01 Testing Backwards Node2vec on DirectedGraphs.ipynb (unless otherwise specified).

__all__ = ['EmailEuNetwork', 'visualize_heatmap', 'SourceSink', 'SmallRandom', 'visualize_graph',
           'DirectedStochasticBlockModel', 'source_graph', 'sink_graph', 'ChainGraph', 'ChainGraph2', 'ChainGraph3',
           'CycleGraph', 'HalfCycleGraph', 'plot_embeddings', 'DirectedStochasticBlockModelHelper',
           'visualize_edge_index']

# Cell
import os
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_gz
from torch_geometric.utils import sort_edge_index

class EmailEuNetwork(InMemoryDataset):
  """"
  email-Eu-core network from Stanford Large Network Dataset Collection

  The network was generated from the email exchanges within a large European research institution.
  Each node represents an individual, and a directional edge from one individual to another represents some email exchanges between them in the specified direction.
  Each individual belongs to exactly one of 42 departments in the institution.
  """
  def __init__(self, transform=None, pre_transform=None):
    super().__init__("./datasets/email_Eu_network", transform, pre_transform)
    self.data, self.slices = torch.load(self.processed_paths[0])

  @property
  def raw_file_names(self):
    return ["email-Eu-core.txt", "email-Eu-core-department-labels.txt"]

  @property
  def processed_file_names(self):
    pre_transformed = '' if self.pre_transform is None else '_pre-transformed'
    return [f"email-Eu-network{pre_transformed}.pt", "never-skip-processing"]

  def download(self):
    for filename in self.raw_file_names:
      download_url(f"https://snap.stanford.edu/data/{filename}.gz", self.raw_dir)
      extract_gz(f"{self.raw_dir}/{filename}.gz", self.processed_dir)
      os.remove(f"{self.raw_dir}/{filename}.gz")

  def process(self):
    # Graph connectivity
    with open(self.raw_paths[0], "r") as f:
      edge_array = [[int(x) for x in line.split()] for line in f.read().splitlines()]
    edge_index = torch.t(torch.tensor(edge_array))
    edge_index = sort_edge_index(edge_index)
    # Ground-truth label
    with open(self.raw_paths[1], "r") as f:
      label_array = [[int(x) for x in line.split()] for line in f.read().splitlines()]
    y = torch.tensor(label_array)
    # Node identity features
    x = torch.eye(y.size(0), dtype=torch.float)
    # Build and save data
    data = Data(x=x, edge_index=edge_index, y=y)
    if self.pre_transform is not None:
      data = self.pre_transform(data)
    self.data, self.slices = self.collate([data])
    torch.save((self.data, self.slices), self.processed_paths[0])

# Cell
import torch
import matplotlib.pyplot as plt
from torch_geometric.utils import to_dense_adj

def visualize_heatmap(edge_index, order_ind=None, cmap="viridis"):
  dense_adj = to_dense_adj(edge_index)[0]
  if order_ind is not None:
    dense_adj = dense_adj[order_ind,:][:,order_ind]
  plt.imshow(dense_adj, cmap=cmap)
  plt.show()

# Cell
import warnings
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import sort_edge_index

class SourceSink(BaseTransform):
  """
  Transform a (directed or undirected) graph into a directed graph
  with a proportion of the nodes with mostly out-edges
  and a porportion of the nodes with mostly in-edges

  Parameters
  ----------
  prob_source : float
      must be between 0 and 1
      Proportion of nodes/communities to turn into source nodes/communities
      (with mostly out-edges)
  prob_sink : float
      must be between 0 and 1
      prob_source and prob_sink must add up to no more than 1
      Proportion of nodes/communities to turn into sink nodes/communities
      (with mostly in-edges)
  adv_prob : float
      must be between 0 and 1
      Probability of in-edges for source nodes and/or out-edges for sink nodes
  remove_prob : float
      must be between 0 and 1
      Probability of removing an in-edge for source nodes and/or out-edges for sink nodes
      1 - remove_prob is the probability of reversing the direction of in-edge for source nodes and/or out-edges for sink nodes
  """
  def __init__(self, prob_source=0.1, prob_sink=0.1, adv_prob=0, remove_prob=0):
    if prob_source + prob_sink > 1:
      warnings.warn("Total probability of source and sink exceeds 1")
      excess = prob_source + prob_sink - 1
      prob_source -= excess/2
      prob_sink -= excess/2
      warnings.warn(f"Adjusted: prob_source = {prob_source}, prob_sink = {prob_sink}")
    self.prob_source = prob_source
    self.prob_sink = prob_sink
    self.adv_prob = adv_prob
    self.remove_prob = remove_prob

  def _has_ground_truth(self, data):
    return data.y is not None and data.y.shape == (data.num_nodes, 2)

  def _wrong_direction(self, labels, sources, sinks, tail, head):
    return (labels[head] in sources and labels[tail] not in sources) \
          or (labels[tail] in sinks and labels[head] not in sinks)

  def __call__(self, data):
    if self._has_ground_truth(data):
      # get ground truth labels
      y = data.y[torch.argsort(data.y[:,0]),:]
      classes = y[:,1].unique()
      # randomly choose source and sink classes
      mask = torch.rand(len(classes))
      source_classes = classes[mask < self.prob_source]
      sink_classes = classes[mask > 1 - self.prob_sink]
      # add source/sink ground-truth label
      y = torch.hstack((y, torch.t(torch.tensor([[1 if c in source_classes else -1 if c in sink_classes else 0 for c in y[:,1]]]))))
      labels = y[:,1]
      sources = source_classes
      sinks = sink_classes
    else:
      warnings.warn("Data has no ground-truth labels")
      # randomly choose source and sink nodes
      nodes = torch.arange(data.num_nodes)
      mask = torch.rand(data.num_nodes)
      source_nodes = nodes[mask < self.prob_source]
      sink_nodes = nodes[mask > 1 - self.prob_sink]
      # add source/sink ground-truth label
      y = torch.tensor([[n, 1 if n in source_nodes else -1 if n in sink_nodes else 0] for n in nodes])
      labels = nodes
      sources = source_nodes
      sinks = sink_nodes

    # correct improper edges
    edge_array = []
    for e in range(data.num_edges):
      tail, head = data.edge_index[:,e]
      if self._wrong_direction(labels, sources, sinks, tail, head) and torch.rand(1)[0] > self.adv_prob:
        if torch.rand(1)[0] < self.remove_prob: # remove the improper edge
          continue
        else: # reverse the improper edge
          edge_array.append([head, tail])
      else: # keep proper edge
        edge_array.append([tail, head])
    edge_index = torch.t(torch.tensor(edge_array))
    data.edge_index = sort_edge_index(edge_index)
    data.y = y
    return data.coalesce()

# Cell
import warnings
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_sparse import SparseTensor
from torch_geometric.utils import remove_self_loops

class SmallRandom(InMemoryDataset):
  def __init__(self, num_nodes=5, prob_edge=0.2, transform=None, pre_transform=None):
    super().__init__(".", transform, pre_transform)

    if num_nodes > 300:
      num_nodes = 300
      warnings.warn(f"Number of nodes is too large for SmallRandom dataset. Reset num_nodes =  {num_nodes}")

    dense_adj = (torch.rand((num_nodes, num_nodes)) < prob_edge).int()
    sparse_adj = SparseTensor.from_dense(dense_adj)
    row, col, _ = sparse_adj.coo()
    edge_index, _ = remove_self_loops(torch.stack([row, col]))

    x = torch.eye(num_nodes, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    if self.pre_transform is not None:
      data = self.pre_transform(data)
    self.data, self.slices = self.collate([data])


# Cell
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
def visualize_graph(data):
  G = to_networkx(data, to_undirected=False)
  nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), arrowsize=20, node_color="#adade0")
  plt.show()

# Cell
import warnings
import torch
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
from torch_sparse import SparseTensor
from torch_geometric.utils import remove_self_loops

class DirectedStochasticBlockModel(InMemoryDataset):
  def __init__(self, num_nodes, num_clusters, aij, bij, transform=None):
    """Directed SBM

    Parameters
    ----------
    num_nodes : int
        _description_
    num_clusters : int
        must evenly divide num_nodes
    aij : num_nodes x num_nodes ndarray
        Probabilities of (undirected) connection between clusters i and j.
        Must be symmetric.
    bij : num_nodes x num_nodes ndarray
        Probabilities with which the edges made via aij are converted to directed edges.
        bij + bji = 1
    transform : _type_, optional
        _description_, by default None
    """
    super().__init__(".", transform)
    cluster = np.repeat(list(range(num_clusters)),num_nodes/num_clusters)
    # print(cluster)
    rand_matrix = torch.rand((num_nodes, num_nodes))
    dense_adj = torch.empty((num_nodes,num_nodes))
    ## Draw inter-cluster undirected edges
    # inefficiently traverse the dense matrix, converting probabilities
    for i in range(num_nodes):
      for j in range(i, num_nodes):
        # print("cluster of i is",cluster[i],"cluster of j is",cluster[j])
        dense_adj[i,j] = 1 if rand_matrix[i,j] < aij[cluster[i],cluster[j]] else 0
        dense_adj[j,i] = dense_adj[i,j] # it's symmetric
    ## Convert undirected edges to directed edges
    rand_matrix = torch.rand((num_nodes,num_nodes))
    for i in range(num_nodes):
      for j in range(i,num_nodes):
        # if an edge exists, assign it a direction
        if dense_adj[i,j] == 1:
          # print('adding direction')
          if rand_matrix[i,j] < bij[cluster[i],cluster[j]]:
            dense_adj[i,j] = 1
            dense_adj[j,i] = 0
          else:
            dense_adj[i,j] = 0
            dense_adj[j,i] = 1
    #
    # print("The adjacency is currently symmetric ",torch.allclose(dense_adj,dense_adj.T))

    sparse_adj = SparseTensor.from_dense(dense_adj)
    row, col, _ = sparse_adj.coo()
    edge_index, _ = remove_self_loops(torch.stack([row, col]))

    x = torch.eye(num_nodes, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    self.data, self.slices = self.collate([data])


# Cell
def source_graph(n_points = 700, num_clusters=7):
  # we'll start with 7 clusters; six on the outside, one on the inside
  aij = np.zeros((num_clusters, num_clusters))
  aij[0,:] = 0.9
  aij[:,0] = 0.9
  np.fill_diagonal(aij, 0.9)
  # aij = np.array(
  #   [[0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
  #    [0.9, 0.9, 0, 0, 0, 0, 0],
  #    [0.9, 0, 0.9, 0, 0, 0, 0],
  #    [0.9, 0, 0, 0.9, 0, 0, 0],
  #    [0.9, 0, 0, 0, 0.9, 0, 0],
  #    [0.9, 0, 0, 0, 0, 0.9, 0],
  #    [0.9, 0, 0, 0, 0, 0, 0.9]]
  # )
  bij = np.zeros((num_clusters, num_clusters))
  bij[0,:] = 1.0
  bij[:,0] = 0.0
  np.fill_diagonal(bij, 0.5)
  # bij = np.array(
  #   [[0.5, 1, 1, 1, 1, 1, 1],
  #    [0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
  #    [0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
  #    [0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
  #    [0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
  #    [0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
  #    [0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]
  # )
  dataset = DirectedStochasticBlockModel(num_nodes=n_points, num_clusters=num_clusters, aij = aij, bij = bij)
  return dataset

# Cell
def sink_graph(n_points = 700, num_clusters=7):
  # we'll start with 7 clusters; six on the outside, one on the inside
  aij = np.zeros((num_clusters, num_clusters))
  aij[0,:] = 0.9
  aij[:,0] = 0.9
  np.fill_diagonal(aij, 0.9)
  # aij = np.array(
  #   [[0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
  #    [0.9, 0.9, 0, 0, 0, 0, 0],
  #    [0.9, 0, 0.9, 0, 0, 0, 0],
  #    [0.9, 0, 0, 0.9, 0, 0, 0],
  #    [0.9, 0, 0, 0, 0.9, 0, 0],
  #    [0.9, 0, 0, 0, 0, 0.9, 0],
  #    [0.9, 0, 0, 0, 0, 0, 0.9]]
  # )
  bij = np.zeros((num_clusters, num_clusters))
  bij[0,:] = 0.0
  bij[:,0] = 1.0
  np.fill_diagonal(bij, 0.5)
  # bij = np.array(
  #   [[0.5,0, 0, 0, 0, 0, 0],
  #    [1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
  #    [1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
  #    [1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
  #    [1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
  #    [1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
  #    [1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]
  # )
  dataset = DirectedStochasticBlockModel(num_nodes=n_points, num_clusters=num_clusters, aij = aij, bij = bij)
  return dataset

# Cell
class ChainGraph(InMemoryDataset):
  def __init__(self,num_nodes = 2,transform=None):
    super().__init__(".", transform)
    dense_adj = torch.tensor(
      [[0,1],[0,0]]
    )
    sparse_adj = SparseTensor.from_dense(dense_adj)
    row, col, _ = sparse_adj.coo()
    edge_index, _ = remove_self_loops(torch.stack([row, col]))

    x = torch.eye(num_nodes, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    self.data, self.slices = self.collate([data])

# Cell
class ChainGraph2(InMemoryDataset):
  def __init__(self,num_nodes = 3,transform=None):
    super().__init__(".", transform)
    dense_adj = torch.tensor(
      [[0,1,0],
      [0,0,1],
      [0,0,0]]
    )
    sparse_adj = SparseTensor.from_dense(dense_adj)
    row, col, _ = sparse_adj.coo()
    edge_index, _ = remove_self_loops(torch.stack([row, col]))

    x = torch.eye(num_nodes, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    self.data, self.slices = self.collate([data])

# Cell
class ChainGraph3(InMemoryDataset):
  def __init__(self,num_nodes = 3,transform=None):
    super().__init__(".", transform)
    dense_adj = torch.tensor(
      [[0,1,0],
      [0,0,0],
      [0,1,0]]
    )
    sparse_adj = SparseTensor.from_dense(dense_adj)
    row, col, _ = sparse_adj.coo()
    edge_index, _ = remove_self_loops(torch.stack([row, col]))

    x = torch.eye(num_nodes, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    self.data, self.slices = self.collate([data])

# Cell
class CycleGraph(InMemoryDataset):
  def __init__(self,num_nodes = 3,transform=None):
    super().__init__(".", transform)
    dense_adj = torch.tensor(
      [[0,1,0],
      [0,0,1],
      [1,0,0]]
    )
    sparse_adj = SparseTensor.from_dense(dense_adj)
    row, col, _ = sparse_adj.coo()
    edge_index, _ = remove_self_loops(torch.stack([row, col]))

    x = torch.eye(num_nodes, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    self.data, self.slices = self.collate([data])

# Cell
class HalfCycleGraph(InMemoryDataset):
  def __init__(self,num_nodes = 3,transform=None):
    super().__init__(".", transform)
    dense_adj = torch.tensor(
      [[0,1,0],
      [1,0,1],
      [1,0,0]]
    )
    sparse_adj = SparseTensor.from_dense(dense_adj)
    row, col, _ = sparse_adj.coo()
    edge_index, _ = remove_self_loops(torch.stack([row, col]))

    x = torch.eye(num_nodes, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    self.data, self.slices = self.collate([data])

# Comes from 03a Node2Vec_with_Backwards_Connection.ipynb, cell
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def plot_embeddings(emb, num_nodes, num_clusters, title=""):
  clusters = np.repeat(list(range(num_clusters)),num_nodes/num_clusters)
  if emb.shape[1] > 2:
    pca = PCA(n_components=2)
    emb= pca.fit_transform(emb)
  plt.figure()
  sc = plt.scatter(emb[:,0], emb[:,1], c=clusters, cmap="Dark2")
  plt.legend(handles = sc.legend_elements()[0], title="Clusters", labels=list(range(num_clusters)))
  plt.suptitle(title)
  plt.show()

# Comes from 03c node2vec graph reversal walk.ipynb, cell
from .datasets import DirectedStochasticBlockModel
from typing import *
import numpy as np
def DirectedStochasticBlockModelHelper(num_nodes: int, num_clusters: int, edge_index: np.ndarray, undir_prob = [0.4], dir_prob = [0.9]):
    """Directed SBM Helper

    Parameters
    ----------
    num_nodes : int
        _description_
    num_clusters : int
        must evenly divide num_nodes
    edge_index : np.ndarray
        see edge_index as in https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html
    undir_prob : List, optional
        Specifies probabilities of (undirected) connection between clusters i and j.  The default probability is 0.4.
    dir_prob : List, optional
        Specifies probabilities with which the edges made via aij are converted to directed edges. The default probability for bij is 0.9 where bij + bji = 1.

    Returns
    ----------
    dataset : DirectedStochasticBlockModel
        See (class) DirectedStochasticBlockModel
    """
    # need to include warnings about the corresponding sizes
    # of edge_index and undir_prob
    # maxN = edge_index.max() + 1

    # construct aij
    aij = np.zeros((num_clusters, num_clusters))

    # default probability of connections across
    # and within specified clusters is 0.4
    if len(undir_prob) == 1:
        np.fill_diagonal(aij, undir_prob[0])
        for x, y in zip(edge_index[0], edge_index[1]):
            aij[x,y] = undir_prob[0]
            aij[y,x] = undir_prob[0]

    # construct bij
    bij = np.zeros((num_clusters, num_clusters))
    np.fill_diagonal(bij, 0.5)
    for x, y in zip(edge_index[0], edge_index[1]):
        bij[x,y] = dir_prob[0]
        bij[y,x] = 1 - dir_prob[0]
    return DirectedStochasticBlockModel(num_nodes, num_clusters, aij=aij, bij=bij)

# Comes from 03c01a Node2Vec_on_MultiTree.ipynb, cell
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import torch
import networkx as nx

def visualize_edge_index(data, num_clusters=7, pos=None):
  num_nodes = data.num_nodes
  nodes_per_cluster = num_nodes//num_clusters
  A = torch.sparse_coo_tensor(data.edge_index,torch.ones(data.edge_index.shape[1])).to_dense()
  row = []
  col = []
  for i in range(num_clusters):
    for j in range(i+1,num_clusters):
      ij_cnt = A[i*nodes_per_cluster:(i+1)*nodes_per_cluster,j*nodes_per_cluster:(j+1)*nodes_per_cluster].sum()
      ji_cnt = A[j*nodes_per_cluster:(j+1)*nodes_per_cluster,i*nodes_per_cluster:(i+1)*nodes_per_cluster].sum()
      if ij_cnt == 0 and ji_cnt == 0:
        continue
      if ij_cnt > ji_cnt:
        row.append(i)
        col.append(j)
      else:
        row.append(j)
        col.append(i)
  edge_index = torch.tensor([row, col])
  cluster_data = Data(x=torch.eye(num_clusters), edge_index=edge_index)
  G = to_networkx(cluster_data, to_undirected=False)
  if pos is None:
    pos = nx.planar_layout(G)
  nx.draw_networkx(G, pos=pos, arrowsize=20, node_color=list(range(num_clusters)), cmap=plt.cm.Dark2, font_color="whitesmoke")
  plt.show()