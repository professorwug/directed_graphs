# AUTOGENERATED! DO NOT EDIT! File to edit: 05b03 Testing the Flow-Affinity Matrix.ipynb (unless otherwise specified).

__all__ = ['EmailEuNetwork', 'visualize_heatmap', 'SourceSink', 'SmallRandom', 'visualize_graph',
           'DirectedStochasticBlockModel', 'source_graph', 'sink_graph', 'ChainGraph', 'ChainGraph2', 'ChainGraph3',
           'CycleGraph', 'HalfCycleGraph', 'xy_tilt', 'directed_circle', 'plot_directed_2d', 'plot_origin_3d',
           'plot_directed_3d', 'directed_prism', 'directed_cylinder', 'directed_spiral', 'directed_swiss_roll',
           'directed_spiral_uniform', 'directed_swiss_roll_uniform', 'angle_x', 'whirlpool',
           'rejection_sample_for_torus', 'torus_with_flow', 'directed_sin_branch', 'static_clusters', 'plot_embeddings',
           'DirectedStochasticBlockModelHelper', 'visualize_edge_index', 'affinity_grid_search']

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

# Cell
def xy_tilt(X, flow, labels, xtilt=0, ytilt=0):
  xrotate = np.array([[1,              0,             0],
                      [0,  np.cos(xtilt), np.sin(xtilt)],
                      [0, -np.sin(xtilt), np.cos(xtilt)]])
  X = X @ xrotate
  flow = flow @ xrotate
  yrotate = np.array([[np.cos(ytilt), 0, -np.sin(ytilt)],
                      [            0, 1,              0],
                      [np.sin(ytilt), 0,  np.cos(ytilt)]])
  X = X @ yrotate
  flow = flow @ yrotate
  return X, flow, labels

# Cell
def directed_circle(num_nodes=100, radius=1, xtilt=0, ytilt=0):
  # sample random angles between 0 and 2pi
  thetas = np.random.uniform(0, 2*np.pi, num_nodes)
  thetas = np.sort(thetas)
  labels = thetas
  # calculate x and y coordinates
  x = np.cos(thetas) * radius
  y = np.sin(thetas) * radius
  z = np.zeros(num_nodes)
  X = np.column_stack((x, y, z))
  # calculate the angle of the tangent
  alphas = thetas + np.pi/2
  # calculate the coordinates of the tangent
  u = np.cos(alphas)
  v = np.sin(alphas)
  w = np.zeros(num_nodes)
  flow = np.column_stack((u, v, w))
  # tilt
  X, flow, labels = xy_tilt(X, flow, labels, xtilt=0, ytilt=0)
  return X, flow, labels

# Cell
import matplotlib.pyplot as plt

def plot_directed_2d(X, flow, labels, mask_prob=0.5):
  num_nodes = X.shape[0]
  fig = plt.figure()
  ax = fig.add_subplot()
  ax.scatter(X[:,0], X[:,1], marker=".", c=labels)
  mask = np.random.rand(num_nodes) > mask_prob
  ax.quiver(X[mask,0], X[mask,1], flow[mask,0], flow[mask,1], alpha=0.1)
  ax.set_aspect("equal")
  plt.show()

# Cell
def plot_origin_3d(ax, lim):
  ax.plot(lim,[0,0],[0,0], color="k", alpha=0.5)
  ax.plot([0,0],lim,[0,0], color="k", alpha=0.5)
  ax.plot([0,0],[0,0],lim, color="k", alpha=0.5)

def plot_directed_3d(X, flow, labels, mask_prob=0.5):
  num_nodes = X.shape[0]
  colors = plt.cm.viridis(labels/(2*np.pi))
  mask = np.random.rand(num_nodes) > mask_prob
  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')
  plot_origin_3d(ax, lim=[-1,1])
  ax.scatter(X[:,0], X[:,1], X[:,2], marker=".", c=labels)
  ax.quiver(X[mask,0], X[mask,1], X[mask,2], flow[mask,0], flow[mask,1], flow[mask,2], alpha=0.1, length=0.5)
  plt.show()

# Cell
def directed_prism(X, flow, labels, height=10):
  num_nodes = X.shape[0]
  z_noise = np.random.uniform(-height/2, height/2, num_nodes)
  X[:,2] = X[:,2] + z_noise
  return X, flow, labels

# Cell
def directed_cylinder(num_nodes=1000, radius=1, height=10, xtilt=0, ytilt=0):
  X, flow, labels = directed_circle(num_nodes, radius, xtilt, ytilt)
  X, flow, labels = directed_prism(X, flow, labels, height)
  return X, flow, labels

# Cell
def directed_spiral(num_nodes=100, num_spirals=2.5, radius=1, xtilt=0, ytilt=0):
  # sample random angles between 0 and num_spirals * 2pi
  thetas = np.random.uniform(0, num_spirals*2*np.pi, num_nodes)
  thetas = np.sort(thetas)
  labels = thetas
  # calculate x and y coordinates
  x = np.cos(thetas) * thetas * radius
  y = np.sin(thetas) * thetas * radius
  z = np.zeros(num_nodes)
  X = np.column_stack((x, y, z))
  # calculate the angle of the tangent
  alphas = thetas + np.pi/2
  # calculate the coordinates of the tangent
  u = np.cos(alphas) * thetas
  v = np.sin(alphas) * thetas
  w = np.zeros(num_nodes)
  flow = np.column_stack((u, v, w))
  # tilt
  X, flow, labels = xy_tilt(X, flow, labels, xtilt, ytilt)
  return X, flow, labels

# Cell
def directed_swiss_roll(num_nodes=1000, num_spirals=2.5, radius=1, height=10, xtilt=0, ytilt=0):
  X, flow, labels = directed_spiral(num_nodes, num_spirals, radius, xtilt, ytilt)
  X, flow, labels = directed_prism(X, flow, labels, height)
  return X, flow, labels

# Cell
def directed_spiral_uniform(num_nodes=100, num_spirals=2.5, radius=1, xtilt=0, ytilt=0):
  # sample random angles between 0 and num_spirals * 2pi
  t1 = np.random.uniform(0, num_spirals*2*np.pi, num_nodes)
  t2 = np.random.uniform(0, num_spirals*2*np.pi, num_nodes)
  thetas = np.maximum(t1, t2)
  thetas = np.sort(thetas)
  labels = thetas
  # calculate x and y coordinates
  x = np.cos(thetas) * thetas * radius
  y = np.sin(thetas) * thetas * radius
  z = np.zeros(num_nodes)
  X = np.column_stack((x, y, z))
  # calculate the angle of the tangent
  alphas = thetas + np.pi/2
  # calculate the coordinates of the tangent
  u = np.cos(alphas)
  v = np.sin(alphas)
  w = np.zeros(num_nodes)
  flow = np.column_stack((u, v, w))
  # tilt
  X, flow, labels = xy_tilt(X, flow, labels, xtilt, ytilt)
  return X, flow, labels

# Cell
def directed_swiss_roll_uniform(num_nodes=1000, num_spirals=2.5, radius=1, height=10, xtilt=0, ytilt=0):
  X, flow, labels = directed_spiral_uniform(num_nodes, num_spirals, radius, xtilt, ytilt)
  X, flow, labels = directed_prism(X, flow, labels, height)
  return X, flow, labels

# Cell
def angle_x(X):
  """Returns angle in [0, 2pi] corresponding to each point X"""
  X_complex = X[:,0] + np.array([1j])*X[:,1]
  return np.angle(X_complex)

# Cell
def whirlpool(X):
  """Generates a whirlpool for flow assignment. Works in both 2d and 3d space.

  Parameters
  ----------
  X : ndarray
      input data, 2d or 3d
  """
  # convert X into angles theta, where 0,0 is 0, and 0,1 is pi/2
  X_angles = angle_x(X)
  # create flows
  flow_x = np.sin(2*np.pi - X_angles)
  flow_y = np.cos(2*np.pi - X_angles)
  output = np.column_stack([flow_x,flow_y])
  if X.shape[1] == 3:
    # data is 3d
    flow_z = np.zeros(X.shape[0])
    output = np.column_stack([output,flow_z])
  return output


# Cell
def rejection_sample_for_torus(n, r, R):
    # Rejection sampling torus method [Sampling from a torus (Revolutions)](https://blog.revolutionanalytics.com/2014/02/sampling-from-a-torus.html)
    xvec = np.random.random(n) * 2 * np.pi
    yvec = np.random.random(n) * (1/np.pi)
    fx = (1 + (r/R)*np.cos(xvec)) / (2*np.pi)
    return xvec[yvec < fx]

def torus_with_flow(n=2000, c=2, a=1, flow_type = 'whirlpool', noise=None, seed=None, use_guide_points = False):
    """
    Sample `n` data points on a torus. Modified from [tadasets.shapes — TaDAsets 0.1.0 documentation](https://tadasets.scikit-tda.org/en/latest/_modules/tadasets/shapes.html#torus)
    Uses rejection sampling.

    In addition to the points, returns a "flow" vector at each point.

    Parameters
    -----------
    n : int
        Number of data points in shape.
    c : float
        Distance from center to center of tube.
    a : float
        Radius of tube.
    flow_type, in ['whirlpool']

    ambient : int, default=None
        Embed the torus into a space with ambient dimension equal to `ambient`. The torus is randomly rotated in this high dimensional space.
    seed : int, default=None
        Seed for random state.
    """

    assert a <= c, "That's not a torus"

    np.random.seed(seed)
    theta = rejection_sample_for_torus(n-2, a, c)
    phi = np.random.random((len(theta))) * 2.0 * np.pi

    data = np.zeros((len(theta), 3))
    data[:, 0] = (c + a * np.cos(theta)) * np.cos(phi)
    data[:, 1] = (c + a * np.cos(theta)) * np.sin(phi)
    data[:, 2] = a * np.sin(theta)

    if use_guide_points:
        data = np.vstack([[[0,-c-a,0],[0,c-a,0],[0,c,a]],data])

    if noise:
        data += noise * np.random.randn(*data.shape)

    if flow_type == 'whirlpool':
        flows = whirlpool(data)
    else:
        raise NotImplementedError
    # compute curvature of sampled torus
    ks = 8*np.cos(theta)/(5 + np.cos(theta))

    return data, flows

# Cell
def directed_sin_branch(num_nodes=1000, xscale=1, yscale=1, sigma=0.25):
  num_nodes_per_branch = num_nodes//3
  # root
  x_root = np.random.uniform(-xscale*np.pi*0.84, 0, num_nodes - 2*num_nodes_per_branch)
  x_root = np.sort(x_root)
  y_root = np.sinh(x_root / xscale) * yscale
  v_root = np.cosh(x_root / xscale) / xscale * yscale
  # branch 1
  x_branch1 = np.random.uniform(0, xscale*np.pi*0.84, num_nodes_per_branch)
  x_branch1 = np.sort(x_branch1)
  y_branch1 = np.sinh(x_branch1 / xscale) * yscale
  v_branch1 = np.cosh(x_branch1 / xscale) / xscale * yscale
  # branch 2
  x_branch2 = np.random.uniform(0, xscale*2*np.pi, num_nodes_per_branch)
  x_branch2 = np.sort(x_branch2)
  y_branch2 = np.sin(x_branch2 / xscale) * yscale
  v_branch2 = np.cos(x_branch2 / xscale) / xscale * yscale
  # stack
  x = np.concatenate((x_branch1, x_branch2, x_root))
  y = np.concatenate((y_branch1, y_branch2, y_root)) + np.random.normal(loc=0, scale=sigma, size=num_nodes)
  v = np.concatenate((v_branch1, v_branch2, v_root))
  z = np.zeros(num_nodes)
  X = np.column_stack((x, y, z))
  u = np.ones(num_nodes)
  w = np.zeros(num_nodes)
  flow = np.column_stack((u, v, w))
  # labels
  labels = np.concatenate((x_branch1 - np.pi*3, x_branch2, x_root + np.pi*3))
  return X, flow, labels


# Cell
def static_clusters(num_nodes=250, num_clusters=5, radius=1, sigma=0.2):
  thetas = np.repeat([2*np.pi*i/num_clusters for i in range(num_clusters)], num_nodes//num_clusters)
  x = np.cos(thetas) * radius + np.random.normal(loc=0, scale=sigma, size=num_nodes)
  y = np.sin(thetas) * radius + np.random.normal(loc=0, scale=sigma, size=num_nodes)
  z = np.zeros(num_nodes)
  X = np.column_stack((x, y, z))
  flow = np.zeros(X.shape)
  return X, flow, thetas

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

# Cell
import matplotlib.pyplot as plt
from .diffusion_flow_embedding import affinity_matrix_from_pointset_to_pointset
def affinity_grid_search(X,flow,sigmas, flow_strengths):
  fig, axs = plt.subplots(len(sigmas),len(flow_strengths), figsize=(len(flow_strengths*6),len(sigmas)*6))
  X = torch.tensor(X)
  flow = torch.tensor(flow)
  for i, s in enumerate(sigmas):
    for j, f in enumerate(flow_strengths):
      A = affinity_matrix_from_pointset_to_pointset(X, X, flow, sigma=s, flow_strength=f)
      A = A.numpy()
      axs[i][j].set_title(f"$\sigma = {s}$ and $f={f}$")
      axs[i][j].imshow(A)
  plt.show()