# AUTOGENERATED! DO NOT EDIT! File to edit: 22_Small_Random_Directed_Graphs.ipynb (unless otherwise specified).

__all__ = ['EmailEuNetwork', 'SourceSink', 'SmallRandom']

# Cell
import os
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_gz
from torch_geometric.utils import sort_edge_index

class EmailEuNetwork(InMemoryDataset):
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
import warnings
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import sort_edge_index

class SourceSink(BaseTransform):
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
      warnings.warn("Data has no groud-truth labels")
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
    return data

# Cell
import warnings
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_sparse import SparseTensor
from torch_geometric.utils import remove_self_loops

class SmallRandom(InMemoryDataset):
  def __init__(self, num_nodes=5, prob_edge=0.2, transform=None, pre_transform=None):
    super().__init__(".", transform, pre_transform)

    if num_nodes > 30:
      num_nodes = 30
      warnings.warn("Number of nodes is too large for SmallRandom dataset. Reset num_nodes = ", num_nodes)

    dense_adj = (torch.rand((num_nodes, num_nodes)) < prob_edge).int()
    sparse_adj = SparseTensor.from_dense(dense_adj)
    row, col, _ = sparse_adj.coo()
    edge_index, _ = remove_self_loops(torch.stack([row, col]))

    x = torch.eye(num_nodes, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    if self.pre_transform is not None:
      data = self.pre_transform(data)
    self.data, self.slices = self.collate([data])
