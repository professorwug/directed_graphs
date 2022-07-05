# AUTOGENERATED! DO NOT EDIT! File to edit: 03c01c Node2Vec_on_Cyclic_Graphs.ipynb (unless otherwise specified).

__all__ = ['BackwardsNode2Vec', 'plot_multiple_embeddings']

# Cell
from node2vec import Node2Vec

class BackwardsNode2Vec(Node2Vec):
  """
  Node2Vec with added backward edges and backward discount
  The weights on the original edges would be `backward_prob` times
  greater than the weights on the ghost backward edges
  """
  def __init__(self, graph, backward_prob=0.1, **kwargs):
    self.backward_prob = backward_prob
    graph = self._add_backward_edges(graph)
    super().__init__(graph, **kwargs)

  def _add_backward_edges(self, graph):
    edges = set(graph.edges)
    for x, y in edges:
      graph[x][y]["weight"] = 1 - self.backward_prob
    for x, y in edges:
      if (y,x) in edges:
        graph[y][x]["weight"] += self.backward_prob
      else:
        graph.add_edge(y, x, weight=self.backward_prob)
    edges = set(graph.edges)
    for x, y in edges:
      if graph[x][y]["weight"] == 0:
        graph.remove_edge(x, y)
    return graph

  def get_embeddings(self, window=1):
    self.window = window
    if not self.quiet:
      self.print_parameters()
    model = self.fit(window=1)
    num_nodes = self.graph.number_of_nodes()
    emb = model.wv[[str(i) for i in range(num_nodes)]]
    return emb

  def print_parameters(self):
    print(f"backward_prob = {self.backward_prob}, dimension = {self.dimensions}, walk_length = {self.walk_length}, num_walks = {self.num_walks}, window = {self.window}")


# Cell
from itertools import product
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from .backwards import BackwardsNode2Vec
from torch_geometric.utils import to_networkx
from tqdm import tqdm

def plot_multiple_embeddings(data, num_clusters, backward_prob_lst, dimensions_lst, walk_length_lst, p_lst, q_lst, num_walk, window, num_repeat=10):
  num_nodes = data.num_nodes
  G_nx = to_networkx(data, to_undirected=False)
  clusters = np.repeat(list(range(num_clusters)),num_nodes/num_clusters)

  num_param_sets =  len(backward_prob_lst) * len(dimensions_lst) * len(walk_length_lst) * len(p_lst) * len(q_lst)
  fig, axes = plt.subplots(num_param_sets, num_repeat, figsize=(num_repeat*8, num_param_sets*6))

  param_sets = set(product(backward_prob_lst, dimensions_lst, walk_length_lst, p_lst, q_lst))
  counter = 0
  for backward_prob, dimensions, walk_length, p, q in param_sets:
    axes[counter,0].set_ylabel(f"B={backward_prob},D={dimensions},L={walk_length},p={p},q={q}")
    node2vec2_model = BackwardsNode2Vec(G_nx, backward_prob=backward_prob, p=p, q=q, dimensions=dimensions, walk_length=walk_length, num_walks=num_walk, workers=4, quiet=True)
    for i in range(num_repeat):
      print(i)
      emb = node2vec2_model.get_embeddings(window=window)
      if emb.shape[1] > 2:
        pca = PCA(n_components=2)
        emb= pca.fit_transform(emb)
      axes[counter, i].scatter(emb[:,0], emb[:,1], c=clusters, cmap="Dark2")
    counter += 1
  plt.show()