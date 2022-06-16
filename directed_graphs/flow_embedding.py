import torch
import torch_geometric
from torch_geometric.utils import to_networkx
import warnings
from torch import nn
import math
from tqdm import trange
import networkx as nx

class FlowEmbedder(torch.nn.Module):
	def __init__(self, graph, model_space, flow_strength):
		"""
		Instantiate a Flow Embedder object, supplying
		graph: a single torch geometric graph object
		model_space: str, one of ["Cartesian Plane","Poincare Disk","Everything Bagel"]
		flow_strength: when moving with the flow, the cost of motion is the euclidean distance divided by this constant.
		"""
		super(FlowEmbedder, self).__init__()
		self.graph = graph
		self.nnodes = graph.num_nodes
		self.flow_strength = flow_strength
		self.ground_truth_distances = torch.empty(self.nnodes, self.nnodes)
		self.disconnected_distance_constant = 1000 # large number we use as the distance between nodes in the graph without a connecting path
		self.degree_polynomial = 3
		self.step_size = 0.1

		# Model parameters
		self.embedded_points = nn.Parameter(torch.rand(self.nnodes,2))
		# Flow field
		self.flow_field_parameters = nn.Parameter(torch.randn(self.degree_polynomial,2))
		# Set up for fitting		
		self.calculate_shortest_path_distances()

	def calculate_shortest_path_distances(self):
		G_nx = to_networkx(self.graph)
		path_lengths = dict(nx.all_pairs_shortest_path_length(G_nx))
		path_distance_array = torch.empty(self.nnodes,self.nnodes)
		for i in range(self.nnodes):
			for j in range(self.nnodes):
				try:
					path_distance_array[i][j] = path_lengths[i][j]
				except KeyError:
					warnings.warn("Graph is not strongly connected. Embedding results are not guaranteed. Consider tweaking the constant for 'disconnected distance'.")
					path_distance_array[i][j] = 1000
	def polynomial_flow_field(self,x,y):
		# compute cross terms of degree n polynomial
		# TODO optimize with vectors
		poly = 0
		for i in range(self.degree_polynomial):
			poly += self.flow_field_parameters[i]*math.comb(self.degree_polynomial,i)*x**(self.degree_polynomial-i)*y**(i)
		return poly
	def cost(self,x1,x2):
		# break down the euclidean path between x1 and x2 into a series of steps of length 10
		euc_distance = torch.linalg.norm(x2-x1)
		num_steps = euc_distance/self.step_size
		steps_to_x2 = [x1+t*0.1*(x2-x1)/euc_distance for t in range(int(num_steps)+1)]
		# evaluate the flow field at each of these points
		cost = 0
		for step in steps_to_x2:
			flow = self.polynomial_flow_field(step[0], step[1])
			cost_normalizer = self.cost_normalizer(torch.dot(flow,(x2-x1)))
			cost += self.step_size / cost_normalizer
		return cost
	def cost_normalizer(self,dot):
		return self.flow_strength*torch.exp(dot)/((self.flow_strength - 1) + torch.exp(dot))
		
	def compute_embedding_distances(self):
		self.embedding_D = torch.empty(self.nnodes,self.nnodes)
		for i in range(self.nnodes):
			for j in range(self.nnodes):
				self.embedding_D[i][j] = self.cost(self.embedded_points[i],self.embedded_points[j])
		
	def loss(self):
		# calculate error in embedded points
		self.embedding_distance_matrix()
		loss = (torch.log(1 + self.ground_truth_distances) - torch.log(1 + self.embedding_D))**2
		return loss
		
	def fit(self,n_steps = 1000):
		# train Flow Embedder on the provided graph
		self.train()
		optim = torch.optim.Adam(self.parameters)
		for step in trange(n_steps):
			optim.zero_grad()
			# compute distances in flow field
			self.compute_embedding_distances()
			# compute loss
			loss = self.loss()
			print("loss is ",loss)
			# compute gradient and step backwards
			loss.backward()
			optim.step()
		print("Exiting training with loss ",loss)
		return self.embedded_points
		
			
		
		
		
		
	
	
		