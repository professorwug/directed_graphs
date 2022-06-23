import torch
import torch_geometric
from torch_geometric.utils import to_networkx
import warnings
from torch import nn
import math
from tqdm import trange
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.has_mps else 'cpu')
class FlowEmbedder(torch.nn.Module):
	def __init__(self, graph, model_space, flow_strength, disconnected_distance_constant = 1000):
		"""
		Instantiate a Flow Embedder object, supplying
		graph: a single torch geometric graph object
		model_space: str, one of ["Cartesian Plane","Poincare Disk","Everything Bagel"]
		flow_strength: when moving with the flow, the cost of motion is the euclidean distance divided by this constant.
		"""
		super(FlowEmbedder, self).__init__()
		self.graph = graph
		self.nnodes = graph.num_nodes
		self.embedding_dimension = 2
		self.flow_strength = flow_strength
		self.ground_truth_distances = torch.empty(self.nnodes, self.nnodes)
		self.disconnected_distance_constant = disconnected_distance_constant # large number we use as the distance between nodes in the graph without a connecting path
		self.degree_polynomial = 1
		self.step_size = 0.1
		self.num_steps = 10

		# Model parameters
		self.embedded_points = nn.Parameter(torch.rand(self.nnodes,2))
		# Flow field
		self.flowfield = nn.Sequential(nn.Linear(2, 10),
		                       nn.Tanh(),
		                       nn.Linear(10, 10),
		                       nn.Tanh(),
		                       nn.Linear(10, 2))
		self.flow_field_parameters = nn.Parameter(torch.randn(self.degree_polynomial + 1))
		self.flow_field_parameters2 = nn.Parameter(torch.randn(self.degree_polynomial + 1))
		
		

		# Set up for fitting		
		self.calculate_shortest_path_distances()

	def calculate_shortest_path_distances(self):
		G_nx = to_networkx(self.graph)
		path_lengths = dict(nx.all_pairs_shortest_path_length(G_nx))
		self.ground_truth_distances = torch.empty(self.nnodes,self.nnodes)
		for i in range(self.nnodes):
			for j in range(self.nnodes):
				try:
					self.ground_truth_distances[i][j] = path_lengths[i][j]
				except KeyError:
					warnings.warn("Graph is not strongly connected. Embedding results are not guaranteed. Consider tweaking the constant for 'disconnected distance'.")
					self.ground_truth_distances[i][j] = 1000
					
	def polynomial_flow_field(self,x,y):
		# compute cross terms of degree n polynomial
		# TODO optimize with vectors
		poly = self.flow_field_parameters[0]
		for i in range(1,self.degree_polynomial+1):
			poly += self.flow_field_parameters[i]*math.comb(self.degree_polynomial,i)*x**(self.degree_polynomial-i)*y**(i)
		return poly
	def polynomial_flow_field2(self, x, y):
		poly = self.flow_field_parameters2[1]
		for i in range(1,self.degree_polynomial+1):
			poly += self.flow_field_parameters2[i]*math.comb(self.degree_polynomial,i)*x**(self.degree_polynomial-i)*y**(i)
		return poly
	
	def cost(self,x1,x2):
		if torch.allclose(x1,x2):
			return 0
		# break down the euclidean path between x1 and x2 into a series of steps of length 10
		euc_distance = torch.linalg.norm(x2-x1)
		num_steps = euc_distance/self.step_size
		steps_to_x2 = [x1+t*0.1*(x2-x1)/euc_distance for t in range(int(num_steps)+1)]
		# evaluate the flow field at each of these points
		cost = 0
		for step in steps_to_x2:
			flow = self.flowfield(step)
			#torch.tensor([self.polynomial_flow_field(step[0], step[1]), self.polynomial_flow_field2(step[0], step[1])])
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

	def fast_compute_embedding_distances(self):
		self.embedding_D = torch.empty(self.nnodes,self.nnodes)
		# Find matrix of delta x
		XX = self.embedded_points.repeat(self.nnodes,1,1)
		XXT = XX.transpose(0,1)
		deltaX = XXT - XX
		# Discretize the line between the points 
		# This is an array with copies of the points
		steps = XXT[:,:,:,None].repeat(1,1,1,self.num_steps)
		# Now add to this the distance travelled in each step
		stepnums = torch.arange(self.num_steps).repeat(self.nnodes,self.nnodes,self.embedding_dimension,1)
		step_distances = deltaX[:,:,:,None].repeat(1,1,1,4) * stepnums
		steps = steps + step_distances
		# Put the features on the end
		steps = steps.transpose(2,3)
		# Evaluate the flows at each step
		flows_per_step = self.flowfield(steps)
		# flows_per_step = flows_per_step.transpose(2,3)
		
	def loss(self):
		# calculate error in embedded points
		self.compute_embedding_distances()
		# print("embedding distances")
		# print(self.embedding_D)
		# print("ground truth distances")
		# print(self.ground_truth_distances)
		loss = torch.linalg.norm(torch.log(1 + self.ground_truth_distances) - torch.log(1 + self.embedding_D))**2
		return loss
		
	def visualize_points(self, ):
		# controls the x and y axes of the plot
		# linspace(min on axis, max on axis, spacing on plot -- large number = more field arrows)
		minx = min(self.embedded_points[:,0].detach().numpy())-1
		maxx = max(self.embedded_points[:,0].detach().numpy())+1
		miny = min(self.embedded_points[:,1].detach().numpy())-1
		maxy = max(self.embedded_points[:,1].detach().numpy())+1
		x, y = np.meshgrid(np.linspace(minx,maxx,20),np.linspace(miny,maxy,20))
		x = torch.tensor(x,dtype=float)
		y = torch.tensor(y,dtype=float)
		xy_t = torch.concat([x[:,:,None],y[:,:,None]],dim=2).float().to(device)
		uv = self.flowfield(xy_t).detach()
		u = uv[:,:,0]
		v = uv[:,:,1]
		""" 
		quiver 
			plots a 2D field of arrows
			quiver([X, Y], U, V, [C], **kw); 
			X, Y define the arrow locations, U, V define the arrow directions, and C optionally sets the color.
		"""
		plt.quiver(x,y,u,v)
		sc = plt.scatter(self.embedded_points[:,0].detach(),self.embedded_points[:,1].detach(), c=list(range(self.nnodes)))
		plt.legend(handles = sc.legend_elements()[0], title="Blobs", labels=list(range(self.nnodes)))
		"""Display all open figures."""
		plt.show()
		
	def fit(self,n_steps = 1000):
		# train Flow Embedder on the provided graph
		self.train()
		optim = torch.optim.Adam(self.parameters())
		for step in trange(n_steps):
			optim.zero_grad()
			# compute loss
			loss = self.loss()
			# print("loss is ",loss)
			# compute gradient and step backwards
			loss.backward()
			optim.step()
		print("Exiting training with loss ",loss)
		return self.embedded_points
		
			
		
		
		
		
	
	
		