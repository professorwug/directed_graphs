"""
Instructions:
To run the script, you'll have to install pytorch (preferably in a virtual environment via conda).
"""
import torch
import warnings
from torch import nn
import math
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Move computations to GPU if available
if torch.__version__[:4] == '1.13':
	# In Pytorch 1.13, support is introduced for apple's MPS accelerators. 
	# If they are available, we'll use them.
	# But if buggy, you can comment out the line including 'mps' and uncomment the below line to default to cpu 
	# device = torch.device('cpu')
	device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.has_mps else 'cpu')
else:
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class OctahedralFitter(torch.nn.Module):
	def __init__(self,num_points):
		"""
		Instantiate a Flow Embedder object, supplying
		graph: a single torch geometric graph object
		model_space: str, one of ["Cartesian Plane","Poincare Disk","Everything Bagel"]
		flow_strength: when moving with the flow, the cost of motion is the euclidean distance divided by this constant.
		"""
		super(OctahedralFitter, self).__init__()
		print("using device ",device)
		# Hyperparameters for the model
		self.num_points = num_points
		self.triangulation = None
		
		# Model parameters, which are tuned by gradient descent
		# You could add the code to compute random points inside the octahedron
		# Currently it just creates random points in 3d; I'd suggest having it sample the theta and phi randomly
		# The nn.Parameter tells Pytorch to update these numbers via gradient decsent
		self.points = nn.Parameter(torch.rand(self.num_points,3))
		# point generation logic here...
		
	def triangulate(self):
		# You can create helper functions in the model 
		# self.triangulation = ...
				
	def cost(self):
		# Calculate the cost here (area of triangulation?)
		# self.triangulate()
		# area = self.compute_area()
		return area
	
		
	def fit(self,n_steps = 1000):
		# train OctahedralFitter to minimize the cost, then return the found points.
		# This is the basic logic for the training loop. Once you have the loss function, you 
		self.train()
		optim = torch.optim.Adam(self.parameters())
		for step in trange(n_steps):
			optim.zero_grad()
			# compute loss
			loss = self.cost()
			# print("loss is ",loss)
			# compute gradient and step backwards
			loss.backward()
			optim.step()
		print("Exiting training with loss ",loss)
		return self.points
		
			
		
		
# activates when run by shell command `python OctahedralFitter.py`
if __name__ == '__main__':
	OF = OctahedralFitter(n_points = 50):
	fit_points = OF.fit(n_steps=1000)
	fit_points = fit_points.detach().numpy()
	np.savetxt('fit_points.out',fit_points)
	# do stuff with the points here/save them for further analysis
	# can be loaded into a notebook with np.loadtxt
	
	
	
		