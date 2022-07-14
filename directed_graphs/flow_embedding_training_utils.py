# AUTOGENERATED! DO NOT EDIT! File to edit: 05c01a Training Utils for Flow Embedding.ipynb (unless otherwise specified).

__all__ = ['visualize_points', 'device', 'FETrainer']

# Cell
from .multiscale_flow_embedder import compute_grid
device = torch.device("cuda" if torch.has_cuda else "cpu")
def visualize_points(embedded_points, flow_artist, labels = None, device = device):
		# computes grid around points
		# TODO: This might create CUDA errors
		grid = compute_grid(embedded_points.to(device))
		# controls the x and y axes of the plot
		# linspace(min on axis, max on axis, spacing on plot -- large number = more field arrows)
		uv = flow_artist(grid).detach().cpu()
		u = uv[:,0].cpu()
		v = uv[:,1].cpu()
		x = grid.detach().cpu()[:,0]
		y = grid.detach().cpu()[:,1]
		# quiver
		# 	plots a 2D field of arrows
		# 	quiver([X, Y], U, V, [C], **kw);
		# 	X, Y define the arrow locations, U, V define the arrow directions, and C optionally sets the color.
		if labels is not None:
			sc = plt.scatter(embedded_points[:,0].detach().cpu(),embedded_points[:,1].detach().cpu(), c=labels)
			plt.legend()
		else:
			sc = plt.scatter(embedded_points[:,0].detach().cpu(),embedded_points[:,1].detach().cpu())
		plt.suptitle("Flow Embedding")
		plt.quiver(x,y,u,v)
		# Display all open figures.
		plt.show()

# Cell
import torch.nn as nn
import torch
from .multiscale_flow_embedder import MultiscaleDiffusionFlowEmbedder

class FETrainer(nn.Module):
  def __init__(self, X, flows):
    self.vizfiz = [
      visualize_points,
    ]
    self.FE = MultiscaleDiffusionFlowEmbedder(
      X = X,
      flows = flows,
      ts = (1, 2, 4, 8),
      sigma_graph = 0.5,
      flow_strength_graph = 5
    )
    self.epochs_between_visualization = 100
    self.total_epochs = 10000

  def fit(self, X, flows):
    for epoch_num in self.total_epochs // self.epochs_between_visualization:
      FE.fit(n_steps = )

  def visualize(self):


