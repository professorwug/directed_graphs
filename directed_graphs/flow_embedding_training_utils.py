# AUTOGENERATED! DO NOT EDIT! File to edit: 05c01a Training Utils for Flow Embedding.ipynb (unless otherwise specified).

__all__ = ['visualize_points', 'device', 'save_embedding_visualization', 'collate_loss', 'FETrainer', 'device']

# Cell
import torch
from .multiscale_flow_embedder import compute_grid
device = torch.device("cuda" if torch.has_cuda else "cpu")
def visualize_points(embedded_points, flow_artist, labels = None, device = device, title = "Flow Embedding", save = False, **kwargs):
		# computes grid around points
		# TODO: This might create CUDA errors
		grid = compute_grid(embedded_points.to(device)).to(device)
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
# 			plt.legend()
		else:
			sc = plt.scatter(embedded_points[:,0].detach().cpu(),embedded_points[:,1].detach().cpu())
		plt.suptitle("Flow Embedding")
		plt.quiver(x,y,u,v)
		# Display all open figures.
		if save:
			plt.savefig(f"visualizations/{title}.jpg")
		else:
			plt.show()
		plt.close()

# Cell
def save_embedding_visualization(embedded_points, flow_artist, labels = None, device = device, title = "Flow Embedding", **kwargs):
  visualize_points(embedded_points=embedded_points, flow_artist = flow_artist, labels = labels, device = device, title = title, save=True)

# Cell
def collate_loss(provided_losses, prior_losses = None, loss_type = "total", ):
		# diffusion_loss,reconstruction_loss, smoothness_loss
		x = []
		k = ""
		if prior_losses is None:
			# if there are no prior losses, initialize a new dictionary to store these
			prior_losses = {}
			for key in provided_losses.keys():
				prior_losses[key] = []
				# k = key
			prior_losses["total"] = []
		for i in range(max([len(provided_losses["diffusion"]), len(provided_losses["smoothness"]),len(provided_losses["diffusion map regularization"]),len(provided_losses["flow cosine loss"])])):
			x.append(i)
			for key in provided_losses.keys():
				try:
					prior_losses[key].append(provided_losses[key][i].detach().cpu().numpy())
				except:
					prior_losses[key].append(0)
		return prior_losses

# Cell
import torch.nn as nn
import torch
import time
import datetime
from .multiscale_flow_embedder import MultiscaleDiffusionFlowEmbedder
from tqdm import trange
import glob
from PIL import Image
import os
import ipywidgets as widgets
import base64
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FETrainer(object):
  def __init__(self, X, flows, labels, device = device):
    #super(FETrainer, self).__init__()
    self.vizfiz = [
      save_embedding_visualization,
    ]
    self.FE = MultiscaleDiffusionFlowEmbedder(
      X = X,
      flows = flows,
      ts = (1, 2, 4, 8),
      sigma_graph = 0.5,
      flow_strength_graph = 5,
      device = device,
      use_embedding_grid = False,
    ).to(device)
    self.losses = None
    self.labels = labels
    self.title = "Vanilla MFE"
    self.epochs_between_visualization = 100
    self.total_epochs = 10000
    self.timestamp = datetime.datetime.now().isoformat()
    os.mkdir(f'visualizations/{self.timestamp}')

  def fit(self):
    num_training_runs = self.total_epochs // self.epochs_between_visualization
    for epoch_num in trange(num_training_runs):
      start = time.time()
      emb_X, flow_artist, losses = self.FE.fit(n_steps = self.epochs_between_visualization)
      stop = time.time()
      title = f"{self.timestamp}/{self.title} Epoch {epoch_num:03d}"
      self.visualize(emb_X, flow_artist, losses, title)
      self.losses = collate_loss(provided_losses=losses, prior_losses=self.losses)
    self.embedded_points = emb_X
    self.flow_artist = flow_artist

  def visualize(self, embedded_points, flow_artist, losses, title):
    for viz_f in self.vizfiz:
      viz_f(embedded_points= embedded_points, flow_artist = flow_artist, losses = losses, title = title, labels = self.labels)

  def training_gif(self, duration = 10):
    frames = [Image.open(image) for image in glob.glob(f"visualizations/{self.timestamp}/*.jpg")]
    frame_one = frames[0]
    frame_one.save(f"visualizations/{self.timestamp}/{self.title}.gif", format="GIF", append_images=frames,
               save_all=True, duration=duration, loop=0)
    # display in jupyter notebook
    b64 = base64.b64encode(open(f"visualizations/{self.timestamp}/{self.title}.gif",'rb').read()).decode('ascii')
    display(widgets.HTML(f'<img src="data:image/gif;base64,{b64}" />'))

  def visualize_embedding(self):
    visualize_points(embedded_points=self.embedded_points, flow_artist = self.flow_artist, labels = self.labels, title = self.title)

  def visualize_loss(self, loss_type="all"):
    if loss_type == "all":
      for key in self.losses.keys():
        plt.plot(self.losses[key])
      plt.legend(self.losses.keys(), loc='upper right')
      plt.title("loss")
    else:
      plt.plot(self.losses[loss_type])
      plt.title(loss_type)


