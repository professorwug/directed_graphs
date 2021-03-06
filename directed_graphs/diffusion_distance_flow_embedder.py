# AUTOGENERATED! DO NOT EDIT! File to edit: 05c05b Learning flow around a fixed diffusion map.ipynb (unless otherwise specified).

__all__ = ['DiffusionDistanceFlowEmbedder', 'compare_distance_matrices', 'DiffusionDistanceFlowEmbedder',
           'FixedDiffusionMapEmbedding', 'FlowEmbedderAroundDiffusionMap']

# Cell
from .multiscale_flow_embedder import MultiscaleDiffusionFlowEmbedder
import torch
from .flow_embedding_training_utils import FETrainer, visualize_points, save_embedding_visualization
class DiffusionDistanceFlowEmbedder(FETrainer):
    def __init__(self, X, flows, labels, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__(X, flows, labels, device = device)
        self.vizfiz = [
            save_embedding_visualization,
            visualize_points,
        ]
        loss_weights = {
            "diffusion":0,
            "smoothness":0,
            "reconstruction":0,
            "diffusion map regularization":1,
            "flow cosine loss": 0,
        }
        self.FE = MultiscaleDiffusionFlowEmbedder(
            X = X,
            flows = flows,
            sigma_graph = 1,
            flow_strength_graph = 1,
            device = device,
            use_embedding_grid = False,
        ).to(device)
        self.title = "Diffusion Distance FE"


# Cell
def compare_distance_matrices(embedded_points, FE, **kwargs):
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(FE.precomputed_distances.cpu().numpy())
    ax[0].set_title("Diff Distances")
    ax[1].imshow(torch.cdist(embedded_points, embedded_points).detach().cpu().numpy())
    ax[1].set_title("Embedding Euclidean")
    plt.show()

# Cell
import torch
import matplotlib.pyplot as plt
from .multiscale_flow_embedder import MultiscaleDiffusionFlowEmbedder
from .flow_embedding_training_utils import FETrainer, visualize_points, save_embedding_visualization
class DiffusionDistanceFlowEmbedder(FETrainer):
    def __init__(self, X, flows, labels, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__(X, flows, labels, device = device)
        self.vizfiz = [
            save_embedding_visualization,
            visualize_points,
            compare_distance_matrices,
        ]
        loss_weights = {
            "diffusion":0,
            "smoothness":0,
            "reconstruction":0,
            "diffusion map regularization":1,
            "flow cosine loss": 0,
        }
        self.FE = MultiscaleDiffusionFlowEmbedder(
            X = X,
            flows = flows,
            sigma_graph = 1,
            flow_strength_graph = 1,
            device = device,
            use_embedding_grid = False,
            loss_weights = loss_weights,
            k_dmap = 15,
            t_dmap = 50,
        ).to(device)
        # visualize diffusion map
        plt.scatter(self.FE.diff_coords[:,0].cpu().numpy(),self.FE.diff_coords[:,1].cpu().numpy(),c=labels)
        self.title = "Diffusion Distance FE"


# Cell
import torch.nn as nn
import torch
from scipy.sparse import diags
from .utils import make_sparse_safe, diffusion_coordinates, diffusion_map_from_points
import numpy as np

class FixedDiffusionMapEmbedding(nn.Module):
    """
    Computes the diffusion map of the provided data.
    If `precompute = True`, computes diffusion map of input graph when initialized
    and returns this map whenever called (assuming that all subsequent input points
    are the same as initially)
    When false, dynamically computes diffusion map on passed points.
    (False is not yet implemented.)
    """

    def __init__(
        self,
        X,
        t=1,
        k=8,
        precompute=True,
        embedding_dimension=2,
        device=torch.device("cpu"),
        **kwargs
    ):
        super().__init__()
        self.t = t
        self.device = device
        # create diffusion map (building off of numpy array of points; this doesn't have to be differentiable)
        X = X.clone().cpu().numpy()
        diff_map = diffusion_map_from_points(X, k=k, t=t)
        self.diff_coords = diff_map[:, :embedding_dimension]
        self.diff_coords = self.diff_coords.real
        # scale to be between 0 and 1
        self.diff_coords = 2 * (self.diff_coords / np.max(self.diff_coords))
        self.diff_coords = torch.tensor(self.diff_coords.copy())
        self.diff_coords = self.diff_coords.to(device)

    def forward(self, X, **kwargs):
        return self.diff_coords

# Cell
from .multiscale_flow_embedder import MultiscaleDiffusionFlowEmbedder
from .flow_embedding_training_utils import (
    FETrainer,
    visualize_points,
    save_embedding_visualization,
)
from .diffusion_flow_embedding import (
    affinity_matrix_from_pointset_to_pointset,
)
import torch.nn.functional as F
import torch


class FlowEmbedderAroundDiffusionMap(FETrainer):
    def __init__(
        self,
        X,
        flows,
        labels,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        sigma_graph=2.13,
        flow_strength_graph=1,
    ):
        super().__init__(X, flows, labels, device=device)
        self.vizfiz = [
            save_embedding_visualization,
            # visualize_points,
        ]
        loss_weights = {
            "diffusion": 1,
            "smoothness": 0,
            "reconstruction": 0,
            "diffusion map regularization": 0,
            "flow cosine loss": 0,
        }
        P_graph = affinity_matrix_from_pointset_to_pointset(
            X, X, flows, sigma=sigma_graph, flow_strength=0
        )
        P_graph = F.normalize(P_graph, p=1, dim=1)
        self.FE = MultiscaleDiffusionFlowEmbedder(
            X=X,
            flows=flows,
            ts=[1],
            sigma_graph=sigma_graph,
            flow_strength_graph=flow_strength_graph,
            device=device,
            use_embedding_grid=False,
            embedder=FixedDiffusionMapEmbedding(X, t=1, k=18, device=device),
            loss_weight = loss_weights,
        ).to(device)
        self.title = "Diffusion Distance FE"
        self.epochs_between_visualization = 1
        self.total_epochs = 1000
