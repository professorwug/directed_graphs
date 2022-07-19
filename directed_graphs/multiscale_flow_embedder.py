# AUTOGENERATED! DO NOT EDIT! File to edit: 05c Multiscale Diffusion Flow Embedding.ipynb (unless otherwise specified).

__all__ = ['compute_grid', 'diffusion_matrix_with_grid_points', 'FeedForwardReLU', 'MultiscaleDiffusionFlowEmbedder']

# Cell
from .diffusion_flow_embedding import (
    affinity_matrix_from_pointset_to_pointset,
    smoothness_of_vector_field,
)


def compute_grid(X, grid_width=20):
    """Returns a grid of points which bounds the points X.
    The grid has 'grid_width' dots in both length and width.
    Accepts X, tensor of shape n x 2
    Returns tensor of shape grid_width^2 x 2"""
    # TODO: This currently only supports
    # find support of points
    minx = float(torch.min(X[:, 0]) - 0.1)  # TODO: use torch.min, try without detach
    maxx = float(torch.max(X[:, 0]) + 0.1)
    miny = float(torch.min(X[:, 1]) - 0.1)
    maxy = float(torch.max(X[:, 1]) + 0.1)
    # form grid around points
    x, y = torch.meshgrid(
        torch.linspace(minx, maxx, steps=grid_width),
        torch.linspace(miny, maxy, steps=grid_width),
        indexing="ij",
    )
    xy_t = torch.concat([x[:, :, None], y[:, :, None]], dim=2).float()
    xy_t = xy_t.reshape(grid_width**2, 2).detach()
    return xy_t

# Cell
import torch
from .diffusion_flow_embedding import (
    affinity_matrix_from_pointset_to_pointset,
    GaussianVectorField,
)
import torch.nn.functional as F


def diffusion_matrix_with_grid_points(X, grid, flow_function, t, sigma, flow_strength):
    n_points = X.shape[0]
    # combine the points and the grid
    points_and_grid = torch.concat([X, grid], dim=0)
    # get flows at each point
    flow_per_point = flow_function(points_and_grid)
    # take a diffusion matrix
    A = affinity_matrix_from_pointset_to_pointset(
        points_and_grid,
        points_and_grid,
        flows=flow_per_point,
        sigma=sigma,
        flow_strength=flow_strength,
    )
    P = F.normalize(A, p=1, dim=-1)
    # TODO: Should we remove self affinities? Probably not, as lazy random walks are advantageous when powering
    # Power the matrix to t steps
    Pt = torch.matrix_power(P, t)
    # Recover the transition probabilities between the points, and renormalize them
    Pt_points = Pt[:n_points, :n_points]
    # Pt_points = torch.diag(1/Pt_points.sum(1)) @ Pt_points
    Pt_points = F.normalize(Pt_points, p=1, dim=1)
    # return diffusion probs between points
    return Pt_points

# Cell
import torch.nn as nn
from collections import OrderedDict


class FeedForwardReLU(nn.Module):
    def __init__(self, shape):
        super(FeedForwardReLU, self).__init__()
        d_len = len(shape) * 2
        d = OrderedDict()
        d[str(0)] = nn.Linear(shape[0], shape[1])
        for i in range(1, d_len - 3):
            if i % 2 == 1:
                d[str(i)] = nn.LeakyReLU()
            else:
                d[str(i)] = nn.Linear(shape[int(i / 2)], shape[int(i / 2) + 1])
        # create MLP
        self.FA = nn.Sequential(d)

    def forward(self, X):
        return self.FA(X)

# Cell
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
from .diffusion_flow_embedding import (
    GaussianVectorField,
    smoothness_of_vector_field,
    FlowArtist,
)
from .diffusion_flow_embedding import (
    diffusion_map_loss,
    flow_cosine_loss,
    directed_neighbors,
    flow_neighbor_loss,
    precomputed_distance_loss,
)
from .utils import (
    diffusion_map_from_points,
    diffusion_map_from_affinities,
)


class MultiscaleDiffusionFlowEmbedder(torch.nn.Module):
    def __init__(
        self,
        X,
        flows,
        ts=(1, 2, 4, 8),
        sigma_graph=0.5,
        sigma_embedding=0.5,
        flow_strength_graph=5,
        flow_strength_embedding=5,
        n_neighbors=20,
        embedding_dimension=2,
        learning_rate=1e-3,
        flow_artist="ReLU",
        flow_artist_shape=(2, 4, 8, 4, 2),
        num_flow_gaussians=25,
        embedder=None,
        decoder=None,
        labels=None,
        loss_weights=None,
        use_embedding_grid=False,
        device=torch.device("cpu"),
        k_dmap=20,
        t_dmap=1,
        dmap_coords_to_use=2,
        use_batches=False,
    ):
        super(MultiscaleDiffusionFlowEmbedder, self).__init__()

        # # Generate default parameters
        embedder = (
            FeedForwardReLU(shape=(3, 4, 8, 4, 2)) if embedder is None else embedder
        )
        # These are the available loss functions.
        self.loss_keys = [
            "diffusion",  # KLD Loss between row-normalized flow-affinity matrix of ambient space and embedded space. Trains both positions and flows.
            "smoothness",  # Laplacian smoothness form v^T L v / v^T v
            "reconstruction",  # MSE reconstruction error if an autoencoder is trained (decoder must be supplied)
            "diffusion map regularization",  # Regularizes points by diffusion distance in ambient space
            "flow cosine loss",  # enforces same angles between pairs of flows in ambient and embedding space (DON'T USE)
            "flow neighbor loss",  # enforces flow in embedding to point towards "flow neighbors" -- points in the ambient space with high flow affinity
        ]
        if loss_weights is None:
            loss_weights = {"diffusion": 1}
        else:
            loss_weights = loss_weights
        for key in self.loss_keys:
            if key not in loss_weights.keys():
                loss_weights[key] = 0
        self.loss_weights = loss_weights
        print("Using loss weights:",self.loss_weights)

        # initialize model parameters: store the data and flows inside the model for future reference by any of our (many) loss functions
        self.X = X
        self.ground_truth_flows = flows
        # Parameter for KLD Diffusion Loss: takes KLD(P^t, P_embedding^t)
        self.ts = ts
        # Build a graph from the input data using our flow affinity function,
        self.sigma_graph = sigma_graph
        self.P_graph = affinity_matrix_from_pointset_to_pointset(
            X, X, flows, sigma=sigma_graph, flow_strength=flow_strength_graph
        )
        self.P_graph = F.normalize(self.P_graph, p=1, dim=1)

        # these are used by KLD Diffusion Loss
        self.sigma_embedding = sigma_embedding
        self.flow_strength = torch.tensor(flow_strength_embedding).float()

        self.nnodes = X.shape[0]
        self.data_dimension = X.shape[1]
        self.use_embedding_grid = use_embedding_grid

        self.eps = 0.001
        self.labels = labels
        self.embedding_dimension = embedding_dimension
        # set device (used for shuffling points around during visualization)
        self.device = device
        # Used for
        self.losses = {}
        for k in self.loss_weights.keys():
            self.losses[k] = []

        self.n_neighbors = n_neighbors
        self.neighbors = directed_neighbors(self.nnodes, self.P_graph, self.n_neighbors)
        # torch.diag(1/self.P_graph.sum(axis=1)) @ self.P_graph
        # compute matrix powers
        # TODO: Could reuse previous powers to speed this up
        self.P_graph_ts = [torch.matrix_power(self.P_graph, t) for t in self.ts]
        self.P_embedding_ts = [None for i in self.ts]
        # Flow field
        self.FlowArtist = FlowArtist(
            flow_artist,
            dim=self.embedding_dimension,
            num_gauss=num_flow_gaussians,
            shape=flow_artist_shape,
            device=self.device,
        ).to(self.device)

        # Autoencoder to embed the points into a low dimension
        self.embedder = embedder.to(self.device)
        if decoder is not None:
            self.decoder = decoder.to(self.device)
        else:
            self.decoder = None

        # Precompute graph distances for any loss functions that regularize against a precomputed embedding
        if self.loss_weights["diffusion map regularization"] > 0:
            P_graph_symmetrized = self.P_graph + self.P_graph.T
            diff_map = diffusion_map_from_affinities(
                P_graph_symmetrized, t=t_dmap, plot_evals=False
            )
            self.diff_coords = diff_map[:, :dmap_coords_to_use]
            self.diff_coords = self.diff_coords.real
            # scale to be between 0 and 1 (by default, the diff coords are tiny, which messes up the flow embedder)
            #             self.diff_coords = 2 * (self.diff_coords / np.max(self.diff_coords))
            self.diff_coords = torch.tensor(self.diff_coords.copy()).to(device)
            self.precomputed_distances = torch.cdist(self.diff_coords, self.diff_coords)
            # scale distances between 0 and 1
            self.precomputed_distances = 2 * (
                self.precomputed_distances / torch.max(self.precomputed_distances)
            )
            self.precomputed_distances = (
                self.precomputed_distances.detach()
            )  # no need to have gradients from this operation

        # training ops
        self.KLD = nn.KLDivLoss(reduction="batchmean", log_target=False)
        self.MSE = nn.MSELoss()
        # self.KLD = homemade_KLD # when running on mac
        self.epsilon = 1e-6  # set zeros to eps
        self.optim = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # split input data into batches

    def diffusion_loss(self):
        # compute grid around points
        if self.use_embedding_grid:
            self.grid = compute_grid(self.embedded_points).to(self.device)
        else:
            # take a convex combination of points # TODO: can be much improved
            self.grid = (
                self.embedded_points + torch.flip(self.embedded_points, dims=[0])
            ) / 2
            self.grid = (
                self.grid.detach()
            )  # don't want gradients flowing back through this
        # normalize embedded points to lie within -self.embedding_bounds, self.embedding_bounds
        # if any are trying to escape, constrain them to lie on the edges
        # self.embedded_points[:,0][torch.abs(self.embedded_points[:,0]) > self.embedding_bounds] = self.embedding_bounds * (self.embedded_points[:,0][torch.abs(self.embedded_points[:,0]) > self.embedding_bounds])/torch.abs(self.embedded_points[:,0][torch.abs(self.embedded_points[:,0]) > self.embedding_bounds])
        # self.embedded_points[:,1][torch.abs(self.embedded_points[:,1]) > self.embedding_bounds] = self.embedding_bounds * (self.embedded_points[:,1][torch.abs(self.embedded_points[:,1]) > self.embedding_bounds])/torch.abs(self.embedded_points[:,0][torch.abs(self.embedded_points[:,1]) > self.embedding_bounds])
        # compute embedding diffusion matrix, using including diffusion to grid points
        for i, t in enumerate(self.ts):
            self.P_embedding_ts[i] = diffusion_matrix_with_grid_points(
                X=self.embedded_points,
                grid=self.grid,
                flow_function=self.FlowArtist,
                t=t,
                sigma=self.sigma_embedding,
                flow_strength=self.flow_strength,
            )
            # set any affinities of zero to a very small amount, to prevent the KL divergence from becoming infinite.
            self.P_embedding_ts[i][
                self.P_embedding_ts[i] == 0
            ] = (
                self.epsilon
            )  # TODO: Perhaps enable later; this didn't cause NaN errors before
            # self.P_embedding_ts[i] = self.P_embedding_ts[i] + self.epsilon
        # take KL divergence between P embedding ts and P graph ts
        diffusion_loss = 0
        """
        for i in range(len(self.ts)):
            log_P_embedding_t = torch.log(self.P_embedding_ts[i])
            log_P_graph_t = torch.log(self.P_graph_ts[i])
            if log_P_embedding_t.is_sparse:
                KL_emb = log_P_embedding_t.to_dense()
                KL_graph = log_P_graph_t.to_dense()
                diffusion_loss_for_t = 0.5*(self.KLD(KL_emb, KL_graph) + self.KLD(KL_graph, KL_emb))
            else:
                A = log_P_embedding_t
                B = log_P_graph_t
                diffusion_loss_for_t = 0.5*(self.KLD(A,B) + self.KLD(B,A))
            diffusion_loss += (2**(-i))*diffusion_loss_for_t
            # print(f"Diffusion loss {i} is {diffusion_loss}")
        """
        for i in range(len(self.ts)):
            log_P_embedding_t = torch.log(self.P_embedding_ts[i])
            if log_P_embedding_t.is_sparse:
                diffusion_loss_for_t = self.KLD(
                    log_P_embedding_t.to_dense(), self.P_graph_ts[i].to_dense()
                )
            else:
                diffusion_loss_for_t = self.KLD(log_P_embedding_t, self.P_graph_ts[i])
            diffusion_loss += (2 ** (-i)) * diffusion_loss_for_t
            # print(f"Diffusion loss {i} is {diffusion_loss}")
        if diffusion_loss.isnan():
            raise NotImplementedError
        return diffusion_loss

    def loss(self):
        # embed points
        self.embedded_points = self.embedder(self.X)

        # compute diffusion loss on embedded points
        if self.loss_weights["diffusion"] != 0:
            self.losses["diffusion"].append(self.diffusion_loss())
        else:
            self.losses["diffusion"].append(0)

        # compute autoencoder loss
        if (self.decoder is not None) and self.loss_weights["reconstruction"] != 0:
            X_reconstructed = self.decoder(self.embedded_points)
            self.losses["reconstruction"].append(
                self.MSE(X_reconstructed, self.X)
            )  # current loss
        else:
            self.losses["reconstruction"].append(0)

        # regularizations
        if self.loss_weights["smoothness"] != 0:
            smoothness_loss = smoothness_of_vector_field(
                self.embedded_points, self.FlowArtist, device=self.device, grid_width=20
            )
            self.losses["smoothness"].append(smoothness_loss)
        else:
            self.losses["smoothness"].append(0)

        if self.loss_weights["diffusion map regularization"] != 0:
            diffmap_loss = precomputed_distance_loss(
                self.precomputed_distances, self.embedded_points
            )
            #           diffmap_loss = diffusion_map_loss(self.P_graph_ts[0], self.embedded_points)
            self.losses["diffusion map regularization"].append(diffmap_loss)
        else:
            self.losses["diffusion map regularization"].append(0)

        if self.loss_weights["flow cosine loss"] != 0:
            flow_loss = flow_cosine_loss(
                self.X, self.ground_truth_flows, self.FlowArtist(self.embedded_points)
            )
            self.losses["flow cosine loss"].append(flow_loss)
        else:
            self.losses["flow cosine loss"].append(0)

        if self.loss_weights["flow neighbor loss"] != 0:
            neighbor_loss = flow_neighbor_loss(
                self.neighbors,
                self.embedded_points,
                self.FlowArtist(self.embedded_points),
            )
            self.losses["flow neighbor loss"].append(neighbor_loss)
        else:
            self.losses["flow neighbor loss"].append(0)
        # iterated form of loss weight summation
        cost = 0
        for loss_name in self.loss_keys:
            cost += self.loss_weights[loss_name] * self.losses[loss_name][-1]
        # cost = (
        #     self.loss_weights["diffusion"] * diffusion_loss
        #     + self.loss_weights["reconstruction"] * reconstruction_loss
        #     + self.loss_weights["smoothness"] * smoothness_loss
        #     + self.loss_weights["diffusion map regularization"] * diffmap_loss
        #     + self.loss_weights["flow cosine loss"] * flow_loss
        #     + self.loss_weights["flow neighbor loss"] * neighbor_loss
        # )
        return cost

    def fit(self, n_steps=1000):
        # train Flow Embedder on the provided graph
        self.train()
        # reset losses
        self.losses = {}
        for k in self.loss_weights.keys():
            self.losses[k] = []
        # self.weight_of_flow = 0
        for step in range(n_steps):
            # if step == 100:
            #      self.weight_of_flow = 1
            # if step == 200:
            #      self.weight_of_flow = 0.5
            self.optim.zero_grad()
            # compute loss
            loss = self.loss()
            if loss.isnan():
                print("Final loss was nan")
                raise NotImplementedError
            # print("loss is ",loss)
            # compute gradient and step backwards
            loss.backward()
            self.optim.step()
            # if step % 500 == 0:
            #      print(f"EPOCH {step}. Loss {loss}. Flow strength {self.flow_strength}. Heatmap of P embedding is ")
            #      self.visualize_diffusion_matrices()
            #      self.visualize_points()
            # TODO: Criteria to automatically end training
        # print("Exiting training with loss ",loss)
        return self.embedded_points, self.FlowArtist, self.losses