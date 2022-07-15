# AUTOGENERATED! DO NOT EDIT! File to edit: 05b03 Testing the Flow-Affinity Matrix.ipynb (unless otherwise specified).

__all__ = ['plot_ribbon', 'sample_ribbon', 'plot_ribbon_samples', 'sample_ribbon_2D', 'plot_ribbon_samples_2D',
           'xy_tilt', 'directed_circle', 'plot_directed_2d', 'plot_origin_3d', 'plot_directed_3d', 'directed_prism',
           'directed_cylinder', 'directed_spiral', 'directed_swiss_roll', 'directed_spiral_uniform',
           'directed_swiss_roll_uniform', 'angle_x', 'whirlpool', 'rejection_sample_for_torus', 'torus_with_flow',
           'directed_one_variable_function', 'directed_sinh_branch', 'static_clusters', 'affinity_grid_search']

# Cell
def plot_ribbon(start=-5, end=5.1, increment=0.2, num_points=1000, dim=3):
    '''
    plots sinusoidal ribbon manifold in 3D
    Inputs
    - start: starting point of sinusoidal function
    - end: ending point of sinusoidal function
    - increment: segment size between start and ed
    - num_points: number of points sampled from the sinusoidal manifold
    '''
    fig = plt.figure(figsize = (12,10))
    ax = plt.axes(projection='3d')

    x = np.arange(start, end, increment)
    y = np.arange(start, end, increment)

    X, Y = np.meshgrid(x, y)
    # define function
    Z = np.sin(X)

    surf = ax.plot_surface(X, Y, Z, cmap = plt.cm.cividis)

    # label axis
    ax.set_xlabel('x', labelpad=20)
    ax.set_ylabel('y', labelpad=20)
    ax.set_zlabel('z', labelpad=20)

    fig.colorbar(surf, shrink=0.5, aspect=8)

    plt.show()

# Cell
def sample_ribbon(start=-5, end=5.1, increment=0.2, num_points=1000, dim=3):
    '''
    sample points uniformly from ribbon
    Inputs
    - start: starting point of sinusoidal function
    - end: ending point of sinusoidal function
    - increment: segment size between start and ed
    - num_points: number of points sampled from the sinusoidal manifold
    Outputs
    - points_mat:
    - flow_mat:
    Examples
    X = sample_ribbon()[0] # points
    X_ = sample_ribbon()[1] # flows
    '''
    # sample points
    sample_x = np.random.uniform(low=start, high=end, size = num_points)
    sample_y = np.random.uniform(low=start, high=end, size = num_points)

    points_mat = np.ndarray(shape=(num_points, dim))
    points_mat[:, 0] = sample_x
    points_mat[:, 1] = sample_y
    points_mat[:, 2] = np.sin(sample_x)

    # calculate flow: unit tangent line at each sampled point
    flow_mat = np.ndarray(shape = (num_points, dim))
    flow_mat[:, 0] = [1] * num_points
    flow_mat[:, 1] = [1] * num_points
    flow_mat[:, 2] = np.cos(sample_x)
    # row normalize
    row_sums = flow_mat.sum(axis=1)
    flow_mat = flow_mat / row_sums[:, np.newaxis]

    return points_mat, flow_mat


# Cell
def plot_ribbon_samples(points_mat, flow_mat):
    '''
    plot points sampled from the manifold with flow at each point,
    where flow is defined as the unit tangent vector at each point
    Inputs
    - points_mat: matrix where each row is a point
    - flow_mat: matrix where each row is a derivative of each corresponding point in points_mat
    '''
    num_points, d = points_mat.shape
    fig = plt.figure(figsize = (12,10))
    ax = fig.add_subplot(projection='3d')

    # plot points
    ax.scatter(points_mat[:, 0], points_mat[:, 1], points_mat[:, 2], c=points_mat[:, 0])

    # plot flow
    mask_prob = 0 # percentage not plotted
    mask = np.random.rand(num_points) > mask_prob
    ax.quiver(points_mat[mask, 0], points_mat[mask, 1], points_mat[mask, 2], flow_mat[mask, 0], flow_mat[mask, 1], flow_mat[mask, 2], alpha=0.8, length=0.5)

    # label axis
    ax.set_xlabel('x', labelpad=20)
    ax.set_ylabel('y', labelpad=20)
    ax.set_zlabel('z', labelpad=20)

    plt.show()



# Cell
def sample_ribbon_2D(start=-5, end=5.1, increment=0.2, num_points=1000, dim=2):
    '''
    sample points uniformly from ribbon
    Inputs
    - start: starting point of sinusoidal function
    - end: ending point of sinusoidal function
    - increment: segment size between start and ed
    - num_points: number of points sampled from the sinusoidal manifold
    Outputs
    - points_mat:
    - flow_mat:
    Examples
    X = sample_ribbon()[0] # points
    X_ = sample_ribbon()[1] # flows
    '''
    # sample points
    sample_x = np.random.uniform(low=start, high=end, size = num_points)

    points_mat = np.ndarray(shape=(num_points, dim))
    points_mat[:, 0] = sample_x
    points_mat[:, 1] = np.sin(sample_x)

    # calculate flow: unit tangent line at each sampled point
    flow_mat = np.ndarray(shape = (num_points, dim))
    flow_mat[:, 0] = [1] * num_points
    flow_mat[:, 1] = np.cos(sample_x)
    # row normalize
    # row_sums = flow_mat.sum(axis=1)
    # flow_mat = flow_mat / row_sums[:, np.newaxis]

    return points_mat, flow_mat


# Cell
def plot_ribbon_samples_2D(points_mat, flow_mat):
    '''
    plot points sampled from the manifold with flow at each point,
    where flow is defined as the unit tangent vector at each point
    Inputs
    - points_mat: matrix where each row is a point
    - flow_mat: matrix where each row is a derivative of each corresponding point in points_mat
    '''
    num_points, d = points_mat.shape
    fig = plt.figure(figsize = (12,10))
    ax = fig.add_subplot()

    # plot points
    plt.scatter(points_mat[:, 0], points_mat[:, 1])

    # plot flow
    mask_prob = 0 # percentage not plotted
    mask = np.random.rand(num_points) > mask_prob
    ax.quiver(points_mat[mask, 0], points_mat[mask, 1], flow_mat[mask, 0], flow_mat[mask, 1], alpha=0.8)

    # label axis
    ax.set_xlabel('x', labelpad=20)
    ax.set_ylabel('y', labelpad=20)

    plt.show()



# Cell
import torch
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
import numpy as np

def directed_circle(num_nodes=100, radius=1, xtilt=0, ytilt=0, twodim=False):
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
  X, flow, labels = xy_tilt(X, flow, labels, xtilt=xtilt, ytilt=ytilt)
  if twodim:
    X = X[:,:2]
    flow = flow[:,:2]
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
def directed_one_variable_function(func, deriv, xlow, xhigh, num_nodes=100, sigma=0.25):
  # positions
  x = np.random.uniform(xlow, xhigh, num_nodes)
  x = np.sort(x)
  labels = x
  y = func(x)
  z = np.zeros(num_nodes)
  # vectors
  u = np.ones(num_nodes)
  v = deriv(x)
  w = np.zeros(num_nodes)
  flow = np.column_stack((u, v, w))
  # noise
  deriv_square = v**2
  noise = np.random.normal(0, sigma, num_nodes) * np.sqrt(deriv_square/(deriv_square + 1))
  x += noise
  y += -1/v * noise
  X = np.column_stack((x, y, z))
  return X, flow, labels

# Cell
from .datasets import xy_tilt
def directed_sinh_branch(num_nodes=1000, xscale=1, yscale=1, sigma=0.25, xtilt=0, ytilt=0):
  num_nodes_per_branch = num_nodes//3
  # root
  X_root, flow_root, labels_root = directed_one_variable_function(
    lambda x: np.sinh(x / xscale) * yscale,
    lambda x: np.cosh(x / xscale) / xscale * yscale,
    xlow=-xscale*np.pi*0.84,
    xhigh=0,
    num_nodes=num_nodes - 2*num_nodes_per_branch,
    sigma=sigma
  )
  # branch 1
  X_branch1, flow_branch1, labels_branch1 = directed_one_variable_function(
    lambda x: np.sinh(x / xscale) * yscale,
    lambda x: np.cosh(x / xscale) / xscale * yscale,
    xlow=0,
    xhigh=xscale*np.pi*0.84,
    num_nodes=num_nodes_per_branch,
    sigma=sigma
  )
  # branch 2
  X_branch2, flow_branch2, labels_branch2 = directed_one_variable_function(
    lambda x: np.sin(x / xscale) * yscale,
    lambda x: np.cos(x / xscale) / xscale * yscale,
    xlow=0,
    xhigh=xscale*np.pi*2,
    num_nodes=num_nodes_per_branch,
    sigma=sigma
  )
  # concatenate
  X = np.concatenate((X_root, X_branch1, X_branch2))
  flow = np.concatenate((flow_root, flow_branch1, flow_branch2))
  labels = np.concatenate((labels_root - np.pi*3, labels_branch1, labels_branch2 + np.pi*3))
  # tilt
  X, flow, labels = xy_tilt(X, flow, labels, xtilt=xtilt, ytilt=ytilt)
  return X, flow, labels


# Cell
def static_clusters(num_nodes=250, num_clusters=5, radius=1, sigma=0.2, xtilt=0, ytilt=0):
  thetas = np.repeat([2*np.pi*i/num_clusters for i in range(num_clusters)], num_nodes//num_clusters)
  x = np.cos(thetas) * radius + np.random.normal(loc=0, scale=sigma, size=num_nodes)
  y = np.sin(thetas) * radius + np.random.normal(loc=0, scale=sigma, size=num_nodes)
  z = np.zeros(num_nodes)
  X = np.column_stack((x, y, z))
  flow = np.zeros(X.shape)
  X, flow, lables = xy_tilt(X, flow, thetas, xtilt=xtilt, ytilt=ytilt)
  return X, flow, lables

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