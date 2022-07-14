import argparse
from directed_graphs.datasets import directed_sinh_branch, directed_swiss_roll_uniform, directed_circle, static_clusters
from directed_graphs.multiscale_flow_embedder import MultiscaleDiffusionFlowEmbedder

parser = argparse.ArgumentParser(description='MDFE Arguments') # collect arguments passed to file
parser.add_argument('--dataset', type=str,
                    help='Dataset to use for training.')
parser.add_argument('--t', type=str,
                    help='List of diffusion ts in multiscale loss')
parser.add_argument('--flow_strength', type=str,
                    help='emphasis on flow in embedding')
parser.add_argument('--lr', action='store_true',
                    help='learning rate')
parser.add_argument('--embedder', type=str,
                    help='point embedder type')
parser.add_argument('--decoder', type=str,
                    help='point embedding decoder type')
parser.add_argument('--loss_weights', type=str,
                    help='weight each kind of lost (0: diffusion, 1: smoothness, 2: reconstruction, 3: diffusion map regularization, 4: flow cosine loss)')
parser.add_argument('--flow_artist', type=str,
                    help='gaussian or ReLU')
parser.add_argument('--fa_shape', type=str, help="shape of ReLU")

parser.add_argument('--out_file', type=str, default='out.pth',
                    help='Where to save the results')
args = parser.parse_args()

def parset(t_string):
    ts = t_string.split(" ")
    t_ints = []
    for t in ts:
        t_ints.append(int(t))
    
    return t_ints

def parselossweights(loss_string):
    losses = loss_string.split(" ")
    loss_dict = {}
    loss_dict["diffusion"] = losses[0]
    loss_dict["smoothness"] = losses[1]
    loss_dict["reconstruction"] = losses[2]
    loss_dict["diffusion map regularization"] = losses[3]
    loss_dict["flow cosine loss"] = losses[4]
    
    return loss_dict
        
def parsefashape(shape_string):
    shape = shape_string.split(" ")
    shape_ints = []
    for layer in shape:
        shape_ints.append(int(layer))
    
def create_dataset(dataset_type):
    s = dataset_type
    if s == "circle2d":
        X, flow, labels = directed_circle(num_nodes=200, radius=1, xtilt=0, ytilt=0, twodim=True)
        sigma = 0.5
        flow_strength = 0.5
    elif s == "circle3d":
        X, flow, labels = directed_circle(num_nodes=200, radius=1, xtilt=np.pi/4, ytilt=0)
        sigma = 0.5
        flow_strength = 0.5
    elif s == "swiss roll":
        X, flow, labels = directed_swiss_roll_uniform(num_nodes=1000, num_spirals=2.5, radius=1, height=5, xtilt=0, ytilt=0)
        sigma = 1
        flow_strength = 1
    elif s == "branch":
        X, flow, labels = directed_sinh_branch(num_nodes=300, xscale=2, yscale=1, sigma=0.5, xtilt=np.pi/4, ytilt=np.pi/4)
        sigma = 1
        flow_strength = 2
    else: # s == "clusters"
        X, flow, lables = static_clusters(num_nodes=800, num_clusters=5, radius=1, sigma=0.2, xtilt=np.pi/4, ytilt=np.pi/4)
        sigma = 0.1
        flow_strength = 1
    
    return X, flow, labels, sigma, flow_strength
    
if __name__ == '__main__':
    dataset = args["dataset"]
    ts = parset(args["t"])
    flow_strength_embedding = int(args["flow_strength"])
    learning_rate = int(args["lr"])
    embedder = args["embedder"]
    decoder = args["decoder"]
    loss_dict = parselossweights(args["loss_weights"])
    flow_artist = args["flow_artist"]
    fa_shape = parsefashape(args["fa_shape"])
    
    # create dataset
    X, flow, labels, sigma_graph, flow_strength_graph = create_datasets(dataset)
    
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # initialize mdfe
    MFE = MultiscaleDiffusionFlowEmbedder(X = X, 
                                          flow = flow,
                                          device=device,
                                          ts = ts,
                                          sigma_graph = sigma_graph,
                                          sigma_embedding = #todo,
                                          flow_strength_graph = flow_strength_graph,
                                          flow_strength_embedding = flow_strength_embedding,
                                          learning_rate = learning_rate,
                                          flow_artist = flow_artist,
                                          flow_artist_shape = fa_shape,
                                          embedder = embedder,
                                          decoder = decoder,
                                          loss_weights = loss_dict,
                                          device = device
                                         ).to(device)
    
    # run MFE fit
    MFE.fit(n_steps=10000)
    
    # TODO: what to save?
    


    
    
    
    
    
    
    
    
    
    