@code_type Python .py
@comment_type #

@title Flow Embeddings Testing Suite

@s Introduction

My good friends, the time has come to dispense with the tedium of note-book based testing and recapitulate the lost art of well-made, logically-structured code modules. 
This file will build a collection of tools to test the `flow_embedder` 05c on a variety of datasets, while compiling helpful details about training, loss values, etc. (-wug)

The goals:
- [ ] be able to run different configurations of the parameters by changing parameters in a json file, and submitting a single job to the cluster.
- [ ] should be possible to set multiple datasets, and have the flow embedder evaluated on each of them.
- [ ] this job should
	- [ ] save a final embedding of the flow embedder, given the parameters, and the dataset
	- [ ] save a GIF of the training progress, compiled every 100 iterations.

# Outline

This file will create the interface to this testing. Its structure will involve
1. Parsing a JSON file to extract the intended training parameters
2. Interfacing with 05c to set up a model
3. Training 05c
4. Saving the output in a readable form

To use this, we can take advantage of the form of 05c's Multiscale Flow Embedder, in which calling `.fit` trains for some number of epochs, and then reports progress back. We can dynamically report progress by training for 500 epochs, visualizing the report, and continuing. 

This should also be able to remove some of the *bloat* from 05c's class definition. We can separate the jobs of encoding the points from those of reporting progress.

Here is the broad outline:

``` directed_graphs/flow_embed.py
@{Imports}

@{Parse JSON file to extract training scheme}

@{Initialize Flow Embedder with Parameters}

@{Train and Visualize}
```

# Parse JSON arguments

I haven't worked with JSON in python before. Here's a reference we'll be following: [Reading & Parsing JSON Data With Python: Tutorial | Oxylabs](x-devonthink-item://6733F023-D601-49E3-B6E7-A2D6FDCA0A65).

To import data from a JSON file, we have to get the name of the file, read it into memory, and then parse it with the json library.

``` Imports 
import json
import argparse
```


``` Parse JSON file to extract training scheme
parser = argparse.ArgumentParser(description = "JSON File Name !!!!")
parser.add_argument("--file", type=str, help = "Dataset to use for training")
args = parser.parse_args()
filename = args.file
with open(filename,'r') as f:
	json_contents = f.read()
	parameter_dict = json.loads(json_contents)
	print(json_contents)
print("Using dataset",parameter_dict['datasets'])
```


# Initialize Flow Embedder
``` Initialize Flow Embedder with Parameters


```

# Train 

``` Train and Visualize



```
