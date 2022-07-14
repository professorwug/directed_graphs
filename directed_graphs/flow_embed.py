#
#
import json
import argparse


#
parser = argparse.ArgumentParser(description = "JSON File Name !!!!")
parser.add_argument("--file", type=str, help = "Dataset to use for training")
args = parser.parse_args()
filename = args.file
with open(filename,'r') as f:
	json_contents = f.read()
	parameter_dict = json.loads(json_contents)
	print(json_contents)
print("Using dataset",parameter_dict['datasets'])


#




#





