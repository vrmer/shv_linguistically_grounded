import os
import sys
import tomli
import numpy as np
from .utils import generate_mask, get_all_means


# load parameters
config_path = sys.argv[1]

with open(config_path, "rb") as infile:
    config = tomli.load(infile)

input_dir = config["input_dir"]
output_dir = config["output_dir"]
filter_type = config["filter_type"]
random_mask_count = config["random_mask_count"]

# get all mean marginal contributions
output_name = f"{filter_type}means"
all_means = get_all_means(input_dir)

# generate masks
if filter_type.startswith("r"):
    for i in range(random_mask_count):
        mask_output_path = os.path.join(output_dir, f"{output_name}.npz")
        mask = {k: generate_mask(v, seed=i, filter_type=filter_type) for k, v in all_means.items()}
        np.savez(mask_output_path, mask)
else:
    mask_output_path = f"{output_name}.npz"
    mask = {k: generate_mask(v, filter_type=filter_type) for k, v in all_means.items()}
    np.savez(mask_output_path, mask)

