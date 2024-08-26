import os
import sys
import tomli
import numpy as np
from functools import partial
from .utils import generate_mask, get_all_means


# load parameters
config_path = sys.argv[1]

with open(config_path, "rb") as infile:
    config = tomli.load(infile)


paths_configs = config["paths"]
attribution_path = paths_configs["input_dir"]
output_dir = paths_configs["output_dir"]
model_name = paths_configs["model_name"]

mask_configs = config["mask"]
threshold = mask_configs["threshold"]
formatting = mask_configs["formatting"]
filter_type = mask_configs["filter_type"]
random_mask_count = mask_configs["random_mask_count"]


generate_mask = partial(
    generate_mask, threshold=threshold,
    formatting=formatting, filter_type=filter_type)

os.makedirs(output_dir, exist_ok=True)
if filter_type.startswith("r"):
    os.makedirs(
        os.path.join(output_dir, "random"), exist_ok=True)

# get all mean marginal contributions
# output_name = f"{filter_type}means"x
all_means = get_all_means(
    os.path.join(attribution_path, model_name))

# generate masks
if filter_type.startswith("r"):
    for i in range(random_mask_count):
        mask_output_path = os.path.join(output_dir, "random", f"{filter_type}{i}.npz")
        mask = {k: generate_mask(v, seed=i) for k, v in all_means.items()}
        np.savez(mask_output_path, mask)
else:
    mask_output_path = os.path.join(output_dir, f"{filter_type}.npz")
    mask = {k: generate_mask(v) for k, v in all_means.items()}
    np.savez(mask_output_path, mask)
