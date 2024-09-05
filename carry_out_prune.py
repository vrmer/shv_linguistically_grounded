import os
import csv
import tomli
import argparse
import subprocess
from tqdm import tqdm
import concurrent.futures
from threading import Lock
from itertools import product


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config",
                        type=str, required=True,
                        help="Path to config file (for prune)")

    parser.add_argument("-n", "--n_heads",
                        type=int, required=False, default=0,
                        help="Number of heads to mask, if not provided, it is taken from the config file")

    parser.add_argument("-m", "--minimum_n_heads",
                        type=int, default=1, required=False,
                        help="Minimum number of heads to prune for parallelisation, it is 1 by default")

    parser.add_argument("-o", "--output_name",
                        type=str, required=False, default="results.csv",
                        help="Path to output file, appended to the config output by default")

    parser.add_argument("-w", "--n_workers",
                        type=int, default=10, required=False,
                        help="Number of parallel workers")

    args = parser.parse_args()


    def run_attribution_combination(target_uid, mask_id, n_heads, config_path):
        command = [
            "python", "-m", "src.attribution", "-c",
            config_path, "-u", target_uid,
            "-m", mask_id, "-n", str(n_heads)
        ]
        result = subprocess.run(
            command, capture_output=True, text=True).stdout.strip()
        return {
            "target_uid": target_uid, "mask_id": mask_id,
            "n_heads": n_heads, "accuracy": float(result)
        }


    def process_combinations(combinations, output_path, config_path, n_workers):
        lock = Lock()

        def safe_write(result):
            with lock:
                with open(output_path, "a") as outfile:
                    writer = csv.DictWriter(outfile, fieldnames=[
                        "target_uid", "mask_id", "n_heads", "accuracy"])
                    writer.writerow(result)

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:

            futures = [
                executor.submit(run_attribution_combination, target_uid, mask_id, n_heads, config_path)
                for target_uid, mask_id, n_heads in combinations]

            with open(output_path, mode="w") as outfile:
                writer = csv.DictWriter(outfile, fieldnames=[
                    "target_uid", "mask_id", "n_heads", "accuracy"])
                writer.writeheader()

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing"):
                result = future.result()
                safe_write(result)


    # define parameters
    with open(args.config, "rb") as infile:
        config = tomli.load(infile)

    device = config["device"]
    seed = config["seed"]

    model_name = config["model_name"]

    n_workers = args.n_workers

    n_heads = config["n_heads"] if args.n_heads == 0 else args.n_heads
    minimum_n_heads = args.minimum_n_heads

    output_path = os.path.join(config["paths"]["prune_path"], model_name, args.output_name)

    with open("paradigms.txt", "r") as infile:
        paradigms = [paradigm.strip() for paradigm in infile.readlines()]

    n_heads_range = range(minimum_n_heads, n_heads + 1)

    combinations = list(
        product(paradigms, paradigms, n_heads_range)
    )

    process_combinations(combinations, output_path, args.config, n_workers)
