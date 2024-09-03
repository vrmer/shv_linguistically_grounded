import os
import re
import csv
import tomli
import argparse
import subprocess
from queue import Queue
from threading import Thread


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config",
                        type=str, required=True,
                        help="Path to config file (for prune)")

    parser.add_argument("-u", "--target_uid",
                        type=str, required=True, default=None,
                        help="Target uid")

    parser.add_argument("-m", "--mask_id",
                        type=str, required=True, default=None,
                        help="Mask id")

    parser.add_argument("-n", "--n_heads",
                        type=int, required=False, default=0,
                        help="Number of heads to mask, if not provided, it is taken from the config file")

    parser.add_argument("--minimum_n_heads",
                        type=int, default=0, required=False,
                        help="Minimum number of heads to prune for parallelisation, it is 0 by default")

    parser.add_argument("-o", "--output",
                        type=str, required=False, default="results.csv",
                        help="Path to output file, appended to the config output by default")

    args = parser.parse_args()


    def writer():
        with open("results.csv", mode="a", newline="") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=[
                "target_uid", "mask_id", "n_heads", "accuracy"])
            writer.writeheader()
            while True:
                result = result_queue.get()
                if result is None:
                    break
                writer.writerow(result)
                result_queue.task_done()


    def run_attribution(target_uid, mask_id, n_heads):
        pattern = r"\d+\.?\d*"  # regex for capturing numbers
        command = [
            "python", "-m", "src.attribution", "-c",
            args.config, "-u", target_uid,
            "-m", mask_id, "-n", n_heads
        ]
        result = subprocess.run(
            command, capture_output=True, text=True).stdout.strip()
        score = float(re.findall(pattern, result)[0])
        result_dict = {
            "target_uid": target_uid, "mask_id": mask_id,
            "n_heads": n_heads, "accuracy": score
        }
        return result_dict


    # define parameters
    with open(args.config, "rb") as infile:
        config = tomli.load(infile)

    device = config["device"]
    seed = config["seed"]

    target_uid = args.target_uid
    mask_id = args.mask_id

    model_name = config["model_name"]

    n_heads = config["n_heads"] if args.n_heads == 0 else args.n_heads
    minimum_n_heads = args.minimum_n_heads

    prune_path = os.path.join(config["paths"]["prune_path"], model_name, args.output)

    # queue to collect output
    result_queue = Queue()

    writer_thread = Thread(target=writer)
    writer_thread.start()

    for i in range(minimum_n_heads, n_heads+1):

        result_dict = run_attribution(
            target_uid=target_uid, mask_id=mask_id, n_heads=str(i))

        result_queue.put(result_dict)

    result_queue.put(None)
    writer_thread.join()
