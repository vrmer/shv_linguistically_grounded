import os
import re
import glob
import json
import numpy as np
import pandas as pd


def compute_metrics(eval_pred, output_dir, metric):
    """
    Compute evaluation metrics and optionally save predictions to a directory.

    This function computes evaluation metrics based on the provided predictions and labels.
    If an output directory is specified, it saves the predictions and true labels to a CSV file.

    Args:
        eval_pred (tuple): A tuple containing logits and true labels.
        output_dir (str): Directory to save the predictions CSV file. If None, no file is saved.
        metric: An evaluation metric object with a compute method that takes predictions and references as arguments.

    Returns:
        dict: The computed metrics.

    Raises:
        ValueError: If the output directory is specified but cannot be created.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    if output_dir:
        predictions_dir = os.path.join(output_dir, "predictions")
        os.makedirs(predictions_dir, exist_ok=True)
        result_df = pd.DataFrame()
        result_df["pred_label"] = predictions
        result_df["true_label"] = labels
        # to create unique filenames
        file_count = len(os.listdir(output_dir))
        result_df.to_csv(os.path.join(
            predictions_dir, f"results-{file_count}.csv"
        ))

    return metric.compute(predictions=predictions, references=labels)


def match_paradigm_category(input_dir: str, column_name: str = "linguistics_term") -> dict:
    """
    Match paradigms to their linguistic categories from JSONL files in a directory.

    This function reads JSONL files from the specified directory, extracts the paradigm name from
    the filename, and matches it to the corresponding linguistic category found in the specified
    column of the file.

    Args:
        input_dir (str): Directory containing the JSONL files.
        column_name (str, optional): The column name to extract the category from. Must be either
                                     'linguistics_term' or 'field'. Default is 'linguistics_term'.

    Returns:
        dict: A dictionary where keys are paradigm names (derived from filenames) and values are
              the corresponding categories from the specified column.

    Raises:
        AssertionError: If the column_name is not 'linguistics_term' or 'field'.
    """
    assert column_name in {"linguistics_term", "field"}, (f"Column name should be 'linguistics_term' "
                                                          f"or 'field', not {column_name}")
    paradigm_dict = dict()

    for path in sorted(glob.glob(
            os.path.join(input_dir, "**"))):
        paradigm = os.path.basename(path)
        paradigm = re.sub(".jsonl", "", paradigm)
        df = pd.read_json(path, lines=True)
        category = df[column_name][0]

        paradigm_dict[paradigm] = category

    return paradigm_dict


def load_contribution_arrays(input_path: str) -> dict:
    """
    Extract and parse contribution arrays from a file.

    This function reads the specified file, and parses its content
    into a dictionary with integer keys, corresponding to attention head IDs.

    Args:
        input_path (str): The file path to read from.

    Returns:
        dict: A dictionary mapping integer keys to their corresponding contribution arrays.
    """
    with open(input_path, "r") as infile:
        arrays = infile.readlines()[-1].lstrip("Contribution Arrays")
    arrays = json.loads(arrays)
    arrays = {int(k): v for k, v in arrays.items()}
    return arrays


def derive_mean_marginal_contributions(contribution_arrays: dict, flatten: bool = True) -> np.ndarray:
    """
    Calculate mean marginal contributions from contribution arrays.

    This function converts a dictionary of contribution arrays into a DataFrame,
    sorts the DataFrame by index, computes the mean of each column, and returns
    these means as a NumPy array.

    Args:
        contribution_arrays (dict): A dictionary where keys represent attention heads
                                    and the values represent mean marginal contributions.
        flatten (bool, optional): Whether to return the mean marginal contributions as a flat array

    Returns:
        numpy.ndarray: An array of mean marginal contributions.
    """
    df = pd.DataFrame.from_dict(contribution_arrays, orient="index")
    # order the heads
    df = df.sort_index().T
    means = df.mean()
    means = means.to_numpy()
    if not flatten:
        return means.reshape((12, 12))
    else:
        return means


def get_all_means(input_dir: str) -> dict:
    """
    Compute mean marginal contributions for all files in a directory.

    Args:
        input_dir (str): The directory containing the files to process.

    Returns:
        dict: A dictionary where keys are derived from filenames
              and values are arrays of mean marginal contributions.
    """
    all_means = dict()
    pattern = os.path.join(input_dir, "**")

    for filepath in glob.glob(pattern):
        pattern_name = os.path.basename(filepath)
        pattern_name = re.sub("_tracker.txt", "", pattern_name)
        contribution_arrays = load_contribution_arrays(filepath)
        means = derive_mean_marginal_contributions(contribution_arrays)
        all_means[pattern_name] = means

    return all_means


def top_n_index(input_array: np.ndarray, top_n: int) -> np.ndarray:
    """
    Identify the indices of the top N values among the mean marginal contributions.

    Args:
        input_array (np.ndarray): The input array from which to find the top N indices.
        top_n (int): The number of top values to identify.

    Returns:
        np.ndarray: An array of indices corresponding to the top N values in the input array.
    """
    return np.argpartition(input_array, -top_n)[-top_n:]
