import torch
import numpy as np
import numpy.random as npr
from .utils import top_n_index, get_all_means


def generate_generic_mask(means: np.ndarray,
                          threshold: float | int = 0,
                          formatting: str = "np",
                          filter_type: str = "+",
                          seed: int | None = None) -> np.ndarray | torch.Tensor:
    """
    Generate an attention head mask based on mean values and specified criteria.

    This function creates a mask array based on the provided means, threshold, and filtering criteria.
    The mask can be formatted as either a NumPy array or a PyTorch tensor.

    Args:
        means (np.ndarray): Array of mean values to base the mask on.
        threshold (float | int, optional): Threshold value for filtering. Default is 0.
        formatting (str, optional): Format of the output mask, either 'np' for NumPy array
                                    or 'pt' for PyTorch tensor. Default is 'np'.
        filter_type (str, optional): Filtering criteria; '+' to exclude means smaller than the threshold,
                                   '-' to exclude means larger than the threshold,
                                   or 'r+'/'r-' for random exclusion. Default is '+'.
        seed (int | None, optional): Seed for random number generator if random filtering is used.
                                     Default is None.

    Returns:
        np.ndarray | torch.Tensor: The generated mask in the specified format.

    Raises:
        ValueError: If the filtering criteria is not valid.
    """
    # initialise the mask
    mask = np.ones_like(means)

    # carry out filtering based on whether to exclude means
    # larger (-) or smaller (+) than the threshold, or exclude random ones
    if filter_type.endswith("+"):
        values_to_exclude = np.where(means < threshold)
    elif filter_type.endswith("-"):
        values_to_exclude = np.where(means > threshold)
    else:
        raise ValueError("Filtering must end with '+' or '-'")
    if filter_type.startswith("r"):
        random_generator = npr.default_rng(seed=seed)
        random_count = len(values_to_exclude[0].tolist())
        values_to_exclude = random_generator.integers(0, 144, size=random_count)

    # exclude relevant values
    mask[values_to_exclude] = 0

    if formatting == "pt":
        mask = torch.from_numpy(mask)

    return mask


def generate_top_mask(means: np.ndarray, n_heads: int, seed: int, random: bool):
    """
    Given an array of marginal contribution means,
    returns a binary mask with a number of top heads
    masked that correspond to n_heads.

    :param means: array of marginal contribution means
    :param n_heads: number of top heads to mask
    :param seed: for reproducibility
    :param random: if True, the mask is randomly generated
    :return: binary mask
    """
    mask = np.ones_like(means)
    if random is True:
        random_generator = np.random.default_rng(seed=seed)
        top_indices = random_generator.integers(0, len(means), size=n_heads)
    else:
        top_indices = top_n_index(means, top_n=n_heads)
    mask[top_indices] = 0
    return mask


def generate_unique_mask(input_dir: str,
                         mask_uid: str,
                         n_heads: int,
                         seed: int = 42,
                         random: bool = False):
    all_means = get_all_means(input_dir)
    mask_means = all_means[mask_uid]
    mask = generate_top_mask(mask_means, n_heads=n_heads, random=random, seed=seed)
    return mask

