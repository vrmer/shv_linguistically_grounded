import torch
import numpy as np
import numpy.random as npr


def generate_mask(means: np.ndarray,
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

