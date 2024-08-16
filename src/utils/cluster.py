import numpy as np
from sklearn.cluster import KMeans
from typing import Tuple, List, Dict
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import plotly.io as pio
# pio.kaleido.scope.mathjax = None


def fit_pca(contribution_array: Dict[int, List[float]]) -> Tuple[PCA, np.ndarray, List[int]]:
    """
    Fit a PCA model to a contribution array.

    This function scales the input contribution array using standard scaling, fits a PCA model to
    the scaled data, and returns the PCA model, the scaled data, and the original keys.

    Args:
        contribution_array (dict): A dictionary where keys are identifiers and values are arrays
                                   representing contributions.

    Returns:
        tuple: A tuple containing:
            - pca (PCA): The fitted PCA model.
            - scaled_values (np.ndarray): The scaled contribution data.
            - keys (list): The original keys from the contribution array.
    """
    scaler = StandardScaler()
    pca = PCA()
    keys, values = list(contribution_array.keys()), list(contribution_array.values())
    values = np.array(values)
    scaled_values = scaler.fit_transform(values)
    pca.fit(scaled_values)
    return pca, scaled_values, keys


def cluster(n_clusters: int,
            reduced_values: np.ndarray,
            method: str = "kmeans") -> Tuple[KMeans|GaussianMixture, np.ndarray]:
    """
    Cluster data using specified clustering method.

    This function applies clustering to the given reduced data using either KMeans or Gaussian Mixture Model (GMM).

    Args:
        n_clusters (int): The number of clusters to form.
        reduced_values (np.ndarray): The input data to be clustered, typically reduced in dimensionality.
        method (str, optional): The clustering method to use, either 'kmeans' or 'gmm'. Default is 'kmeans'.

    Returns:
        tuple: A tuple containing:
            - model: The fitted clustering model (KMeans or GaussianMixture).
            - cluster_labels (np.ndarray): The cluster labels for each data point.

    Raises:
        ValueError: If the specified method is not 'kmeans' or 'gmm'.
    """
    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters)
    elif method == "gmm":
        model = GaussianMixture(n_components=n_clusters)
    else:
        raise ValueError(f"Method has to be kmeans or gmm, not {method}")
    model.fit(reduced_values)
    cluster_labels = model.predict(reduced_values)
    return model, cluster_labels
