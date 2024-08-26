import numpy as np
from sklearn.cluster import KMeans
from typing import Tuple, List, Dict
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def apply_scaler(contribution_array: Dict[str, List[float]]) -> Tuple[List[str], List[float]]:
    """
    Scales the input contribution array using standard scaling.

    Args:
        contribution_array (dict): Dictionary with identifiers as keys and
                                                     contribution arrays as values.

    Returns:
        Tuple[List[str], List[float]]: Tuple containing the list of keys and the scaled values.
    """
    scaler = StandardScaler()
    keys, values = list(contribution_array.keys()), list(contribution_array.values())
    values = np.array(values)
    scaled_values = scaler.fit_transform(values)
    return keys, scaled_values


def fit_pca(scaled_values: List[float],
            n_components: None | str | int = 2,
            svd_solver: str = "arpack",
            ) -> PCA:
    """
    Fit a PCA model to scaled data.

    Args:
        scaled_values (List[float]): Scaled data.
        n_components (None | str | int, optional): Number of components to keep. If None, all components are kept.
            If 'mle', Minka’s MLE is used to guess the dimension. If an integer, it is the number of components to keep.
            Defaults to 2.
        svd_solver (str, optional): The solver to use for the decomposition.
                                    Must be one of {'auto', 'full', 'arpack', 'randomized'}.
                                    Defaults to 'arpack'.

    Returns:
        np.ndarray: The transformed data.
    """
    pca = PCA(n_components=n_components, svd_solver=svd_solver)
    reduced_values = pca.fit_transform(scaled_values)
    return reduced_values


def cluster(n_clusters: int,
            values: np.ndarray,
            method: str = "kmeans") -> Tuple[KMeans | GaussianMixture, np.ndarray]:
    """
    Cluster data using specified clustering method.

    This function applies clustering to the given reduced data using either KMeans or Gaussian Mixture Model (GMM).

    Args:
        n_clusters (int): The number of clusters to form.
        values (np.ndarray): The input data to be clustered, typically reduced in dimensionality.
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
    model.fit(values)
    cluster_labels = model.predict(values)
    return model, cluster_labels
