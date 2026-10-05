"""Elbow selection of k_nn and k. Neither uses the sensitive attribute."""

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from .parpic import find_knee


def select_n_neighbors_by_elbow(X, candidates=(3, 5, 7, 10, 12, 15, 18, 20, 25, 30)):
    """k_nn at the knee of the mean k_nn-distance curve (as for DBSCAN)."""
    mean_kdist = []
    for k_nn in candidates:
        dist, _ = NearestNeighbors(n_neighbors=k_nn + 1).fit(X).kneighbors(X)
        mean_kdist.append(dist[:, k_nn].mean())
    return find_knee(candidates, mean_kdist)


def select_k_by_elbow(Z, candidates=tuple(range(2, 16)), seed=42):
    """k at the knee of the k-means WCSS curve of the (unconstrained) embedding."""
    inertias = [KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z).inertia_
                for k in candidates]
    return find_knee(candidates, inertias)
