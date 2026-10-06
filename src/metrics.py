"""Evaluation metrics used in the paper."""

import numpy as np
from sklearn.metrics import silhouette_score

from .fairness import balance_objective  # re-exported

__all__ = ["balance_objective", "wcss", "price_of_fairness",
           "substantive_clusters", "silhouette"]


def wcss(Z, labels):
    """Within-cluster sum of squares in the embedding."""
    return float(sum(((Z[labels == c] - Z[labels == c].mean(axis=0)) ** 2).sum()
                     for c in np.unique(labels)))


def price_of_fairness(Z, labels_fair, labels_kmeans):
    """PoF = (WCSS(C_FA; Z) - WCSS(C_k-means; Z)) / WCSS(C_k-means; Z), where Z
    is the embedding on which both clusterings are computed (Section 6). Can be
    slightly negative, since k-means is a heuristic."""
    w_km = wcss(Z, labels_kmeans)
    return (wcss(Z, labels_fair) - w_km) / w_km


def substantive_clusters(labels, min_share=0.01):
    """Number of clusters holding at least `min_share` of the vertices; exposes
    degenerate solutions (balance alone rewards a single cluster)."""
    _, counts = np.unique(labels, return_counts=True)
    return int(np.sum(counts >= min_share * len(labels)))


def silhouette(Z, labels, sample_size=None, seed=42):
    """Silhouette score in the embedding (Rousseeuw, 1987)."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(silhouette_score(Z, labels, sample_size=sample_size, random_state=seed))
