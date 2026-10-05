"""Stage 1 of Fair ParPIC: the centered group-indicator matrix F and the
fairness projector Pi_F onto null(F^T), plus the balance objective."""

import numpy as np


def build_group_indicator_matrix(group_ids, n_groups):
    """F(i, g) = 1{i in S_g} - alpha_g for g = 1, ..., G-1 (Eq. 6 of the paper).

    Returns F (n x (G-1)) and the group proportions alpha (length G)."""
    n = len(group_ids)
    alpha = np.bincount(group_ids, minlength=n_groups) / n
    F = np.zeros((n, n_groups - 1))
    for j in range(n_groups - 1):
        F[:, j] = (group_ids == j).astype(float) - alpha[j]
    return F, alpha


def build_fairness_projector(F):
    """Pi_F = I - F (F^T F)^{-1} F^T, applied without forming an n x n matrix:
    each application costs O(n d G)."""
    FtF_inv = np.linalg.inv(F.T @ F)
    return lambda M: M - F @ (FtF_inv @ (F.T @ M))


def balance_objective(labels, group_ids):
    """balance(C) = min_c min_g |C_c cap S_g| / |C_c|  (Eq. 5 of the paper).
    The best achievable value is alpha_min (Lemma: balance ceiling)."""
    vals = []
    for c in np.unique(labels):
        idx_c = labels == c
        for g in np.unique(group_ids):
            vals.append(np.sum(idx_c & (group_ids == g)) / np.sum(idx_c))
    return float(np.min(vals))
