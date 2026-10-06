"""Stage 1 of Fair ParPIC: the centered group-indicator matrix F, the
pi-orthogonal projector Pi_pi onto N_F = null(F^T), and the balance objective.

Notation follows the paper: pi = pi_(nu) = (nu + xi) / 1^T (nu + xi) is the
stationary distribution of the P-RW operator P_(nu), and <u, v>_pi = u^T D_pi v
is the inner product in which P_(nu) is self-adjoint.
"""

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


def build_fairness_projector(F, pi=None):
    """The pi-orthogonal projector onto N_F = null(F^T) (Eq. 7 of the paper),

        Pi_pi = I_n - D_pi^{-1} F (F^T D_pi^{-1} F)^{-1} F^T,

    returned as a function M -> Pi_pi M that never forms an n x n matrix:
    each application costs O(n d G). Pi_pi M is the closest matrix to M with
    F^T Y = 0 in the pi-weighted norm (Lemma 2(c)), and the projected power
    iteration Z <- Pi_pi P_(nu) Z is a power iteration of a self-adjoint
    compression of P_(nu) (Theorem 1).

    Parameters
    ----------
    F : (n, G-1) array
        Centered group-indicator matrix.
    pi : (n,) array, optional
        Stationary distribution pi_(nu) of the P-RW operator (any positive
        multiple gives the same projector). If omitted, the uniform measure is
        used, which gives the Euclidean projector I - F (F^T F)^{-1} F^T; this
        is NOT the method of the paper and is kept only for ablations.
    """
    if pi is None:
        DiF = F
    else:
        pi = np.asarray(pi, dtype=float)
        if np.any(pi <= 0):
            raise ValueError("pi must be strictly positive (Remark 1).")
        DiF = F / pi[:, None]                       # D_pi^{-1} F
    gram_inv = np.linalg.inv(F.T @ DiF)             # (F^T D_pi^{-1} F)^{-1}
    return lambda M: M - DiF @ (gram_inv @ (F.T @ M))


def balance_objective(labels, group_ids):
    """balance(C) = min_c min_g |C_c cap S_g| / |C_c|  (Eq. 5 of the paper).
    The best achievable value is alpha_min (Lemma 5, balance ceiling)."""
    vals = []
    for c in np.unique(labels):
        idx_c = labels == c
        for g in np.unique(group_ids):
            vals.append(np.sum(idx_c & (group_ids == g)) / np.sum(idx_c))
    return float(np.min(vals))
