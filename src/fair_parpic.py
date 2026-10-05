"""Fair ParPIC: fair-projected power iteration (Stage 1) followed by the
alternating fair assignment (Stage 2)."""

import numpy as np
import scipy.sparse as sp

from .fair_assignment import fair_kmeans
from .fairness import (balance_objective, build_fairness_projector,
                       build_group_indicator_matrix)
from .parpic import (build_knn_digraph, build_p_rw_operator,
                     degree_vertex_measure, natural_random_walk,
                     select_diffusion_time)


def p_rw_from_graph(W, gamma=0.5):
    """Natural random walk of a digraph W -> ParPIC's reversible operator."""
    P, d_out, d_in = natural_random_walk(W)
    return build_p_rw_operator(P, degree_vertex_measure(d_out, d_in, gamma))


def parpic_embeddings(X, group_ids, n_groups, n_neighbors=None, gamma=0.5,
                      t_max=40, seed=42, W=None, d=None):
    """ParPIC and Fair-Projected ParPIC embeddings from one shared graph,
    operator, diffusion time and random start Z^(0); the only difference is
    the projection Pi_F after every power-iteration step.

    Pass either a point cloud X (a directed k-NN digraph with n_neighbors is
    built) or a directed adjacency matrix W.

    Returns (Z_parpic, Z_fair, t_star, max_ij |(F^T Z_fair)_ij|)."""
    if W is None:
        W = build_knn_digraph(X, n_neighbors)
    W = sp.csr_matrix(W)
    n = W.shape[0]
    d = int(np.ceil(np.sqrt(n))) if d is None else d
    P_nu = p_rw_from_graph(W, gamma)
    t_star = select_diffusion_time(P_nu, t_max, d, seed=seed)

    F, _ = build_group_indicator_matrix(group_ids, n_groups)
    Pi_F = build_fairness_projector(F)

    Z0 = np.random.default_rng(seed).standard_normal((n, d))
    Zb, Zf = Z0.copy(), Z0.copy()
    for _ in range(t_star):
        Zb = P_nu @ Zb                 # ParPIC:       Z <- P_(nu) Z
        Zf = Pi_F(P_nu @ Zf)           # Fair ParPIC:  Z <- Pi_F P_(nu) Z
    return Zb, Zf, t_star, float(np.max(np.abs(F.T @ Zf)))


class FairParPIC:
    """Fair Parametrized Power-Iteration Clustering.

    Parameters
    ----------
    n_clusters : int
        Number of clusters k.
    n_neighbors : int
        k_nn of the directed k-NN digraph (ignored when ``adjacency`` is given).
    delta : float in [0, 1)
        Fairness tolerance: the output satisfies
        (1 - delta) alpha_min <= balance <= alpha_min.
    gamma : float in [0, 1)
        Vertex-measure parameter of nu_gamma.
    t_max : int
        Largest diffusion time scanned by the entropy criterion.
    solver : {"decomposition", "milp"}
        Solver of the fair assignment problem (P_delta).
    max_iter : int
        Maximum number of alternations s_max.
    random_state : int

    Attributes (after ``fit``)
    --------------------------
    labels_, embedding_, t_star_, balance_, alpha_min_, floor_,
    fairness_residual_ (max |F^T Z|), groups_ (original group labels),
    solver_info_ (certified gap etc.).
    """

    def __init__(self, n_clusters, n_neighbors=10, delta=0.05, gamma=0.5,
                 t_max=40, solver="decomposition", max_iter=10, random_state=42):
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.delta = delta
        self.gamma = gamma
        self.t_max = t_max
        self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X=None, sensitive=None, adjacency=None):
        if sensitive is None:
            raise ValueError("`sensitive` (one group label per vertex) is required.")
        self.groups_, group_ids = np.unique(np.asarray(sensitive), return_inverse=True)
        G = len(self.groups_)
        if G < 2:
            raise ValueError("At least two sensitive groups are required.")
        _, Zf, t_star, resid = parpic_embeddings(
            X, group_ids, G, self.n_neighbors, self.gamma, self.t_max,
            self.random_state, W=adjacency)
        labels, info = fair_kmeans(Zf, group_ids, G, self.n_clusters, self.delta,
                                   solver=self.solver, seed=self.random_state,
                                   max_iter=self.max_iter, return_info=True)
        alpha = np.bincount(group_ids, minlength=G) / len(group_ids)
        self.labels_ = labels
        self.embedding_ = Zf
        self.t_star_ = t_star
        self.fairness_residual_ = resid
        self.alpha_min_ = float(alpha.min())
        self.floor_ = (1 - self.delta) * self.alpha_min_
        self.balance_ = balance_objective(labels, group_ids)
        self.solver_info_ = info
        return self

    def fit_predict(self, X=None, sensitive=None, adjacency=None):
        return self.fit(X, sensitive, adjacency).labels_
