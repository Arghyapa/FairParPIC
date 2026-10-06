"""ParPIC machinery (Debaussart-Joniec et al., 2026): directed k-NN graph,
parametrized random-walk (P-RW) operator, entropy-based diffusion time.

Everything here is fairness-agnostic; the fair projection lives in
``fairness.py`` and the embeddings in ``fair_parpic.py``.
"""

import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors


def find_knee(x, y):
    """Knee of a curve: the point of maximum distance to the chord joining its
    endpoints (KNEEDLE's difference curve without smoothing). Used for the
    diffusion time and for both elbow selections."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if y.max() - y.min() < 1e-12:
        return int(x[0])
    xn = (x - x.min()) / (x.max() - x.min())
    yn = (y - y.min()) / (y.max() - y.min())
    x0, y0, x1, y1 = xn[0], yn[0], xn[-1], yn[-1]
    num = np.abs((y1 - y0) * xn - (x1 - x0) * yn + x1 * y0 - y1 * x0)
    den = np.sqrt((y1 - y0) ** 2 + (x1 - x0) ** 2)
    return int(x[int(np.argmax(num / (den if den > 1e-12 else 1.0)))])


def build_knn_digraph(X, n_neighbors):
    """Unweighted directed k-NN digraph: W_ij = 1 iff j is among the
    n_neighbors nearest neighbors of i (no self-loops)."""
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)
    W = nn.kneighbors_graph(X, mode="connectivity")
    W.setdiag(0)
    W.eliminate_zeros()
    return sp.csr_matrix(W)


def natural_random_walk(W):
    """P = D_out^{-1} W, together with the out- and in-degrees."""
    W = sp.csr_matrix(W, dtype=float)
    d_out = np.asarray(W.sum(axis=1)).flatten()
    d_in = np.asarray(W.sum(axis=0)).flatten()
    d_out_safe = np.where(d_out > 0, d_out, 1.0)
    return sp.csr_matrix(sp.diags(1.0 / d_out_safe) @ W), d_out, d_in


def degree_vertex_measure(d_out, d_in, gamma=0.5):
    """nu_gamma = gamma * d_in / sum(d_in) + (1 - gamma) * d_out / sum(d_out)
    (Eq. 2 of the paper). The floor 1e-12 is inactive for gamma < 1 on a
    k_nn-NN digraph, where nu_gamma(i) >= (1 - gamma) / n; gamma = 1 is not
    covered by the method (Appendix A.4)."""
    nu = gamma * d_in / d_in.sum() + (1.0 - gamma) * d_out / d_out.sum()
    return np.maximum(nu, 1e-12)


def build_p_rw_operator(P, nu, return_pi=False):
    """P_(nu) = (D_nu + D_xi)^{-1} (D_nu P + P^T D_nu), with xi = P^T nu (Eq. 1).

    Row-stochastic and reversible with respect to nu + xi, equivalently with
    respect to its normalization pi_(nu) = (nu + xi) / 1^T (nu + xi)
    (Remark 1(a)). With ``return_pi=True`` also returns pi_(nu), the measure
    of the inner product <u, v>_pi used by the fairness projector Pi_pi."""
    D_nu = sp.diags(nu)
    numerator = D_nu @ P + P.T @ D_nu
    xi = np.asarray(nu @ P).flatten()
    mass = np.where(nu + xi > 1e-15, nu + xi, 1e-15)
    P_nu = sp.csr_matrix(sp.diags(1.0 / mass) @ numerator)
    if return_pi:
        return P_nu, mass / mass.sum()
    return P_nu


def _row_entropy(row):
    p = row[row > 1e-15]
    return 0.0 if len(p) == 0 else float(-np.sum(p * np.log(p)))


def select_diffusion_time(P_nu, t_max=40, n_probes=None, seed=42):
    """Knee of the estimated row-entropy curve
    H_hat(t) = (n / q) * sum_b H_{i_b}(t), from q probe rows of P_(nu)^t
    computed by sparse vector-matrix products (P_(nu)^t is never formed)."""
    n = P_nu.shape[0]
    if n_probes is None:
        n_probes = int(np.ceil(np.sqrt(n)))
    rng = np.random.default_rng(seed)
    probe = rng.choice(n, size=min(n_probes, n), replace=False)
    V = sp.csr_matrix((np.ones(len(probe)), (np.arange(len(probe)), probe)),
                      shape=(len(probe), n))
    H = np.zeros(t_max + 1)
    for t in range(1, t_max + 1):
        V = V @ P_nu
        Vd = V.toarray()
        H[t] = (n / len(probe)) * sum(_row_entropy(Vd[r]) for r in range(Vd.shape[0]))
    return find_knee(np.arange(1, t_max + 1), H[1:])
