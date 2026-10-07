"""
Appendix E.3, Table S3 -- numerical check of Theorems 1 and 2 on the three
datasets (n = 3000, seed 42, defaults of the comparison): full spectra of
P_(nu) and of the compression T_pi = Pi_pi P_(nu) restricted to N_F.

Reports: self-adjointness defects, interlacing (Eq. 9), eigenvalue gaps, the
number of surviving modes at every time t <= 40 and threshold eps
(Theorem 2(c)), and the diffusion-distance identity (Eq. 10) for the projected
operator K_pi = Pi_pi P_(nu) Pi_pi at t*.

Uses dense eigendecompositions (n x n), so it is meant for n = 3000.
Needs the CSV files in data/ (or in $DATA_DIR); run from the repository root.
DATASET=Diabetes,Bank runs a subset.   Output: results/spectrum.json
"""
import json
import os
import sys

import numpy as np
import scipy.linalg as sla

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import load_dataset                                   # noqa: E402
from src.fairness import build_group_indicator_matrix               # noqa: E402
from src.parpic import (build_knn_digraph, build_p_rw_operator,     # noqa: E402
                        degree_vertex_measure, natural_random_walk,
                        select_diffusion_time)

DATA_DIR = os.environ.get("DATA_DIR", "data")
ONLY = os.environ.get("DATASET")
CFG = {"Diabetes": ("diabetes", 12), "Census": ("census", 10), "Bank": ("bank", 12)}   # name: (dataset, elbow k_nn)


def defect(A):
    """||A - A^T||_F / ||A||_F; zero exactly when A is symmetric."""
    return float(np.linalg.norm(A - A.T) / np.linalg.norm(A))


def check(dataset, k_nn):
    X, gid, groups = load_dataset(dataset, n_sample=3000, seed=42, data_dir=DATA_DIR)
    G, n = len(groups), X.shape[0]
    W = build_knn_digraph(X, k_nn)
    P0, d_out, d_in = natural_random_walk(W)
    nu = degree_vertex_measure(d_out, d_in, 0.5)
    P_sparse, pi = build_p_rw_operator(P0, nu, return_pi=True)
    P = P_sparse.toarray()
    t_star = select_diffusion_time(P_sparse, 40, int(np.ceil(np.sqrt(n))), seed=42)
    F, _ = build_group_indicator_matrix(gid, G)

    sq = np.sqrt(pi)
    DP = pi[:, None] * P                                         # matrix of <u, P v>_pi
    S = sq[:, None] * P / sq[None, :]
    lam = np.sort(np.linalg.eigvalsh((S + S.T) / 2))[::-1]
    B = sla.null_space(F.T / sq[None, :]) / sq[:, None]          # pi-orthonormal basis of N_F
    A = B.T @ DP @ B                                             # matrix of <u, T_pi v>_pi
    zeta, V = np.linalg.eigh((A + A.T) / 2)
    order = np.argsort(zeta)[::-1]
    zeta, V = zeta[order], V[:, order]
    m = len(zeta)
    interlacing = bool(np.all(zeta <= lam[:m] + 1e-10) and np.all(zeta >= lam[G - 1:G - 1 + m] - 1e-10))

    worst, at_tstar = 0, None                                     # Theorem 2(c)
    for t in range(1, 41):
        for eps in (0.5, 0.2, 0.1, 0.05, 0.01, 0.001):
            c = eps ** (1.0 / t)
            a = int((np.abs(lam) >= c).sum())
            b = int((np.abs(zeta) >= c).sum())
            assert b <= a <= b + 2 * (G - 1)
            worst = max(worst, a - b)
            if t == t_star and eps == 0.01:
                at_tstar = (a, b)

    # diffusion-distance identity (Eq. 10) for K_pi at t = t*, on 200 random pairs
    DF = F / pi[:, None]
    Pi_pi = np.eye(n) - DF @ np.linalg.inv(F.T @ DF) @ F.T
    K = Pi_pi @ P @ Pi_pi
    Kt = np.linalg.matrix_power(K, int(t_star))
    Psi = B @ V
    rng = np.random.default_rng(0)
    rel = 0.0
    for _ in range(200):
        i, j = rng.choice(n, 2, replace=False)
        lhs = (((Kt[i] - Kt[j]) ** 2) / pi).sum()
        rhs = ((zeta ** (2 * t_star)) * (Psi[i] - Psi[j]) ** 2).sum()
        rel = max(rel, abs(lhs - rhs) / rhs)

    return {"G": G, "t_star": int(t_star), "pi_ratio": float(pi.max() / pi.min()),
            "defect_P": defect(DP), "defect_T_pi": defect(A), "interlacing": interlacing,
            "zeta_1": float(zeta[0]), "zeta_min": float(zeta[-1]), "lambda_min": float(lam[-1]),
            "max_gap_top50": float(np.abs(zeta[:50] - lam[:50]).max()),
            "max_gap_all": float(np.abs(zeta - lam[:m]).max()),
            "max_mode_count_difference": int(worst), "modes_at_tstar_eps0.01": at_tstar,
            "diffusion_identity_rel_error": float(rel),
            "negative_entries_share_K": float((K < -1e-12).mean())}


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    out = {}
    for name, (dataset, k_nn) in CFG.items():
        if ONLY and name not in ONLY.split(","):
            continue
        out[name] = check(dataset, k_nn)
        print(name, json.dumps(out[name]), flush=True)
        with open("results/spectrum.json", "w") as fh:
            json.dump(out, fh, indent=1)
