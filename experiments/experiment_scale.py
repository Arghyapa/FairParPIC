"""
Appendix E.1, Table S1 -- Stage 1 at larger scale: ParPIC vs. FP-ParPIC
(Stage 1 only, k-means discretization) on

    Diabetes  stratified subsample of 70,000 vertices   (gender, G = 2)
    Census    full dataset                               (gender, G = 2)
    Bank      full dataset                               (marital, G = 3)

with k_nn = 10, k = 10, gamma = 0.5, d = ceil(sqrt(n)), t_max = 40.
One run per dataset (seed 42). Both methods share the graph, the operator,
the diffusion time and the random start; they differ only in the projection
Pi_pi = I - D_pi^-1 F (F^T D_pi^-1 F)^-1 F^T (orthogonal in the pi inner product).

Reported per dataset: n, G, alpha_min, t*, and for each method the balance,
the number of substantive clusters (>= 1% of n), the largest distance between
two group centroids in the embedding, max |F^T Z|, and the time spent in the
power iteration (to show what the projection adds).

Fully self-contained. Needs the CSV files in data/ (or in $DATA_DIR); run from
the repository root. Run one dataset with
    python experiments/experiment_scale.py Bank
Output: results/scale_<dataset>.json
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

SEED = 42
DATA_DIR = os.environ.get("DATA_DIR", "data")   # folder with the CSV files (data/README.md)
K = 10
K_NN = 10
GAMMA = 0.5
T_MAX = 40
DATASETS = {   # name: (csv, sensitive column, categories, separator, sample size or None = full)
    "Diabetes": ("diabetic_data.csv", "gender", ("Male", "Female"), ",", 70000),
    "Census":   ("uci_census.csv", "gender", ("Male", "Female"), ",", None),
    "Bank":     ("bank-full.csv", "marital", ("married", "single", "divorced"), ";", None),
}


def load_dataset(csv_path, sensitive_col, valid_categories, sep, n_sample):
    df = pd.read_csv(os.path.join(DATA_DIR, csv_path), sep=sep)
    df = df.replace("?", np.nan).replace("unknown", np.nan)
    df = df[df[sensitive_col].isin(valid_categories)]
    missing = df.isnull().mean()
    df = df.drop(columns=missing[missing > 0.4].index)
    numeric_cols = df.select_dtypes(exclude=["object"]).columns
    df = df[list(numeric_cols) + [sensitive_col]]
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    if n_sample is not None:                         # stratified subsample
        parts = []
        for _, g in df.groupby(sensitive_col):
            parts.append(g.sample(n=min(len(g), int(round(n_sample * len(g) / len(df)))),
                                  random_state=SEED))
        df = pd.concat(parts, axis=0).reset_index(drop=True)
    groups, group_ids = np.unique(df[sensitive_col].astype(str), return_inverse=True)
    X = StandardScaler().fit_transform(df.drop(columns=[sensitive_col]).values)
    return X, group_ids, len(groups)


def find_knee(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if y.max() - y.min() < 1e-12:
        return int(x[0])
    xn = (x - x.min()) / (x.max() - x.min())
    yn = (y - y.min()) / (y.max() - y.min())
    num = np.abs((yn[-1] - yn[0]) * xn - (xn[-1] - xn[0]) * yn + xn[-1] * yn[0] - yn[-1] * xn[0])
    return int(x[int(np.argmax(num))])


def balance_objective(labels, group_ids):
    return float(min(np.sum((labels == c) & (group_ids == g)) / np.sum(labels == c)
                     for c in np.unique(labels) for g in np.unique(group_ids)))


def centroid_gap(Z, group_ids, G):
    """Largest Euclidean distance between two group centroids."""
    mu = np.array([Z[group_ids == g].mean(axis=0) for g in range(G)])
    return float(max(np.linalg.norm(mu[a] - mu[b]) for a in range(G) for b in range(a + 1, G)))


def run(name):
    csv, sens, cats, sep, n_sample = DATASETS[name]
    X, gid, G = load_dataset(csv, sens, cats, sep, n_sample)
    n = X.shape[0]
    d = int(np.ceil(np.sqrt(n)))
    alpha = np.bincount(gid, minlength=G) / n
    print(f"{name}: n = {n}, features = {X.shape[1]}, G = {G}, alpha = {np.round(alpha, 4).tolist()}", flush=True)

    # ---- graph and P-RW operator
    t0 = time.time()
    W = NearestNeighbors(n_neighbors=K_NN + 1).fit(X).kneighbors_graph(X, mode="connectivity")
    W.setdiag(0)
    W.eliminate_zeros()
    W = sp.csr_matrix(W)
    d_out = np.asarray(W.sum(axis=1)).ravel()
    d_in = np.asarray(W.sum(axis=0)).ravel()
    P = sp.csr_matrix(sp.diags(1.0 / np.where(d_out > 0, d_out, 1.0)) @ W)
    nu = np.maximum(GAMMA * d_in / d_in.sum() + (1 - GAMMA) * d_out / d_out.sum(), 1e-12)
    xi = np.asarray(nu @ P).ravel()
    P_nu = sp.csr_matrix(sp.diags(1.0 / np.maximum(nu + xi, 1e-15))
                         @ (sp.diags(nu) @ P + P.T @ sp.diags(nu)))
    t_graph = time.time() - t0
    one_way = 1.0 - W.multiply(W.T).nnz / W.nnz
    print(f"   graph: {W.nnz} edges, {100 * one_way:.1f}% without a reverse edge, "
          f"max in-degree {int(d_in.max())}   ({t_graph:.0f}s)", flush=True)

    # ---- diffusion time (entropy criterion)
    t0 = time.time()
    probe = np.random.default_rng(SEED).choice(n, size=d, replace=False)
    V = sp.csr_matrix((np.ones(d), (np.arange(d), probe)), shape=(d, n))
    H = []
    for _ in range(T_MAX):
        V = V @ P_nu
        Vd = V.toarray()
        H.append(sum(-np.sum(r[r > 1e-15] * np.log(r[r > 1e-15])) for r in Vd))
    t_star = find_knee(np.arange(1, T_MAX + 1), H)
    del V, Vd
    print(f"   t* = {t_star}   ({time.time() - t0:.0f}s)", flush=True)

    # ---- embeddings
    F = np.column_stack([(gid == j) - alpha[j] for j in range(G - 1)]).astype(float)
    pi = nu + xi                                     # P_(nu) is reversible w.r.t. pi
    DF = F / pi[:, None]                             # D_pi^{-1} F
    FtF_inv = np.linalg.inv(F.T @ DF)
    out = {"n": n, "features": int(X.shape[1]), "G": G, "alpha_min": float(alpha.min()),
           "d": d, "t_star": t_star, "edges": int(W.nnz), "one_way_edges": float(one_way),
           "graph_seconds": t_graph}
    for method in ("ParPIC", "FP-ParPIC"):
        Z = np.random.default_rng(SEED).standard_normal((n, d))
        t0 = time.time()
        for _ in range(t_star):
            Z = P_nu @ Z
            if method == "FP-ParPIC":
                Z -= DF @ (FtF_inv @ (F.T @ Z))             # Pi_pi
        t_iter = time.time() - t0
        t0 = time.time()
        labels = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit_predict(Z)
        t_km = time.time() - t0
        sizes = np.bincount(labels, minlength=K)
        out[method] = {
            "balance": balance_objective(labels, gid),
            "substantive_clusters": int((sizes >= 0.01 * n).sum()),
            "sizes": sizes.tolist(),
            "centroid_gap": centroid_gap(Z, gid, G),
            "max_FtZ": float(np.abs(F.T @ Z).max()),
            "iteration_seconds": t_iter, "kmeans_seconds": t_km,
        }
        r = out[method]
        print(f"   {method:10s} balance {r['balance']:.4f}   clusters {r['substantive_clusters']}   "
              f"centroid gap {r['centroid_gap']:.2e}   max|F^T Z| {r['max_FtZ']:.1e}   "
              f"iteration {t_iter:.1f}s   k-means {t_km:.0f}s", flush=True)
        del Z
    os.makedirs("results", exist_ok=True)
    with open(f"results/scale_{name}.json", "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    for name in (sys.argv[1:] or DATASETS):
        run(name)
