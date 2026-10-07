"""
Synthetic, naturally directed networks with weak-to-strong group signal
(Appendix E.2 and Table S2 of the paper): ParPIC, FP-ParPIC (Stage 1 alone,
with k-means) and Fair ParPIC (Algorithm 1), all with the pi-orthogonal
projector Pi_pi of the paper.

Every network is a directed stochastic block model (no k_nn-NN graph, no
features). Every vertex receives a distinct arrival time, and the edge i -> j
is present independently with probability (Eq. S23)

    p(i -> j) = min{1, sigma * beta(i, j) * o(i, j) * rho(j)}

    block factor        beta(i, j) = p_cl(i, j) * (1 + eta)  if i, j are in the same group
                                     p_cl(i, j) * (1 - eta)  otherwise,
                        p_cl(i, j) = p_in  = 0.030 if i, j are in the same planted cluster
                                     p_out = 0.006 otherwise;
    orientation factor  o(i, j)    = 2 p_b if j arrived before i, 2 (1 - p_b) otherwise,
                        so a fraction p_b of the edge mass points from later
                        to earlier vertices, as citations do;
    popularity factor   rho(j) > 0 with mean one;
    scale factor        sigma = 3000 / n keeps the expected degree independent of n.

Self-loops are excluded, and a vertex without outgoing edges is linked to one
random vertex of its cluster.

The GROUP SIGNAL eta in [0, 1) is the only thing that changes within a network
type: eta = 0 means the protected groups play no role in the links; as eta
grows, vertices link preferentially inside their own group, and for
eta > 2/3 a same-group link ACROSS clusters is more likely than a cross-group
link INSIDE a cluster (the groups become the dominant community structure).

Three network types:
  A  "citation":   n = 10000, 5 equal clusters, 2 groups (30/70) with the
     population shares in every cluster (the planted clustering is fair);
     p_b = 0.9, rho = 1.
  B  "popularity": n = 8000, 8 clusters of unequal size (25% down to 4%),
     3 groups (15/35/50) in every cluster; p_b = 0.9 and a heavy-tailed
     popularity rho (Pareto with shape 2.5, normalized to mean one).
  C  "hyperlink":  n = 6000, 4 equal clusters, 2 groups (30/70 overall) spread
     UNEVENLY over the clusters (minority share 15%, 25%, 35%, 45%), so the
     planted clustering itself is unfair (balance 0.150); p_b = 0.7, i.e. more
     reciprocity, rho = 1.

For each network: balance and ARI to the planted clusters of ParPIC + k-means,
FP-ParPIC + k-means and Fair ParPIC, and the price of fairness of Stage 2.
k = number of planted clusters, delta = 0.05, gamma = 0.5, d = ceil(sqrt(n)),
t* from the entropy criterion. One run per network (seed 42).

Run a subset with   TYPES=A,B LEVELS=none,strong python experiment_synthetic.py
Output: results/synthetic_suite.json
"""

import json
import os
import time

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from src.fair_assignment import fair_kmeans
from src.fair_parpic import parpic_embeddings
from src.metrics import balance_objective, wcss

# ======================================================================
# Settings
# ======================================================================
SEED = 42
GAMMA = 0.5
DELTA = 0.05
T_MAX = 40
MAX_ALTERNATIONS = 10
P_IN, P_OUT = 0.030, 0.006            # at n = 3000; scaled by sigma = 3000 / n
SIGNALS = {"none": 0.0, "weak": 0.3, "moderate": 0.6, "strong": 0.9}   # eta

NETWORKS = {
    "A citation": dict(
        n=10000, cluster_shares=[0.2] * 5,
        group_shares=[[0.3, 0.7]] * 5, p_b=0.9, pareto=None),
    "B popularity": dict(
        n=8000, cluster_shares=[0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04],
        group_shares=[[0.15, 0.35, 0.50]] * 8, p_b=0.9, pareto=2.5),
    "C hyperlink": dict(
        n=6000, cluster_shares=[0.25] * 4,
        group_shares=[[0.15, 0.85], [0.25, 0.75], [0.35, 0.65], [0.45, 0.55]],
        p_b=0.7, pareto=None),
}


# ======================================================================
# Directed network generator (Eq. S23)
# ======================================================================
def directed_network(n, cluster_shares, group_shares, p_b, pareto, eta):
    rng = np.random.default_rng(SEED)
    sizes = np.round(np.array(cluster_shares) * n).astype(int)
    sizes[0] += n - sizes.sum()
    cluster = np.repeat(np.arange(len(sizes)), sizes)
    group = np.empty(n, dtype=int)
    for c, size in enumerate(sizes):                # group shares inside cluster c
        counts = np.round(np.array(group_shares[c]) * size).astype(int)
        counts[-1] += size - counts.sum()
        group[cluster == c] = rng.permutation(np.repeat(np.arange(len(counts)), counts))
    arrival = rng.permutation(n)
    rho = np.ones(n)                                # popularity factor
    if pareto is not None:                          # heavy-tailed, normalized to mean one
        rho = rng.pareto(pareto, n) + 1
        rho /= rho.mean()
    sigma = 3000 / n                                # scale factor

    rows, cols = [], []
    for start in range(0, n, 1000):                 # row blocks keep memory small
        i = np.arange(start, min(start + 1000, n))
        same_c = cluster[i, None] == cluster[None, :]
        same_g = group[i, None] == group[None, :]
        p_cl = np.where(same_c, P_IN, P_OUT)
        beta = p_cl * np.where(same_g, 1 + eta, 1 - eta)                     # block factor
        o = np.where(arrival[None, :] < arrival[i, None], 2 * p_b, 2 * (1 - p_b))  # orientation
        prob = np.minimum(sigma * beta * o * rho[None, :], 1.0)
        hit = rng.random(prob.shape) < prob
        hit[np.arange(len(i)), i] = False           # no self-loops
        r, c = np.nonzero(hit)
        rows.append(i[r]); cols.append(c)
    W = sp.csr_matrix((np.ones(sum(len(r) for r in rows)),
                       (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    W = sp.lil_matrix(W)
    for i in np.where(np.asarray(W.sum(axis=1)).ravel() == 0)[0]:   # no dangling vertices
        cand = np.where((cluster == cluster[i]) & (np.arange(n) != i))[0]
        W[i, rng.choice(cand)] = 1
    return sp.csr_matrix(W), cluster, group


# ======================================================================
# One network: ParPIC, FP-ParPIC and Fair ParPIC on a shared embedding run
# ======================================================================
def run_network(spec, eta):
    W, cluster, group = directed_network(eta=eta, **spec)
    n = W.shape[0]
    K = len(spec["cluster_shares"])
    G = len(spec["group_shares"][0])
    alpha_min = float((np.bincount(group) / n).min())
    d_in = np.asarray(W.sum(axis=0)).ravel()
    rec = {"n": n, "K": K, "G": G, "eta": eta, "edges": int(W.sum()),
           "alpha_min": alpha_min, "floor": (1 - DELTA) * alpha_min,
           "one_way_edges": float(1 - W.multiply(W.T).sum() / W.sum()),
           "max_in_degree": int(d_in.max()),
           "planted balance": balance_objective(cluster, group)}

    # Shared graph, operator, diffusion time and random start; the FP-ParPIC
    # embedding differs only by the projection Pi_pi after every step.
    Z_parpic, Z_fp, t_star, max_FtZ = parpic_embeddings(
        None, group, G, gamma=GAMMA, t_max=T_MAX, seed=SEED, W=W)
    rec["t_star"] = int(t_star)
    rec["max_FtZ"] = max_FtZ

    km_parpic = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit_predict(Z_parpic)
    km_fp = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit_predict(Z_fp)
    fair = fair_kmeans(Z_fp, group, G, K, DELTA, seed=SEED, max_iter=MAX_ALTERNATIONS)
    for name, labels in [("ParPIC", km_parpic), ("FP-ParPIC", km_fp), ("Fair ParPIC", fair)]:
        rec[f"{name} balance"] = balance_objective(labels, group)
        rec[f"{name} ARI"] = float(adjusted_rand_score(cluster, labels))
    w_km = wcss(Z_fp, km_fp)                        # PoF of Stage 2, as in Table 3
    rec["Fair ParPIC pof"] = (wcss(Z_fp, fair) - w_km) / w_km
    return rec


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    out_path = "results/synthetic_suite.json"
    wanted = os.environ.get("TYPES")
    levels = os.environ.get("LEVELS")
    for name, spec in NETWORKS.items():
        if wanted and name[0] not in wanted.split(","):
            continue
        for level, eta in SIGNALS.items():
            if levels and level not in levels.split(","):
                continue
            t0 = time.time()
            rec = run_network(spec, eta)
            rec["seconds"] = round(time.time() - t0, 1)
            results = json.load(open(out_path)) if os.path.exists(out_path) else {}
            results[f"{name} | {level}"] = rec
            with open(out_path, "w") as fh:
                json.dump(results, fh, indent=2)
            print(f"{name:13s} {level:9s} eta={eta:.1f} n={rec['n']} edges={rec['edges']} "
                  f"one-way={100 * rec['one_way_edges']:.0f}% t*={rec['t_star']} "
                  f"floor={rec['floor']:.4f} planted={rec['planted balance']:.4f} | balance "
                  f"ParPIC {rec['ParPIC balance']:.4f}  FP-ParPIC {rec['FP-ParPIC balance']:.4f}  "
                  f"Fair ParPIC {rec['Fair ParPIC balance']:.4f} | ARI {rec['ParPIC ARI']:.3f} / "
                  f"{rec['FP-ParPIC ARI']:.3f} / {rec['Fair ParPIC ARI']:.3f} | "
                  f"PoF {100 * rec['Fair ParPIC pof']:.2f}%  ({rec['seconds']}s)", flush=True)
