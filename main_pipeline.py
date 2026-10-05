"""Real-data experiment of the paper (Section 6, Tables 2 and 3).

For each dataset and seed it runs, on one shared graph, operator, diffusion
time and random start:

    ParPIC        + k-means          (unconstrained baseline)
    FP-ParPIC     + k-means          (Stage 1 alone)
    ParPIC        + fair assignment  (ablation: Stage 2 without the projection)
    Fair ParPIC   = FP-ParPIC + fair assignment   (Algorithm 1)

and reports balance, number of substantive clusters, price of fairness,
silhouette score, max |F^T Z| and the certified optimality gap.

Examples
--------
    python main_pipeline.py --dataset census
    python main_pipeline.py --dataset all --k 7 --seeds 42 0 1 2 3
    python main_pipeline.py --dataset bank --delta 0.1 --solver milp
"""

import argparse
import json
import os
import time

import numpy as np
from sklearn.cluster import KMeans

from src.data import DATASETS, load_dataset
from src.fair_assignment import fair_kmeans
from src.fair_parpic import parpic_embeddings
from src.metrics import (balance_objective, price_of_fairness, silhouette,
                         substantive_clusters)
from src.model_selection import select_k_by_elbow, select_n_neighbors_by_elbow

METHODS = ["ParPIC", "FP-ParPIC", "ParPIC + FA", "Fair ParPIC"]


def run_once(X, gid, G, n_neighbors, k, delta, solver, seed, gamma, t_max):
    Zb, Zf, t_star, max_FtZ = parpic_embeddings(X, gid, G, n_neighbors, gamma, t_max, seed)
    out = {"t_star": t_star, "max_FtZ": max_FtZ}
    for emb, Z, km_name, fa_name in [("parpic", Zb, "ParPIC", "ParPIC + FA"),
                                     ("fair", Zf, "FP-ParPIC", "Fair ParPIC")]:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
        fa, info = fair_kmeans(Z, gid, G, k, delta, solver=solver, seed=seed,
                               init_centers=km.cluster_centers_, return_info=True)
        for name, lab in [(km_name, km.labels_), (fa_name, fa)]:
            out[name] = {"balance": balance_objective(lab, gid),
                         "clusters": substantive_clusters(lab),
                         "silhouette": silhouette(Z, lab, seed=seed)}
        out[fa_name]["pof"] = price_of_fairness(Z, fa, km.labels_)
        out[fa_name]["gap"] = info.get("gap")
        out[fa_name]["alternations"] = info.get("n_iter")
    return out


def run_dataset(name, args):
    print(f"\n{'=' * 78}\n{name}  (delta = {args.delta}, solver = {args.solver}, n = {args.n_sample})")
    # hyper-parameters fixed once on the seed-42 subsample (elbow, no sensitive attribute)
    X, gid, groups = load_dataset(name, n_sample=args.n_sample, seed=42, data_dir=args.data_dir)
    G = len(groups)
    alpha = np.bincount(gid, minlength=G) / len(gid)
    nn = args.n_neighbors or select_n_neighbors_by_elbow(X)
    if args.k:
        k = args.k
    else:
        Zb, _, _, _ = parpic_embeddings(X, gid, G, nn, args.gamma, args.t_max, 42)
        k = select_k_by_elbow(Zb)
    floor, ceiling = (1 - args.delta) * alpha.min(), alpha.min()
    print(f"groups = {list(groups)}, alpha = {np.round(alpha, 3).tolist()}, "
          f"k_nn = {nn}, k = {k}, floor = {floor:.4f}, ceiling = {ceiling:.4f}")

    runs = []
    for s in args.seeds:
        t0 = time.time()
        X, gid, _ = load_dataset(name, n_sample=args.n_sample, seed=s, data_dir=args.data_dir)
        r = run_once(X, gid, G, nn, k, args.delta, args.solver, s, args.gamma, args.t_max)
        r["seed"], r["seconds"] = s, round(time.time() - t0, 1)
        runs.append(r)
        print(f"  seed {s:>2}: t*={r['t_star']:>2}  " +
              "  ".join(f"{m}={r[m]['balance']:.3f}" for m in METHODS) +
              f"  |F^T Z|={r['max_FtZ']:.1e}  PoF {100 * r['ParPIC + FA']['pof']:.1f}% -> "
              f"{100 * r['Fair ParPIC']['pof']:.1f}%  ({r['seconds']}s)", flush=True)

    summary = {"k": k, "n_neighbors": nn, "alpha": alpha.tolist(), "floor": floor,
               "ceiling": ceiling, "runs": runs}
    print(f"  mean ± std over {len(runs)} seeds:")
    for m in METHODS:
        b = np.array([r[m]["balance"] for r in runs])
        cl = np.mean([r[m]["clusters"] for r in runs])
        sil = np.array([r[m]["silhouette"] for r in runs])
        line = (f"    {m:<12} balance {b.mean():.3f} ± {b.std():.3f}   Cl. {cl:.1f}   "
                f"silhouette {sil.mean():.3f} ± {sil.std():.3f}")
        if "pof" in runs[0][m]:
            p = 100 * np.array([r[m]["pof"] for r in runs])
            line += f"   PoF {p.mean():.1f}% ± {p.std():.1f}%"
        print(line)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="all", choices=list(DATASETS) + ["all"])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--n-sample", type=int, default=3000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 0, 1, 2, 3])
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--k", type=int, default=None, help="number of clusters (default: elbow)")
    ap.add_argument("--n-neighbors", type=int, default=None, help="k_nn (default: elbow)")
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--t-max", type=int, default=40)
    ap.add_argument("--solver", default="decomposition", choices=["decomposition", "milp"])
    ap.add_argument("--out", default="results/real_data.json")
    args = ap.parse_args()

    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    results = {name: run_dataset(name, args) for name in names}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
