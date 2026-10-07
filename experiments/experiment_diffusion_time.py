"""
Appendix E.3, Fig. S1 -- behavior of the projected iteration as the
diffusion time changes.

ParPIC selects the diffusion time t* at the knee of the entropy of the
UNPROJECTED operator P. FP-ParPIC iterates T_pi = Pi_pi P but keeps
that t*. This experiment runs both iterations for t = 1, ..., 40 from the
same start and records, at each checkpoint t:

  effective rank   exp(entropy of the normalized squared singular values of
                   the centered embedding): how many modes are still alive.
                   If the projection left the time scales of the diffusion
                   unchanged, the two curves coincide.
  balance          ParPIC + k-means, and Fair ParPIC (projection + fair
                   assignment, delta = 0.05, k = 7)
  price of fairness of Fair ParPIC and of ParPIC + fair assignment
  ARI to t*        adjusted Rand index between the Fair ParPIC clustering at
                   t and the one at t* (stability around the selected time)

Fully self-contained. One run per setting (seed 42, n = 3000). Needs the
CSV files in data/ (or in $DATA_DIR); run from the repository root.
Run one dataset with  DATASET=Census python experiments/experiment_diffusion_time.py
(default: all three). Outputs results/sweep_time_<dataset>.json and, when all
three exist, figures/fig_time.pdf/.png
"""

import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from matplotlib.lines import Line2D
from scipy.optimize import Bounds, LinearConstraint, milp
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# ======================================================================
# Settings
# ======================================================================
SEED = 42
N_SAMPLE = 3000
DELTA = 0.05
K = 7
T_MAX = 40
MAX_ALTERNATIONS = 10
GAMMA = 0.5
T_VALUES = [1, 2, 3, 5, 7, 15, 20, 30, 40]     # plus the selected t* of each dataset

DATASETS = {         # name: (csv file, sensitive column, categories, separator, elbow k_nn)
    "Diabetes": ("diabetic_data.csv", "gender", ("Male", "Female"), ",", 12),
    "Census":   ("uci_census.csv", "gender", ("Male", "Female"), ",", 10),
    "Bank":     ("bank-full.csv", "marital", ("married", "single", "divorced"), ";", 12),
}


DATA_DIR = os.environ.get("DATA_DIR", "data")   # folder with the CSV files (data/README.md)
ONLY = os.environ.get("DATASET")                # e.g. DATASET=Diabetes,Bank runs a subset


def selected(datasets):
    """The datasets to run: all of them, or those named in $DATASET."""
    return [(n, v) for n, v in datasets.items() if not ONLY or n in ONLY.split(",")]


# ======================================================================
# 1. Data
# ======================================================================
def load_dataset(csv_path, sensitive_col, valid_categories, sep):
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

    parts = []
    for _, g in df.groupby(sensitive_col):           # stratified subsample
        n_g = min(len(g), int(round(N_SAMPLE * len(g) / len(df))))
        parts.append(g.sample(n=n_g, random_state=SEED))
    df = pd.concat(parts, axis=0).reset_index(drop=True)

    groups, group_ids = np.unique(df[sensitive_col].astype(str), return_inverse=True)
    X = StandardScaler().fit_transform(df.drop(columns=[sensitive_col]).values)
    return X, group_ids, len(groups)


# ======================================================================
# 2. Stage 1: ParPIC and FP-ParPIC embeddings
# ======================================================================
def find_knee(x, y):
    """Point of maximum distance to the chord joining the curve's endpoints."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if y.max() - y.min() < 1e-12:
        return int(x[0])
    xn = (x - x.min()) / (x.max() - x.min())
    yn = (y - y.min()) / (y.max() - y.min())
    num = np.abs((yn[-1] - yn[0]) * xn - (xn[-1] - xn[0]) * yn
                 + xn[-1] * yn[0] - yn[-1] * xn[0])
    return int(x[int(np.argmax(num))])


def p_rw_operator(X, k_nn, gamma):
    """k_nn-NN digraph -> natural random walk P -> P-RW operator P_(nu)."""
    W = NearestNeighbors(n_neighbors=k_nn + 1).fit(X).kneighbors_graph(X, mode="connectivity")
    W.setdiag(0)
    W.eliminate_zeros()
    W = sp.csr_matrix(W)
    d_out = np.asarray(W.sum(axis=1)).ravel()
    d_in = np.asarray(W.sum(axis=0)).ravel()
    P = sp.diags(1.0 / np.where(d_out > 0, d_out, 1.0)) @ W
    nu = np.maximum(gamma * d_in / d_in.sum() + (1 - gamma) * d_out / d_out.sum(), 1e-12)
    xi = np.asarray(nu @ P).ravel()
    num = sp.diags(nu) @ P + P.T @ sp.diags(nu)
    p_rw_operator.pi = nu + xi                   # P_(nu) is reversible w.r.t. pi = nu + xi
    return sp.csr_matrix(sp.diags(1.0 / np.maximum(nu + xi, 1e-15)) @ num)


def diffusion_time(P_nu, t_max, n_probes):
    """Knee of the row-wise operator entropy H(t), estimated on probe rows."""
    n = P_nu.shape[0]
    probe = np.random.default_rng(SEED).choice(n, size=min(n_probes, n), replace=False)
    V = sp.csr_matrix((np.ones(len(probe)), (np.arange(len(probe)), probe)),
                      shape=(len(probe), n))
    H = []
    for _ in range(t_max):
        V = V @ P_nu
        Vd = V.toarray()
        H.append(sum(-np.sum(r[r > 1e-15] * np.log(r[r > 1e-15])) for r in Vd))
    return find_knee(np.arange(1, t_max + 1), H)


# ======================================================================
# 3. Stage 2: fair assignment (LP relaxation -> count rounding ->
#    per-group transportation), inside alternating fair k-means
# ======================================================================
def fair_assignment(Z, centers, group_ids, G, alpha, delta):
    n, k = Z.shape[0], centers.shape[0]
    cost = ((Z[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    cost = cost / cost.mean()
    lower = (1 - delta) * alpha
    upper = np.minimum(alpha / (1 - delta), 1.0)

    # Step 1: LP relaxation of the fair assignment problem
    rows = list(np.repeat(np.arange(n), k))
    cols = list(np.arange(n * k))
    vals = [1.0] * (n * k)
    lb, ub, r = [1.0] * n, [1.0] * n, n
    for c in range(k):
        var = np.arange(n) * k + c
        for g in range(G):
            in_g = (group_ids == g).astype(float)
            for coef, lo, hi in [(in_g - lower[g], 0.0, np.inf),
                                 (in_g - upper[g], -np.inf, 0.0)]:
                rows += [r] * n; cols += list(var); vals += list(coef)
                lb.append(lo); ub.append(hi); r += 1
        rows += [r] * n; cols += list(var); vals += [1.0] * n
        lb.append(1.0); ub.append(np.inf); r += 1
    A = sp.csr_matrix((vals, (rows, cols)), shape=(r, n * k))
    lp = milp(cost.ravel(), constraints=LinearConstraint(A, lb, ub),
              integrality=np.zeros(n * k), bounds=Bounds(0, 1))
    x_frac = lp.x.reshape(n, k)
    m_tilde = np.vstack([x_frac[group_ids == g].sum(axis=0) for g in range(G)])

    # Step 2: integer counts m[g, c] closest to the fractional counts
    nv = G * k
    R, C, V, L2, U2 = [], [], [], [], []

    def add(coefs, lo, hi):
        row = len(L2)
        for j, v in coefs:
            R.append(row); C.append(j); V.append(v)
        L2.append(lo); U2.append(hi)

    n_g = np.bincount(group_ids, minlength=G)
    for g in range(G):
        add([(g * k + c, 1.0) for c in range(k)], n_g[g], n_g[g])
    for c in range(k):
        for g in range(G):
            add([(h * k + c, float(h == g) - lower[g]) for h in range(G)], 0, np.inf)
            add([(h * k + c, float(h == g) - upper[g]) for h in range(G)], -np.inf, 0)
        add([(g * k + c, 1.0) for g in range(G)], 1, np.inf)
    for j in range(nv):                          # t_j >= |m_j - m~_j|
        g, c = divmod(j, k)
        add([(nv + j, 1.0), (j, -1.0)], -m_tilde[g, c], np.inf)
        add([(nv + j, 1.0), (j, 1.0)], m_tilde[g, c], np.inf)
    A2 = sp.csr_matrix((V, (R, C)), shape=(len(L2), 2 * nv))
    cnt = milp(np.r_[np.zeros(nv), np.ones(nv)], constraints=LinearConstraint(A2, L2, U2),
               integrality=np.r_[np.ones(nv), np.zeros(nv)], bounds=Bounds(0, np.inf))
    m = np.rint(cnt.x[:nv]).astype(int).reshape(G, k)

    # Step 3: per-group transportation problems (integral LP optimum)
    labels = np.empty(n, dtype=int)
    for g in range(G):
        idx = np.where(group_ids == g)[0]
        ng = len(idx)
        rr = np.r_[np.repeat(np.arange(ng), k), ng + np.tile(np.arange(k), ng)]
        cc = np.r_[np.arange(ng * k), np.arange(ng * k)]
        At = sp.csr_matrix((np.ones(2 * ng * k), (rr, cc)), shape=(ng + k, ng * k))
        bt = np.r_[np.ones(ng), m[g]].astype(float)
        tr = milp(cost[idx].ravel(), constraints=LinearConstraint(At, bt, bt),
                  integrality=np.ones(ng * k), bounds=Bounds(0, 1))
        labels[idx] = np.argmax(tr.x.reshape(ng, k), axis=1)
    return labels


def fair_kmeans(Z, group_ids, G, k, delta):
    alpha = np.bincount(group_ids, minlength=G) / len(group_ids)
    centers = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Z).cluster_centers_
    labels = None
    for _ in range(MAX_ALTERNATIONS):
        new = fair_assignment(Z, centers, group_ids, G, alpha, delta)
        if labels is not None and np.array_equal(new, labels):
            break
        labels = new
        centers = np.vstack([Z[labels == c].mean(axis=0) for c in range(k)])
    return labels


# ======================================================================
# 4. Metrics
# ======================================================================
def balance_objective(labels, group_ids):
    vals = []
    for c in np.unique(labels):
        in_c = labels == c
        for g in np.unique(group_ids):
            vals.append(np.sum(in_c & (group_ids == g)) / np.sum(in_c))
    return float(np.min(vals))


def wcss(Z, labels):
    return float(sum(((Z[labels == c] - Z[labels == c].mean(axis=0)) ** 2).sum()
                     for c in np.unique(labels)))


# ======================================================================
# 5. Sweep over the diffusion time
# ======================================================================
def effective_rank(Z):
    s = np.linalg.svd(Z - Z.mean(axis=0), compute_uv=False) ** 2
    p = s / s.sum()
    p = p[p > 1e-15]
    return float(np.exp(-(p * np.log(p)).sum()))


def sweep_time(csv, sens, cats, sep, k_nn):
    X, group_ids, G = load_dataset(csv, sens, cats, sep)
    n = X.shape[0]
    d = int(np.ceil(np.sqrt(n)))
    alpha_min = float((np.bincount(group_ids) / n).min())
    P_nu = p_rw_operator(X, k_nn, GAMMA)
    t_star = diffusion_time(P_nu, T_MAX, d)
    times = sorted(set(T_VALUES) | {t_star})

    alpha = np.bincount(group_ids, minlength=G) / n
    F = np.column_stack([(group_ids == j) - alpha[j] for j in range(G - 1)]).astype(float)
    DF = F / p_rw_operator.pi[:, None]              # D_pi^{-1} F
    FtF_inv = np.linalg.inv(F.T @ DF)

    keys = ["ParPIC km", "FP-ParPIC FA", "ParPIC FA pof", "FP-ParPIC FA pof",
            "ParPIC erank", "FP-ParPIC erank", "FP-ParPIC km", "ParPIC FA"]
    rec = {"alpha_min": alpha_min, "x": times, "t_star": t_star, **{k: [] for k in keys}}
    Z_parpic = np.random.default_rng(SEED).standard_normal((n, d))
    Z_fair = Z_parpic.copy()
    fair_labels = {}
    for t in range(1, max(times) + 1):
        Z_parpic = P_nu @ Z_parpic
        Z_fair = P_nu @ Z_fair
        Z_fair = Z_fair - DF @ (FtF_inv @ (F.T @ Z_fair))
        if t not in times:
            continue
        for name, Z in [("ParPIC", Z_parpic), ("FP-ParPIC", Z_fair)]:
            km = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit_predict(Z)
            fa = fair_kmeans(Z, group_ids, G, K, DELTA)
            w_km = wcss(Z, km)
            rec[f"{name} km"].append(balance_objective(km, group_ids))
            rec[f"{name} FA"].append(balance_objective(fa, group_ids))
            rec[f"{name} FA pof"].append((wcss(Z, fa) - w_km) / w_km)
            rec[f"{name} erank"].append(effective_rank(Z))
            if name == "FP-ParPIC":
                fair_labels[t] = fa
        print(f"   t={t:<3d}{'*' if t == t_star else ' '} effective rank {rec['ParPIC erank'][-1]:.2f} / "
              f"{rec['FP-ParPIC erank'][-1]:.2f}   ParPIC k-means balance={rec['ParPIC km'][-1]:.4f}   "
              f"Fair ParPIC balance={rec['FP-ParPIC FA'][-1]:.4f}   price of fairness "
              f"{100 * rec['ParPIC FA pof'][-1]:.2f}% / {100 * rec['FP-ParPIC FA pof'][-1]:.2f}%", flush=True)
    rec["ARI to t_star"] = [float(adjusted_rand_score(fair_labels[t_star], fair_labels[t])) for t in times]
    return rec


# ======================================================================
# 6. Figure
# ======================================================================
# Top row (balance):            Fair ParPIC (ours) vs ParPIC + k-means
# Bottom row (price of fairness): Fair ParPIC (ours) vs ParPIC + fair assignment
# Ours: filled circles. Baselines: squares (hollow for k-means); fair assignment
# baseline is also dashed. Marker shape is a second
# cue besides colour, so the figure also reads in greyscale / for CVD.
BLUE, ORANGE = "#2a78d6", "#eb6834"
GREEN, RED = "#008300", "#e34948"
GREY, INK = "#6b6a66", "#1f1f1e"
DASHED = (0, (4, 2.5))
FIG_WIDTH, FIG_HEIGHT = 6.75, 4.0   # inches; 6.75 = AISTATS text width
STYLE = {   # key: (legend label, colour, line style, marker, hollow marker)
    "FP-ParPIC FA":     ("Fair ParPIC (ours)",       BLUE,   "-",    "o", False),
    "ParPIC km":        ("ParPIC + $k$-means",       ORANGE, "-",    "s", True),
    "FP-ParPIC FA pof": ("Fair ParPIC (ours)",       GREEN,  "-",    "o", False),
    "ParPIC FA pof":    ("ParPIC + fair assignment", RED,    DASHED, "s", False),
}
TOP_ROW = ["ParPIC km", "FP-ParPIC FA"]              # drawing order: ours on top
BOTTOM_ROW = ["ParPIC FA pof", "FP-ParPIC FA pof"]


def set_paper_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7.5,
        "text.color": INK, "axes.labelcolor": INK,
        "axes.edgecolor": "#8a8984", "axes.linewidth": 0.6,
        "xtick.color": "#52514e", "ytick.color": "#52514e",
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.5,
        "lines.linewidth": 1.5, "lines.markersize": 4.5, "legend.frameon": False,
        "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def style_kwargs(key, line=True):
    label, colour, ls, marker, hollow = STYLE[key]
    return dict(color=colour, linestyle=ls if line else "none", marker=marker, label=label,
                markerfacecolor="white" if hollow else colour,
                markeredgecolor=colour if hollow else "white", markeredgewidth=0.8)


def floor_ceiling(ax, floor, ceiling):
    ax.axhline(floor, color=GREY, linestyle=":", linewidth=1.1, zorder=2)
    ax.axhline(ceiling, color=GREY, linewidth=0.8, zorder=2)


def pof_limits(ax, values):
    """Axis from 0, extended (with a zero line) if a value is negative, which
    happens when the fair clustering beats k-means' own local optimum."""
    low, high = min(values), max(values)
    if low < 0:
        ax.axhline(0, color=GREY, linewidth=0.6, zorder=2)
        ax.set_ylim(low - 0.06 * (high - low), high + 0.08 * (high - low))
    else:
        ax.set_ylim(0, high * 1.12 if high > 0 else 1)


def row_legends(fig, axes, floor_label, ceiling_label=r"Ceiling $\alpha_{\min}$"):
    """One legend per row, centred directly above the row it describes."""
    top = [Line2D([], [], **style_kwargs(k)) for k in TOP_ROW[::-1]] + [
        Line2D([], [], color=GREY, linestyle=":", linewidth=1.1, label=floor_label),
        Line2D([], [], color=GREY, linewidth=0.8, label=ceiling_label)]
    bottom = [Line2D([], [], **style_kwargs(k)) for k in BOTTOM_ROW[::-1]]
    top_y = axes[0, 0].get_position().y1
    mid_y = 0.5 * (axes[0, 0].get_position().y0 + axes[1, 0].get_position().y1)
    kw = dict(handlelength=2.8, columnspacing=1.8, handletextpad=0.6)
    fig.legend(handles=top, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, top_y + 0.045), **kw)
    fig.legend(handles=bottom, loc="center", ncol=2, bbox_to_anchor=(0.5, mid_y), **kw)


def new_figure(n_cols, sharex=True, width_ratios=None):
    set_paper_style()
    gs = {"hspace": 0.45, "wspace": 0.32}
    if width_ratios is not None:
        gs.update(width_ratios=width_ratios, wspace=0.22)
    return plt.subplots(2, n_cols, figsize=(FIG_WIDTH, FIG_HEIGHT), sharex=sharex,
                        gridspec_kw=gs, squeeze=False)


def save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path + ".pdf")
    fig.savefig(path + ".png")
    plt.close(fig)
    print(f"saved {path}.pdf and {path}.png")


def plot_time(results, path):
    fig, axes = new_figure(len(results))
    for j, (name, r) in enumerate(results.items()):
        x, amin = r["x"], r["alpha_min"]
        for ax in axes[:, j]:                               # selected diffusion time
            ax.axvline(r["t_star"], color=GREY, linewidth=1.0, linestyle=(0, (1, 2)), zorder=2.5)

        ax = axes[0, j]                                      # balance
        floor_ceiling(ax, (1 - DELTA) * amin, amin)
        for key in TOP_ROW:
            ax.plot(x, r[key], zorder=3, **style_kwargs(key))
        ax.set_title(name, pad=4)
        ax.set_ylim(0, amin * 1.12)
        ax.annotate(r"$t^\star$", (r["t_star"], 0.02 * amin), xytext=(3, 0),
                    textcoords="offset points", fontsize=7, color=GREY)

        ax = axes[1, j]                                      # price of fairness
        values = []
        for key in BOTTOM_ROW:
            y = 100 * np.array(r[key])
            values += list(y)
            ax.plot(x, y, zorder=3, **style_kwargs(key))
        pof_limits(ax, values)
        ax.set_xlabel(r"Diffusion time $t$")
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10, 20, 40])
        ax.set_xticklabels(["1", "2", "5", "10", "20", "40"])
        ax.minorticks_off()

    axes[0, 0].set_ylabel("Balance")
    axes[1, 0].set_ylabel("Price of fairness (%)")
    row_legends(fig, axes, rf"Floor $(1-\delta)\,\alpha_{{\min}}$, $\delta={DELTA:g}$")
    save(fig, path)


# ======================================================================
# 7. Main
# ======================================================================
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    for name, (csv, sens, cats, sep, k_nn) in selected(DATASETS):
        print(f"[diffusion-time sweep] {name}", flush=True)
        rec = sweep_time(csv, sens, cats, sep, k_nn)
        with open(f"results/sweep_time_{name}.json", "w") as fh:
            json.dump(rec, fh, indent=2)
    files = {name: f"results/sweep_time_{name}.json" for name in DATASETS}
    if all(os.path.exists(f) for f in files.values()):
        plot_time({name: json.load(open(f)) for name, f in files.items()}, "figures/fig_time")
