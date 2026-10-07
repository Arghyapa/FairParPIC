"""
Appendix E.4.4, Fig. S5 -- robustness to the vertex-measure parameter gamma.

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
GAMMA_VALUES = [0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]

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


def embeddings(X, group_ids, G, k_nn, gamma):
    """ParPIC and FP-ParPIC embeddings from the same operator,
    diffusion time and random start."""
    n = X.shape[0]
    d = int(np.ceil(np.sqrt(n)))
    P_nu = p_rw_operator(X, k_nn, gamma)
    t_star = diffusion_time(P_nu, T_MAX, d)

    alpha = np.bincount(group_ids, minlength=G) / n
    F = np.column_stack([(group_ids == j) - alpha[j] for j in range(G - 1)]).astype(float)
    DF = F / p_rw_operator.pi[:, None]              # D_pi^{-1} F
    FtF_inv = np.linalg.inv(F.T @ DF)

    def project(M):                                  # Pi_pi M
        return M - DF @ (FtF_inv @ (F.T @ M))

    Z_parpic = np.random.default_rng(SEED).standard_normal((n, d))
    Z_fair = Z_parpic.copy()
    for _ in range(t_star):
        Z_parpic = P_nu @ Z_parpic
        Z_fair = project(P_nu @ Z_fair)
    return Z_parpic, Z_fair


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


def evaluate(Z_parpic, Z_fair, group_ids, G, k, record, tag):
    """Clusters both embeddings with k-means and fair assignment; appends
    balance and price of fairness to `record`."""
    for name, Z in [("ParPIC", Z_parpic), ("FP-ParPIC", Z_fair)]:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(Z)
        fa = fair_kmeans(Z, group_ids, G, k, DELTA)
        w_km = wcss(Z, km)
        record[f"{name} km"].append(balance_objective(km, group_ids))
        record[f"{name} FA"].append(balance_objective(fa, group_ids))
        record[f"{name} FA pof"].append((wcss(Z, fa) - w_km) / w_km)
        print(f"   {tag:10s} {name:10s} k-means balance={record[f'{name} km'][-1]:.4f}  "
              f"fair balance={record[f'{name} FA'][-1]:.4f}  "
              f"price of fairness={100 * record[f'{name} FA pof'][-1]:.2f}%", flush=True)


def empty_record(alpha_min, xs):
    keys = ["ParPIC km", "FP-ParPIC km", "ParPIC FA", "FP-ParPIC FA",
            "ParPIC FA pof", "FP-ParPIC FA pof"]
    return {"alpha_min": alpha_min, "x": xs, **{k: [] for k in keys}}


# ======================================================================
# 5. Sweep
# ======================================================================
def sweep_gamma(csv, sens, cats, sep, k_nn):
    X, group_ids, G = load_dataset(csv, sens, cats, sep)
    alpha_min = float((np.bincount(group_ids) / len(group_ids)).min())
    rec = empty_record(alpha_min, GAMMA_VALUES)
    rec["pi_ratio"] = []
    for gamma in GAMMA_VALUES:
        Z_parpic, Z_fair = embeddings(X, group_ids, G, k_nn, gamma)
        pi = p_rw_operator.pi
        rec["pi_ratio"].append(float(pi.max() / pi.min()))
        print(f"   gamma={gamma:g}: max(pi) / min(pi) = {rec['pi_ratio'][-1]:.3g}", flush=True)
        evaluate(Z_parpic, Z_fair, group_ids, G, K, rec, f"gamma={gamma:g}")
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


def plot_sweep(results, xlabel, path):
    """Fig. S5: gamma < 1 only, as equally spaced categories (gamma = 1 is the
    degenerate boundary case discussed in the text)."""
    fig, axes = new_figure(len(results))
    for j, (name, r_all) in enumerate(results.items()):
        keep = [i for i, g in enumerate(r_all["x"]) if g < 1]
        r = {key: ([v[i] for i in keep] if isinstance(v, list) else v) for key, v in r_all.items()}
        gammas, amin = r["x"], r["alpha_min"]
        x = list(range(len(gammas)))

        ax = axes[0, j]                                      # balance
        floor_ceiling(ax, (1 - DELTA) * amin, amin)
        for key in TOP_ROW:
            ax.plot(x, r[key], zorder=3, **style_kwargs(key))
        ax.set_title(name, pad=4)
        ax.set_ylim(0, amin * 1.12)

        ax = axes[1, j]                                      # price of fairness
        values = []
        for key in BOTTOM_ROW:
            y = 100 * np.array(r[key])
            values += list(y)
            ax.plot(x, y, zorder=3, **style_kwargs(key))
        pof_limits(ax, values)
        ax.set_xlabel(xlabel)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{g:g}" for g in gammas])

    axes[0, 0].set_ylabel("Balance")
    axes[1, 0].set_ylabel("Price of fairness (%)")
    row_legends(fig, axes, rf"Floor $(1-\delta)\,\alpha_{{\min}}$, $\delta={DELTA:g}$")
    save(fig, path)


# ======================================================================
# 7. Main
# ======================================================================
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    results = {}
    for name, (csv, sens, cats, sep, k_nn) in selected(DATASETS):
        print(f"[gamma sweep] {name}", flush=True)
        results[name] = sweep_gamma(csv, sens, cats, sep, k_nn)
        with open("results/sweep_gamma.json", "w") as fh:
            json.dump(results, fh, indent=2)
    plot_sweep(results, r"Vertex-measure parameter $\gamma$", "figures/fig_gamma")
