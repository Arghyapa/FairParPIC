"""
Appendix E.4.5, Fig. S6 and Table S4 -- number of protected groups G (Census).
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
K_NN = 10
GAMMA = 0.5
T_MAX = 40
MAX_ALTERNATIONS = 10
DELTA_GRID = [0.05, 0.075, 0.1, 0.15, 0.2, 0.3]   # fallback if DELTA is infeasible
DATA_DIR = os.environ.get("DATA_DIR", "data")   # folder with the CSV files (data/README.md)
CSV = "uci_census.csv"
STRATIFY_COL = "gender"
SENSITIVE_COLS = ["gender", "race", "marital_status"]

RACE4 = {"White": "White", "Black": "Black", "Asian-Pac-Islander": "Asian-Pac-Islander",
         "Amer-Indian-Eskimo": "Other", "Other": "Other"}
RACE3 = {"White": "White", "Black": "Black", "Asian-Pac-Islander": "Other",
         "Amer-Indian-Eskimo": "Other", "Other": "Other"}

# name: function of the sampled data frame -> one group label per row
GROUPINGS = {
    "race2":        lambda s: np.where(s["race"] == "White", "White", "Non-White"),
    "race3":        lambda s: s["race"].map(RACE3).values,
    "race4":        lambda s: s["race"].map(RACE4).values,
    "race5":        lambda s: s["race"].values,
    "gender":       lambda s: s["gender"].values,
    "marital6":     lambda s: s["marital_status"].replace(
                        {"Married-AF-spouse": "Married-civ-spouse"}).values,
    "gender_race8": lambda s: (s["gender"] + " | " + s["race"].map(RACE4)).values,
}
VIEW_RACE = ["race2", "race3", "race4", "race5"]
VIEW_ATTR = ["gender", "race5", "marital6", "gender_race8"]
ATTR_LABEL = {"gender": "Gender", "race5": "Race", "marital6": "Marital",
              "gender_race8": "Gender$\\times$Race"}


# ======================================================================
# 1. Data
# ======================================================================
def load_census():
    """Same cleaning and gender-stratified sample as every other experiment;
    also returns the sensitive columns of the sampled rows."""
    df = pd.read_csv(os.path.join(DATA_DIR, CSV))
    df = df.replace("?", np.nan).replace("unknown", np.nan)
    df = df[df[STRATIFY_COL].isin(("Male", "Female"))]

    missing = df.isnull().mean()
    df = df.drop(columns=missing[missing > 0.4].index)
    numeric_cols = list(df.select_dtypes(exclude=["object"]).columns)
    df = df[numeric_cols + SENSITIVE_COLS]
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    parts = []
    for _, g in df.groupby(STRATIFY_COL):            # stratified subsample
        n_g = min(len(g), int(round(N_SAMPLE * len(g) / len(df))))
        parts.append(g.sample(n=n_g, random_state=SEED))
    df = pd.concat(parts, axis=0).reset_index(drop=True)

    X = StandardScaler().fit_transform(df[numeric_cols].values)
    return X, df[SENSITIVE_COLS]


def encode(labels):
    names, group_ids = np.unique(np.asarray(labels).astype(str), return_inverse=True)
    return group_ids, list(names)


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


def fair_projected_embedding(P_nu, t_star, Z0, group_ids, G):
    """Z^(tau) = Pi_pi (P_nu Z^(tau-1)); F depends on the groups."""
    n = Z0.shape[0]
    alpha = np.bincount(group_ids, minlength=G) / n
    F = np.column_stack([(group_ids == j) - alpha[j] for j in range(G - 1)]).astype(float)
    DF = F / p_rw_operator.pi[:, None]              # D_pi^{-1} F
    FtF_inv = np.linalg.inv(F.T @ DF)
    Z = Z0.copy()
    for _ in range(t_star):
        Z = P_nu @ Z
        Z = Z - DF @ (FtF_inv @ (F.T @ Z))
    return Z


def centroid_parity_gap(Z, group_ids, G):
    """Largest distance between a group centroid and the global centroid
    (zero, up to round-off, for the fair-projected embedding)."""
    mu = Z.mean(axis=0)
    return float(max(np.linalg.norm(Z[group_ids == g].mean(axis=0) - mu) for g in range(G)))


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


def fair_counts_feasible(n_g, k, delta):
    """Is there any k-clustering (all clusters non-empty) whose every
    group-proportion lies in [(1-delta)*alpha_g, alpha_g/(1-delta)]?
    Only the integer counts m[g, c] matter: any counts can be realized by
    some assignment, so this small integer program decides feasibility."""
    n_g = np.asarray(n_g)
    G = len(n_g)
    alpha = n_g / n_g.sum()
    lower = (1 - delta) * alpha
    upper = np.minimum(alpha / (1 - delta), 1.0)
    R, C, V, L, U = [], [], [], [], []

    def add(coefs, lo, hi):
        row = len(L)
        for j, v in coefs:
            R.append(row); C.append(j); V.append(v)
        L.append(lo); U.append(hi)

    for g in range(G):
        add([(g * k + c, 1.0) for c in range(k)], n_g[g], n_g[g])
    for c in range(k):
        for g in range(G):
            add([(h * k + c, float(h == g) - lower[g]) for h in range(G)], 0, np.inf)
            add([(h * k + c, float(h == g) - upper[g]) for h in range(G)], -np.inf, 0)
        add([(g * k + c, 1.0) for g in range(G)], 1, np.inf)
    A = sp.csr_matrix((V, (R, C)), shape=(len(L), G * k))
    res = milp(np.zeros(G * k), constraints=LinearConstraint(A, L, U),
               integrality=np.ones(G * k), bounds=Bounds(0, np.inf))
    return res.status == 0


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
# 5. Experiment
# ======================================================================
def run(names=None, results=None):
    """Runs the groupings in `names` (default: all), adding to `results`."""
    names = list(GROUPINGS) if names is None else names
    results = {} if results is None else results
    X, sens = load_census()
    n = X.shape[0]
    d = int(np.ceil(np.sqrt(n)))

    # Everything group-independent is computed once.
    P_nu = p_rw_operator(X, K_NN, GAMMA)
    t_star = diffusion_time(P_nu, T_MAX, d)
    Z0 = np.random.default_rng(SEED).standard_normal((n, d))
    Z_parpic = Z0.copy()
    for _ in range(t_star):
        Z_parpic = P_nu @ Z_parpic
    km_parpic = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit_predict(Z_parpic)
    print(f"n={n}  d={d}  diffusion time t*={t_star}", flush=True)

    for name in names:
        group_ids, groups = encode(GROUPINGS[name](sens))
        G = len(groups)
        n_g = np.bincount(group_ids, minlength=G)
        assert n_g.min() >= K, f"{name}: a group has fewer than k={K} members"
        alpha = n_g / n
        delta = next(dl for dl in DELTA_GRID if fair_counts_feasible(n_g, K, dl))
        Z_fair = fair_projected_embedding(P_nu, t_star, Z0, group_ids, G)

        rec = {"G": G, "groups": groups, "group_sizes": n_g.tolist(),
               "alpha_min": float(alpha.min()), "delta": delta,
               "centroid_gap ParPIC": centroid_parity_gap(Z_parpic, group_ids, G),
               "centroid_gap FP-ParPIC": centroid_parity_gap(Z_fair, group_ids, G)}
        print(f"[{name}] G={G}  alpha_min={alpha.min():.4f}  sizes={n_g.tolist()}", flush=True)
        if delta != DELTA:
            print(f"   no fair {K}-clustering exists at delta={DELTA:g}; "
                  f"using smallest feasible delta={delta:g}", flush=True)
        for emb, Z in [("ParPIC", Z_parpic), ("FP-ParPIC", Z_fair)]:
            km = km_parpic if emb == "ParPIC" else \
                KMeans(n_clusters=K, n_init=10, random_state=SEED).fit_predict(Z)
            fa = fair_kmeans(Z, group_ids, G, K, delta)
            w_km = wcss(Z, km)
            rec[f"{emb} km"] = balance_objective(km, group_ids)
            rec[f"{emb} FA"] = balance_objective(fa, group_ids)
            rec[f"{emb} FA pof"] = (wcss(Z, fa) - w_km) / w_km
            print(f"   {emb:10s} k-means balance={rec[f'{emb} km']:.4f}  "
                  f"fair balance={rec[f'{emb} FA']:.4f}  "
                  f"(floor {(1 - delta) * alpha.min():.4f})  "
                  f"price of fairness={100 * rec[f'{emb} FA pof']:.2f}%", flush=True)
        results[name] = rec
        os.makedirs("results", exist_ok=True)
        with open("results/groups_census.json", "w") as fh:
            json.dump(results, fh, indent=2)
    return results


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


OFFSET = {"ParPIC km": 0.07, "FP-ParPIC FA": -0.07,
          "ParPIC FA pof": 0.07, "FP-ParPIC FA pof": -0.07}


def plot_groups(results, path):
    """Top row: balance divided by alpha_min, so groupings with different
    alpha_min share one scale (floor 1 - delta, ceiling 1). Bottom row: price
    of fairness. Left: nested race groupings (lines, x = G). Right: different
    attributes (categorical x, so points only)."""
    fig, axes = new_figure(2, sharex="col", width_ratios=[1, 1.15])
    for j, (view, title) in enumerate([(VIEW_RACE, "(a) Race, nested groupings"),
                                       (VIEW_ATTR, "(b) Different sensitive attributes")]):
        rs = [results[v] for v in view]
        lines = j == 0
        x = np.array([r["G"] for r in rs], float) if lines else np.arange(len(rs), dtype=float)

        def xs(key):
            return x if lines else x + OFFSET[key]

        ax = axes[0, j]                                      # normalized balance
        floor_ceiling(ax, 1 - DELTA, 1.0)
        for key in TOP_ROW:
            y = [r[key] / r["alpha_min"] for r in rs]
            ax.plot(xs(key), y, zorder=3, **style_kwargs(key, line=lines))
        for xi, r in zip(x, rs):          # groupings where DELTA was infeasible
            if r["delta"] != DELTA:
                ax.hlines(1 - r["delta"], xi - 0.32, xi + 0.32, color=GREY,
                          linestyle=":", linewidth=1.1, zorder=2)
                ax.annotate(rf"$\delta={r['delta']:g}$" + "\n" + rf"({DELTA:g} infeasible)",
                            (xi, 0.72), ha="center", va="center", fontsize=6.5, color=GREY)
        ax.set_ylim(-0.03, 1.12)
        ax.set_title(title, pad=4)

        ax = axes[1, j]                                      # price of fairness
        values = []
        for key in BOTTOM_ROW:
            y = [100 * r[key] for r in rs]
            values += y
            ax.plot(xs(key), y, zorder=3, **style_kwargs(key, line=lines))
        pof_limits(ax, values)
        ax.set_xticks(x)
        if lines:
            ax.set_xticklabels([f"{int(g)}\n{r['alpha_min']:.3f}" for g, r in zip(x, rs)])
            ax.set_xlabel(r"Number of groups $G$  (second line: $\alpha_{\min}$)")
        else:
            ax.set_xticklabels([f"{ATTR_LABEL[v]}\n$G={r['G']}$, {r['alpha_min']:.3f}"
                                for v, r in zip(view, rs)])
            ax.set_xlim(-0.45, len(rs) - 0.55)
            ax.set_xlabel(r"Sensitive attribute  (second line: $G$, $\alpha_{\min}$)")

    axes[0, 0].set_ylabel(r"Balance / $\alpha_{\min}$")
    axes[1, 0].set_ylabel("Price of fairness (%)")
    row_legends(fig, axes, rf"Floor $(1-\delta)\,\alpha_{{\min}}$, $\delta={DELTA:g}$")
    save(fig, path)


# ======================================================================
# 7. Main
# ======================================================================
if __name__ == "__main__":
    results = run()
    plot_groups(results, "figures/fig_groups")
