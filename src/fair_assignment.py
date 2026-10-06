"""Stage 2 of Fair ParPIC: the fair assignment problem (P_delta) and the
alternating fair k-means.

For fixed centers mu_1..mu_k, (P_delta) is (Eqs. 13-16 of the paper)

    min_x  J(x, mu) = sum_{i,c} x_ic ||z_i - mu_c||^2
    s.t.   sum_c x_ic = 1                                        for every vertex i
           r_lo_g |C_c| <= |C_c cap S_g| <= r_hi_g |C_c|          for every c, g
           |C_c| >= 1                                             for every c
           x_ic in {0, 1}

with the share bounds r_lo_g = (1 - delta) alpha_g and
r_hi_g = min(1, alpha_g / (1 - delta)) (written underline-r_g and overline-r_g
in the paper). Every feasible solution satisfies
(1 - delta) alpha_min <= balance <= alpha_min (Theorem 3), so fairness never
depends on how exactly (P_delta) is solved.

Two solvers are provided:

* ``"decomposition"`` (default, Section 5.2 of the paper): LP relaxation ->
  integer rounding (Q) of the G x k group counts n_gc -> one transportation
  problem per group. The share bounds hold exactly and (J(x, mu) - LB) / LB is
  a certified bound on the relative sub-optimality (Proposition 6).
* ``"milp"``: the full mixed-integer program solved by HiGHS, with a relative
  optimality gap of 0.5%. Exact but slow beyond a few thousand vertices.
"""

import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment, milp
from sklearn.cluster import KMeans


class InfeasibleFairnessError(RuntimeError):
    """(P_delta) has no feasible solution for this (k, delta) and these groups.
    Feasibility requires n_g >= k for every group (Proposition 2), and this is
    not sufficient (Remark S2); use a larger delta or a smaller k."""


def share_bounds(alpha, delta):
    """Share bounds (r_lo_g, r_hi_g) = ((1 - delta) alpha_g, min(1, alpha_g / (1 - delta)))."""
    lower = (1.0 - delta) * alpha
    upper = np.minimum(alpha / (1.0 - delta), 1.0)
    return lower, upper


def _assignment_constraints(n, k, group_ids, n_groups, lower, upper):
    """Sparse constraint matrix of (P_delta); variable x_ic sits at i * k + c."""
    rows, cols, vals, lb, ub = [], [], [], [], []
    r = 0
    for i in range(n):                                   # each vertex once
        rows += [r] * k
        cols += list(range(i * k, i * k + k))
        vals += [1.0] * k
        lb.append(1.0); ub.append(1.0); r += 1
    idx_all = np.arange(n)
    for c in range(k):
        var = idx_all * k + c
        for g in range(n_groups):
            in_g = (group_ids == g).astype(float)
            rows += [r] * n; cols += var.tolist(); vals += (in_g - lower[g]).tolist()
            lb.append(0.0); ub.append(np.inf); r += 1        # share >= r_lo_g
            rows += [r] * n; cols += var.tolist(); vals += (in_g - upper[g]).tolist()
            lb.append(-np.inf); ub.append(0.0); r += 1       # share <= r_hi_g
        rows += [r] * n; cols += var.tolist(); vals += [1.0] * n
        lb.append(1.0); ub.append(np.inf); r += 1            # non-empty
    A = sp.csr_matrix((vals, (rows, cols)), shape=(r, n * k))
    return A, lb, ub


def fair_assignment_milp(Z, centers, group_ids, n_groups, alpha, delta,
                         time_limit=60, mip_rel_gap=5e-3):
    """Exact MILP formulation solved with HiGHS (scipy.optimize.milp).
    Any returned solution (optimal, or best found at the gap / time limit) is
    integer-feasible, so the balance floor holds regardless."""
    n, k = Z.shape[0], centers.shape[0]
    cost = ((Z[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    cost = cost / (cost.mean() if cost.mean() > 0 else 1.0)
    lower, upper = share_bounds(alpha, delta)
    A, lb, ub = _assignment_constraints(n, k, group_ids, n_groups, lower, upper)
    res = milp(c=cost.ravel(), constraints=LinearConstraint(A, lb, ub),
               integrality=np.ones(n * k), bounds=Bounds(0, 1),
               options={"time_limit": time_limit, "disp": False,
                        "mip_rel_gap": mip_rel_gap})
    if res.x is None:
        raise InfeasibleFairnessError(f"Fair assignment MILP failed: {res.message}")
    labels = np.argmax(res.x.reshape(n, k), axis=1)
    return labels, {"gap": float(getattr(res, "mip_gap", np.nan) or 0.0)}


def fair_assignment_decomposition(Z, centers, group_ids, n_groups, alpha, delta):
    """Three-step solver of Section 5.2 (exact share bounds, certified gap)."""
    n, k = Z.shape[0], centers.shape[0]
    cost = ((Z[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    cost = cost / (cost.mean() if cost.mean() > 0 else 1.0)
    lower, upper = share_bounds(alpha, delta)

    # Step 1: LP relaxation of (P_delta); LB is its optimal value
    A, lb, ub = _assignment_constraints(n, k, group_ids, n_groups, lower, upper)
    lp = milp(cost.ravel(), constraints=LinearConstraint(A, lb, ub),
              integrality=np.zeros(n * k), bounds=Bounds(0, 1))
    if lp.x is None:
        raise InfeasibleFairnessError(f"LP relaxation failed: {lp.message}")
    LB = float(lp.fun)
    x_frac = lp.x.reshape(n, k)
    n_tilde = np.vstack([x_frac[group_ids == g].sum(axis=0) for g in range(n_groups)])

    # Step 2: count rounding (Q): integer counts n_gc closest in L1 to the
    # fractional counts n~_gc that satisfy the share bounds and non-emptiness
    # written on counts.
    nv = n_groups * k
    R, C, V, L2, U2 = [], [], [], [], []

    def add(coefs, lo, hi):
        row = len(L2)
        for j, v in coefs:
            R.append(row); C.append(j); V.append(v)
        L2.append(lo); U2.append(hi)

    n_g = np.bincount(group_ids, minlength=n_groups)
    for g in range(n_groups):
        add([(g * k + c, 1.0) for c in range(k)], n_g[g], n_g[g])
    for c in range(k):
        for g in range(n_groups):
            add([(h * k + c, float(h == g) - lower[g]) for h in range(n_groups)], 0, np.inf)
            add([(h * k + c, float(h == g) - upper[g]) for h in range(n_groups)], -np.inf, 0)
        add([(g * k + c, 1.0) for g in range(n_groups)], 1, np.inf)
    for j in range(nv):                                   # t_j >= |n_gc - n~_gc|
        g, c = divmod(j, k)
        add([(nv + j, 1.0), (j, -1.0)], -n_tilde[g, c], np.inf)
        add([(nv + j, 1.0), (j, 1.0)], n_tilde[g, c], np.inf)
    A2 = sp.csr_matrix((V, (R, C)), shape=(len(L2), 2 * nv))
    cnt = milp(np.r_[np.zeros(nv), np.ones(nv)], constraints=LinearConstraint(A2, L2, U2),
               integrality=np.r_[np.ones(nv), np.zeros(nv)], bounds=Bounds(0, np.inf))
    if cnt.x is None:
        # (Q) is feasible iff (P_delta) is (Proposition 6(a)): integrality makes it infeasible.
        raise InfeasibleFairnessError(
            "No clustering satisfies the share bounds for this (k, delta): "
            "increase delta or decrease k.")
    n_gc = np.rint(cnt.x[:nv]).astype(int).reshape(n_groups, k)

    # Step 3: per-group transportation, solved exactly as a linear assignment of
    # the group's vertices to n_gc[g, c] slots of every cluster.
    labels = np.empty(n, dtype=int)
    for g in range(n_groups):
        idx = np.where(group_ids == g)[0]
        slots = np.repeat(np.arange(k), n_gc[g])
        row, col = linear_sum_assignment(cost[idx][:, slots])
        labels[idx[row]] = slots[col]

    J = float(cost[np.arange(n), labels].sum())
    gap = (J - LB) / LB if LB > 0 else 0.0
    return labels, {"LB": LB, "J": J, "gap": gap}


SOLVERS = {"decomposition": fair_assignment_decomposition,
           "milp": fair_assignment_milp}


def fair_kmeans(Z, group_ids, n_groups, k, delta, solver="decomposition",
                seed=42, max_iter=10, init_centers=None, return_info=False):
    """Alternating fair k-means (Algorithm 1, Stage 2): start from the k-means
    centers, then alternate (solve P_delta for fixed centers) <-> (move each
    center to the mean of its cluster) until the assignment stops changing.
    J(x, mu) is non-increasing and every iterate satisfies the balance
    certificate (Proposition 4)."""
    assign = SOLVERS[solver]
    alpha = np.bincount(group_ids, minlength=n_groups) / len(group_ids)
    if init_centers is None:
        init_centers = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z).cluster_centers_
    centers = init_centers
    labels, info, n_iter = None, {}, 0
    for n_iter in range(1, max_iter + 1):
        new_labels, info = assign(Z, centers, group_ids, n_groups, alpha, delta)
        if labels is not None and np.array_equal(new_labels, labels):
            break
        labels = new_labels
        centers = np.vstack([Z[labels == c].mean(axis=0) for c in range(k)])
    info = dict(info, n_iter=n_iter)
    return (labels, info) if return_info else labels
