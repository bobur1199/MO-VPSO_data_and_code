"""
nsga2.py
========
NSGA-II implemented from the published specification:

  Deb, K.; Pratap, A.; Agarwal, S.; Meyarivan, T. "A fast and elitist
  multiobjective genetic algorithm: NSGA-II." IEEE Trans. Evol. Comput.
  2002, 6(2), 182-197.

Components follow the paper exactly:
  - fast non-dominated sort (Section III-A, O(MN^2))
  - crowding distance assignment (Section III-B)
  - crowded-comparison operator: lower rank wins; ties broken by larger
    crowding distance (Section III-C)
  - binary tournament selection on the crowded-comparison operator
  - simulated binary crossover (SBX) and polynomial mutation (Deb & Agrawal),
    the standard real-coded operators used with NSGA-II
  - (mu + lambda) elitist survival: parents and offspring merged, then
    truncated by rank and crowding distance

Constraint handling uses Deb's constrained-domination principle (the same
rule used by MO-VPSO), so the comparison is like-for-like:
  a constraint-dominates b if
    (i)  a feasible, b infeasible; or
    (ii) both infeasible and a has smaller violation; or
    (iii) both feasible and a Pareto-dominates b.

Default operator parameters are the values recommended in the NSGA-II
literature: p_c = 0.9, eta_c = 20, p_m = 1/D, eta_m = 20.
"""
import numpy as np


def fast_non_dominated_sort(F, V):
    """Vectorized fast non-dominated sort (Deb 2002, Sec. III-A).
    Domination matrix computed with numpy; ranks identical to the loop form."""
    N = len(F)
    if N == 0:
        return []
    feas = V <= 0
    # Pareto domination among objective vectors
    le = np.all(F[:, None, :] <= F[None, :, :], axis=2)
    lt = np.any(F[:, None, :] < F[None, :, :], axis=2)
    pdom = le & lt
    # constrained domination matrix D[p,q] = p dominates q
    both_f = feas[:, None] & feas[None, :]
    both_i = (~feas[:, None]) & (~feas[None, :])
    f_over_i = feas[:, None] & (~feas[None, :])
    less_v = V[:, None] < V[None, :]
    D = (both_f & pdom) | f_over_i | (both_i & less_v)
    np.fill_diagonal(D, False)

    dom_count = D.sum(axis=0)          # how many dominate p
    fronts = []
    remaining = np.ones(N, dtype=bool)
    counts = dom_count.copy()
    while remaining.any():
        cur = np.where(remaining & (counts == 0))[0]
        if len(cur) == 0:
            cur = np.where(remaining)[0]      # safety: cycle guard
        fronts.append(cur)
        remaining[cur] = False
        counts = counts - D[cur].sum(axis=0)
    return fronts


def _cdom(fa, va, fb, vb):
    """Constrained domination (Deb 2002, Section V-A)."""
    if va <= 0 and vb > 0:
        return True
    if va > 0 and vb <= 0:
        return False
    if va > 0 and vb > 0:
        return va < vb
    return np.all(fa <= fb) and np.any(fa < fb)


def crowding_distance(F):
    N, M = F.shape
    if N == 0:
        return np.zeros(0)
    d = np.zeros(N)
    for m in range(M):
        order = np.argsort(F[:, m])
        d[order[0]] = d[order[-1]] = np.inf
        span = F[order[-1], m] - F[order[0], m]
        if span <= 0:
            continue
        for i in range(1, N - 1):
            d[order[i]] += (F[order[i+1], m] - F[order[i-1], m]) / span
    return d


def sbx(p1, p2, lo, hi, rng, eta=20.0, pc=0.9):
    c1, c2 = p1.copy(), p2.copy()
    if rng.random() > pc:
        return c1, c2
    for i in range(len(p1)):
        if rng.random() > 0.5:
            continue
        if abs(p1[i] - p2[i]) < 1e-14:
            continue
        x1, x2 = min(p1[i], p2[i]), max(p1[i], p2[i])
        u = rng.random()
        beta = 1.0 + 2.0 * (x1 - lo[i]) / (x2 - x1)
        alpha = 2.0 - beta ** -(eta + 1)
        bq = ((u * alpha) ** (1/(eta+1)) if u <= 1/alpha
              else (1/(2 - u*alpha)) ** (1/(eta+1)))
        y1 = 0.5 * ((x1 + x2) - bq * (x2 - x1))
        beta = 1.0 + 2.0 * (hi[i] - x2) / (x2 - x1)
        alpha = 2.0 - beta ** -(eta + 1)
        bq = ((u * alpha) ** (1/(eta+1)) if u <= 1/alpha
              else (1/(2 - u*alpha)) ** (1/(eta+1)))
        y2 = 0.5 * ((x1 + x2) + bq * (x2 - x1))
        y1, y2 = np.clip(y1, lo[i], hi[i]), np.clip(y2, lo[i], hi[i])
        if rng.random() < 0.5:
            c1[i], c2[i] = y2, y1
        else:
            c1[i], c2[i] = y1, y2
    return c1, c2


def poly_mutation(x, lo, hi, rng, eta=20.0, pm=None):
    y = x.copy()
    D = len(x)
    if pm is None:
        pm = 1.0 / D
    for i in range(D):
        if rng.random() > pm:
            continue
        yl, yu = lo[i], hi[i]
        if yu - yl < 1e-14:
            continue
        delta1 = (y[i] - yl) / (yu - yl)
        delta2 = (yu - y[i]) / (yu - yl)
        u = rng.random()
        mut_pow = 1.0 / (eta + 1.0)
        if u <= 0.5:
            xy = 1.0 - delta1
            val = 2.0*u + (1.0 - 2.0*u) * xy**(eta+1)
            dq = val**mut_pow - 1.0
        else:
            xy = 1.0 - delta2
            val = 2.0*(1.0-u) + 2.0*(u-0.5) * xy**(eta+1)
            dq = 1.0 - val**mut_pow
        y[i] = np.clip(y[i] + dq*(yu - yl), yl, yu)
    return y


def tournament(rank, crowd, rng, n):
    """Binary tournament on the crowded-comparison operator."""
    a = rng.integers(0, n, n)
    b = rng.integers(0, n, n)
    win = np.where(
        rank[a] < rank[b], a,
        np.where(rank[b] < rank[a], b,
                 np.where(crowd[a] >= crowd[b], a, b)))
    return win


def run_nsga2(eval_fn, D, lo, hi, seed, pop_size=50, max_gen=100,
              pc=0.9, eta_c=20.0, eta_m=20.0):
    """eval_fn(X) -> (F (N,M), V (N,)). Budget: pop_size*(max_gen+1) evals,
    matching a PSO run of pop_size particles over max_gen iterations."""
    rng = np.random.default_rng(seed)
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    X = rng.uniform(lo, hi, (pop_size, D))
    F, V = eval_fn(X)

    for gen in range(max_gen):
        fronts = fast_non_dominated_sort(F, V)
        rank = np.empty(len(X), dtype=int)
        crowd = np.zeros(len(X))
        for r, fr in enumerate(fronts):
            rank[fr] = r
            crowd[fr] = crowding_distance(F[fr])

        sel = tournament(rank, crowd, rng, len(X))
        off = np.empty_like(X)
        for i in range(0, pop_size, 2):
            p1, p2 = X[sel[i]], X[sel[min(i+1, pop_size-1)]]
            c1, c2 = sbx(p1, p2, lo, hi, rng, eta_c, pc)
            off[i] = poly_mutation(c1, lo, hi, rng, eta_m)
            if i + 1 < pop_size:
                off[i+1] = poly_mutation(c2, lo, hi, rng, eta_m)
        Foff, Voff = eval_fn(off)

        # (mu + lambda) elitist survival
        Xall = np.vstack([X, off])
        Fall = np.vstack([F, Foff])
        Vall = np.concatenate([V, Voff])
        fronts = fast_non_dominated_sort(Fall, Vall)
        keep = []
        for fr in fronts:
            if len(keep) + len(fr) <= pop_size:
                keep.extend(fr.tolist())
            else:
                cd = crowding_distance(Fall[fr])
                order = fr[np.argsort(-cd)]
                keep.extend(order[:pop_size - len(keep)].tolist())
                break
        keep = np.array(keep, dtype=int)
        X, F, V = Xall[keep], Fall[keep], Vall[keep]

    fronts = fast_non_dominated_sort(F, V)
    best = fronts[0]
    feas = V[best] <= 1e-9
    return {'X': X, 'F': F, 'V': V,
            'front_X': X[best][feas], 'front_F': F[best][feas],
            'n_feasible': int(feas.sum())}
