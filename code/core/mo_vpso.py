"""
mo_vpso.py
==========
Multi-objective building blocks shared by all particle swarm optimizers.

  dominates         constrained Pareto dominance (Deb, 2002): a feasible
                    solution dominates an infeasible one, the less violating
                    of two infeasible solutions dominates, and two feasible
                    solutions are compared by Pareto dominance on (f1, f2)
  crowding_distance crowding distance of an archive (Deb et al., 2002)
  update_archive    merge candidates, keep the constrained-non-dominated set,
                    prune to the capacity by crowding distance
  _mo_eval          f1 = path length (Eq. 5), f2 = mean absolute turning
                    angle (Eq. 6), violation = penetration depth into the
                    inflated obstacles + 50 x samples outside the workspace (Eq. 7)
  run_mo_vpso       stand-alone MO-VPSO with clamp factor 0.05 (Section 3.4)
"""
import numpy as np
from metrics import hypervolume_2d  # noqa: F401  (re-exported)


def dominates(a, b, va, vb):
    """Constrained dominance: a dominates b. a,b: (f1,f2) objective pairs.
    va, vb: constraint violation (0 = feasible)."""
    if va < vb:
        return True
    if va > vb:
        return False
    # both equal violation (usually both 0, i.e. both feasible)
    not_worse = (a[0] <= b[0]) and (a[1] <= b[1])
    strictly_better = (a[0] < b[0]) or (a[1] < b[1])
    return not_worse and strictly_better


def crowding_distance(F):
    """F: (N,2) objective values of the archive. Returns (N,) distances."""
    N = len(F)
    if N <= 2:
        return np.full(N, np.inf)
    d = np.zeros(N)
    for m in range(2):
        order = np.argsort(F[:, m])
        d[order[0]] = d[order[-1]] = np.inf
        span = F[order[-1], m] - F[order[0], m]
        if span <= 0:
            continue
        for i in range(1, N - 1):
            d[order[i]] += (F[order[i+1], m] - F[order[i-1], m]) / span
    return d


def update_archive(archive_X, archive_F, archive_V, X, F, V, cap=60):
    """Merge new candidates into the archive, keep only non-dominated
    (constrained) points, prune to `cap` by crowding distance."""
    allX = np.vstack([archive_X, X]) if len(archive_X) else X
    allF = np.vstack([archive_F, F]) if len(archive_F) else F
    allV = np.concatenate([archive_V, V]) if len(archive_V) else V
    n = len(allX)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j or not keep[j]:
                continue
            if dominates(allF[j], allF[i], allV[j], allV[i]):
                keep[i] = False
                break
    allX, allF, allV = allX[keep], allF[keep], allV[keep]
    if len(allX) > cap:
        d = crowding_distance(allF)
        order = np.argsort(-d)[:cap]
        allX, allF, allV = allX[order], allF[order], allV[order]
    return allX, allF, allV


def _mo_eval(env, X):
    """Vectorized evaluation of a population X (P, D).
    Returns F (P, 2) = [path length (Eq. 5), mean turning angle (Eq. 6)]
    and the constraint violation (P,) of Eq. (7) with lambda_e = 50."""
    W, H, rr = env.width, env.height, env.rr
    scale = np.array([W, H])
    B = env.basis
    P = X.shape[0]
    cp = X.reshape(P, -1, 2) * scale
    pts = np.concatenate(
        [np.broadcast_to(env.start, (P, 1, 2)), cp,
         np.broadcast_to(env.goal, (P, 1, 2))], axis=1)
    paths = np.einsum('rn,pnd->prd', B, pts)
    outx = (paths[..., 0] < rr) | (paths[..., 0] > W - rr)
    outy = (paths[..., 1] < rr) | (paths[..., 1] > H - rr)
    envv = (outx | outy).sum(axis=1)
    paths[..., 0] = np.clip(paths[..., 0], rr, W - rr)
    paths[..., 1] = np.clip(paths[..., 1], rr, H - rr)

    seg = np.diff(paths, axis=1)
    lengths = np.linalg.norm(seg, axis=2).sum(axis=1)

    depth = np.zeros(paths.shape[:2])
    if env.circles:
        d = np.linalg.norm(paths[:, :, None, :] - env.c_centers[None, None], axis=3)
        depth += np.maximum(env.c_radii[None, None] - d, 0).sum(axis=2)
    if env.rects:
        q = (np.abs(paths[:, :, None, :] - env.r_centers[None, None])
             - env.r_half[None, None])
        outside = np.linalg.norm(np.maximum(q, 0), axis=3)
        inside = np.minimum(np.maximum(q[..., 0], q[..., 1]), 0)
        sd = outside + inside
        depth += np.maximum(rr - sd, 0).sum(axis=2)
    violation = depth.sum(axis=1) + 50.0 * envv

    v = seg
    n = np.linalg.norm(v, axis=2)
    n = np.where(n < 1e-12, 1e-12, n)
    cosang = np.clip((v[:, :-1] * v[:, 1:]).sum(axis=2) / (n[:, :-1] * n[:, 1:]),
                     -1, 1)
    smooth = np.abs(np.arccos(cosang)).mean(axis=1)

    F = np.stack([lengths, smooth], axis=1)
    return F, violation


def run_mo_vpso(env, seed, pop_size=50, max_iter=100, num_cp=4,
                archive_cap=60, reset_every=25):
    """Stand-alone MO-VPSO with velocity clamp factor 0.05, used in the
    multi-objective versus weighted-sum comparison of Section 3.4."""
    D = 2 * num_cp
    rng = np.random.default_rng(seed)
    lo, hi = np.zeros(D), np.ones(D)
    vmax = 0.05 * (hi - lo)

    X = rng.uniform(lo, hi, (pop_size, D))
    V = np.zeros((pop_size, D))
    F, viol = _mo_eval(env, X)
    pbX, pbF, pbV = X.copy(), F.copy(), viol.copy()

    archive_X = np.empty((0, D)); archive_F = np.empty((0, 2))
    archive_V = np.empty((0,))
    archive_X, archive_F, archive_V = update_archive(
        archive_X, archive_F, archive_V, X, F, viol, cap=archive_cap)

    for t in range(1, max_iter + 1):
        w = 0.9 - 0.5 * t / max_iter
        p = np.clip(t / max_iter, 0, 1) ** 1.75
        c1, c2 = 2.0 - p*1.5, 0.5 + p*1.5

        if len(archive_X) > 0:
            d = crowding_distance(archive_F)
            leaders = np.empty((pop_size, D))
            for i in range(pop_size):
                i1, i2 = rng.integers(0, len(archive_X), 2)
                leaders[i] = archive_X[i1] if d[i1] >= d[i2] else archive_X[i2]
        else:
            leaders = np.tile(pbX[np.argmin(pbV + pbF.sum(1))], (pop_size, 1))

        r1 = rng.random((pop_size, D)); r2 = rng.random((pop_size, D))
        V = np.clip(w*V + c1*r1*(pbX - X) + c2*r2*(leaders - X), -vmax, vmax)
        X = np.clip(X + V, lo, hi)

        if reset_every and t % reset_every == 0:
            # restart the worst 20% by aggregate rank (violation, then sum of objectives)
            rank_key = pbV + pbF.sum(1)
            k = max(1, int(0.2 * pop_size))
            worst = np.argpartition(rank_key, -k)[-k:]
            X[worst] = rng.uniform(lo, hi, (k, D)); V[worst] = 0.

        F, viol = _mo_eval(env, X)
        for i in range(pop_size):
            if dominates(F[i], pbF[i], viol[i], pbV[i]):
                pbX[i], pbF[i], pbV[i] = X[i].copy(), F[i].copy(), viol[i]
            elif not dominates(pbF[i], F[i], pbV[i], viol[i]):
                if rng.random() < 0.5:
                    pbX[i], pbF[i], pbV[i] = X[i].copy(), F[i].copy(), viol[i]
        archive_X, archive_F, archive_V = update_archive(
            archive_X, archive_F, archive_V, X, F, viol, cap=archive_cap)

    feasible_mask = archive_V <= 1e-9
    return {
        'archive_X': archive_X, 'archive_F': archive_F, 'archive_V': archive_V,
        'feasible': feasible_mask, 'n_feasible': int(feasible_mask.sum()),
    }
