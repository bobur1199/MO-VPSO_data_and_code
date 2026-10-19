"""
mo_framework.py
===============
Shared multi-objective particle swarm framework of Section 2.2 (Fig. 1).

The Pareto machinery -- constrained dominance, external archive capped at 60
entries and pruned by crowding distance, leader selection by binary
tournament on crowding distance, and the personal-best update rule -- is
identical for every optimizer. Only the parameter schedule (w, c1, c2) and the
restart rule differ, so differences between MO-PSO, MO-IPSO, MO-APSO,
MO-RandWPSO and MO-VPSO are attributable to the schedule alone (Section 2.3).

Objectives (both minimized): f1 = path length (Eq. 5), f2 = mean absolute
turning angle (Eq. 6). Constraint: penetration depth plus boundary term
(Eq. 7), handled by constrained dominance (Deb, 2002).
"""
import numpy as np
from mo_vpso import dominates, crowding_distance, update_archive, _mo_eval
from algorithms import get_schedule


def run_mo(env, algo, seed, k=0.5, pop_size=50, max_iter=100, num_cp=4,
           archive_cap=60):
    """One run of the multi-objective framework with schedule `algo`
    (a name from algorithms.ALGOS / VARIANTS). Returns the final archive
    (objectives, decision vectors) and its feasibility mask."""
    cfg = get_schedule(algo)
    D = 2 * num_cp
    rng = np.random.default_rng(seed)
    lo, hi = np.zeros(D), np.ones(D)
    vmax = k * (hi - lo)

    X = rng.uniform(lo, hi, (pop_size, D))
    V = np.zeros((pop_size, D))
    F, viol = _mo_eval(env, X)
    pbX, pbF, pbV = X.copy(), F.copy(), viol.copy()

    aX = np.empty((0, D)); aF = np.empty((0, 2)); aV = np.empty((0,))
    aX, aF, aV = update_archive(aX, aF, aV, X, F, viol, cap=archive_cap)

    for t in range(1, max_iter + 1):
        w = cfg['w'](t, max_iter, rng)
        c1, c2 = cfg['c'](t, max_iter)

        if len(aX) > 0:
            dcr = crowding_distance(aF)
            idx1 = rng.integers(0, len(aX), pop_size)
            idx2 = rng.integers(0, len(aX), pop_size)
            pick = np.where(dcr[idx1] >= dcr[idx2], idx1, idx2)
            leaders = aX[pick]
        else:
            leaders = np.tile(pbX[np.argmin(pbV + pbF.sum(1))], (pop_size, 1))

        V = np.clip(w*V + c1*rng.random((pop_size, D))*(pbX - X)
                    + c2*rng.random((pop_size, D))*(leaders - X), -vmax, vmax)
        X = np.clip(X + V, lo, hi)

        if cfg['reset'] and t % cfg['reset'] == 0:
            kk = max(1, int(0.2 * pop_size))
            rank_key = pbV + pbF.sum(1)
            worst = np.argpartition(rank_key, -kk)[-kk:]
            X[worst] = rng.uniform(lo, hi, (kk, D)); V[worst] = 0.

        F, viol = _mo_eval(env, X)
        for i in range(pop_size):
            if dominates(F[i], pbF[i], viol[i], pbV[i]):
                pbX[i], pbF[i], pbV[i] = X[i].copy(), F[i].copy(), viol[i]
            elif not dominates(pbF[i], F[i], pbV[i], viol[i]):
                if rng.random() < 0.5:
                    pbX[i], pbF[i], pbV[i] = X[i].copy(), F[i].copy(), viol[i]
        aX, aF, aV = update_archive(aX, aF, aV, X, F, viol, cap=archive_cap)

    feas = aV <= 1e-9
    return {'archive_F': aF, 'archive_X': aX, 'feasible': feas,
            'n_feasible': int(feas.sum())}
