"""
run_mo_vs_so.py
===============
Section 3.4 (pre-registered confirmatory study; analysis plan in
preregistration/PREREGISTRATION_multi_objective_confirmatory.md).

The same VPSO update rule is run once with the Pareto archive and
crowding-distance leader selection (MO-VPSO) and once with the weighted-sum
cost (SO-VPSO) on O2, O4, H7 and H8; population 50, 100 iterations, four
control points, clamp factor 0.05 (the original protocol). The single-
objective solution is scored on the same (length, smoothness) pair and the
same feasibility rule. Hypervolume uses a reference point pooled over both
methods within each environment (1.1 x the per-objective maximum).

  stage 'exploratory'   seeds 1-20   -> data/05_mo_vs_so_exploratory.csv
  stage 'confirmatory'  seeds 41-90  -> data/06_mo_vs_so_confirmatory.csv

Usage: python run_mo_vs_so.py exploratory|confirmatory
"""
import sys
import numpy as np
import pandas as pd

import _common as C
from vpso_core import FastEnvironment
from algorithms import make_batched_path_cost, run_pso
from mo_vpso import run_mo_vpso, _mo_eval
from metrics import hypervolume_2d, reference_point

ENVS = ['O2', 'O4', 'H7', 'H8']
STAGES = {'exploratory': (range(1, 21), '05_mo_vs_so_exploratory.csv'),
          'confirmatory': (range(41, 91), '06_mo_vs_so_confirmatory.csv')}
POP, IT, CP = 50, 100, 4


def main(stage):
    seeds, fname = STAGES[stage]
    rows = []
    for ek in ENVS:
        env = FastEnvironment(C.ALL_ENVS[ek], CP, 100)
        cost = make_batched_path_cost(env)
        so, mo = {}, {}
        for s in seeds:
            r = run_pso(cost, 2 * CP, 0.0, 1.0, 'VPSO', s, pop_size=POP,
                        max_iter=IT, k=0.05)
            F, viol = _mo_eval(env, r['best_x'][None])
            so[s] = (F[0], int(viol[0] <= 1e-9))
            mo[s] = run_mo_vpso(env, s, pop_size=POP, max_iter=IT, num_cp=CP)
        ref = reference_point(
            [m['archive_F'][m['feasible']] for m in mo.values()]
            + [f[None] for f, fe in so.values() if fe])
        for s in seeds:
            f, fe = so[s]
            rows.append(dict(env=ek, method='SO-VPSO', seed=s,
                             length=round(float(f[0]), 4),
                             smoothness=round(float(f[1]), 5), feasible=fe,
                             n_front=fe,
                             hv=round(hypervolume_2d(f[None], ref), 4) if fe else 0.0))
        for s in seeds:
            m = mo[s]
            Ff = m['archive_F'][m['feasible']]
            if len(Ff):
                i = int(Ff[:, 0].argmin())
                rows.append(dict(env=ek, method='MO-VPSO', seed=s,
                                 length=round(float(Ff[i, 0]), 4),
                                 smoothness=round(float(Ff[i, 1]), 5), feasible=1,
                                 n_front=len(Ff),
                                 hv=round(hypervolume_2d(Ff, ref), 4)))
            else:
                rows.append(dict(env=ek, method='MO-VPSO', seed=s, length=np.nan,
                                 smoothness=np.nan, feasible=0, n_front=0, hv=0.0))
        print(stage, ek, 'done', flush=True)
    out = C.data_path(fname)
    pd.DataFrame(rows).to_csv(out, index=False)
    print('wrote', out)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'confirmatory')
