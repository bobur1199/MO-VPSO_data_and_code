"""
run_mo_baselines.py
===================
Section 3.3; Table 8; Figure 13.

MO-VPSO is compared with NSGA-II, MOPSO and SMPSO on the ten environments,
seeds 1-30, population 50 and 100 generations/iterations (5050 function
evaluations per run for every optimizer).

Indicators per run
  hv        hypervolume against a reference point shared by the four
            optimizers within each environment (1.1 x the per-objective
            maximum over all their feasible solutions)
  igd_plus  IGD+ against the non-dominated union of all feasible solutions
            found by the four optimizers in that environment, objectives
            normalized between the ideal point of that reference front and the
            hypervolume reference point (runs without a feasible path: empty)

Outputs
  data/fronts/mo_baselines_<ENV>.npz   final feasible front of every run
  data/03_mo_baselines.csv             one row per (environment, optimizer, seed)

Usage
  python run_mo_baselines.py [H6,H7,...]   run the listed environments (default: all)
  python run_mo_baselines.py --table       rebuild the CSV from the stored fronts
"""
import os
import sys
import numpy as np
import pandas as pd

import _common as C
from vpso_core import FastEnvironment
from mo_vpso import _mo_eval
from mo_framework import run_mo
from nsga2 import run_nsga2
from mopso_smpso import run_mopso, run_smpso
from metrics import hypervolume_2d, reference_point, nondominated, igd_plus

SEEDS = range(1, 31)
POP, IT, CP = 50, 100, 4
OPTIMIZERS = ['MO-VPSO', 'NSGA-II', 'MOPSO', 'SMPSO']
OUT = C.data_path('03_mo_baselines.csv')


def run_env(ek):
    env = FastEnvironment(C.ALL_ENVS[ek], CP, 100)
    D = 2 * CP
    lo, hi = np.zeros(D), np.ones(D)

    def ev(X):
        return _mo_eval(env, X)

    store = {}
    for s in SEEDS:
        r = run_mo(env, 'VPSO', s, k=0.5, pop_size=POP, max_iter=IT, num_cp=CP)
        store[f'MO-VPSO|s{s}|F'] = r['archive_F'][r['feasible']]
        store[f'MO-VPSO|s{s}|X'] = r['archive_X'][r['feasible']]
        for name, fn in [('NSGA-II', lambda: run_nsga2(ev, D, lo, hi, s, pop_size=POP, max_gen=IT)),
                         ('MOPSO', lambda: run_mopso(ev, D, lo, hi, s, pop_size=POP, max_iter=IT)),
                         ('SMPSO', lambda: run_smpso(ev, D, lo, hi, s, pop_size=POP, max_iter=IT))]:
            out = fn()
            store[f'{name}|s{s}|F'] = out['front_F']
            store[f'{name}|s{s}|X'] = out['front_X']
    np.savez_compressed(C.front_path(f'mo_baselines_{ek}.npz'), **store)
    print(f'{ek} done', flush=True)


def build_table():
    rows = []
    for ek in C.ENV_ORDER:
        fn = C.front_path(f'mo_baselines_{ek}.npz')
        if not os.path.exists(fn):
            raise SystemExit(f'missing {fn}; run the environment first')
        z = np.load(fn)
        fronts = {(a, s): z[f'{a}|s{s}|F'] for a in OPTIMIZERS for s in SEEDS}
        ref = reference_point(list(fronts.values()))
        pool = np.vstack([f for f in fronts.values() if len(f)])
        ref_front = pool[nondominated(pool)]
        ideal = ref_front.min(axis=0)
        for a in OPTIMIZERS:
            for s in SEEDS:
                F = fronts[(a, s)]
                feas = int(len(F) > 0)
                rows.append(dict(
                    env=ek, optimizer=a, seed=s, feasible=feas, n_front=len(F),
                    hv=round(hypervolume_2d(F, ref), 6),
                    igd_plus=round(igd_plus(F, ref_front, ideal, ref), 6) if feas else np.nan,
                    best_len=round(float(F[:, 0].min()), 4) if feas else np.nan,
                    ref_f1=round(float(ref[0]), 4), ref_f2=round(float(ref[1]), 6)))
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print('wrote', OUT)


if __name__ == '__main__':
    if '--table' in sys.argv:
        build_table()
    else:
        for ek in C.parse_envs(sys.argv):
            run_env(ek)
        if all(os.path.exists(C.front_path(f'mo_baselines_{e}.npz'))
               for e in C.ENV_ORDER):
            build_table()
