"""
run_mo_schedules.py
===================
Sections 3.1, 3.2 and 3.5 (restart ablation); Tables 3-7; Figures 4-12.

The five parameter schedules of Table 1 (MO-PSO, MO-IPSO, MO-APSO,
MO-RandWPSO, MO-VPSO) and the ablation variant MO-VPSO without restart are
run in the identical multi-objective framework (code/core/mo_framework.py)
on the ten environments, seeds 1-30, population 50, 100 iterations, four
control points, clamp factor 0.5.

Hypervolume uses one reference point per environment: 1.1 x the maximum of
each objective over all feasible archive members found by the FIVE compared
schedules in all runs. The no-restart variant is scored against the same
reference point and does not contribute to it.

Outputs
  data/fronts/mo_schedules_<ENV>.npz   final feasible archive of every run
                                       (objectives F and decision vectors X)
  data/02_mo_schedules.csv             one row per (environment, optimizer, seed)

Usage
  python run_mo_schedules.py [H6,H7,...]   run the listed environments (default: all)
  python run_mo_schedules.py --table       rebuild the CSV from the stored archives
"""
import os
import sys
import numpy as np
import pandas as pd

import _common as C
from vpso_core import FastEnvironment
from mo_framework import run_mo
from metrics import hypervolume_2d, reference_point

SEEDS = range(1, 31)
SCHEDULES = ['PSO', 'IPSO', 'APSO', 'RandWPSO', 'VPSO']
ABLATION = ['VPSO-noRestart']
LABEL = {'PSO': 'MO-PSO', 'IPSO': 'MO-IPSO', 'APSO': 'MO-APSO',
         'RandWPSO': 'MO-RandWPSO', 'VPSO': 'MO-VPSO',
         'VPSO-noRestart': 'MO-VPSO-noRestart'}
OUT = C.data_path('02_mo_schedules.csv')


def run_env(ek):
    env = FastEnvironment(C.ALL_ENVS[ek], 4, 100)
    store = {}
    for algo in SCHEDULES + ABLATION:
        for s in SEEDS:
            r = run_mo(env, algo, s, k=0.5, pop_size=50, max_iter=100, num_cp=4)
            store[f'{LABEL[algo]}|s{s}|F'] = r['archive_F'][r['feasible']]
            store[f'{LABEL[algo]}|s{s}|X'] = r['archive_X'][r['feasible']]
    np.savez_compressed(C.front_path(f'mo_schedules_{ek}.npz'), **store)
    print(f'{ek} done', flush=True)


def build_table():
    rows = []
    for ek in C.ENV_ORDER:
        fn = C.front_path(f'mo_schedules_{ek}.npz')
        if not os.path.exists(fn):
            raise SystemExit(f'missing {fn}; run the environment first')
        z = np.load(fn)
        ref = reference_point([z[f'{LABEL[a]}|s{s}|F'] for a in SCHEDULES
                               for s in SEEDS])
        for algo in SCHEDULES + ABLATION:
            for s in SEEDS:
                F = z[f'{LABEL[algo]}|s{s}|F']
                feas = int(len(F) > 0)
                rows.append(dict(
                    env=ek, optimizer=LABEL[algo], seed=s, feasible=feas,
                    n_front=len(F),
                    hv=round(hypervolume_2d(F, ref), 6),
                    best_len=round(float(F[:, 0].min()), 4) if feas else np.nan,
                    best_smooth=round(float(F[:, 1].min()), 6) if feas else np.nan,
                    ref_f1=round(float(ref[0]), 4), ref_f2=round(float(ref[1]), 6)))
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print('wrote', OUT)


if __name__ == '__main__':
    if '--table' in sys.argv:
        build_table()
    else:
        for ek in C.parse_envs(sys.argv):
            run_env(ek)
        if all(os.path.exists(C.front_path(f'mo_schedules_{e}.npz'))
               for e in C.ENV_ORDER):
            build_table()
