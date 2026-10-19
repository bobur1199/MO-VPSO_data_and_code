"""
run_clamp_sweep.py
==================
Section 3.6; Table 10; Figure 14.

The clamp factor k of Eq. (2) is swept over {0.05, 0.15, 0.5, 1.0} with all
other settings fixed: the five schedules of Table 1, weighted-sum cost,
environments H6-H9, seeds 1-30, population 50, 100 iterations, four control
points. Saturation is the fraction of velocity components that exceed the
clamp before clipping, averaged over all iterations and over iterations 1-25.

Output: data/09_clamp_sweep.csv
"""
import pandas as pd

import _common as C
from vpso_core import FastEnvironment
from algorithms import make_batched_path_cost, run_pso, SCHEDULES

KAPPAS = [0.05, 0.15, 0.5, 1.0]
ENVS = ['H6', 'H7', 'H8', 'H9']


def main():
    rows = []
    for k in KAPPAS:
        for ek in ENVS:
            env = FastEnvironment(C.ALL_ENVS[ek], 4, 100)
            cost = make_batched_path_cost(env)
            for algo in SCHEDULES:
                for s in range(1, 31):
                    r = run_pso(cost, 8, 0.0, 1.0, algo, s, pop_size=50,
                                max_iter=100, k=k, record_saturation=True)
                    rows.append(dict(
                        kappa=k, env=ek, optimizer=algo, seed=s,
                        best_cost=round(r['best_cost'], 4),
                        path_length=round(float(r['aux']['length']), 4),
                        feasible=int(r['aux']['feasible']),
                        saturation_all=round(float(r['saturation'].mean()), 5),
                        saturation_1_25=round(float(r['saturation'][:25].mean()), 5)))
            print(f'kappa={k} {ek} done', flush=True)
    out = C.data_path('09_clamp_sweep.csv')
    pd.DataFrame(rows).to_csv(out, index=False)
    print('wrote', out)


if __name__ == '__main__':
    main()
