"""
run_so_schedules.py
===================
Section 3.4; Table 9.

The five schedules of Table 1 in the conventional single-objective
formulation (weighted-sum cost: path length + 500 x penetration depth +
50 x samples outside the workspace), with the budget and seeds of the
multi-objective comparison: ten environments, seeds 1-30, population 50,
100 iterations, four control points, clamp factor 0.5.

Output: data/04_so_schedules.csv
"""
import pandas as pd

import _common as C
from vpso_core import FastEnvironment
from algorithms import make_batched_path_cost, run_pso, SCHEDULES


def main():
    rows = []
    for ek in C.ENV_ORDER:
        env = FastEnvironment(C.ALL_ENVS[ek], 4, 100)
        cost = make_batched_path_cost(env)
        for algo in SCHEDULES:
            for s in range(1, 31):
                r = run_pso(cost, 8, 0.0, 1.0, algo, s, pop_size=50,
                            max_iter=100, k=0.5)
                rows.append(dict(
                    env=ek, optimizer=algo, seed=s,
                    best_cost=round(r['best_cost'], 4),
                    path_length=round(float(r['aux']['length']), 4),
                    smoothness=round(env.smoothness(env.path_from_cp(r['best_x'])), 5),
                    feasible=int(r['aux']['feasible'])))
        print(ek, 'done', flush=True)
    out = C.data_path('04_so_schedules.csv')
    pd.DataFrame(rows).to_csv(out, index=False)
    print('wrote', out)


if __name__ == '__main__':
    main()
