"""
run_design_sensitivity.py
=========================
Section 3.7 (design sensitivity). Environments O1-O5, seeds 1-10.

  stage 'earlier'  first-stage sweeps with the earlier formulation of the
                   optimizer (vpso_core.VPSO: two-phase inertia weight,
                   transfer exponent 1/1.75, population 300, clamp 0.05):
                     cp4 ... cp12        number of control points, binary
                                         per-sample penalty, weight 500
                     count50/200/2000    binary per-sample penalty weight, 8 control points
                     depth500/2000       penetration-depth penalty weight, 8 control points
                   -> data/10_design_sensitivity_earlier.csv
  stage 'final'    repetition with the final schedule (VPSO of Table 1,
                   population 50, 100 iterations, clamp 0.5):
                     cp4 ... cp12        number of control points, depth penalty 500
                     depth500/2000, count500/2000   penalty form and weight, 4 control points
                   -> data/11_design_sensitivity_final.csv

Usage: python run_design_sensitivity.py earlier|final
"""
import sys
import pandas as pd

import _common as C
from vpso_core import FastEnvironment, VPSO, make_cost_function_v2
from algorithms import make_batched_path_cost, run_pso

ENVS = ['O1', 'O2', 'O3', 'O4', 'O5']
SEEDS = range(1, 11)


def earlier_configs():
    cfg = [(f'cp{n}', n, 'count', 500.0) for n in (4, 6, 8, 10, 12)]
    cfg += [(f'count{int(p)}', 8, 'count', p) for p in (50.0, 200.0, 2000.0)]
    cfg += [(f'depth{int(p)}', 8, 'depth', p) for p in (500.0, 2000.0)]
    return cfg


def final_configs():
    cfg = [(f'cp{n}', n, 'depth', 500.0) for n in (4, 6, 8, 10, 12)]
    cfg += [(f'{m}{int(p)}', 4, m, p) for m in ('depth', 'count') for p in (500.0, 2000.0)]
    return cfg


def main(stage):
    rows = []
    configs = earlier_configs() if stage == 'earlier' else final_configs()
    for name, ncp, mode, pen in configs:
        for ek in ENVS:
            env = FastEnvironment(C.ALL_ENVS[ek], ncp, 100)
            for s in SEEDS:
                if stage == 'earlier':
                    fn = make_cost_function_v2(env, collision_pen=pen,
                                               collision_mode=mode)
                    r = VPSO({'var_min': 0.0, 'var_max': 1.0, 'num_var': 2 * ncp,
                              'cost_function': fn}, s, pop_size=300, max_iter=100)
                    d = r['details']
                    length, feas, col = d['length'], d['feasible'], d['collisions']
                else:
                    cost = make_batched_path_cost(env, collision_pen=pen,
                                                  collision_mode=mode)
                    r = run_pso(cost, 2 * ncp, 0.0, 1.0, 'VPSO', s, pop_size=50,
                                max_iter=100, k=0.5)
                    a = r['aux']
                    length, feas, col = a['length'], a['feasible'], a['collisions']
                rows.append(dict(
                    config=name, control_points=ncp, penalty_form=mode,
                    penalty_weight=pen, env=ek, seed=s,
                    best_cost=round(float(r['best_cost']), 4),
                    path_length=round(float(length), 4),
                    collisions=int(col), feasible=int(feas),
                    smoothness=round(env.smoothness(env.path_from_cp(r['best_x'])), 5)))
        print(stage, name, 'done', flush=True)
    out = C.data_path('10_design_sensitivity_earlier.csv' if stage == 'earlier'
                      else '11_design_sensitivity_final.csv')
    pd.DataFrame(rows).to_csv(out, index=False)
    print('wrote', out)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'final')
