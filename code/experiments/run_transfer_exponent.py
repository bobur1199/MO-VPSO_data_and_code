"""
run_transfer_exponent.py
========================
Section 3.5 (transfer-exponent ablation).

Single-objective weighted-sum cost, population 50, 100 iterations, four
control points, clamp factor 0.5, linearly decreasing inertia weight.
Variants differ only in the learning-factor schedule of Eq. (10):

  VPSO-e1.75       gamma = 1.75 (proposed, back-loaded), C in [0.5, 2.0], restart
  VPSO-e1/1.75     gamma = 1/1.75 (front-loaded),         C in [0.5, 2.0], restart
  VPSO-e1.0        gamma = 1 (linear),                    C in [0.5, 2.0], restart
  VPSO-e1.0-wide   gamma = 1 (linear),                    C in [0.5, 2.5], restart
  APSO             gamma = 1 (linear),                    C in [0.5, 2.5], no restart

  stage 'exploratory'   H6, H8, H9; seeds 1-30; all five variants
                        -> data/07_transfer_exponent_exploratory.csv
  stage 'confirmatory'  pre-specified comparison of VPSO-e1.75 with
                        VPSO-e1/1.75 (and APSO) on fresh seeds: 101-180 for H6
                        and H7, 101-160 for H8 and H9 (sample as executed)
                        -> data/08_transfer_exponent_confirmatory.csv

Usage: python run_transfer_exponent.py exploratory|confirmatory
"""
import sys
import pandas as pd

import _common as C
from vpso_core import FastEnvironment
from algorithms import make_batched_path_cost, run_pso, transfer_schedule, linear_w


def schedule(gamma, c_max, reset):
    return dict(w=linear_w,
                c=lambda t, T: transfer_schedule(t, T, gamma, c_max, 0.5),
                reset=reset)


VARIANTS = {
    'APSO':           schedule(1.0, 2.5, None),
    'VPSO-e1/1.75':   schedule(1 / 1.75, 2.0, 25),
    'VPSO-e1.0':      schedule(1.0, 2.0, 25),
    'VPSO-e1.75':     schedule(1.75, 2.0, 25),
    'VPSO-e1.0-wide': schedule(1.0, 2.5, 25),
}
STAGES = {
    'exploratory': dict(design={'H6': range(1, 31), 'H8': range(1, 31),
                                'H9': range(1, 31)},
                        variants=list(VARIANTS),
                        out='07_transfer_exponent_exploratory.csv'),
    'confirmatory': dict(design={'H6': range(101, 181), 'H7': range(101, 181),
                                 'H8': range(101, 161), 'H9': range(101, 161)},
                         variants=['APSO', 'VPSO-e1/1.75', 'VPSO-e1.75'],
                         out='08_transfer_exponent_confirmatory.csv'),
}


def main(stage):
    cfg = STAGES[stage]
    rows = []
    for ek, seeds in cfg['design'].items():
        env = FastEnvironment(C.ALL_ENVS[ek], 4, 100)
        cost = make_batched_path_cost(env)
        for v in cfg['variants']:
            for s in seeds:
                r = run_pso(cost, 8, 0.0, 1.0, VARIANTS[v], s, pop_size=50,
                            max_iter=100, k=0.5)
                rows.append(dict(env=ek, variant=v, seed=s,
                                 feasible=int(r['aux']['feasible']),
                                 length=round(float(r['aux']['length']), 4)))
        print(stage, ek, 'done', flush=True)
    out = C.data_path(cfg['out'])
    pd.DataFrame(rows).to_csv(out, index=False)
    print('wrote', out)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'confirmatory')
