"""
run_zdt_verification.py
=======================
Section 2.4; Table 2.

Verifies the NSGA-II, MOPSO and SMPSO implementations on ZDT1-ZDT3
(30 variables), whose Pareto-optimal fronts are known analytically:
population 100, 250 generations/iterations, seeds 1-10. The indicator is the
inverted generational distance (IGD) to a dense sample of the true front.

Output: data/01_zdt_verification.csv
"""
import numpy as np
import pandas as pd

import _common as C
from nsga2 import run_nsga2
from mopso_smpso import run_mopso, run_smpso
from metrics import igd, nondominated

D, POP, GEN, SEEDS = 30, 100, 250, range(1, 11)


def _g(X):
    return 1 + 9 * X[:, 1:].sum(1) / (X.shape[1] - 1)


def zdt1(X):
    f1 = X[:, 0]; g = _g(X)
    return np.stack([f1, g * (1 - np.sqrt(f1 / g))], 1), np.zeros(len(X))


def zdt2(X):
    f1 = X[:, 0]; g = _g(X)
    return np.stack([f1, g * (1 - (f1 / g) ** 2)], 1), np.zeros(len(X))


def zdt3(X):
    f1 = X[:, 0]; g = _g(X)
    f2 = g * (1 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1))
    return np.stack([f1, f2], 1), np.zeros(len(X))


def true_fronts():
    x = np.linspace(0, 1, 500)
    t = {'ZDT1': np.stack([x, 1 - np.sqrt(x)], 1),
         'ZDT2': np.stack([x, 1 - x ** 2], 1)}
    f1 = np.linspace(0, 1, 2000)
    P = np.stack([f1, 1 - np.sqrt(f1) - f1 * np.sin(10 * np.pi * f1)], 1)
    t['ZDT3'] = P[nondominated(P)]
    return t


def main():
    TRUE = true_fronts()
    lo, hi = np.zeros(D), np.ones(D)
    runners = {
        'NSGA-II': lambda fn, s: run_nsga2(fn, D, lo, hi, s, pop_size=POP, max_gen=GEN),
        'MOPSO': lambda fn, s: run_mopso(fn, D, lo, hi, s, pop_size=POP, max_iter=GEN),
        'SMPSO': lambda fn, s: run_smpso(fn, D, lo, hi, s, pop_size=POP, max_iter=GEN),
    }
    rows = []
    for name, fn in [('ZDT1', zdt1), ('ZDT2', zdt2), ('ZDT3', zdt3)]:
        for algo, run in runners.items():
            for s in SEEDS:
                r = run(fn, s)
                rows.append(dict(problem=name, optimizer=algo, seed=s,
                                 n_front=len(r['front_F']),
                                 igd=round(igd(r['front_F'], TRUE[name]), 6)))
            print(name, algo, 'done', flush=True)
    out = C.data_path('01_zdt_verification.csv')
    pd.DataFrame(rows).to_csv(out, index=False)
    print('wrote', out)


if __name__ == '__main__':
    main()
