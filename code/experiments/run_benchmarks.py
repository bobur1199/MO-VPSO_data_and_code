"""
run_benchmarks.py
=================
Section 3.8; Tables 11-12; Figure 15.

The five schedules of Table 1 in single-objective form on six benchmark
functions (Griewank 30-D, Bohachevsky 2-D, Bukin N.6 2-D, Ackley 30-D,
Rastrigin 30-D, Levy 30-D): 30 runs each, population 300, 100 iterations,
clamp factor 0.05, identical initialization and seeds for every schedule.

Outputs
  data/12_benchmarks.csv                  final best objective value per run
  data/fronts/benchmark_histories.npz     best-so-far value per iteration
                                          (key '<schedule>|<function>', 30 x 100)
"""
import numpy as np
import pandas as pd

import _common as C
from algorithms import BENCHMARKS, run_pso, SCHEDULES


def main():
    rows, hist = [], {}
    for fname, b in BENCHMARKS.items():
        for algo in SCHEDULES:
            hs = []
            for s in range(1, 31):
                r = run_pso(lambda X: (b['f'](X), None), b['dim'], b['lo'], b['hi'],
                            algo, s, pop_size=300, max_iter=100, k=0.05)
                rows.append(dict(function=fname, optimizer=algo, seed=s,
                                 final_value=float(f"{r['best_cost']:.6e}")))
                hs.append(r['history'])
            hist[f'{algo}|{fname}'] = np.array(hs)
        print(fname, 'done', flush=True)
    out = C.data_path('12_benchmarks.csv')
    pd.DataFrame(rows).to_csv(out, index=False)
    np.savez_compressed(C.front_path('benchmark_histories.npz'), **hist)
    print('wrote', out)


if __name__ == '__main__':
    main()
