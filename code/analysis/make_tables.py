"""
make_tables.py
==============
Recomputes every table and every statistic reported in the manuscript from
the raw data in data/, and writes

  results/tables/Table02_zdt_verification.csv ... Table12_benchmarks.csv
  results/statistics_report.txt      all test statistics quoted in the text

Usage: python make_tables.py
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'code', 'core'))
from stats_tools import (friedman, wilcoxon_paired, van_elteren,  # noqa: E402
                         binomial_paired, mann_whitney, holm)

DATA = os.path.join(ROOT, 'data')
TAB = os.path.join(ROOT, 'results', 'tables')
os.makedirs(TAB, exist_ok=True)

ENVS = ['H6', 'H7', 'H8', 'H9', 'H10', 'O1', 'O2', 'O3', 'O4', 'O5']
HARD = ENVS[:5]
MO = ['MO-PSO', 'MO-IPSO', 'MO-APSO', 'MO-RandWPSO', 'MO-VPSO']
SO = ['PSO', 'IPSO', 'APSO', 'RandWPSO', 'VPSO']
EST = ['MO-VPSO', 'NSGA-II', 'MOPSO', 'SMPSO']

REPORT = []


def say(*lines):
    for ln in lines:
        REPORT.append(ln)
        print(ln)


def sci(p):
    return f'{p:.2g}' if p >= 1e-3 else f'{p:.1e}'


def save(df, name):
    df.to_csv(os.path.join(TAB, name))


def read(name):
    return pd.read_csv(os.path.join(DATA, name))


# --------------------------------------------------------------- Table 2 ----
def table2():
    d = read('01_zdt_verification.csv')
    t = d.pivot_table(index='problem', columns='optimizer', values='igd',
                      aggfunc='mean')[['NSGA-II', 'MOPSO', 'SMPSO']]
    save(t.round(4), 'Table02_zdt_verification.csv')
    say('', '=== Table 2: mean IGD on ZDT problems (10 runs) ===', t.round(4).to_string())


# ----------------------------------------------------------- Tables 3-7 -----
def tables3to7():
    d = read('02_mo_schedules.csv')
    m = d[d.optimizer.isin(MO)]
    hv = m.pivot_table(index='env', columns='optimizer', values='hv').loc[ENVS, MO]
    save(hv.round(2), 'Table03_mo_hypervolume.csv')
    say('', '=== Table 3: mean hypervolume ===', hv.round(2).to_string())

    fr = friedman(hv.values, higher_is_better=True)
    say(f'Friedman over environments (N={fr["n_blocks"]}): chi2={fr["chi2"]:.2f}, '
        f'p={sci(fr["p"])}, Kendall W={fr["W"]:.2f}',
        'Mean ranks: ' + ', '.join(f'{a} {r:.1f}' for a, r in zip(MO, fr['mean_ranks'])))
    blk = m.pivot_table(index=['env', 'seed'], columns='optimizer', values='hv')[MO]
    fr2 = friedman(blk.values, higher_is_better=True)
    say(f'Friedman over runs (N={fr2["n_blocks"]}): chi2={fr2["chi2"]:.2f}, '
        f'p={sci(fr2["p"])}, W={fr2["W"]:.2f}; mean rank MO-VPSO {fr2["mean_ranks"][-1]:.2f}')

    rows = []
    for subset, envs in [('All 10 env.', ENVS), ('Hard set', HARD)]:
        b = blk.loc[envs]
        ps, rs = [], []
        for a in MO[:-1]:
            p, r = wilcoxon_paired(b['MO-VPSO'], b[a])
            ps.append(p); rs.append(r)
        for a, p, pa, r in zip(MO[:-1], ps, holm(ps), rs):
            rows.append(dict(comparison=f'vs {a}', subset=subset, n_pairs=len(b),
                             p_raw=p, p_holm=pa, r=round(r, 3)))
    t4 = pd.DataFrame(rows)
    save(t4.set_index(['comparison', 'subset']), 'Table04_mo_wilcoxon.csv')
    say('', '=== Table 4: Wilcoxon signed-rank, MO-VPSO vs each (Holm over 4) ===',
        t4.assign(p_holm=t4.p_holm.map(sci), p_raw=t4.p_raw.map(sci)).to_string(index=False))

    sr = 100 * m.pivot_table(index='env', columns='optimizer', values='feasible').loc[ENVS, MO]
    save(sr.round(0), 'Table05_mo_success_rate.csv')
    say('', '=== Table 5: success rate (%) ===', sr.round(0).to_string())
    pooled = 100 * m.groupby('optimizer').feasible.mean()[MO]
    fb = m.pivot_table(index=['env', 'seed'], columns='optimizer', values='feasible')[MO]
    ps, info = [], []
    for a in MO[:-1]:
        n1, n0, p = binomial_paired(fb['MO-VPSO'], fb[a]); ps.append(p); info.append((a, n1, n0))
    say('Pooled success rate: ' + ', '.join(f'{a} {v:.1f}%' for a, v in pooled.items()))
    for (a, n1, n0), p, pa in zip(info, ps, holm(ps)):
        say(f'  MO-VPSO vs {a}: discordant {n1} vs {n0}, exact binomial p={sci(p)}, p_Holm={sci(pa)}')

    f = m[m.feasible == 1]
    nf = f.pivot_table(index='env', columns='optimizer', values='n_front').reindex(ENVS)[MO]
    save(nf.round(1), 'Table06_mo_front_size.csv')
    say('', '=== Table 6: mean number of non-dominated collision-free paths (feasible runs) ===',
        nf.round(1).to_string())
    ln = f.pivot_table(index='env', columns='optimizer', values='best_len').reindex(ENVS)[MO]
    save(ln.round(2), 'Table07_mo_shortest_path.csv')
    say('', '=== Table 7: mean shortest collision-free path length (feasible runs) ===',
        ln.round(2).to_string())

    # restart ablation (Section 3.5)
    ab = d[d.optimizer.isin(['MO-VPSO', 'MO-VPSO-noRestart'])]
    pv = ab.pivot_table(index=['env', 'seed'], columns='optimizer',
                        values=['hv', 'feasible', 'n_front'])
    p, r = wilcoxon_paired(pv['hv']['MO-VPSO'], pv['hv']['MO-VPSO-noRestart'])
    n1, n0, pb = binomial_paired(pv['feasible']['MO-VPSO-noRestart'], pv['feasible']['MO-VPSO'])
    fe = ab[ab.feasible == 1].groupby('optimizer').n_front.mean()
    say('', '=== Section 3.5: restart ablation (scored against the Table 3 reference points) ===',
        f'Mean HV: without restart {pv["hv"]["MO-VPSO-noRestart"].mean():.2f}, '
        f'with restart {pv["hv"]["MO-VPSO"].mean():.2f}; Wilcoxon p={p:.2f}, r={r:+.2f}',
        f'Success rate: without {100*pv["feasible"]["MO-VPSO-noRestart"].mean():.1f}%, '
        f'with {100*pv["feasible"]["MO-VPSO"].mean():.1f}%; discordant {n1} vs {n0}, p={pb:.2f}',
        f'Mean front size (feasible runs): without {fe["MO-VPSO-noRestart"]:.1f}, with {fe["MO-VPSO"]:.1f}')


# --------------------------------------------------------------- Table 8 ----
def table8():
    d = read('03_mo_baselines.csv')
    hv = d.pivot_table(index='env', columns='optimizer', values='hv').loc[ENVS, EST]
    sr = 100 * d.pivot_table(index='env', columns='optimizer', values='feasible').loc[ENVS, EST]
    f = d[d.feasible == 1]
    npf = f.pivot_table(index='env', columns='optimizer', values='n_front').reindex(ENVS)[EST]
    lm = f.pivot_table(index='env', columns='optimizer', values='best_len').reindex(ENVS)[EST]
    t = pd.concat({'HV': hv.round(2), 'SR (%)': sr.round(0), '|P|': npf.round(1),
                   'L_min': lm.round(1)}, axis=1)

    def ranks_wins(tab, higher):
        key = -tab if higher else tab
        rk = key.rank(axis=1, method='average')
        best = key.eq(key.min(axis=1), axis=0)
        all_tie = best.all(axis=1)
        wins = best[~all_tie].sum()
        return rk.mean(), wins
    extra = {}
    for name, tab, higher in [('HV', hv, True), ('SR (%)', sr.round(0), True),
                              ('|P|', npf, True), ('L_min', lm.round(1), False)]:
        r, w = ranks_wins(tab, higher)
        extra[name] = (r, w)
    t.loc['Mean rank'] = np.concatenate([extra[k][0].values for k in ['HV', 'SR (%)', '|P|', 'L_min']])
    t.loc['Wins'] = np.concatenate([extra[k][1].values for k in ['HV', 'SR (%)', '|P|', 'L_min']])
    save(t, 'Table08_established_optimizers.csv')
    say('', '=== Table 8: comparison with established optimizers ===', t.to_string())

    fr = friedman(hv.values, True)
    say(f'HV Friedman over environments: chi2={fr["chi2"]:.2f}, p={sci(fr["p"])}, W={fr["W"]:.2f}; '
        'mean ranks ' + ', '.join(f'{a} {r:.1f}' for a, r in zip(EST, fr['mean_ranks'])))
    blk = d.pivot_table(index=['env', 'seed'], columns='optimizer', values='hv')[EST]
    fr2 = friedman(blk.values, True)
    say(f'HV Friedman over runs (N={fr2["n_blocks"]}): chi2={fr2["chi2"]:.2f}, '
        f'p={sci(fr2["p"])}, W={fr2["W"]:.2f}')
    ps, pw = [], []
    for a in EST[1:]:
        _, p = van_elteren([d[(d.env == e) & (d.optimizer == 'MO-VPSO')].hv for e in ENVS],
                           [d[(d.env == e) & (d.optimizer == a)].hv for e in ENVS])
        ps.append(p)
        pw.append(wilcoxon_paired(blk['MO-VPSO'], blk[a])[0])
    for a, pa, wa in zip(EST[1:], holm(ps), holm(pw)):
        say(f'  HV MO-VPSO vs {a}: van Elteren p_Holm={sci(pa)}; paired Wilcoxon p_Holm={sci(wa)}')
    pooled = 100 * d.groupby('optimizer').feasible.mean()[EST]
    say('Pooled success rate: ' + ', '.join(f'{a} {v:.1f}%' for a, v in pooled.items()))

    ig = f.groupby('optimizer').igd_plus.mean()[EST]
    ige = f.pivot_table(index='env', columns='optimizer', values='igd_plus').loc[ENVS, EST]
    save(ige.round(3), 'Table08b_igd_plus_per_environment.csv')
    wins = (ige.idxmin(axis=1) == 'MO-VPSO').sum()
    fr3 = friedman(ige.values, higher_is_better=False)
    say('', 'IGD+ (runs with a feasible path): ' + ', '.join(f'{a} {v:.3f}' for a, v in ig.items()),
        f'MO-VPSO lowest per-environment mean in {wins} of 10 environments; '
        f'Friedman over environments p={fr3["p"]:.3f}', ige.round(3).to_string())
    ps = []
    for a in EST[1:]:
        _, p = van_elteren([f[(f.env == e) & (f.optimizer == 'MO-VPSO')].igd_plus for e in ENVS],
                           [f[(f.env == e) & (f.optimizer == a)].igd_plus for e in ENVS])
        ps.append(p)
    for a, pa in zip(EST[1:], holm(ps)):
        say(f'  IGD+ MO-VPSO vs {a}: van Elteren p_Holm={sci(pa)}')


# --------------------------------------------------------------- Table 9 ----
def table9():
    d = read('04_so_schedules.csv')
    sr = 100 * d.pivot_table(index='env', columns='optimizer', values='feasible').loc[ENVS, SO]
    save(sr.round(0), 'Table09_so_success_rate.csv')
    say('', '=== Table 9: single-objective success rate (%) ===', sr.round(0).to_string())
    pooled = 100 * d.groupby('optimizer').feasible.mean()[SO]
    say('Pooled: ' + ', '.join(f'{a} {v:.1f}%' for a, v in pooled.items()))
    fb = d.pivot_table(index=['env', 'seed'], columns='optimizer', values='feasible')[SO]
    ps, info = [], []
    for a in SO[:-1]:
        n1, n0, p = binomial_paired(fb['VPSO'], fb[a]); ps.append(p); info.append((a, n1, n0))
    for (a, n1, n0), pa in zip(info, holm(ps)):
        say(f'  VPSO vs {a}: discordant {n1} vs {n0}, p_Holm={sci(pa)}')


# -------------------------------------------------- Section 3.4 MO vs SO ----
def mo_vs_so():
    for fname, label in [('05_mo_vs_so_exploratory.csv', 'exploratory (seeds 1-20)'),
                         ('06_mo_vs_so_confirmatory.csv', 'confirmatory (seeds 41-90)')]:
        d = read(fname)
        p = d.pivot_table(index=['env', 'seed'], columns='method', values=['hv', 'feasible'])
        both = p[(p['feasible']['MO-VPSO'] == 1) & (p['feasible']['SO-VPSO'] == 1)]
        say('', f'=== Section 3.4: MO-VPSO vs SO-VPSO, {label} ===',
            f'Paired feasible runs: {len(both)}; mean HV MO-VPSO {both["hv"]["MO-VPSO"].mean():.2f}, '
            f'SO-VPSO {both["hv"]["SO-VPSO"].mean():.2f}')
        tests = []
        for e in ['H7', 'H8', 'O2', 'O4']:
            g = both.loc[e] if e in both.index.get_level_values(0) else None
            if g is None or len(g) < 2:
                continue
            pr, r = wilcoxon_paired(g['hv']['MO-VPSO'], g['hv']['SO-VPSO'])
            tests.append((e, len(g), pr, r))
        pr, r = wilcoxon_paired(both['hv']['MO-VPSO'], both['hv']['SO-VPSO'])
        tests.append(('pooled', len(both), pr, r))
        for (e, n, pr, r), pa in zip(tests, holm([t[2] for t in tests])):
            say(f'  {e}: n={n}, Wilcoxon p_Holm={sci(pa)}, r={r:+.2f}')
        fe = 100 * d.groupby('method').feasible.mean()
        say(f'  Feasibility: MO-VPSO {fe["MO-VPSO"]:.1f}%, SO-VPSO {fe["SO-VPSO"]:.1f}%')
        fb = d.pivot_table(index=['env', 'seed'], columns='method', values='feasible')
        ps = []
        envs = ['H7', 'H8', 'O2', 'O4']
        for e in envs:
            ps.append(binomial_paired(fb.loc[e]['MO-VPSO'], fb.loc[e]['SO-VPSO'])[2])
        ps.append(binomial_paired(fb['MO-VPSO'], fb['SO-VPSO'])[2])
        say('  Feasibility exact binomial p_Holm: ' + ', '.join(
            f'{e} {sci(pa)}' for e, pa in zip(envs + ['pooled'], holm(ps))))


# ------------------------------------------ Section 3.5 transfer exponent ---
def exponent():
    e = read('07_transfer_exponent_exploratory.csv')
    fe = 100 * e.groupby('variant').feasible.mean()
    say('', '=== Section 3.5: transfer exponent, exploratory (H6, H8, H9; seeds 1-30) ===',
        ', '.join(f'{k} {v:.1f}%' for k, v in fe.items()),
        f'Back-loaded minus front-loaded: {fe["VPSO-e1.75"] - fe["VPSO-e1/1.75"]:+.1f} points')
    c = read('08_transfer_exponent_confirmatory.csv')
    q = c.pivot_table(index=['env', 'seed'], columns='variant', values='feasible')
    fe = 100 * q.mean()
    n1, n0, p = binomial_paired(q['VPSO-e1.75'], q['VPSO-e1/1.75'])
    m1, m0, pa = binomial_paired(q['VPSO-e1.75'], q['APSO'])
    say('', f'=== Section 3.5: transfer exponent, confirmatory ({len(q)} paired blocks) ===',
        f'gamma=1.75 {fe["VPSO-e1.75"]:.1f}%, gamma=1/1.75 {fe["VPSO-e1/1.75"]:.1f}%, APSO {fe["APSO"]:.1f}%',
        f'Difference {fe["VPSO-e1.75"] - fe["VPSO-e1/1.75"]:+.1f} points; discordant {n1} vs {n0}, '
        f'exact binomial p={sci(p)}',
        f'Against APSO {fe["VPSO-e1.75"] - fe["APSO"]:+.1f} points; discordant {m1} vs {m0}, p={pa:.2f}',
        (100 * c.pivot_table(index='env', columns='variant', values='feasible')).round(1).to_string(),
        'Seeds per environment: ' + ', '.join(
            f'{k} {g.seed.min()}-{g.seed.max()}' for k, g in c.groupby('env')))


# -------------------------------------------------------------- Table 10 ----
def table10():
    d = read('09_clamp_sweep.csv')
    rows = []
    for k, g in d.groupby('kappa'):
        pv = g.pivot_table(index=['env', 'seed'], columns='optimizer', values='best_cost')[SO]
        fr = friedman(pv.values, higher_is_better=False)
        rows.append(dict(kappa=k, saturation_all=g.saturation_all.mean(),
                         saturation_1_25=g.saturation_1_25.mean(),
                         friedman_p=fr['p'], kendall_W=fr['W'], n_blocks=fr['n_blocks']))
    t = pd.DataFrame(rows).set_index('kappa')
    save(t, 'Table10_clamp_sweep.csv')
    say('', '=== Table 10: velocity clamp sweep ===', t.to_string(
        formatters={'saturation_all': '{:.3f}'.format, 'saturation_1_25': '{:.3f}'.format,
                    'friedman_p': sci, 'kendall_W': '{:.3f}'.format}),
        f'Kendall W ratio (k=1.0 / k=0.05): {t.kendall_W.iloc[-1] / t.kendall_W.iloc[0]:.1f}')


# ----------------------------------------------- Section 3.7 sensitivity ----
def sensitivity():
    for fname, label in [('10_design_sensitivity_earlier.csv', 'earlier formulation'),
                         ('11_design_sensitivity_final.csv', 'final schedule')]:
        d = read(fname)
        t = (100 * d.groupby('config', sort=False).feasible.mean()).round(0)
        save(t.to_frame('feasibility_pct'), f'Section3.7_{fname[3:-4]}.csv')
        say('', f'=== Section 3.7: design sensitivity, {label} (O1-O5, seeds 1-10) ===',
            ', '.join(f'{k} {v:.0f}%' for k, v in t.items()))


# ----------------------------------------------------------- Table 11-12 ----
def table12():
    d = read('12_benchmarks.csv')
    funcs = ['Griewank', 'Bohachevsky', 'Bukin', 'Ackley', 'Rastrigin', 'Levy']
    mean = d.pivot_table(index='function', columns='optimizer', values='final_value',
                         aggfunc='mean').loc[funcs, SO]
    sd = d.pivot_table(index='function', columns='optimizer', values='final_value',
                       aggfunc=lambda x: np.std(x, ddof=1)).loc[funcs, SO]
    out = mean.copy().astype(object)
    for f in funcs:
        for a in SO:
            out.loc[f, a] = f'{mean.loc[f, a]:.3g} ({sd.loc[f, a]:.3g})'
    save(out, 'Table12_benchmarks.csv')
    say('', '=== Table 12: benchmark functions, mean (SD) over 30 runs ===', out.to_string())
    for f in funcs:
        g = d[d.function == f]
        ps = [mann_whitney(g[g.optimizer == 'VPSO'].final_value,
                           g[g.optimizer == a].final_value) for a in SO[:-1]]
        say(f'  {f}: VPSO vs ' + ', '.join(
            f'{a} p_Holm={sci(pa)}' for a, pa in zip(SO[:-1], holm(ps))))


if __name__ == '__main__':
    for fn in [table2, tables3to7, table8, table9, mo_vs_so, exponent, table10,
               sensitivity, table12]:
        try:
            fn()
        except FileNotFoundError as err:
            say('', f'[skipped {fn.__name__}: {err.filename} not found]')
    with open(os.path.join(ROOT, 'results', 'statistics_report.txt'), 'w') as fh:
        fh.write('\n'.join(REPORT) + '\n')
