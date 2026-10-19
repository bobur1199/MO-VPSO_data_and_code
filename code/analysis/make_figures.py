"""
make_figures.py
===============
Regenerates Figures 1-15 of the manuscript from the data in data/ and writes
each figure at 600 dpi as PNG and TIFF (LZW), plus the plotted values of every
data figure as a source-data CSV.

  figures/Fig01.png, Fig01.tif ... Fig15.png, Fig15.tif
  figures/source_data/Fig04_source_data.csv ...

Usage: python make_figures.py [01,02,...]     (default: all figures)
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.ticker
matplotlib.use('Agg')
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle  # noqa: E402
from matplotlib.lines import Line2D                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'code', 'core'))
from vpso_core import FastEnvironment, ENVIRONMENTS   # noqa: E402
from hard_envs import HARD_ENVIRONMENTS               # noqa: E402
from metrics import nondominated                      # noqa: E402

DATA = os.path.join(ROOT, 'data')
FRONTS = os.path.join(DATA, 'fronts')
FIG = os.path.join(ROOT, 'figures')
SRC = os.path.join(FIG, 'source_data')
os.makedirs(SRC, exist_ok=True)
DPI = 600

ALL_ENVS = {**{f'H{k}': v for k, v in HARD_ENVIRONMENTS.items()},
            **{f'O{k}': v for k, v in ENVIRONMENTS.items()}}
ENVS = ['H6', 'H7', 'H8', 'H9', 'H10', 'O1', 'O2', 'O3', 'O4', 'O5']
MO = ['MO-PSO', 'MO-IPSO', 'MO-APSO', 'MO-RandWPSO', 'MO-VPSO']
SO = ['PSO', 'IPSO', 'APSO', 'RandWPSO', 'VPSO']
EST = ['MO-VPSO', 'NSGA-II', 'MOPSO', 'SMPSO']

COLOR = ['#1f77b4', '#E69F00', '#2ca02c', '#7f7f7f', '#d62728']
STYLE = [(0, (1, 1.5)), (0, (5, 2.5)), (0, (6, 2, 1, 2)),
         (0, (6, 2, 1, 2, 1, 2)), '-']
WIDTH = [1.4, 1.4, 1.4, 1.4, 2.0]
EST_STYLE = {'MO-VPSO': ('#D55E00', 'o'), 'NSGA-II': ('#0072B2', '^'),
             'MOPSO': ('#009E73', 's'), 'SMPSO': ('#E69F00', 'D')}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Liberation Sans', 'Arial', 'Helvetica', 'DejaVu Sans'],
    'mathtext.fontset': 'custom', 'mathtext.rm': 'Liberation Sans',
    'mathtext.it': 'Liberation Sans:italic', 'mathtext.bf': 'Liberation Sans:bold',
    'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9,
    'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'legend.fontsize': 8,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': '#dddddd', 'grid.linestyle': ':',
    'grid.linewidth': 0.6, 'legend.frameon': False,
    'savefig.dpi': DPI, 'figure.dpi': 100,
})


def save(fig, n, source=None):
    base = os.path.join(FIG, f'Fig{n:02d}')
    fig.savefig(base + '.png', dpi=DPI, bbox_inches='tight', pad_inches=0.04)
    fig.savefig(base + '.tif', dpi=DPI, bbox_inches='tight', pad_inches=0.04,
                pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)
    if source is not None:
        source.to_csv(os.path.join(SRC, f'Fig{n:02d}_source_data.csv'),
                      index=False)
    print(f'Fig{n:02d} written')


def read(name):
    return pd.read_csv(os.path.join(DATA, name))


# ------------------------------------------------------------- Figure 1 -----
def fig01():
    steps = [
        ('Initialize swarm: positions $X$, velocities $V = 0$', '#e7eefb'),
        ('Decode control points \u2192 clamped cubic spline (Eq. 4)', '#e7eefb'),
        ('Evaluate $f_1$ (Eq. 5), $f_2$ (Eq. 6)\nand constraint violation $V_c$ (Eq. 7)', '#fdf2e6'),
        ('Initialize external Pareto archive\n(constrained non-dominated solutions)', '#e8f2e8'),
        ('Update $w$ (Eq. 9) and $C_1, C_2$ (Eq. 10)', '#e7eefb'),
        ('Select leader per particle:\nbinary tournament on crowding distance', '#e8f2e8'),
        ('Update velocity (Eq. 12) and position (Eq. 1);\nclamp velocity (Eq. 2)', '#e7eefb'),
        ('Every $T_r$ iterations: restart worst 20% (Eq. 13)', '#fbe6e8'),
        ('Evaluate; update personal bests\nby constrained dominance', '#fdf2e6'),
        ('Update archive; prune by crowding distance', '#e8f2e8'),
        ('Return Pareto set of collision-free paths', '#e8f2e8'),
    ]
    fig, ax = plt.subplots(figsize=(3.3, 4.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 11.5); ax.axis('off')
    x0, w, h, gap = 0.1, 8.5, 0.78, 0.22
    tops = []
    for i, (txt, col) in enumerate(steps):
        y = 11.2 - i * (h + gap) - h
        tops.append(y)
        ax.add_patch(FancyBboxPatch((x0, y), w, h, boxstyle='round,pad=0,rounding_size=0.12',
                                    fc=col, ec='#3a3a3a', lw=0.75))
        ax.text(x0 + w / 2, y + h / 2, txt, ha='center', va='center', fontsize=5.9,
                linespacing=1.15)
        if i < len(steps) - 1:
            ax.annotate('', xy=(x0 + w / 2, y - gap + 0.02), xytext=(x0 + w / 2, y),
                        arrowprops=dict(arrowstyle='-|>', color='#3a3a3a', lw=0.9,
                                        mutation_scale=7, shrinkA=0, shrinkB=0))
    red = '#c62828'
    yl, yu = tops[9] + h / 2, tops[4] + h / 2
    xr = x0 + w + 0.6
    ax.plot([x0 + w, xr, xr], [yl, yl, yu], color=red, lw=1.2)
    ax.annotate('', xy=(x0 + w + 0.1, yu), xytext=(xr, yu),
                arrowprops=dict(arrowstyle='-|>', color=red, lw=1.2, mutation_scale=8,
                                shrinkA=0, shrinkB=0))
    ax.text(xr + 0.35, (yl + yu) / 2, 'iterate until $T_{\\max}$', rotation=90,
            color=red, ha='center', va='center', fontsize=6.6)
    save(fig, 1)


# ------------------------------------------------------------- Figure 2 -----
def fig02():
    T = 100
    t = np.arange(0, T + 1)
    w = 0.9 - 0.5 * t / T
    p = (t / T) ** 1.75
    c1, c2 = 2.0 - 1.5 * p, 0.5 + 1.5 * p
    lim1 = 4 * (1 + w)
    lim2 = 24 * (1 - w ** 2) / (7 - 5 * w)
    tc = T * 0.5 ** (1 / 1.75)
    t2 = T * (0.9 - 0.84) / 0.5
    fig, axs = plt.subplots(1, 3, figsize=(6.95, 2.3))
    ax = axs[0]
    ax.plot(t, w, color=COLOR[4], lw=1.6)
    ax.set_title('(a) Inertia weight', loc='left'); ax.set_ylabel('$w(t)$')
    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ax = axs[1]
    ax.plot(t, c1, color=COLOR[0], lw=1.6, label='$C_1$ (cognitive)')
    ax.plot(t, c2, color=COLOR[4], lw=1.6, label='$C_2$ (social)')
    ax.axvline(tc, color='gray', ls='--', lw=0.8)
    ax.text(tc - 2, 1.92, f'crossover\n$t \\approx {tc:.0f}$', ha='right', va='top', fontsize=7.5)
    ax.set_title('(b) Learning factors, $\\gamma = 1.75$', loc='left')
    ax.set_ylabel('Learning factor')
    ax.legend(loc='center left', bbox_to_anchor=(0.0, 0.56), fontsize=7, handlelength=1.5)
    ax = axs[2]
    ax.axvspan(0, t2, color='#f6c6c0', alpha=0.6, lw=0)
    ax.plot(t, c1 + c2, color=COLOR[2], lw=2.0, label='$C_1 + C_2 = 2.5$')
    ax.plot(t, lim1, color='k', ls='--', lw=1.0, label='order-1 limit $4(1+w)$')
    ax.plot(t, lim2, color='#7b1fa2', ls=':', lw=1.6, label='order-2 limit')
    ax.text(t2 + 2, 0.35, f'order-2 limit\nexceeded for $t < {t2:.0f}$', fontsize=7)
    ax.set_ylim(0, 11.5); ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.03), fontsize=7, handlelength=1.8,
              labelspacing=0.3)
    ax.set_title('(c) Coefficient sum and stability limits', loc='left')
    for ax in axs:
        ax.set_xlabel('Iteration $t$'); ax.set_xticks([0, 50, 100])
    fig.tight_layout(w_pad=1.5)
    save(fig, 2, pd.DataFrame({'iteration': t, 'w': w, 'C1': c1, 'C2': c2,
                               'C1_plus_C2': c1 + c2, 'order1_limit': lim1,
                               'order2_limit': lim2}))


# ------------------------------------------------------------- Figure 3 -----
def fig03():
    T = 100
    t = np.arange(0, T + 1)
    fig, ax = plt.subplots(figsize=(4.3, 2.85))
    src = {'iteration': t}
    for g, col, ls, lab in [(1 / 1.75, COLOR[0], (0, (5, 2.5)), '$\\gamma = 1/1.75$ (front-loaded)'),
                            (1.75, COLOR[4], '-', '$\\gamma = 1.75$ (proposed, back-loaded)')]:
        p = (t / T) ** g
        c1, c2 = 2.0 - 1.5 * p, 0.5 + 1.5 * p
        ax.plot(t, c1, color=col, ls=ls, lw=2.0, label=lab)
        ax.plot(t, c2, color=col, ls=ls, lw=2.0, alpha=0.4)
        tc = T * 0.5 ** (1 / g)
        ax.plot(tc, 1.25, 'o', color=col, ms=6)
        ax.text(tc + 3, 1.36, f'crossover $t \\approx {tc:.0f}$', color=col, fontsize=8.5)
        key = 'gamma_1_over_1.75' if g < 1 else 'gamma_1.75'
        src[f'C1_{key}'] = c1; src[f'C2_{key}'] = c2
    ax.set_xlabel('Iteration $t$')
    ax.set_ylabel('Learning factor ($C_1$ solid, $C_2$ faded)')
    ax.legend(loc='lower left')
    save(fig, 3, pd.DataFrame(src))


# ---------------------------------------------------------- Figures 4-5 -----
def pooled_front(z, label):
    pts = [z[f'{label}|s{s}|F'] for s in range(1, 31) if len(z[f'{label}|s{s}|F'])]
    if not pts:
        return np.empty((0, 2))
    P = np.vstack(pts)
    P = np.unique(P, axis=0)
    P = P[nondominated(P)]
    return P[np.argsort(P[:, 0])]


def fig_front(n, ek, legend_loc):
    z = np.load(os.path.join(FRONTS, f'mo_schedules_{ek}.npz'))
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    rows = []
    for a, col, ls, lw in zip(MO, COLOR, STYLE, WIDTH):
        P = pooled_front(z, a)
        ax.plot(P[:, 0], P[:, 1], color=col, ls=ls, lw=lw if a == 'MO-VPSO' else 1.6,
                marker='o', ms=4, label=a)
        rows += [dict(optimizer=a, f1_path_length=f1, f2_mean_turning_angle_rad=f2)
                 for f1, f2 in P]
    ax.set_xlabel('$f_1$: path length')
    ax.set_ylabel('$f_2$: mean turning angle (rad)')
    ax.legend(loc=legend_loc)
    save(fig, n, pd.DataFrame(rows))


def fig04():
    fig_front(4, 'H9', 'upper right')


def fig05():
    fig_front(5, 'O1', 'upper left')


# ------------------------------------------------------------- Figure 6 -----
def fig06():
    d = read('02_mo_schedules.csv')
    hv = d[d.optimizer.isin(MO)].pivot_table(index='env', columns='optimizer',
                                             values='hv').loc[ENVS, MO]
    fig, ax = plt.subplots(figsize=(6.8, 2.45))
    x = np.arange(len(ENVS)); bw = 0.16
    for i, (a, col) in enumerate(zip(MO, COLOR)):
        ax.bar(x + (i - 2) * bw, hv[a], bw, color=col, label=a)
    ax.axvline(4.5, color='k', ls='--', lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(ENVS)
    ax.set_ylabel('Mean hypervolume'); ax.grid(axis='x', visible=False)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2.5))
    ax.legend(ncol=5, loc='lower left', bbox_to_anchor=(0.0, 1.0), handlelength=2.2)
    save(fig, 6, hv.reset_index().melt(id_vars='env', var_name='optimizer',
                                       value_name='mean_hypervolume'))


# ------------------------------------------------------------- Figure 7 -----
def fig07():
    d = read('02_mo_schedules.csv')
    sr = 100 * d[d.optimizer.isin(MO)].pivot_table(index='env', columns='optimizer',
                                                   values='feasible').loc[ENVS, MO]
    ang = 2 * np.pi * np.arange(len(ENVS)) / len(ENVS)
    ang = np.append(ang, ang[0])
    fig = plt.figure(figsize=(4.4, 3.45))
    ax = fig.add_subplot(111, projection='polar')
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    for a, col, ls, lw in zip(MO, COLOR, STYLE, WIDTH):
        v = np.append(sr[a].values, sr[a].values[0])
        ax.plot(ang, v, color=col, ls=ls, lw=1.1 if a != 'MO-VPSO' else 1.8, label=a)
        if a == 'MO-VPSO':
            ax.fill(ang, v, color=col, alpha=0.12, lw=0)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(ENVS, fontsize=9)
    ax.set_ylim(0, 100); ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=7)
    ax.set_rlabel_position(18)
    ax.grid(color='#b0b0b0', ls='-', lw=0.8)
    ax.tick_params(axis='x', pad=6)
    ax.legend(loc='upper left', bbox_to_anchor=(1.16, 1.0))
    save(fig, 7, sr.reset_index().melt(id_vars='env', var_name='optimizer',
                                       value_name='success_rate_pct'))


# ------------------------------------------------------------- Figure 8 -----
def fig08():
    d = read('02_mo_schedules.csv')
    f = d[d.optimizer.isin(MO) & (d.feasible == 1)]
    nf = f.pivot_table(index='env', columns='optimizer', values='n_front').reindex(ENVS)[MO]
    M = nf.T.values                                # optimizers x envs
    fig = plt.figure(figsize=(6.5, 2.25))
    ax_l = fig.add_axes([0.13, 0.12, 0.345, 0.84])
    ax_r = fig.add_axes([0.482, 0.12, 0.345, 0.84])
    cax = fig.add_axes([0.845, 0.12, 0.018, 0.84])
    vmin, vmax = np.nanmin(M), np.nanmax(M)
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('#dddddd')
    best = np.nanargmax(M, axis=0)
    for ax, cols in [(ax_l, range(0, 5)), (ax_r, range(5, 10))]:
        sub = np.ma.masked_invalid(M[:, cols])
        im = ax.imshow(sub, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_xticks(range(5)); ax.set_xticklabels([ENVS[c] for c in cols])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(True)
        for i in range(M.shape[0]):
            for jj, j in enumerate(cols):
                v = M[i, j]
                isbest = (i == best[j])
                if np.isnan(v):
                    ax.text(jj, i, '—', ha='center', va='center', fontsize=7.5)
                    continue
                rgba = cmap((v - vmin) / (vmax - vmin))
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                ax.text(jj, i, f'{v:.1f}', ha='center', va='center', fontsize=7.5,
                        color='k' if lum > 0.55 else 'w',
                        fontweight='bold' if isbest else 'normal')
                if isbest:
                    ax.add_patch(Rectangle((jj - 0.5, i - 0.5), 1, 1, fill=False,
                                           ec='k', lw=1.4))
        ax.tick_params(length=3)
    ax_l.set_yticks(range(5)); ax_l.set_yticklabels(MO)
    ax_r.set_yticks([])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('Mean number of paths'); cb.outline.set_linewidth(0.6)
    cax.grid(False); cax.minorticks_off()
    save(fig, 8, nf.reset_index().melt(id_vars='env', var_name='optimizer',
                                       value_name='mean_front_size_feasible_runs'))


# --------------------------------------------------------- Figures 9-12 -----
def draw_env(ax, ek):
    cfg = ALL_ENVS[ek]
    for o in cfg['obstacles']:
        if 'radius' in o:
            ax.add_patch(Circle(o['center'], o['radius'], fc='#9e9e9e', ec='#505050',
                                lw=0.7, zorder=2))
        else:
            cx, cy = o['center']
            ax.add_patch(Rectangle((cx - o['width'] / 2, cy - o['height'] / 2),
                                   o['width'], o['height'], fc='#9e9e9e', ec='#505050',
                                   lw=0.7, zorder=2))
    ax.set_xlim(0, cfg['width']); ax.set_ylim(0, cfg['height'])
    ax.set_aspect('equal')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(True)


def best_paths(ek):
    """Shortest feasible path of the highest-hypervolume run, per optimizer."""
    d = read('02_mo_schedules.csv')
    z = np.load(os.path.join(FRONTS, f'mo_schedules_{ek}.npz'))
    env = FastEnvironment(ALL_ENVS[ek], 4, 100)
    out = {}
    for a in MO:
        g = d[(d.env == ek) & (d.optimizer == a) & (d.feasible == 1)]
        if g.empty:
            continue
        s = int(g.sort_values(['hv', 'seed'], ascending=[False, True]).iloc[0].seed)
        F, X = z[f'{a}|s{s}|F'], z[f'{a}|s{s}|X']
        out[a] = (s, env.path_from_cp(X[int(np.argmin(F[:, 0]))]))
    return out


def path_panel(ax, ek, rows):
    cfg = ALL_ENVS[ek]
    draw_env(ax, ek)
    for a, col, ls, lw in zip(MO, COLOR, STYLE, WIDTH):
        bp = best_paths.cache.setdefault(ek, best_paths(ek))
        if a not in bp:
            continue
        s, P = bp[a]
        ax.plot(P[:, 0], P[:, 1], color=col, ls=ls, lw=lw, zorder=3 + (a == 'MO-VPSO'))
        rows += [dict(env=ek, optimizer=a, seed=s, point=i, x=x, y=y)
                 for i, (x, y) in enumerate(P)]
    ax.plot(*cfg['start'], 's', color='#d62728', mec='k', mew=0.8, ms=7, zorder=6)
    ax.plot(*cfg['goal'], '*', color='#1f77b4', mec='k', mew=0.8, ms=12, zorder=6)
    ax.set_title(f'{ek}: {cfg["name"]}', fontsize=7.8)
    ax.set_xlabel('X (units)'); ax.set_ylabel('Y (units)')


best_paths.cache = {}


def path_legend_handles():
    h = [Line2D([], [], color=c, ls=s, lw=w) for c, s, w in zip(COLOR, STYLE, WIDTH)]
    h += [Line2D([], [], marker='s', color='#d62728', mec='k', ls='', ms=7),
          Line2D([], [], marker='*', color='#1f77b4', mec='k', ls='', ms=11)]
    return h, MO + ['Start', 'Goal']


def fig_paths(n, envs):
    fig, axs = plt.subplots(1, 3, figsize=(6.8, 2.6))
    rows = []
    for ax, ek in zip(axs, envs):
        path_panel(ax, ek, rows)
        ax.set_xticks([0, 25, 50, 75, 100])
    fig.tight_layout(w_pad=1.8)
    h, lab = path_legend_handles()
    fig.legend(h, lab, ncol=7, loc='upper center', bbox_to_anchor=(0.5, 0.02),
               fontsize=7.5, handlelength=2.6, columnspacing=1.2)
    save(fig, n, pd.DataFrame(rows))


def fig09():
    fig_paths(9, ['H6', 'H7', 'H8'])


def fig10():
    fig_paths(10, ['H9', 'H10', 'O1'])


def fig11():
    fig_paths(11, ['O2', 'O3', 'O4'])


def fig12():
    fig, ax = plt.subplots(figsize=(2.95, 2.95))
    rows = []
    path_panel(ax, 'O5', rows)
    h, lab = path_legend_handles()
    order = [0, 1, 2, 3, 4, 5, 6]
    fig.legend([h[i] for i in order], [lab[i] for i in order], ncol=2,
               loc='upper center', bbox_to_anchor=(0.55, 0.0), fontsize=8,
               handlelength=2.6, columnspacing=2.0)
    save(fig, 12, pd.DataFrame(rows))


# ------------------------------------------------------------ Figure 13 -----
def fig13():
    d = read('03_mo_baselines.csv')
    hv = d.pivot_table(index='env', columns='optimizer', values='hv').loc[ENVS, EST]
    sr = 100 * d.pivot_table(index='env', columns='optimizer', values='feasible').loc[ENVS, EST]
    f = d[d.feasible == 1]
    npf = f.pivot_table(index='env', columns='optimizer', values='n_front').reindex(ENVS)[EST]
    lm = f.pivot_table(index='env', columns='optimizer', values='best_len').reindex(ENVS)[EST]
    rel = hv.div(hv.max(axis=1), axis=0)
    exc = 100 * (lm.div(lm.min(axis=1), axis=0) - 1)
    panels = [(rel, '(a) Relative hypervolume',
               'Hypervolume relative to the best in each environment'),
              (sr, '(b) Success rate', 'Collision-free success rate (%)'),
              (npf, '(c) Paths per run', 'Mean number of non-dominated collision-free paths'),
              (exc, '(d) Excess path length',
               'Excess shortest-path length over the best (%), lower is better')]
    fig, axs = plt.subplots(2, 2, figsize=(6.4, 7.0))
    axs[0, 0].set_xlim(0.2, 1.03); axs[0, 1].set_xlim(-3, 103); axs[1, 0].set_xlim(0, None)
    off = {'MO-VPSO': -0.33, 'NSGA-II': -0.11, 'MOPSO': 0.11, 'SMPSO': 0.33}
    for ax, (tab, title, xl) in zip(axs.flat, panels):
        for i, e in enumerate(ENVS):
            v = tab.loc[e].dropna()
            ax.plot([v.min(), v.max()], [i, i], color='#d9d9d9', lw=1.3, zorder=1)
            for a in EST:
                if not np.isnan(tab.loc[e, a]):
                    col, mk = EST_STYLE[a]
                    ax.plot(tab.loc[e, a], i + off[a], mk, color=col, ms=5,
                            mec='none', zorder=3)
        ax.axhline(4.5, color='gray', ls='--', lw=0.8)
        ax.set_yticks(range(len(ENVS))); ax.set_yticklabels(ENVS)
        ax.set_ylim(len(ENVS) - 0.5, -0.6)
        ax.set_title(title, loc='left', fontweight='bold')
        ax.set_xlabel(xl, fontsize=8.5)
        ax.grid(axis='y', visible=False); ax.grid(axis='x', ls='-', color='#ebebeb')
        ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
    handles = [Line2D([], [], marker=EST_STYLE[a][1], color=EST_STYLE[a][0], ls='',
                      ms=6, mec='none') for a in EST]
    fig.tight_layout(h_pad=2.0, w_pad=2.0)
    fig.legend(handles, EST, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 0.0),
               fontsize=8.5, columnspacing=3)
    src = pd.concat({'relative_hv': rel, 'success_rate_pct': sr,
                     'mean_front_size': npf, 'excess_length_pct': exc}, axis=1)
    src.columns = [f'{a}_{b}' for a, b in src.columns]
    save(fig, 13, src.reset_index())


# ------------------------------------------------------------ Figure 14 -----
def fig14():
    sys.path.insert(0, os.path.join(ROOT, 'code', 'core'))
    from stats_tools import friedman
    d = read('09_clamp_sweep.csv')
    rows = []
    for k, g in d.groupby('kappa'):
        pv = g.pivot_table(index=['env', 'seed'], columns='optimizer', values='best_cost')[SO]
        rows.append(dict(kappa=k, saturation_all=g.saturation_all.mean(),
                         kendall_W=friedman(pv.values, False)['W']))
    t = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(4.3, 2.8))
    ax.plot(t.kappa, t.saturation_all, '-o', color='#333333', lw=1.8, ms=5,
            label='Clamp saturation')
    ax.set_xscale('log'); ax.set_xticks(t.kappa)
    ax.set_xticklabels(['0.05', '0.15', '0.5', '1'])
    ax.set_xlabel('Velocity clamp factor $\\kappa$')
    ax.set_ylabel('Saturation (fraction of components)')
    ax2 = ax.twinx()
    ax2.plot(t.kappa, t.kendall_W, 's--', color='#d62728', lw=1.8, ms=5.5,
             label="Kendall's $W$")
    ax2.set_ylabel("Kendall's $W$ (separation of schedules)", color='#d62728')
    ax2.tick_params(axis='y', colors='#d62728')
    ax2.spines['right'].set_visible(True); ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper center')
    save(fig, 14, t)


# ------------------------------------------------------------ Figure 15 -----
def fig15():
    z = np.load(os.path.join(FRONTS, 'benchmark_histories.npz'))
    funcs = [('Griewank', 'Griewank'), ('Bohachevsky', 'Bohachevsky'),
             ('Bukin', 'Bukin N.6'), ('Ackley', 'Ackley'),
             ('Rastrigin', 'Rastrigin'), ('Levy', 'Levy')]
    fig, axs = plt.subplots(2, 3, figsize=(6.8, 4.1))
    rows = {'iteration': np.arange(1, 101)}
    for ax, (key, title) in zip(axs.flat, funcs):
        for a, col, ls, lw in zip(SO, COLOR, STYLE, WIDTH):
            m = z[f'{a}|{key}'].mean(axis=0)
            ax.plot(np.arange(1, 101), np.where(m > 0, m, np.nan), color=col, ls=ls,
                    lw=lw if a == 'VPSO' else 1.2, label=a)
            rows[f'{key}_{a}'] = m
        ax.set_yscale('log'); ax.set_title(title)
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xlabel('Iteration')
    for ax in axs[:, 0]:
        ax.set_ylabel('Mean objective value')
    fig.tight_layout(h_pad=1.2)
    h, lab = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, ncol=5, loc='upper center', bbox_to_anchor=(0.5, 0.01),
               handlelength=2.8, columnspacing=2.0)
    save(fig, 15, pd.DataFrame(rows))


FIGS = {1: fig01, 2: fig02, 3: fig03, 4: fig04, 5: fig05, 6: fig06, 7: fig07,
        8: fig08, 9: fig09, 10: fig10, 11: fig11, 12: fig12, 13: fig13,
        14: fig14, 15: fig15}

if __name__ == '__main__':
    sel = [int(x) for x in sys.argv[1].split(',')] if len(sys.argv) > 1 else list(FIGS)
    for n in sel:
        try:
            FIGS[n]()
        except FileNotFoundError as err:
            print(f'Fig{n:02d} skipped: {err.filename} not found')
