"""
stats_tools.py
==============
Non-parametric tests used in the manuscript (Derrac et al., 2011 protocol).

  friedman          Friedman test with Kendall's W and mean ranks
  wilcoxon_paired   Wilcoxon signed-rank test with rank-biserial correlation
  van_elteren       van Elteren stratified rank-sum test (normal approximation)
  binomial_paired   exact binomial (McNemar) test on discordant binary pairs
  mann_whitney      two-sided Mann-Whitney U test
  holm              Holm-Bonferroni adjustment
"""
import numpy as np
from scipy import stats


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min((m - rank) * p[idx], 1.0))
        adj[idx] = running
    return adj


def friedman(table, higher_is_better=True):
    """table: (blocks x treatments) array or DataFrame.
    Returns dict(chi2, p, W, mean_ranks) where rank 1 is the best treatment."""
    A = np.asarray(table, float)
    n, k = A.shape
    chi2, p = stats.friedmanchisquare(*[A[:, j] for j in range(k)])
    key = -A if higher_is_better else A
    ranks = np.apply_along_axis(stats.rankdata, 1, key)
    return {'chi2': float(chi2), 'p': float(p),
            'W': float(chi2 / (n * (k - 1))),
            'mean_ranks': ranks.mean(axis=0), 'n_blocks': n}


def rank_biserial(x, y):
    """Matched-pairs rank-biserial correlation of d = x - y.
    Positive when x tends to exceed y."""
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def wilcoxon_paired(x, y):
    """Two-sided Wilcoxon signed-rank test of x vs y (zero differences
    discarded, scipy default) and the rank-biserial correlation r
    (positive favours x when larger is better)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if np.all(x == y):
        return 1.0, 0.0
    p = stats.wilcoxon(x, y).pvalue
    return float(p), rank_biserial(x, y)


def van_elteren(x_by_stratum, y_by_stratum, tie_correction=False):
    """van Elteren (1960) stratified Wilcoxon rank-sum test.

    x_by_stratum, y_by_stratum: lists of 1-D arrays, one pair per stratum.
    Within each stratum the observations receive mid-ranks and the rank sum of
    x is weighted by 1/(N_s + 1). The statistic is referred to the normal
    distribution with the variance of van Elteren (1960), without tie
    correction unless tie_correction=True. Returns (z, two-sided p); z > 0
    when x tends to be larger than y."""
    T = ET = VT = 0.0
    for x, y in zip(x_by_stratum, y_by_stratum):
        x = np.asarray(x, float); y = np.asarray(y, float)
        n, m = len(x), len(y)
        if n == 0 or m == 0:
            continue
        N = n + m
        allv = np.concatenate([x, y])
        r = stats.rankdata(allv)
        w = 1.0 / (N + 1.0)
        T += w * r[:n].sum()
        ET += w * n * (N + 1) / 2.0
        tie = 0.0
        if tie_correction and N > 1:
            _, counts = np.unique(allv, return_counts=True)
            tie = (counts ** 3 - counts).sum() / (N * (N - 1))
        VT += w * w * n * m / 12.0 * ((N + 1) - tie)
    if VT <= 0:
        return 0.0, 1.0
    z = (T - ET) / np.sqrt(VT)
    return float(z), float(2 * stats.norm.sf(abs(z)))


def binomial_paired(a, b):
    """Exact two-sided binomial (McNemar) test on discordant pairs of two
    paired binary vectors. Returns (n_a_only, n_b_only, p)."""
    a = np.asarray(a).astype(int); b = np.asarray(b).astype(int)
    n10 = int(((a == 1) & (b == 0)).sum())
    n01 = int(((a == 0) & (b == 1)).sum())
    p = stats.binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
    return n10, n01, float(p)


def mann_whitney(x, y):
    """Two-sided Mann-Whitney U test."""
    return float(stats.mannwhitneyu(x, y, alternative='two-sided').pvalue)
