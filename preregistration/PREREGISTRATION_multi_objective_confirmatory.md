# Pre-registration — Confirmatory study, MO-VPSO vs SO-VPSO

**Status: written and frozen BEFORE the confirmatory runs were executed.**

## 1. Motivation

An exploratory study (seeds 1-20, 4 environments: orig2, orig4, hard7, hard8)
found MO-VPSO (Pareto-archive leader selection) had higher hypervolume than
SO-VPSO (weighted-sum) in all 4 environments, significant in 3 of 4 where
enough paired feasible runs existed (orig2 p=0.008 n=8, orig4 p=0.0001 n=18,
hard7 p=0.0008 n=17; hard8 had only n=3 paired feasible runs, untested).
Feasibility favoured MO-VPSO in 2 environments (orig4, hard8) and SO-VPSO in
1 (orig2), with hard7 tied. This confirmatory study tests these findings on
an independent sample.

## 2. Independence from the exploratory sample

Exploratory seeds: 1-20. Confirmatory seeds: **41-90 (50 fresh seeds)**.

## 3. Fixed design

- Environments: orig2, orig4, hard7, hard8 (the exploratory set; unchanged).
- Methods: SO-VPSO (revised weighted-sum VPSO) and MO-VPSO (Pareto/MOPSO
  variant of the same particle update rule), as implemented in
  `algorithms.py` / `mo_vpso.py`, unmodified since the exploratory run.
- Budget: population 50, 100 iterations, 4 control points, per method.
- Reference point for hypervolume: computed pooled across both methods
  within each environment (max observed length x1.1, max observed
  smoothness x1.1), consistent with the exploratory protocol.
- Total: 2 methods x 4 environments x 50 seeds = 400 runs.

## 4. Hypotheses

- **H1 (primary).** MO-VPSO attains higher hypervolume than SO-VPSO,
  pooled and per-environment, on paired feasible runs.
- **H2 (secondary).** MO-VPSO attains higher feasibility (success rate)
  than SO-VPSO.

## 5. Analysis plan

**Primary (H1).** Paired Wilcoxon signed-rank test on hypervolume, run-by-run
(same seed), restricted to runs where BOTH methods produced a feasible
solution (as in the exploratory analysis). Reported per environment and
pooled across environments (blocks = environment x seed). Two-sided,
alpha = 0.05, Holm-Bonferroni correction across the 4 per-environment tests
plus the pooled test (5 comparisons total). Rank-biserial correlation as
effect size.

**Secondary (H2).** Exact binomial test (McNemar) on discordant feasibility
pairs, per environment and pooled, Holm-corrected alongside H1's tests only
within their own family (i.e. H1's 5 tests get their own Holm correction,
H2's 5 tests get their own).

## 6. Stopping rule

Fixed at 50 seeds per environment. No interim looks, no extension based on
results.

## 7. Pre-specified interpretation

- If H1 confirmed pooled and in >=3/4 environments: the hypervolume
  advantage is robust and the manuscript may report it as a validated
  finding of the multi-objective reformulation.
- If H1 confirmed pooled only, or in <3/4 environments: the effect is
  environment-dependent and must be reported with that qualification, not
  as a blanket claim.
- If H1 is not confirmed pooled: the exploratory hypervolume advantage did
  not replicate and must be withdrawn as a general claim; per-environment
  results (if any survive) may still be reported as exploratory only.
- H2 is not expected to show a clean pattern (exploratory results were mixed
  by environment) and will be reported descriptively regardless of outcome.
