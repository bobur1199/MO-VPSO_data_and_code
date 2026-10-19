# MO-VPSO: data and code

Data, source code and analysis scripts for the article

> Abdiev, B., Khamidov, B., Yakhshiev, S., Ravshanov, J., Isaev, D., Akhmedov, X.,
> Toirova, N., Ataullaev, A., Rakhmanova, N., Bozorov, R. & Hamidov, A.
> **MO-VPSO: A Multi-Objective Particle Swarm Optimizer with Back-Loaded
> Learning Factors for Collision-Free Mobile Robot Path Planning.**
> Submitted to *Scientific Reports*.

Every table, figure and test statistic in the article can be regenerated from
this package. All experiments are seeded, so rerunning an experiment script
reproduces its data file exactly; the analysis scripts regenerate the tables,
the statistics and the figures from the data files without rerunning any
experiment.

Corresponding author: Bobur Abdiev, Department of Mechanical Engineering,
Navoi State University of Mining and Technologies, Navoi, Uzbekistan
(b.abdiev@nsumt.uz).

## Contents

```
README.md                  this file
LICENSE                    MIT licence (source code)
LICENSE-DATA               CC BY 4.0 licence (data, figures, documentation)
CITATION.cff               citation metadata
requirements.txt           Python dependencies
reproduce_all.sh           reruns every experiment, then all tables and figures
code/
  core/                    algorithms, environments, metrics and statistical tests
  experiments/             seeded runners; each writes one data file in data/
  analysis/                make_tables.py (Tables 2-12, all statistics)
                           make_figures.py (Figures 1-15, 600 dpi)
data/                      raw per-run results (CSV) and Pareto fronts (NPZ);
                           column definitions in data/README.md
results/                   tables and statistics regenerated from data/
figures/                   Figures 1-15 (PNG and TIFF, 600 dpi) and the
                           source data of every data figure (figures/source_data/)
preregistration/           analysis plan of the confirmatory study of Section 3.4,
                           written before its data were collected
```

## Code (`code/core/`)

| File | Contents |
|---|---|
| `vpso_core.py` | Conventional environments O1-O5; `FastEnvironment` (clamped cubic-spline path through start, control points and goal, Eq. 4, evaluated as one matrix product); the earlier formulation of the optimizer used only in the first-stage sweeps of Section 3.7 |
| `hard_envs.py` | Discriminating environments H6-H10, each built from a stated geometric principle (Section 2.3) |
| `algorithms.py` | Parameter schedules of Table 1 (PSO, IPSO, APSO, RandWPSO, VPSO with gamma = 1.75 and periodic restart), the weighted-sum path cost, the six benchmark functions and the batched single-objective PSO runner |
| `mo_vpso.py` | Objectives f1 (Eq. 5) and f2 (Eq. 6), constraint violation (Eqs. 7-8), constrained dominance, crowding distance and archive update |
| `mo_framework.py` | The shared multi-objective framework of Section 2.2 (Fig. 1): every schedule runs in the same archive, leader-selection and personal-best machinery |
| `nsga2.py` | NSGA-II (Deb et al., 2002) |
| `mopso_smpso.py` | MOPSO (Coello Coello et al., 2004) and SMPSO (Nebro et al., 2009) |
| `metrics.py` | Hypervolume, reference point, IGD and IGD+ |
| `stats_tools.py` | Friedman test with Kendall's W, Wilcoxon signed-rank with rank-biserial r, van Elteren stratified test, exact binomial test on discordant pairs, Mann-Whitney U, Holm-Bonferroni correction |

## Where each result comes from

| Manuscript item | Data file (`data/`) | Experiment script (`code/experiments/`) |
|---|---|---|
| Table 2 (ZDT verification) | `01_zdt_verification.csv` | `run_zdt_verification.py` |
| Tables 3-7, Figures 4-12; restart ablation (Section 3.5) | `02_mo_schedules.csv`, `fronts/mo_schedules_*.npz` | `run_mo_schedules.py` |
| Table 8, Figure 13, IGD+ (Section 3.3) | `03_mo_baselines.csv`, `fronts/mo_baselines_*.npz` | `run_mo_baselines.py` |
| Table 9 | `04_so_schedules.csv` | `run_so_schedules.py` |
| MO versus weighted-sum study (Section 3.4) | `05_mo_vs_so_exploratory.csv`, `06_mo_vs_so_confirmatory.csv` | `run_mo_vs_so.py` |
| Transfer-exponent ablation (Section 3.5) | `07_transfer_exponent_exploratory.csv`, `08_transfer_exponent_confirmatory.csv` | `run_transfer_exponent.py` |
| Table 10, Figure 14 | `09_clamp_sweep.csv` | `run_clamp_sweep.py` |
| Design sensitivity (Section 3.7) | `10_design_sensitivity_earlier.csv`, `11_design_sensitivity_final.csv` | `run_design_sensitivity.py` |
| Tables 11-12, Figure 15 | `12_benchmarks.csv`, `fronts/benchmark_histories.npz` | `run_benchmarks.py` |
| Figures 1-3 (algorithm and schedules) | analytical, no data | `code/analysis/make_figures.py` |

`code/analysis/make_tables.py` writes every table to `results/tables/` and
every statistic quoted in the text (Friedman, Wilcoxon, van Elteren, exact
binomial and Mann-Whitney tests, Holm-corrected p-values and effect sizes) to
`results/statistics_report.txt`.

## Experimental settings

| Setting | Value |
|---|---|
| Workspace, robot radius | 100 x 100 units, 1.5 |
| Path encoding | 4 control points (D = 8), clamped cubic spline sampled at 100 points |
| Population, iterations | 50, 100 (5050 evaluations per run) for path planning; 300, 100 for benchmark functions |
| Velocity clamp factor | 0.5 for all path-planning comparisons; 0.05 for the benchmark functions and the weighted-sum confirmatory study (original protocol); swept in Section 3.6 |
| Archive | 60 members, pruned by crowding distance; leader by binary tournament on crowding distance |
| MO-VPSO schedule | w: 0.9 -> 0.4 linear; C_max = 2.0, C_min = 0.5, gamma = 1.75; worst 20 % restarted every 25 iterations |
| Independent runs | 30 per configuration (seeds 1-30) unless stated otherwise |
| Hypervolume reference point | 1.1 x the per-objective maximum over all feasible solutions of the compared optimizers in that environment |

## How to reproduce

Requirements: Python 3.10 or newer with the packages in `requirements.txt`
(tested with Python 3.13, NumPy 2.5, SciPy 1.18, pandas 3.0, Matplotlib 3.11).

```bash
pip install -r requirements.txt

# tables, statistics and figures from the included data (about one minute)
python code/analysis/make_tables.py
python code/analysis/make_figures.py

# rerun every experiment from scratch (several hours on one CPU core)
bash reproduce_all.sh
```

Each experiment script can also be run on its own, for example
`python code/experiments/run_mo_schedules.py H6,H7` to rerun two environments;
`python code/experiments/run_mo_schedules.py --table` then rebuilds the CSV
from the stored fronts. Scripts may be run from any directory.

Random numbers are drawn from NumPy's PCG64 generator
(`numpy.random.default_rng(seed)`), one independent generator per run, so a run
depends only on its seed and settings and not on the order in which runs are
executed. Runs with the same seed start from the same initial swarm for every
particle swarm schedule.

## Data availability statement (for the manuscript)

> All data generated and analysed during this study, including the per-run
> results of every experiment, the final Pareto fronts and the pre-registered
> analysis plan, are available in the [repository name] repository,
> [DOI or URL].

## Code availability statement (for the manuscript)

> The source code of MO-VPSO, of the baseline implementations and of all
> experiment and analysis scripts is available in the [repository name]
> repository, [DOI or URL], under the MIT licence.

## Licence

Source code: MIT (`LICENSE`). Data, figures and documentation: Creative
Commons Attribution 4.0 International (`LICENSE-DATA`).
