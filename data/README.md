# Data dictionary

All files are plain CSV (comma-separated, UTF-8, header row, `.` as decimal
separator, empty field = not applicable). Every row is one independent run
identified by its environment (or benchmark function), optimizer and random
seed; the seed initializes `numpy.random.default_rng(seed)`, and every run is
reproducible from it with the script named below.

Environments: `H6`-`H10` are the five discriminating environments and
`O1`-`O5` the five conventional environments of Section 2.3 (geometry in
`code/core/hard_envs.py` and `code/core/vpso_core.py`). Workspace 100 x 100
units, robot radius 1.5. Path lengths are in workspace units; turning angles
in radians.

## 01_zdt_verification.csv — Table 2 (script `run_zdt_verification.py`)
| Column | Meaning |
|---|---|
| problem | ZDT1, ZDT2 or ZDT3 (30 decision variables) |
| optimizer | NSGA-II, MOPSO or SMPSO |
| seed | 1-10 |
| n_front | number of points on the returned non-dominated front |
| igd | inverted generational distance to the analytical Pareto front (lower is better) |

## 02_mo_schedules.csv — Tables 3-7, Figures 4-12, restart ablation (script `run_mo_schedules.py`)
| Column | Meaning |
|---|---|
| env | environment |
| optimizer | MO-PSO, MO-IPSO, MO-APSO, MO-RandWPSO, MO-VPSO, or MO-VPSO-noRestart (ablation, Section 3.5) |
| seed | 1-30 |
| feasible | 1 if the final archive contains at least one collision-free path, else 0 |
| n_front | number of non-dominated collision-free paths in the final archive |
| hv | hypervolume of the feasible archive (0 if none) against (ref_f1, ref_f2) |
| best_len | length of the shortest collision-free path on the front (empty if none) |
| best_smooth | smallest mean turning angle on the front, rad (empty if none) |
| ref_f1, ref_f2 | hypervolume reference point of the environment: 1.1 x the maximum of each objective over all feasible solutions of the five compared schedules in all 30 runs |

## 03_mo_baselines.csv — Table 8, Figure 13 (script `run_mo_baselines.py`)
| Column | Meaning |
|---|---|
| env, seed | environment, seed 1-30 |
| optimizer | MO-VPSO, NSGA-II, MOPSO or SMPSO |
| feasible, n_front, hv, best_len | as in file 02; hv uses a reference point shared by these four optimizers |
| igd_plus | IGD+ against the non-dominated union of the feasible solutions of the four optimizers in that environment, objectives normalized between its ideal point and the hypervolume reference point (empty if no feasible path) |
| ref_f1, ref_f2 | hypervolume reference point of the environment |

## 04_so_schedules.csv — Table 9 (script `run_so_schedules.py`)
| Column | Meaning |
|---|---|
| env, seed | environment, seed 1-30 |
| optimizer | PSO, IPSO, APSO, RandWPSO, VPSO (single-objective weighted-sum formulation) |
| best_cost | final weighted-sum cost (length + 500 x penetration depth + 50 x samples outside the workspace) |
| path_length | length of the returned path |
| smoothness | mean turning angle of the returned path, rad |
| feasible | 1 if the returned path is collision-free and inside the workspace |

## 05_mo_vs_so_exploratory.csv, 06_mo_vs_so_confirmatory.csv — Section 3.4 (script `run_mo_vs_so.py`)
Seeds 1-20 (exploratory) and 41-90 (pre-registered confirmatory study; plan in
`preregistration/`). The pre-registration refers to the environments as
orig2, orig4, hard7 and hard8; these are O2, O4, H7 and H8.

| Column | Meaning |
|---|---|
| env, seed | environment, seed |
| method | MO-VPSO (Pareto archive) or SO-VPSO (weighted sum) |
| length | MO-VPSO: shortest path on the front; SO-VPSO: length of the returned path |
| smoothness | mean turning angle of that path, rad |
| feasible | 1 if a collision-free path was returned |
| n_front | number of collision-free paths returned (1 or 0 for SO-VPSO) |
| hv | hypervolume against a reference point pooled over both methods within the environment |

## 07_transfer_exponent_exploratory.csv, 08_transfer_exponent_confirmatory.csv — Section 3.5 (script `run_transfer_exponent.py`)
| Column | Meaning |
|---|---|
| env, seed | environment; seeds 1-30 (exploratory), 101-180 for H6/H7 and 101-160 for H8/H9 (confirmatory) |
| variant | VPSO-e1.75 (gamma = 1.75, proposed), VPSO-e1/1.75 (front-loaded), VPSO-e1.0 (linear), VPSO-e1.0-wide (linear, coefficients in [0.5, 2.5]), APSO |
| feasible | 1 if the returned path is collision-free |
| length | length of the returned path |

## 09_clamp_sweep.csv — Table 10, Figure 14 (script `run_clamp_sweep.py`)
| Column | Meaning |
|---|---|
| kappa | velocity clamp factor (Eq. 2): 0.05, 0.15, 0.5 or 1.0 |
| env, optimizer, seed | H6-H9; the five schedules of Table 1; seeds 1-30 |
| best_cost, path_length, feasible | as in file 04 |
| saturation_all | fraction of velocity components clipped by the clamp, averaged over iterations 1-100 |
| saturation_1_25 | the same, averaged over iterations 1-25 |

## 10_design_sensitivity_earlier.csv, 11_design_sensitivity_final.csv — Section 3.7 (script `run_design_sensitivity.py`)
| Column | Meaning |
|---|---|
| config | configuration label (cpN = N control points; countW / depthW = binary per-sample or penetration-depth penalty with weight W) |
| control_points | number of spline control points |
| penalty_form | count (binary per-sample) or depth (penetration depth) |
| penalty_weight | collision penalty weight |
| env, seed | O1-O5, seeds 1-10 |
| best_cost, path_length, collisions, feasible, smoothness | final cost, path length, number of colliding samples, feasibility flag and mean turning angle (rad) of the returned path |

File 10 uses the earlier formulation of the optimizer (population 300, clamp
factor 0.05); file 11 repeats the sweeps with the final schedule
(population 50, clamp factor 0.5).

## 12_benchmarks.csv — Tables 11-12, Figure 15 (script `run_benchmarks.py`)
| Column | Meaning |
|---|---|
| function | Griewank, Bohachevsky, Bukin (N.6), Ackley, Rastrigin, Levy |
| optimizer | PSO, IPSO, APSO, RandWPSO, VPSO |
| seed | 1-30 |
| final_value | best objective value after 100 iterations (global minimum 0) |

## fronts/ — Pareto fronts and convergence histories (NumPy `.npz`)
| File | Contents |
|---|---|
| `mo_schedules_<ENV>.npz` | key `<optimizer>\|s<seed>\|F`: objectives (n x 2: length, turning angle) of the final collision-free archive; key `...\|X`: the corresponding decision vectors (n x 8, normalized control points). Used for Figures 4-5 and 9-12. |
| `mo_baselines_<ENV>.npz` | the same for MO-VPSO, NSGA-II, MOPSO and SMPSO (Section 3.3) |
| `benchmark_histories.npz` | key `<optimizer>\|<function>`: best-so-far objective value, 30 runs x 100 iterations (Figure 15) |

Load with `numpy.load(path)`; a path is reconstructed from a decision vector
with `FastEnvironment(cfg, 4, 100).path_from_cp(x)` (`code/core/vpso_core.py`).
