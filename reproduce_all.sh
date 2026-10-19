#!/usr/bin/env bash
# Reruns every experiment, then regenerates all tables, statistics and figures.
# Each experiment overwrites its data file in data/.
set -euo pipefail
cd "$(dirname "$0")/code/experiments"

python run_zdt_verification.py
python run_mo_schedules.py
python run_mo_baselines.py
python run_so_schedules.py
python run_mo_vs_so.py exploratory
python run_mo_vs_so.py confirmatory
python run_transfer_exponent.py exploratory
python run_transfer_exponent.py confirmatory
python run_clamp_sweep.py
python run_design_sensitivity.py earlier
python run_design_sensitivity.py final
python run_benchmarks.py

cd ../analysis
python make_tables.py
python make_figures.py
