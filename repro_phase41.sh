#!/bin/bash
# Phase 4.1 REPRODUCIBILITY AUDIT — on-demand, NOT part of the default
# test suite. Reruns the depth-2 (recovery v2) and depth-3 suites
# verbatim into *_repro.json (nothing archived is touched), renders
# separate figures into results/figs_repro/, and writes the
# field-by-field comparison report:
#     results/repro/phase41_repro_report.md
# Expect "REPRODUCED (all runs identical)"; anything else is a
# regression signal. ~1-1.5 h total. Do NOT run while another
# experiment owns the CPU.
# Launch from WSL:   bash repro_phase41.sh
ENV=/home/el_bardo/anaconda3/envs/mcmc_koopman
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
mkdir -p results/repro
{
  echo "=== depth-2 (recovery v2) repro ==="
  "$ENV/bin/mpiexec" -n 1 "$ENV/bin/python" -u repro_phase41_depth2.py
  echo "=== depth-3 repro ==="
  "$ENV/bin/mpiexec" -n 1 "$ENV/bin/python" -u repro_phase41_depth3.py
  echo "=== figures + comparison report ==="
  "$ENV/bin/python" -u repro_phase41_plots.py
  echo "REPRO_EXIT=$?"
} 2>&1 | tee results/repro/phase41_repro_run.log
