#!/bin/bash
# Phase 4.2b hero run — CBC eccentric dephasing, frequency domain
# STATISTICAL-POWER SPEC: 8 temperatures, {x,log,sqrt,exp,sin,cos} x
# safe-4 ops + the A_pow = c0*x**c1 template atom (free '**' removed
# after the 2026-07-15 burn-in forensics), Stage A 800 / Stage B-C 1600
# rounds x 6 steps, 8 MPI processes in the burn-in (7 workers + master).
# (6 runs: Stage A x2 seeds, Stage B x2 seeds, Stage C e0-scan x2;
#  expect ~9-10 h total — run overnight.)
# Diagnostics afterwards:  python exp_phase42_cbc_plots.py  (fig12-18)
# Launch from WSL, any shell:   bash run_phase42_cbc.sh
# Progress appears live AND is saved to results/phase42_cbc_run.log:
#   - a "### RUN k/6 ..." banner per run, with elapsed time and ETA
#   - burn-in progress bars (stage-2 combos)
#   - a status line every 25 rounds: round n/N, wall time, current best
#     cold-rung log-posterior and its expression
#   - a "[checkpoint] k/6 runs saved" line after every finished run
#     (partial results always in results/phase42_cbc_partial.json)
# Final output: results/phase42_cbc.json + summary incl. the recovered
# eccentric PN exponent per run (truth -34/9 = -3.7778) and the
# e0^2-scaling check of the correction coefficient.
ENV=/home/el_bardo/anaconda3/envs/mcmc_koopman
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
"$ENV/bin/mpiexec" -n 1 "$ENV/bin/python" -u exp_phase42_cbc.py 2>&1 \
  | tee results/phase42_cbc_run.log
echo "PYEXIT=${PIPESTATUS[0]}"
