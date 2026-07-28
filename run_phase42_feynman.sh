#!/bin/bash
# Phase 4.2a hero run — Feynman univariate set (12 runs, ~4-6 h).
# Launch from WSL, any shell:   bash run_phase42_feynman.sh
# Progress appears live AND is saved to results/phase42_feynman_run.log:
#   - a "### RUN k/12 ..." banner per run, with elapsed time and ETA
#   - burn-in progress bars (stage-2 combos)
#   - a status line every 25 rounds: round n/800, wall time, current
#     best cold-rung log-posterior and its expression
#   - a "[checkpoint] k/12 runs saved" line after every finished run
#     (partial results always in results/phase42_feynman_partial.json,
#     so nothing is lost if the machine sleeps or the run is killed)
# Final output: results/phase42_feynman.json + console summary table.
ENV=/home/el_bardo/anaconda3/envs/mcmc_koopman
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
"$ENV/bin/mpiexec" -n 1 "$ENV/bin/python" -u exp_phase42_feynman.py 2>&1 \
  | tee results/phase42_feynman_run.log
echo "PYEXIT=${PIPESTATUS[0]}"
