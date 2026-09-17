"""REPRODUCIBILITY RERUN — Phase 4.1 depth-2 recovery suite (v2).

NOT part of the default test suite: run ON DEMAND (e.g. after kernel
changes) to check that the archived v2 results still reproduce.

Reruns exp_phase41_recovery.py UNMODIFIED — same targets, seeds (17,
71), budgets, protocol, and burn-in worker count (2; pinned because
the burn-in ranking is worker-count sensitive) — writing to a SEPARATE
output so nothing archived is touched:

    results/phase41_recovery_v2_repro.json

Expectation: BIT-IDENTICAL reproduction of results/phase41_recovery_v2
.json (the pipeline is deterministic at fixed seeds; kernel guards
added 2026-07-15 are inert on this vocabulary). Any deviation is a
REGRESSION SIGNAL: compare with repro_phase41_plots.py, which renders
separate figures and prints a field-by-field diff report.

Run (module dir, MPI-capable env — spawn burn-in!):
    mpiexec -n 1 python repro_phase41_depth2.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# neutralize argv BEFORE import: the suite module parses --long/--only
# at import time
sys.argv = ["repro_phase41_depth2"]

import exp_phase41_recovery as suite

OUT = os.path.join(_HERE, "results", "phase41_recovery_v2_repro.json")


def main():
    assert suite.BURN_IN_WORKERS == 2, (
        "worker count drifted (%r) — the original v2 ran with 2; "
        "reproduction requires the same" % suite.BURN_IN_WORKERS)
    assert not suite.LONG and suite.ONLY is None
    suite.OUT_JSON = OUT
    print("[repro-d2] rerunning the v2 suite verbatim -> %s"
          % os.path.basename(OUT), flush=True)
    suite.main()
    print("[repro-d2] done. Compare with: python repro_phase41_plots.py",
          flush=True)


if __name__ == "__main__":
    main()
