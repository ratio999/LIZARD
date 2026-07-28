"""REPRODUCIBILITY RERUN — Phase 4.1 depth-3 discovery suite.

NOT part of the default test suite: run ON DEMAND (e.g. after kernel
changes) to check that the archived depth-3 results still reproduce.

Reruns exp_phase41_depth3.py UNMODIFIED — same targets (poly3,
inv_quad, mixed), seeds (17, 71), 400-round budget, and burn-in worker
count (2; pinned — the burn-in ranking is worker-count sensitive) —
writing to a SEPARATE output so nothing archived is touched:

    results/phase41_depth3_repro.json

Expectation: BIT-IDENTICAL reproduction of results/phase41_depth3.json.
Any deviation is a REGRESSION SIGNAL: compare with
repro_phase41_plots.py (separate figures + field-by-field diff report).

Run (module dir, MPI-capable env — spawn burn-in!):
    mpiexec -n 1 python repro_phase41_depth3.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

sys.argv = ["repro_phase41_depth3"]   # neutralize --only parsing

import exp_phase41_depth3 as suite

OUT = os.path.join(_HERE, "results", "phase41_depth3_repro.json")


def main():
    assert suite.BURN_IN_WORKERS == 2, (
        "worker count drifted (%r) — the original ran with 2; "
        "reproduction requires the same" % suite.BURN_IN_WORKERS)
    assert suite.ONLY is None
    suite.OUT_JSON = OUT
    print("[repro-d3] rerunning the depth-3 suite verbatim -> %s"
          % os.path.basename(OUT), flush=True)
    suite.main()
    print("[repro-d3] done. Compare with: python repro_phase41_plots.py",
          flush=True)


if __name__ == "__main__":
    main()
