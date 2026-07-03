"""Phase 2.12 (MPI-parallel EMC sampling loop) test.

Runnable as a plain script (NO mpirun needed -- the master spawns its own
workers via MPI.COMM_SELF.Spawn, exactly like the burn-in spawn path):

    python test_phase2mpi.py

What is being tested
--------------------
`Forest.run_mpi_spawn(y, n_rounds, n_steps_per_round, ...)` runs the EMC
loop with ONE spawned MPI worker per temperature rung: the mutation phase
(the expensive part) runs concurrently; swap + crossover decisions are
computed on the master with the synced Polya and the master RNG, mirroring
the serial acceptance math exactly. The shared-Theta rigor (R1) is kept by
deterministic replay: workers collect accepted construction paths
(polya_defer=True) and every replica replays the same ordered list each
round.

  (a) End-to-end integration: the MPI run completes (no deadlock), returns
      sane stats, and leaves the master Forest in a fully usable state:
      every tree valid, every posterior finite, best_tree() works.
  (b) Polya replay consistency: master polya.t advanced by EXACTLY the
      number of replayed paths (t_before + paths == t_after) -- the
      counter is the fingerprint of the deterministic-replay protocol.
  (c) Progress: with a non-trivial number of rounds the population must
      explore (>= 1 distinct structure change from the initial state).
  (d) Serial reference: the same Forest configuration also runs serially
      (run_all + attempt_swaps + attempt_crossovers) without error --
      guarding against any accidental behavioural change to the serial
      path from the polya_defer / _ctor additions (both default-off).
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np

from groveC import Forest
import gp_movesC


# ---------------------------------------------------------------------------
# Fixtures (small + safe-ops so every step stays fast; mirrors test_phase29)
# ---------------------------------------------------------------------------
_FUNCS = ["sin(x)", "cos(x)", "x"]
_OPS_SAFE = ["+", "-", "*", "/"]


def _make_xy(n=40, seed=3):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.5, 3.5, n).astype(np.float64)
    y = (np.sin(x) + 0.5 * x + rng.normal(0.0, 0.05, n)).astype(np.float64)
    return x, y


def _make_forest(x, y, temps=(1.0, 2.0), size=2, depth=2, seed=11):
    return Forest(
        x, y, np.asarray(temps, dtype=np.float64), size, depth, _FUNCS,
        Npars=1, init_value=1.0, operations=_OPS_SAFE,
        base_temp=1.0, use_gradients=False, seed=seed,
    )


def _sig_snapshot(forest):
    """Set of structure descriptions across all groves/slots."""
    out = set()
    for g in forest.groves:
        for t in g.trees:
            out.add(t.tree_get_description())
    return out


# ---------------------------------------------------------------------------
def test_mpi_sampler(n_rounds=25, n_steps=4, seed=7):
    print("[2.12] MPI-parallel EMC sampling loop (spawn model)")
    x, y = _make_xy()
    forest = _make_forest(x, y, seed=seed)
    forest.compute_posteriors_all(y)

    sigs_before = _sig_snapshot(forest)
    t_before = forest.polya.t

    stats = forest.run_mpi_spawn(
        y, n_rounds, n_steps, seed=seed + 1,
        # Same caps as the other phase-2.9/2.8 tests: bound the sympy
        # expression size so every worker step stays fast.
        max_tree_depth=4, subtree_rebuild_max_height=2,
    )
    print("[2.12] stats:", stats)

    # (a) master Forest fully usable after the run.
    n_finite = 0
    for gi, g in enumerate(forest.groves):
        for si, t in enumerate(g.trees):
            assert t is not None, "grove %d slot %d is None" % (gi, si)
            assert t.is_valid(x), \
                "grove %d slot %d invalid after MPI run" % (gi, si)
            assert np.isfinite(g.root_posteriors[si]), \
                "grove %d slot %d non-finite posterior" % (gi, si)
            n_finite += 1
    best = forest.best_tree()
    assert best is not None, "best_tree() returned None"
    print("[2.12] (a) PASS  %d slots valid+finite; best: %s"
          % (n_finite, best.tree_get_description()))

    # (b) Polya deterministic-replay fingerprint: the urn counter advanced
    #     by exactly the number of replayed paths.
    t_after = forest.polya.t
    assert t_after == t_before + stats['polya_paths_replayed'], (
        "polya.t mismatch: %d + %d replayed != %d (replay protocol broken?)"
        % (t_before, stats['polya_paths_replayed'], t_after))
    print("[2.12] (b) PASS  polya.t: %d -> %d (== +%d replayed paths)"
          % (t_before, t_after, stats['polya_paths_replayed']))

    # (c) the population actually moved.
    sigs_after = _sig_snapshot(forest)
    moved = (sigs_after != sigs_before) or stats['polya_paths_replayed'] > 0
    assert moved, "MPI run produced zero accepted moves across all rounds"
    print("[2.12] (c) PASS  population explored (paths=%d, swaps=%d, xover=%d)"
          % (stats['polya_paths_replayed'], stats['swap_accepted'],
             stats['crossover_accepted']))


def test_serial_reference(n_rounds=10, n_steps=4, seed=7):
    """(d) The serial path still works untouched with the same config."""
    print("[2.12d] serial reference run (guards serial path regression)")
    x, y = _make_xy()
    forest = _make_forest(x, y, seed=seed)
    forest.compute_posteriors_all(y)

    _sd = getattr(gp_movesC, "set_max_tree_depth", None)
    _gd = getattr(gp_movesC, "get_max_tree_depth", None)
    _sr = getattr(gp_movesC, "set_subtree_rebuild_max_height", None)
    _gr = getattr(gp_movesC, "get_subtree_rebuild_max_height", None)
    _od = _gd() if _gd else None
    _or = _gr() if _gr else None
    try:
        if _sd:
            _sd(4)
        if _sr:
            _sr(2)
        for _ in range(n_rounds):
            forest.run_all(y, n_steps)
            forest.attempt_swaps(y)
            forest.attempt_crossovers(y)
    finally:
        if _sd and _od is not None:
            _sd(_od)
        if _sr and _or is not None:
            _sr(_or)

    for gi, g in enumerate(forest.groves):
        for si, t in enumerate(g.trees):
            assert t.is_valid(x), \
                "serial: grove %d slot %d invalid" % (gi, si)
    # polya_defer defaults to False -> serial reinforcement still immediate.
    for g in forest.groves:
        assert not g.polya_defer, "polya_defer must default to False"
        assert g.drain_deferred_paths() == [], \
            "deferred paths must be empty on the serial path"
    print("[2.12d] PASS  serial loop unchanged (polya.t=%d)" % forest.polya.t)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    test_serial_reference()
    test_mpi_sampler()
    print("\nAll Phase 2.12 (MPI sampling) tests passed.")
