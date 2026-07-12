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
  (e) Phase 2.10 Stage 3b: opaque atomic primitives cross the wire as
      ('skel3b', skeleton, description) payloads -- a seeded atom tree
      survives the startup wire, the per-round gathers/scatters, and the
      final collection, and the sampler keeps fitting the signal only the
      atom can represent.

NOTE: in environments where singleton MPI_Comm_spawn is not configured,
run under the launcher instead:  mpiexec -n 1 python test_phase2mpi.py
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


def test_mpi_atoms(n_rounds=6, n_steps=25, seed=5):
    """(e) Phase 2.10 Stage 3b: opaque atomic primitives cross the MPI wire.

    A two-rung Forest carries the Asin = c0*sin(c1*x) template and a signal
    only the atom can fit (safe-4 ops: no compose/pow, so a frequency-scaled
    sinusoid is unreachable by combining the plain vocabulary). Slot 0 of
    the cold grove is seeded with the PLANTED atom tree (theta at the true
    values), so the STARTUP wire for that slot must already be a 'skel3b'
    skeleton payload; workers rebuild it as a unit, run their kernels with
    atoms, and the per-round gathers + final collection must bring opaque
    leaves home intact.
    """
    print("[2.10q] MPI wire with opaque primitives (Stage 3b)")
    import real_functionC as rfc
    import function_treeC as ftc

    x = np.linspace(0.1, 6.0, 50).astype(np.float64)
    y = (2.3 * np.sin(1.3 * x)).astype(np.float64)
    spec = rfc.AtomSpec("Asin", "c0*sin(c1*x)")
    funcs = ["sin(x)", "x"]

    # freeze_vocab=True: this is the "paper mode" combination — closed
    # psi_const vocabulary (Stage 3c) + atoms + the skeleton wire — so the
    # test also exercises the worker-side set_state restore of
    # unify_params and the master's pool-registration skip.
    forest = Forest(x, y, np.asarray([1.0, 2.0]), 3, 1, list(funcs),
                    sigma=1.0, Npars=1, init_value=1.0,
                    operations=["+", "-", "*", "/"], base_temp=1.0,
                    use_gradients=False, seed=seed, atoms=[spec],
                    freeze_vocab=True)
    assert forest.polya.primitives.count("Asin") == 1
    assert forest.polya.unify_params and not forest.polya.auto_register
    vocab0 = list(forest.polya.primitives)

    # Plant the true atom in the cold grove: the startup wire for this slot
    # is a skeleton payload by construction.
    g0 = forest.groves[0]
    anode = ftc.Node(rfc.instantiate_atom(spec, x, ["a0", "a1"]))
    atree = ftc.Tree(0, list(funcs), x, node=anode, sigma=g0.sigma)
    anode.theta["a0"] = 2.3
    anode.theta["a1"] = 1.3
    atree.tree_update(atree.get_root())
    atree.set_g_tau(g0.g_tau)
    g0.set_single_tree(atree, 0)
    g0.root_posteriors[0] = atree.log_marginal_bic(y, optimize=False)
    assert np.isfinite(g0.root_posteriors[0])

    stats = forest.run_mpi_spawn(y, n_rounds, n_steps, seed=seed + 1,
                                 max_tree_depth=4,
                                 subtree_rebuild_max_height=2)
    print("[2.10q] stats:", stats)

    # (i) machinery sane after the run
    for gi, g in enumerate(forest.groves):
        for si, t in enumerate(g.trees):
            assert t is not None and t.is_valid(x), \
                "grove %d slot %d broken after atom-MPI run" % (gi, si)
    # (ii) the frozen vocabulary is BIT-STABLE across the whole MPI run
    #      (master replay + every worker replica): no second Asin cell, no
    #      instantiated description, no parameter name — nothing at all.
    assert list(forest.polya.primitives) == vocab0, \
        "frozen vocabulary changed during the MPI run"
    for prim in forest.polya.primitives:
        assert "*" not in prim, "instantiated atom leaked: %r" % prim
    # (iii) opaque leaves came home: the planted near-perfect atom tree
    #       dominates everything the safe-4 vocabulary can build, so at
    #       least one final tree must be atomic.
    n_atomic = sum(
        1 for g in forest.groves for t in g.trees
        if any(lf.is_atomic() for lf in t.get_leaves()))
    assert n_atomic >= 1, "opaque leaf lost across the MPI wire"
    # (iv) and the master's best tree still fits the signal only the atom
    #      can represent
    best = forest.best_tree()
    resid = y - np.asarray(best.evaluate(), dtype=np.float64)
    mse = float(np.nanmean(resid * resid))
    assert mse < 0.05, "best tree %r mse %.4f" % (
        best.tree_get_description(), mse)
    print("[2.10q] PASS  n_atomic=%d best=%s mse=%.2e"
          % (n_atomic, best.tree_get_description(), mse))


def test_mpi_burnin_atoms(seed=13):
    """(f) Phase 2.10 (final): the SPAWN burn-in seeds opaque templates and
    preserves them across all three wire crossings (depth-1 trees shipped
    to workers, worker results gathered back, master-side rebuild) — each
    uses the Stage 3b dual-payload wire for atom-bearing trees. The
    rank-parallel (non-spawn) variant rebuilds candidates from description
    strings and must refuse atoms loudly.
    """
    print("[2.10f] spawn burn-in with opaque primitives")
    import real_functionC as rfc
    from groveC import Grove

    x = np.linspace(0.1, 6.0, 50).astype(np.float64)
    y = (2.3 * np.sin(1.3 * x)).astype(np.float64)
    spec = rfc.AtomSpec("Asin", "c0*sin(c1*x)")

    # WORKLOAD-SIZE CONSTRAINT: the spawn master's progress-wait loop only
    # exits after receiving at least one worker progress message, and
    # workers send one every 50 processed items — so every spawn burn-in
    # needs >= ~50 stage-2 combinations per worker chunk or the master
    # waits forever (a latent edge never hit by real vocabularies; see
    # PLAN §2.10). Three ops push the n_top^2 enumeration well past that
    # while keeping the recovery argument intact: polynomials from {x} via
    # {+,-,*} at depth <= 2 cannot fit a multi-period sinusoid, so a good
    # fit is still reachable ONLY through the seeded atom.
    grove = Grove(x, y, 1.0, 6, 1, ["x"], sigma=1.0,
                  operations=["+", "-", "*"],
                  use_gradients=False, burn_in=True, use_mpi=2,
                  use_spawn=True, seed=seed, atoms=[spec])
    assert grove.size > 0 and grove.trees, "spawn burn-in produced no trees"
    best = grove.best_tree()
    atomics = [lf for lf in best.get_leaves() if lf.is_atomic()]
    assert len(atomics) >= 1, (
        "spawn burn-in best tree %r carries no opaque leaf"
        % best.tree_get_description())
    resid = y - np.asarray(best.evaluate(), dtype=np.float64)
    mse = float(np.nanmean(resid * resid))
    assert mse < 1e-2, (
        "spawn burn-in failed to fit through the atom (best %r, mse %.4f)"
        % (best.tree_get_description(), mse))
    n_atomic = sum(1 for t in grove.trees
                   if any(lf.is_atomic() for lf in t.get_leaves()))
    print("[2.10f] spawn PASS  (best: %s, mse %.2e, %d atomic trees)"
          % (best.tree_get_description(), mse, n_atomic))

    # rank-parallel variant refuses atoms (guarded, not silent decomposition)
    raised = False
    try:
        Grove(x, y, 1.0, 4, 1, ["x"], sigma=1.0, operations=["+"],
              use_gradients=False, burn_in=True, use_mpi=2,
              use_spawn=False, seed=seed, atoms=[spec])
    except NotImplementedError:
        raised = True
    assert raised, "_burn_in_stage2_parallel must refuse atoms"
    print("[2.10f] PASS  (rank-parallel guard raised as required)")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    test_serial_reference()
    test_mpi_sampler()
    test_mpi_atoms()
    test_mpi_burnin_atoms()
    print("\nAll Phase 2.12 (MPI sampling) tests passed.")
