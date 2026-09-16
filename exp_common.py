"""Shared infrastructure for the Phase 4 experiment scripts.

Encodes the agreed experimental protocol (2026-07-13):

  * SIGMA CONVENTION — benchmarks score with sigma FIXED AT THE KNOWN
    NOISE LEVEL (Forest/Grove sigma = noise_sd). The sigma=1.0 test-suite
    convention flattens the likelihood ~1/noise_sd^2-fold and makes
    low-amplitude structure legitimately lose to constants (see PLAN
    §4.1, results/phase41_recovery_rational_v2.json).

  * SHARED BURN-IN — the exhaustive depth-<=2 burn-in runs ONCE per
    (target, seed) in a standalone Grove using the spawn variant
    (_burn_in_stage2_parallel_spawn), and every Forest rung is seeded
    with the SAME top-K candidates. Burn-in scoring is pure BIC
    (temperature-independent), so sharing is exact, not approximate.
    NB: do NOT build Forest(burn_in=True) — that runs one redundant
    burn-in per rung and balloons every grove to the full candidate set.

  * SEEDING — node_lists entries are DEEPCOPIES per rung (aliased Nodes
    across groves would be corrupted by in-place moves), and the caller
    contract is that seeded_forest() already ran compute_posteriors_all
    (the node_list init path leaves posteriors as None placeholders that
    Grove._step cannot digest).

  * TRACES — emc_loop() can record per-round snapshots: best posterior
    per rung, Polya urn counter t, Polya trie size, and the root-cell
    Dirichlet (the "what did the urn learn" diagnostic).

Import from the module directory; the spawn burn-in needs an MPI-capable
environment (activated conda shell, or run under `mpiexec -n 1`).
"""

import time
from copy import deepcopy

import numpy as np

import gp_movesC
from groveC import Grove, Forest

# Bounded-support caps used by every Phase 4 experiment (must match the
# fixtures the ladder is tuned under).
MAX_TREE_DEPTH = 4
REBUILD_MAX_HEIGHT = 2


def set_caps(max_depth=MAX_TREE_DEPTH, rebuild=REBUILD_MAX_HEIGHT):
    gp_movesC.set_max_tree_depth(max_depth)
    gp_movesC.set_subtree_rebuild_max_height(rebuild)


def spawn_burn_in(x, y, funcs, ops, sigma, seed, atoms=None, n_workers=3,
                  depth=2, burn_in_top_k=0, burn_in_cap=0):
    """Run the shared burn-in ONCE and return (roots, posts, wall_s).

    roots : candidate root Nodes ranked by BIC posterior (best first) —
            deepcopy before installing anywhere;
    posts : matching posteriors;
    wall_s: burn-in wall time.

    Spawn-variant constraint (PLAN §2.10): each worker chunk must reach
    >= 50 stage-2 combinations or the master's progress loop never
    breaks; any realistic vocabulary (>= 3 primitives, safe-4 ops)
    clears this by orders of magnitude.

    burn_in_top_k / burn_in_cap (both 0 = exhaustive, the historical
    behaviour) bound stage 2: keep the K best depth-1 candidates, and/or
    sample at most `cap` stage-2 combinations under the seed. Needed for
    the d=3 pooled search, where the exhaustive stage 2 reaches ~5e5
    combinations on 1e4+ rows. Keep cap / n_workers >= 50 (see above).
    """
    t0 = time.perf_counter()
    g = Grove(np.ascontiguousarray(x, dtype=np.float64),
              np.ascontiguousarray(y, dtype=np.float64),
              1.0, 8, depth, list(funcs), sigma=sigma,
              operations=list(ops), use_gradients=False,
              burn_in=True, use_mpi=int(n_workers), use_spawn=True,
              seed=seed, atoms=atoms,
              burn_in_top_k=int(burn_in_top_k), burn_in_cap=int(burn_in_cap))
    wall = time.perf_counter() - t0
    roots = [t.get_root() for t in g.trees]
    posts = [float(p) for p in np.asarray(g.root_posteriors)[:len(roots)]]
    return roots, posts, wall


def seeded_forest(x, y, ladder, size_per_grove, funcs, ops, sigma, seed,
                  top_roots, atoms=None, freeze_vocab=True,
                  flex_atoms=None, flex_root_lock=False):
    """Forest with EVERY rung seeded from the same top-K burned-in roots.

    Each rung receives its own deepcopies; posteriors are computed here
    (warm-started from the burn-in theta-hats, so this is cheap). The
    rungs then evolve by within-grove MH, coupled each round by swaps and
    trans-grove crossovers under the one shared Polya prior.

    flex_atoms (optional, default None = unchanged behaviour): scaffold
    templates forwarded to the Forest, for populations seeded with
    transparent scaffolds — burn-in never runs here, so the flex
    burn-in refusal is not in play.
    """
    k = int(size_per_grove)
    picks = [top_roots[i % len(top_roots)] for i in range(k)]
    node_lists = [[deepcopy(r) for r in picks] for _ in range(len(ladder))]
    forest = Forest(np.ascontiguousarray(x, dtype=np.float64),
                    np.ascontiguousarray(y, dtype=np.float64),
                    np.asarray(ladder, dtype=np.float64), k, 2,
                    list(funcs), sigma=sigma, node_lists=node_lists,
                    operations=list(ops), use_gradients=False, seed=seed,
                    freeze_vocab=freeze_vocab, atoms=atoms,
                    flex_atoms=flex_atoms, flex_root_lock=flex_root_lock)
    forest.compute_posteriors_all(y)
    return forest


def _polya_num_nodes(polya):
    if polya is None:
        return 0
    n = 0
    stack = [polya.root]
    while stack:
        node = stack.pop()
        n += 1
        stack.extend(node.children.values())
    return n


def snapshot(forest, unique_counts=None):
    """Light per-round trace record (all fields JSON-serializable).

    2026-07-13 additive fields (older JSONs simply lack them):
      accepts        per-rung cumulative accepted within-grove moves
      depths / nodes per-rung lists of tree depth and node count
      unique_structs per-rung cumulative distinct structures visited
                     (accumulated by emc_loop, passed in here)
    """
    best = []
    for g in forest.groves:
        arr = np.asarray([p if p is not None else np.nan
                          for p in g.root_posteriors], dtype=np.float64)
        best.append(float(np.nanmax(arr)) if arr.size else float("nan"))
    rec = {"best_post": best}
    rec["accepts"] = [int(sum(g.move_accepts.values()))
                      for g in forest.groves]
    rec["depths"] = [[int(t.get_root().depth()) for t in g.trees]
                     for g in forest.groves]
    rec["nodes"] = [[int(t.node_count()) for t in g.trees]
                    for g in forest.groves]
    if unique_counts is not None:
        rec["unique_structs"] = [int(c) for c in unique_counts]
    if forest.polya is not None:
        rec["polya_t"] = int(forest.polya.t)
        rec["polya_nodes"] = _polya_num_nodes(forest.polya)
        rec["root_alphas"] = [float(a) for a in forest.polya.root.alphas]
    return rec


def emc_loop(forest, y, rounds, steps_per_round, swaps_per_round,
             xovers_per_round=1, trace_every=1, label="", t0=None,
             progress_every=50):
    """The canonical serial EMC loop; returns the list of trace snapshots."""
    traces = []
    seen = [set() for _ in forest.groves]
    if t0 is None:
        t0 = time.perf_counter()
    for rnd in range(rounds):
        forest.run_all(y, steps_per_round)
        for _ in range(swaps_per_round):
            forest.attempt_swaps(y)
        for _ in range(xovers_per_round):
            forest.attempt_crossovers(y)
        # cumulative distinct structures per rung (cheap: cached strings)
        for gi, g in enumerate(forest.groves):
            for t in g.trees:
                seen[gi].add(t.tree_get_description())
        if trace_every and (rnd + 1) % trace_every == 0:
            traces.append(snapshot(forest, [len(s) for s in seen]))
        if progress_every and (rnd + 1) % progress_every == 0:
            g0 = forest.groves[0]
            arr = np.asarray([p if p is not None else np.nan
                              for p in g0.root_posteriors], dtype=np.float64)
            cold = float(np.nanmax(arr)) if arr.size else float("nan")
            print("    %s: round %d/%d (%.1f s)  best cold log-post %.3f  "
                  "cold best: %s"
                  % (label, rnd + 1, rounds, time.perf_counter() - t0,
                     cold, g0.trees[g0.best_index()].tree_get_description()),
                  flush=True)
    return traces


# ---------------------------------------------------------------------------
# Smoke test: burn once, seed a 3-rung forest, run a few rounds, print the
# diagnostics the parts 2-3 scripts will rely on.
# ---------------------------------------------------------------------------
def _smoke():
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    set_caps()
    rng = np.random.default_rng(3)
    x = np.linspace(0.5, 3.5, 60)
    noise_sd = 0.1
    y = np.sin(x) + 0.5 * x + rng.normal(0.0, noise_sd, x.size)

    funcs = ["sin(x)", "cos(x)", "x"]
    ops = ["+", "-", "*", "/"]

    print("[smoke] shared spawn burn-in ...", flush=True)
    roots, posts, wall = spawn_burn_in(x, y, funcs, ops, noise_sd, seed=5,
                                       n_workers=2)
    print("[smoke] burn-in: %d candidates in %.1f s; top-3:" %
          (len(roots), wall))
    for r, p in list(zip(roots, posts))[:3]:
        print("    %10.2f   %s" % (p, r.get_fun().get_description()))

    forest = seeded_forest(x, y, [1.0, 2.0, 5.0], 4, funcs, ops, noise_sd,
                           seed=7, top_roots=roots)
    # every rung must hold the SAME top-K start (distinct objects)
    d0 = [t.tree_get_description() for t in forest.groves[0].trees]
    for g in forest.groves[1:]:
        assert [t.tree_get_description() for t in g.trees] == d0
    assert forest.groves[0].trees[0] is not forest.groves[1].trees[0]
    print("[smoke] seeded 3 rungs x 4 slots with identical top-K "
          "(distinct copies); start: %s" % d0[0])

    traces = emc_loop(forest, y, rounds=12, steps_per_round=4,
                      swaps_per_round=2, label="smoke", progress_every=6)
    stats = forest.ladder_stats()
    print("[smoke] 12 rounds done; swap attempts per pair: %s"
          % stats["swap_attempts"])
    print("[smoke] trace[0] keys: %s" % sorted(traces[0].keys()))
    print("[smoke] polya_t %d -> %d, trie %d -> %d nodes"
          % (traces[0]["polya_t"], traces[-1]["polya_t"],
             traces[0]["polya_nodes"], traces[-1]["polya_nodes"]))
    best = forest.best_tree()
    print("[smoke] best after 12 rounds: %s" % best.tree_get_description())
    print("[smoke] PASS")


if __name__ == "__main__":
    _smoke()
