"""Phase 2.8 (adaptive move-rate calibration) tests.

Runnable as a plain script (no pytest):

    python test_phase28.py

The adaptive move-weight machinery ALREADY lives in groveC (the ``adapt_moves``
flag plus ``_adapted_move_probs`` / ``_adapt_move`` / ``_move_logits`` /
``_move_group``). It shipped OFF by default and was never exercised. Phase 2.8
is therefore a *validation* task, not a feature-writing one: switch adaptation
ON and prove it is (i) invariant-preserving, (ii) reproducible, and, crucially,
(iii) target-preserving.

What each test checks
---------------------
  (a) Kernel invariants hold at init AND after adaptation:
        * birth/death PAIR EQUALITY  p_add == p_leaf_hoist  and
          p_split == p_internal_hoist  (keeps _step's RJMCMC move-selection
          term identically zero);
        * CONTAINMENT: every physical move probability >= adapt_floor.
      Together these are the two Roberts & Rosenthal (2007) conditions
      (diminishing adaptation + containment) that keep the adaptive chain
      ergodic for the same target.
  (b) Adaptation actually moves the group logits and is bit-reproducible under
      a fixed seed (it consumes only accept/reject outcomes, no extra RNG).
  (c) TARGET INVARIANCE (the payoff): turning adaptation ON must not move the
      cold-chain stationary distribution. Measured null-calibrated on the
      low-variance size (leaf-count) marginal, exactly as in test_phase29 2.9c.

Design mirrors test_phase29: frozen Pólya (fixed target), safe glue ops, and
hard depth / subtree-rebuild caps so every run stays fast and bounded.
"""

import os
import re
import sys
import numpy as np
import sympy as sp

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_SRC = os.path.join(os.path.dirname(_HERE), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from groveC import Grove
from polya_treeC import PolyaTree
import gp_movesC  # noqa: F401  (imported so a missing build fails loudly here)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
_FUNCS = ["sin(x)", "cos(x)", "x"]
# Safe glue ops only: build_tree's internal pow/compose is the expensive part,
# and compose-as-glue nests the whole-tree expression over many rounds. +,-,*,/
# keep the accepted expression growing additively (same rationale as 2.9c).
_OPS_SAFE = ["+", "-", "*", "/"]

_PARAM_RE = re.compile(r"[ab]\d+")

# Physical move order returned by Grove.get_move_probs() / _choose_move dispatch.
_MOVE_LABELS = ["point", "leaf_hoist", "internal_hoist",
                "add", "split", "subtree", "cross"]
_IDX_ADD, _IDX_LEAF = 3, 1          # add_node  <-> leaf_hoist   (birth/death pair)
_IDX_SPLIT, _IDX_INT = 4, 2         # split     <-> internal_hoist(birth/death pair)


def _make_xy(n=40, seed=3):
    """Positive domain so every primitive/op stays defined."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0.5, 3.5, n).astype(np.float64)
    y = (np.sin(x) + 0.5 * x + rng.normal(0.0, 0.05, n)).astype(np.float64)
    return x, y


def _structure_sig(tree):
    """Discrete structural signature of a tree (parameter labels erased)."""
    return _PARAM_RE.sub("θ", tree.tree_get_description())


def _frozen_polya(cap=8):
    """A shared-style Pólya prior with the counter pinned so the target is
    FIXED: t = 1e8 => gamma_t = 1/(1+t) ~ 1e-8, so update_path is a practical
    no-op. Keeps the per-node size penalty (bounds bloat) while fixing the
    stationary target across adapt-on / adapt-off runs."""
    polya = PolyaTree(_OPS_SAFE, list(_FUNCS), alpha_0=1.0,
                      auto_register=True, strict_ops=True)
    names = [f"a{k}" for k in range(cap)] + [f"b{k}" for k in range(cap)]
    polya.add_primitives(names)
    polya.t = 10 ** 8
    return polya


def _make_grove(adapt, seed, depth=2, size=3, T=1.0):
    x, y = _make_xy()
    g = Grove(x, y, T, size, depth, _FUNCS,
              operations=_OPS_SAFE, Npars=1, init_value=1.0, base_temp=1.0,
              use_gradients=False, seed=seed, polya=_frozen_polya(),
              adapt_moves=adapt)
    g.compute_posteriors(y)
    return g, y


# Cap setters (getattr keeps this working against an older compiled gp_movesC).
def _get_cap_fns():
    return (getattr(gp_movesC, "set_max_tree_depth", None),
            getattr(gp_movesC, "get_max_tree_depth", None),
            getattr(gp_movesC, "set_subtree_rebuild_max_height", None),
            getattr(gp_movesC, "get_subtree_rebuild_max_height", None))


def _check_invariants(p, floor=None, label=""):
    """Assert the two structural invariants on a physical move-prob 7-vector."""
    assert abs(float(np.sum(p)) - 1.0) < 1e-9, \
        "%s: move probs do not sum to 1 (got %.6f)" % (label, float(np.sum(p)))
    # Birth/death pair equality (exact, by construction of _adapted_move_probs).
    assert abs(p[_IDX_ADD] - p[_IDX_LEAF]) < 1e-12, (
        "%s: p_add != p_leaf_hoist (%.3e vs %.3e) -> RJMCMC move-selection "
        "term would no longer cancel" % (label, p[_IDX_ADD], p[_IDX_LEAF]))
    assert abs(p[_IDX_SPLIT] - p[_IDX_INT]) < 1e-12, (
        "%s: p_split != p_internal_hoist (%.3e vs %.3e) -> RJMCMC move-"
        "selection term would no longer cancel"
        % (label, p[_IDX_SPLIT], p[_IDX_INT]))
    if floor is not None:
        assert float(np.min(p)) >= floor - 1e-12, (
            "%s: containment violated -- min prob %.4f < floor %.4f"
            % (label, float(np.min(p)), floor))


# ---------------------------------------------------------------------------
# (a) Kernel invariants at init and after adaptation
# ---------------------------------------------------------------------------
def test_2_8a_kernel_invariants(n_steps=400, seed=7):
    print("[2.8a] adaptive kernel keeps pair-equality + containment")
    _sd, _gd, _sr, _gr = _get_cap_fns()
    _od = _gd() if _gd else None
    _or = _gr() if _gr else None
    try:
        if _sd:
            _sd(4)
        if _sr:
            _sr(2)
        g, y = _make_grove(adapt=True, seed=seed)

        # Invariants must hold BEFORE any adaptation.
        p0 = np.asarray(g.get_move_probs(), dtype=np.float64)
        _check_invariants(p0, floor=g.adapt_floor, label="init")
        logits0 = np.array(g._move_logits, copy=True)
        assert g._adapt_t == 0, "adapt counter should start at 0"

        g.run(y, n_steps)

        # Invariants must STILL hold after adaptation drifts the logits.
        p1 = np.asarray(g.get_move_probs(), dtype=np.float64)
        _check_invariants(p1, floor=g.adapt_floor, label="post-run")

        # Adaptation must actually have happened.
        assert g._adapt_t > 0, "no adaptive updates occurred (adapt_moves off?)"
        assert not np.allclose(g._move_logits, logits0), (
            "group logits never moved -> adaptation is a no-op")
    finally:
        if _sd and _od is not None:
            _sd(_od)
        if _sr and _or is not None:
            _sr(_or)
    print("[2.8a] PASS  adapt_t=%d  logits moved; pair-equality + floor(%.3f) OK"
          % (g._adapt_t, g.adapt_floor))


# ---------------------------------------------------------------------------
# (b) Adaptation is deterministic under a fixed seed
# ---------------------------------------------------------------------------
def test_2_8b_determinism(n_steps=300, seed=11):
    print("[2.8b] adaptive trajectory is reproducible under a fixed seed")
    _sd, _gd, _sr, _gr = _get_cap_fns()
    _od = _gd() if _gd else None
    _or = _gr() if _gr else None
    try:
        if _sd:
            _sd(4)
        if _sr:
            _sr(2)
        g1, y1 = _make_grove(adapt=True, seed=seed)
        g2, y2 = _make_grove(adapt=True, seed=seed)
        g1.run(y1, n_steps)
        g2.run(y2, n_steps)

        # Same seed -> identical adapted logits, move stats, and structures.
        assert np.allclose(g1._move_logits, g2._move_logits, atol=0.0), \
            "identical seeds gave different adapted logits"
        assert g1.move_stats() == g2.move_stats(), \
            "identical seeds gave different move_stats()"
        s1 = [t.tree_get_description() for t in g1.trees]
        s2 = [t.tree_get_description() for t in g2.trees]
        assert s1 == s2, "identical seeds gave different final trees"
    finally:
        if _sd and _od is not None:
            _sd(_od)
        if _sr and _or is not None:
            _sr(_or)
    print("[2.8b] PASS  identical seed -> identical logits + move_stats + trees")


# ---------------------------------------------------------------------------
# (c) Target invariance: adaptation ON must not move the stationary law
# ---------------------------------------------------------------------------
def _sample(adapt, seed, rounds=400):
    """Run a single cold Grove and collect (structure, size) histograms from
    all its trees after burn-in. Returns (struct, sizes, adapted_moved)."""
    burn = rounds // 4
    g, y = _make_grove(adapt=adapt, seed=seed)
    logits0 = np.array(g._move_logits, copy=True)
    struct = {}
    sizes = {}
    for r in range(rounds):
        g.run(y, 1)
        if r >= burn:
            for tr in g.trees:
                s = _structure_sig(tr)
                struct[s] = struct.get(s, 0) + 1
                k = len(tr.get_leaves())          # size marginal (leaf count)
                sizes[k] = sizes.get(k, 0) + 1
    moved = bool(not np.allclose(g._move_logits, logits0))
    return struct, sizes, moved


def _tv(c1, c2):
    n1 = float(sum(c1.values())) or 1.0
    n2 = float(sum(c2.values())) or 1.0
    ks = set(c1) | set(c2)
    return 0.5 * sum(abs(c1.get(k, 0) / n1 - c2.get(k, 0) / n2) for k in ks)


def test_2_8c_target_invariance(rounds=400, slack=0.15, seed=99):
    """Adding adaptive move weights must not change the fixed cold target.

    Calibrate against the sampling-noise floor measured by a SECOND independent
    adaptation-OFF run, and assert on the low-variance size marginal (exactly
    the test_phase29 2.9c methodology, but the perturbation here is 'adaptation
    on/off' rather than 'crossover on/off'). Diminishing adaptation +
    containment (Roberts & Rosenthal 2007) guarantee the same stationary law;
    this is the empirical check of that guarantee.
    """
    print("[2.8c] adaptation preserves the fixed cold target (statistical)")
    _sd, _gd, _sr, _gr = _get_cap_fns()
    _od = _gd() if _gd else None
    _or = _gr() if _gr else None
    try:
        if _sd:
            _sd(4)
        if _sr:
            _sr(2)
        # Two independent adaptation-OFF runs measure the noise floor; the
        # adaptation-ON run adds only the move-weight adaptation.
        _sA, zA, _ = _sample(adapt=False, seed=seed,       rounds=rounds)
        _sA2, zA2, _ = _sample(adapt=False, seed=seed + 101, rounds=rounds)
        _sB, zB, moved = _sample(adapt=True, seed=seed + 1, rounds=rounds)
    finally:
        if _sd and _od is not None:
            _sd(_od)
        if _sr and _or is not None:
            _sr(_or)

    # The test is only meaningful if adaptation actually drifted the weights.
    assert moved, ("adaptation never changed the move weights; test is vacuous "
                   "-- did adapt_moves compile / toggle on?")

    tv_null = _tv(zA, zA2)     # noise floor, size marginal
    tv_test = _tv(zA, zB)      # floor + any adaptation-induced size shift
    assert tv_test <= tv_null + slack, (
        "turning adaptation ON shifted the cold SIZE marginal beyond the "
        "sampling-noise floor: TV(off, on)=%.3f > TV(off, off)=%.3f + %.2f "
        "(adaptation may not preserve the target)"
        % (tv_test, tv_null, slack))
    print("[2.8c] PASS  size: TV_test=%.3f vs TV_null=%.3f (+%.2f)"
          % (tv_test, tv_null, slack))


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Keep BLAS/OpenMP single-threaded for reproducibility parity with the
    # rest of the suite.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    test_2_8a_kernel_invariants()
    test_2_8b_determinism()
    test_2_8c_target_invariance()
    print("\nAll Phase 2.8 (adaptive move-rate calibration) tests passed.")
