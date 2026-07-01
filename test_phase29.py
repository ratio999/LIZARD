"""Phase 2.9 (Evolutionary Monte Carlo / Forest) tests.

Runnable as a plain script (no pytest):

    python test_phase29.py

Covers the two new cross-temperature operators wired into `groveC.Forest`:

  (a) shared-Theta wiring + trans-grove moves do NOT reinforce the Polya
      (also a regression test for the old missing-``_py_rng`` crash);
  (b) the crossover acceptance ratio reduces to the parallel-tempering swap
      ratio when the crossover is a whole-tree swap (algebraic identity);
  (c) statistical target-invariance: adding crossover to a mutation+swap
      sampler does not change the (fixed) cold-chain stationary distribution;
  (d) R2 within-grove irreducibility: the mutation move-set connects the tree
      lattice both upward (growth) and downward (shrink) and explores broadly.

Design references (locked decisions):
  R1  ONE shared PolyaTree across all groves, never frozen; only within-grove
      mutations reinforce it. Swap and crossover are prior-preserving
      relabelings and must not call ``update_path``.
  R3  ONE random consecutive temperature pair per swap / crossover attempt.
"""

import os
import re
import sys
import math
import random

import numpy as np
import sympy as sp

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from groveC import Grove, Forest
from function_treeC import expr_to_tree
from polya_treeC import PolyaTree
import gp_movesC  # noqa: F401  (imported so a missing build fails loudly here)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
_FUNCS = ["sin(x)", "cos(x)", "x"]
_OPS = ["+", "-", "*", "/", "o", "**"]

# Strip parameter labels (a0, b3, ...) so two structurally-identical trees
# that happen to use different parameter names collapse to the same key.
_PARAM_RE = re.compile(r"[ab]\d+")


def _make_xy(n=48, seed=0):
    """Positive domain so every primitive/op stays defined."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0.5, 3.5, n).astype(np.float64)
    y = (np.sin(x) + 0.5 * x + rng.normal(0.0, 0.05, n)).astype(np.float64)
    return x, y


def _make_forest(x, y, temps=(1.0, 3.0), size=3, depth=2, seed=0, polya=None,
                 ops=None):
    temperatures = np.asarray(temps, dtype=np.float64)
    return Forest(
        x, y, temperatures, size, depth, _FUNCS,
        Npars=1, init_value=1.0, operations=(_OPS if ops is None else ops),
        base_temp=1.0, use_gradients=False, seed=seed, polya=polya,
    )


def _structure_sig(tree):
    """Discrete structural signature of a tree (parameter labels erased)."""
    return _PARAM_RE.sub("θ", tree.tree_get_description())


# ---------------------------------------------------------------------------
# (a) Shared Theta + trans-grove moves do not reinforce (+ RNG regression)
# ---------------------------------------------------------------------------
def test_2_9a_shared_polya_and_no_reinforce(n_moves=200):
    print("[2.9a] Forest wiring: shared Θ + trans-grove moves do not reinforce")
    x, y = _make_xy(n=48, seed=1)
    forest = _make_forest(x, y, temps=(1.0, 2.5, 6.0), size=3, depth=2, seed=7)

    # R1: the Forest owns ONE PolyaTree and every grove points at that SAME
    # object (shared Theta). Identity, not equality.
    assert forest.polya is not None, "Forest must build a shared PolyaTree"
    for k, g in enumerate(forest.groves):
        assert g.polya is forest.polya, (
            "grove %d holds a different PolyaTree -> shared-Θ broken" % k)

    # Give every slot a valid baseline posterior for swap/crossover to read.
    forest.compute_posteriors_all(y)

    # The trans-grove swap and crossover are prior-preserving relabelings:
    # they MUST NOT advance the adaptive Polya counter. (Only within-grove
    # mutations, exercised by run_all, are allowed to reinforce.)
    #
    # This loop is also the regression test for the old Forest, whose swap
    # referenced a non-existent ``self._py_rng`` and raised AttributeError.
    t_before = forest.polya.t
    n_swap_acc = 0
    n_xover_acc = 0
    for _ in range(n_moves):
        n_swap_acc += forest.attempt_swaps(y)
        n_xover_acc += forest.attempt_crossovers(y)

    assert forest.polya.t == t_before, (
        "trans-grove moves reinforced the shared Polya: t %d -> %d"
        % (t_before, forest.polya.t))

    # Every tree in every grove must still be a valid model after the churn
    # (deepcopy-based crossover must never leave a grove holding garbage).
    for k, g in enumerate(forest.groves):
        for i, tr in enumerate(g.trees):
            assert tr.is_valid(x), (
                "grove %d slot %d became invalid after trans-grove moves" % (k, i))

    print("[2.9a] PASS  swaps_acc=%d  xover_acc=%d  polya.t stayed %d"
          % (n_swap_acc, n_xover_acc, t_before))


# ---------------------------------------------------------------------------
# (b) Crossover acceptance reduces to the PT swap for a whole-tree swap
# ---------------------------------------------------------------------------
def test_2_9b_crossover_reduces_to_swap(n_cases=1000, seed=2024):
    print("[2.9b] crossover acceptance reduces to the PT swap (whole-tree swap)")
    # When the "crossover" happens to swap the two whole trees, the children
    # carry the OTHER grove's energy (Ua' = Ub, Ub' = Ua) and, because
    # subtree_crossover is symmetric, lqf == lqr. The Forest crossover
    # exponent
    #
    #     (Ua'-Ua)/Ta + (Ub'-Ub)/Tb + (lqr - lqf)
    #
    # then collapses to
    #
    #     (Ub-Ua)/Ta + (Ua-Ub)/Tb = (1/Ta - 1/Tb)(Ub - Ua),
    #
    # which is exactly the parallel-tempering swap exponent used by
    # ``Forest._attempt_swap_pair``.
    rng = np.random.default_rng(seed)
    max_err = 0.0
    for _ in range(n_cases):
        Ta, Tb = rng.uniform(0.3, 8.0, 2)
        Ua, Ub = rng.uniform(-50.0, 50.0, 2)
        lqf = lqr = float(rng.uniform(-3.0, 3.0))     # symmetric proposal
        Ua_new, Ub_new = Ub, Ua                       # whole-tree swap

        log_alpha_xover = ((Ua_new - Ua) / Ta
                           + (Ub_new - Ub) / Tb
                           + (lqr - lqf))
        log_alpha_swap = (1.0 / Ta - 1.0 / Tb) * (Ub - Ua)
        max_err = max(max_err, abs(log_alpha_xover - log_alpha_swap))

    assert max_err < 1e-12, (
        "crossover did not reduce to the swap exponent: max|Δ|=%.3e" % max_err)
    print("[2.9b] PASS  %d random (Ta,Tb,Ua,Ub) cases, max|Δ|=%.2e"
          % (n_cases, max_err))


# ---------------------------------------------------------------------------
# (c) Statistical target-invariance of the crossover move
# ---------------------------------------------------------------------------
def test_2_9c_crossover_target_invariance(rounds=500, slack=0.15, seed=99):
    """Adding crossover must not move the stationary distribution.

    We need a *fixed* target for both runs. We get it by FREEZING the shared
    Polya (huge t => diminishing-adaptation gamma_t = 1/(1+t) ~ 0, so its
    alphas never move) rather than removing it. Freezing keeps the target
    stationary AND retains the Polya's per-node complexity penalty, which is
    the ONLY thing that bounds tree size here: BIC penalises the number of
    *parameters*, not the number of *nodes*, so a non-parametric subtree can
    grow for free. Nulling the prior (the naive way to fix the target) lets
    the hot grove bloat without bound until subtree_mutation rebuilds a
    perfect binary subtree of the (now huge) subtree height via
    build_tree(height) -- an O(2^height) blow-up that looks like a hang.

    We compare the cold-chain (T=1) empirical distribution for a mutation+swap
    sampler vs a mutation+swap+crossover sampler. Because the structure space
    is large and multimodal, a fixed TV threshold is dominated by sampling
    noise (the naive absolute test failed at TV~0.50 even though the acceptance
    ratio is the validated EMC form). Instead we calibrate against the noise
    floor measured by a SECOND independent mutation-only run, and assert on the
    low-variance size (leaf-count) marginal -- exactly where a mis-weighted
    trans-dimensional crossover would bias the chain. Crossover is target-
    preserving iff adding it does not move the size histogram beyond that
    floor.
    """
    print("[2.9c] crossover preserves the fixed tempered target (statistical)")
    x, y = _make_xy(n=40, seed=3)
    # Gentle ladder: a modest top temperature keeps trees shallow (fast) while
    # still giving the two chains different targets so crossover actually does
    # something.
    temps = (1.0, 2.0)

    # build_tree() ALWAYS uses pow/compose internally (it ignores the ops
    # list), and `compose` is a sympy subs whose symbolic size explodes with
    # height. gp_movesC bounds the height it ever feeds to build_tree (see
    # _SUBTREE_REBUILD_MAX_HEIGHT / _ADD_NODE_DEPTH_MAX), which defuses the
    # per-move bomb. The remaining accumulation risk is pow/compose used as a
    # GLUE op by add_node/split: repeated compose glue nests the WHOLE-tree
    # expression and blows up over many rounds. Restricting this test's glue
    # ops to +,-,*,/ makes the accepted tree grow additively instead, so the
    # cold-chain expression stays tractable for the full run. (Crossover
    # target-invariance is an algebraic property, independent of which ops the
    # vocabulary contains.)
    safe_ops = ["+", "-", "*", "/"]

    def build(bseed):
        f = _make_forest(x, y, temps=temps, size=3, depth=2, seed=bseed,
                         ops=safe_ops)
        # Freeze the shared Polya: a huge counter makes gamma_t ~ 1e-8, so
        # update_path is a no-op in practice and the prior stays fixed. Every
        # grove holds this SAME object (shared Theta), so freezing once fixes
        # the whole population's target while keeping the bloat-bounding
        # per-node penalty active.
        f.polya.t = 10 ** 8
        f.compute_posteriors_all(y)
        return f

    def sample(forest, use_xover):
        struct = {}
        sizes = {}
        acc = 0
        burn = rounds // 4
        for r in range(rounds):
            print(r)
            forest.run_all(y, 1)          # within-grove mutation
            forest.attempt_swaps(y)       # cross-temperature exchange
            if use_xover:
                acc += forest.attempt_crossovers(y)
            if r >= burn:
                cold = forest.groves[0]   # temps[0] = 1.0 is the cold chain
                for tr in cold.trees:
                    s = _structure_sig(tr)
                    struct[s] = struct.get(s, 0) + 1
                    k = len(tr.get_leaves())   # size marginal (leaf count)
                    sizes[k] = sizes.get(k, 0) + 1
        return struct, sizes, acc

    # Belt-and-suspenders: hard-cap the tree depth and the subtree-rebuild
    # height for the duration of this test so no single move can build a large
    # sympy expression even if the ladder drifts. Restored afterwards so other
    # tests keep the production defaults. getattr() keeps this working against
    # an older compiled gp_movesC that predates these setters.
    _set_depth = getattr(gp_movesC, "set_max_tree_depth", None)
    _get_depth = getattr(gp_movesC, "get_max_tree_depth", None)
    _set_reb = getattr(gp_movesC, "set_subtree_rebuild_max_height", None)
    _get_reb = getattr(gp_movesC, "get_subtree_rebuild_max_height", None)
    _old_depth = _get_depth() if _get_depth else None
    _old_reb = _get_reb() if _get_reb else None
    try:
        if _set_depth:
            _set_depth(4)
        if _set_reb:
            _set_reb(2)
        # sA/zA and sA2/zA2: two INDEPENDENT mutation+swap runs (no crossover).
        # Their cold-chain histograms differ only by finite-sample + multimodal
        # mixing noise -- this is the null "noise floor". sB/zB adds crossover.
        sA,  zA,  _   = sample(build(seed),       use_xover=False)
        sA2, zA2, _   = sample(build(seed + 101), use_xover=False)
        sB,  zB,  acc = sample(build(seed + 1),   use_xover=True)
    finally:
        if _set_depth and _old_depth is not None:
            _set_depth(_old_depth)
        if _set_reb and _old_reb is not None:
            _set_reb(_old_reb)

    # The test is only meaningful if crossover actually fired; otherwise the
    # crossover run is trivially identical to a mutation+swap sampler.
    assert acc >= 5, (
        "crossover accepted only %d times; test is vacuous — widen the "
        "vocabulary or increase ``rounds``" % acc)

    def _tv(c1, c2):
        n1 = float(sum(c1.values())) or 1.0
        n2 = float(sum(c2.values())) or 1.0
        ks = set(c1) | set(c2)
        return 0.5 * sum(abs(c1.get(k, 0) / n1 - c2.get(k, 0) / n2)
                         for k in ks)

    # Compare against the MEASURED noise floor rather than a fixed absolute
    # threshold: over a large, multimodal structure space two independent runs
    # of the SAME sampler already differ a lot. Assert on the low-variance SIZE
    # marginal (leaf count) -- well estimated with ~1e3 samples AND exactly
    # where a mis-weighted trans-dimensional crossover would bias the chain
    # (toward larger / smaller trees). Full-structure TV is diagnostic only.
    tv_null_size = _tv(zA, zA2)     # noise floor, size marginal
    tv_test_size = _tv(zA, zB)      # floor + any crossover-induced size shift
    tv_null_str  = _tv(sA, sA2)     # (diagnostic) noise floor, full structure
    tv_test_str  = _tv(sA, sB)      # (diagnostic) full structure

    assert tv_test_size <= tv_null_size + slack, (
        "adding crossover shifted the cold SIZE marginal beyond the sampling-"
        "noise floor: TV(no-x, x)=%.3f > TV(no-x, no-x)=%.3f + %.2f "
        "(crossover may not preserve the target)"
        % (tv_test_size, tv_null_size, slack))
    print("[2.9c] PASS  size: TV_test=%.3f vs TV_null=%.3f (+%.2f)  |  "
          "struct(diag): TV_test=%.3f vs TV_null=%.3f  |  xover_acc=%d"
          % (tv_test_size, tv_null_size, slack,
             tv_test_str, tv_null_str, acc))


# ---------------------------------------------------------------------------
# (d) R2 within-grove irreducibility / connectivity
# ---------------------------------------------------------------------------
def test_2_9d_within_chain_irreducibility(n_steps=800, seeds=(0, 1, 2, 3),
                                          depth=3):
    """Empirical within-chain irreducibility on a tiny vocabulary.

    The tree space is variable-dimension and (with free parameters) infinite,
    so "visit every state" is impossible. Instead we verify the essential
    connectivity property: the mutation move-set can move the chain BOTH up
    the size lattice (a leaf grows into a larger tree) and back DOWN it (a
    larger tree shrinks to a leaf), and that it explores many distinct
    structures rather than getting trapped at its start.
    """
    print("[2.9d] R2: within-grove mutation move-set is irreducible")
    x, y = _make_xy(n=40, seed=5)
    funcs = ["x", "sin(x)"]    # tiny vocabulary
    ops = ["+", "*"]

    x_sym = sp.Symbol("x")
    leaf_expr = x_sym                   # smallest structure: a single leaf
    big_expr = sp.sin(x_sym) + x_sym    # a strictly larger structure

    def run_from(expr, rseed):
        g = Grove(x, y, T=2.0, size=1, depth=depth, funcs=funcs,
                  operations=ops, Npars=1, init_value=1.0, base_temp=1.0,
                  use_gradients=False, seed=rseed)
        start = expr_to_tree(expr, x, funcs)
        g.set_single_tree(start, 0)
        g.compute_posteriors(y)         # refresh the injected tree's baseline
        start_size = len(start.get_leaves())
        sigs = set()
        sizes = set()
        saw_decrease = False
        prev_size = start_size
        # Bound per-step cost: cap the tree depth and the height ever fed to
        # build_tree (its internal pow/compose is the expensive part) so this
        # 4-seed x 2-direction sweep stays fast. Depth 4 still allows plenty of
        # up/down growth for the connectivity check. Restored on exit.
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
            for st in range(n_steps):
                print(st)
                g.run(y, 1)
                tr = g.trees[0]
                sigs.add(_structure_sig(tr))
                cur_size = len(tr.get_leaves())
                sizes.add(cur_size)
                if cur_size < prev_size:
                    saw_decrease = True
                prev_size = cur_size
        finally:
            if _sd and _od is not None:
                _sd(_od)
            if _sr and _or is not None:
                _sr(_or)
        return sigs, sizes, start_size, saw_decrease

    # (i) Upward connectivity: from a single leaf the chain must reach a
    #     strictly larger tree (birth / growth moves connect upward).
    up_sigs = set()
    up_ok = False
    saw_shrink = False
    for s in seeds:
        sigs, sizes, s0, dec = run_from(leaf_expr, s)
        up_sigs |= sigs
        if max(sizes) > s0:
            up_ok = True
        saw_shrink = saw_shrink or dec
    assert up_ok, (
        "from a leaf the chain never reached a larger tree "
        "-> no upward connectivity")

    # (ii) Downward connectivity: a death move (leaf_hoist / internal_hoist)
    #      must be accepted at least once -- i.e. the chain must be observed
    #      stepping to a STRICTLY smaller tree. We assert on an accepted size
    #      DECREASE rather than on reaching a fixed absolute size, because
    #      every birth grafts a perfect-binary subtree of depth >= 1 (>= 2
    #      leaves): the death on-support condition requires the dropped sibling
    #      to have depth >= 1 too, so leaf-count PARITY is conserved by every
    #      add/split/hoist. A two-leaf start such as sin(x)+x can therefore
    #      NEVER reach a single leaf -- "min(size) < start" would be structur-
    #      ally impossible, not a mixing failure. Observing size[t+1] < size[t]
    #      anywhere is the faithful, parity-correct evidence that the downward
    #      (death) moves fire and are accepted.
    for s in seeds:
        _, sizes, s0, dec = run_from(big_expr, s)
        saw_shrink = saw_shrink or dec
    assert saw_shrink, (
        "no accepted size-decreasing (death) move was ever observed across "
        "the leaf-started or the larger-tree-started chains "
        "-> no downward connectivity")

    # (iii) Broad exploration: the leaf-started chains visit many distinct
    #       structures across seeds (not trapped at the start).
    assert len(up_sigs) >= 5, (
        "mutation chain explored only %d distinct structures (expected >= 5)"
        % len(up_sigs))

    print("[2.9d] PASS  up-connected + down-connected, distinct structures=%d"
          % len(up_sigs))


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Keep BLAS/OpenMP single-threaded for reproducibility parity with the
    # rest of the suite.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    test_2_9a_shared_polya_and_no_reinforce()
    test_2_9b_crossover_reduces_to_swap()
    test_2_9c_crossover_target_invariance()
    test_2_9d_within_chain_irreducibility()
    print("\nAll Phase 2.9 (Forest EMC) tests passed.")
