"""
Phase 3.3 tests.

(a) Determinism smoke test: same seed + same g_tau -> identical Grove
    trajectory (per-tree topology, move counters, root posteriors).

(b) Birth-density g unit test: verify g_tau plumbing (Grove -> Tree),
    default = std(y), and that point_mutation's BIRTH branch emits
    log_q_fwd = log N(theta_new; 0, g_tau^2) and log_q_rev = -log M
    (forward/reverse densities per Green 1995 RJMCMC).

(c) Per-move proposal log-density checks: reconstruct (lqf, lqr)
    analytically from tree counts and compare to the values returned
    by each public move:
      * subtree_mutation: lqr - lqf == log(size_T) - log(size_Tp),
        where size_T = #non-parameter-leaf nodes (selection density).
      * add_node:  lqr == -log n_leaves(T')   (reverse is leaf_hoist).
      * split:     lqr == -log(n_int(T') - 1) (reverse is internal_hoist).
      * leaf_hoist:     lqf == -log n_leaves(T).
      * internal_hoist: lqf == -log(n_int(T) - 1).
      * point_mutation:    symmetric branches return (0, 0) exactly;
        kind-crossing branches obey the BIRTH/DEATH formulas already
        validated in (b).

These tests are runnable as a plain script (matching the style of
test_cython4.py).  They DO NOT use pytest.

Reproducibility caveats (mirrored from Grove docstring):
    - Single-threaded BLAS/OMP recommended (OMP_NUM_THREADS=1).
    - Determinism here is process-local; cross-machine bit-equality
      requires identical numpy/scipy builds.
"""

import math
import os
import sys
import random
import numpy as np

# Allow running as `python tests/test_phase23.py` from repo root
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, os.pardir)))

import sympy as sp
import gp_movesC
from gp_movesC import (point_mutation, subtree_mutation, add_node,
                       split, leaf_hoist, internal_hoist)
from groveC import Grove
from function_treeC import Tree, expr_to_tree
from polya_treeC import PolyaTree


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
def _make_xy(n=64, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(-2.0, 2.0, n, dtype=np.float64)
    y = (np.sin(x) + 0.5 * x).astype(np.float64)
    y = y + rng.normal(0.0, 0.05, size=n).astype(np.float64)
    return x, y


_FUNCS = ["sin(x)", "cos(x)", "exp(x)", "log(x)", "x"]
_OPS = ["+", "-", "*", "/", "o", "**"]


def _make_grove(x, y, seed, g_tau=None, size=4, depth=2):
    return Grove(
        x, y, T=1.0, size=size, depth=depth, funcs=_FUNCS,
        Npars=1, init_value=1.0, operations=_OPS, base_temp=1.0,
        seed=seed, g_tau=g_tau,
    )


def _grove_state(g):
    """Snapshot of grove state used for equality comparison."""
    tops = [t.tree_get_description() for t in g.trees]
    props = dict(g.move_proposals)
    accs = dict(g.move_accepts)
    posts = np.asarray(g.root_posteriors, dtype=np.float64).copy()
    return tops, props, accs, posts


# ---------------------------------------------------------------------------
# (a) Determinism
# ---------------------------------------------------------------------------
def test_3_3a_determinism(n_steps=40, seed=12345, g_tau=0.7):
    print("[3.3a] determinism: seed=%d, n_steps=%d, g_tau=%g"
          % (seed, n_steps, g_tau))
    x, y = _make_xy(n=48, seed=0)

    g1 = _make_grove(x, y, seed=seed, g_tau=g_tau)
    g2 = _make_grove(x, y, seed=seed, g_tau=g_tau)

    # g_tau must match exactly (set by ctor, not by RNG)
    assert g1.g_tau == g_tau, "g1.g_tau mismatch: %r vs %r" % (g1.g_tau, g_tau)
    assert g2.g_tau == g_tau, "g2.g_tau mismatch: %r vs %r" % (g2.g_tau, g_tau)

    g1.run(y, n_steps=n_steps)
    g2.run(y, n_steps=n_steps)

    t1, p1, a1, post1 = _grove_state(g1)
    t2, p2, a2, post2 = _grove_state(g2)

    assert t1 == t2, "tree topologies differ:\n  g1=%s\n  g2=%s" % (t1, t2)
    assert p1 == p2, "move_proposals differ: %r vs %r" % (p1, p2)
    assert a1 == a2, "move_accepts differ: %r vs %r" % (a1, a2)
    # root_posteriors carry NaNs for empty slots; compare with equal_nan
    assert np.allclose(post1, post2, atol=1e-12, equal_nan=True), \
        "root_posteriors differ:\n  g1=%s\n  g2=%s" % (post1, post2)

    # Sanity: a different seed should generally produce a different state.
    g3 = _make_grove(x, y, seed=seed + 1, g_tau=g_tau)
    g3.run(y, n_steps=n_steps)
    t3, p3, a3, _ = _grove_state(g3)
    if (t1, p1, a1) == (t3, p3, a3):
        # Not a hard failure (small chance), but flag it for the operator.
        print("  WARN: seed+1 produced identical state; "
              "consider increasing n_steps or grove size for stronger signal.")
    print("[3.3a] PASS")


# ---------------------------------------------------------------------------
# (b) Birth-density g
# ---------------------------------------------------------------------------
def _log_g(theta, tau):
    """Reference: log N(theta; 0, tau^2)."""
    return -0.5 * math.log(2.0 * math.pi * tau * tau) - 0.5 * (theta / tau) ** 2


def test_3_3b_g_tau_plumbing():
    """Grove -> Tree: every tree carries the grove's g_tau."""
    print("[3.3b.1] g_tau plumbing")
    x, y = _make_xy(n=32, seed=1)

    # Explicit g_tau wins.
    g = _make_grove(x, y, seed=7, g_tau=0.33)
    assert g.g_tau == 0.33
    for i, t in enumerate(g.trees):
        assert t.g_tau == 0.33, "tree %d g_tau mismatch: %r" % (i, t.g_tau)

    # Default g_tau == std(y) (with the >0 guard).
    g_def = _make_grove(x, y, seed=7, g_tau=None)
    expected = float(np.asarray(y).std())
    assert math.isclose(g_def.g_tau, expected, rel_tol=1e-12, abs_tol=0.0), \
        "default g_tau %r != std(y) %r" % (g_def.g_tau, expected)
    for t in g_def.trees:
        assert math.isclose(t.g_tau, expected, rel_tol=1e-12, abs_tol=0.0)

    # Invalid g_tau rejected.
    raised = False
    try:
        _make_grove(x, y, seed=7, g_tau=0.0)
    except ValueError:
        raised = True
    assert raised, "Grove must reject g_tau=0"
    raised = False
    try:
        _make_grove(x, y, seed=7, g_tau=-1.0)
    except ValueError:
        raised = True
    assert raised, "Grove must reject g_tau<0"
    print("[3.3b.1] PASS")


def test_3_3b_birth_density(max_seeds=400):
    """point_mutation BIRTH branch emits the correct g-density.

    Strategy
    --------
    Build a small random tree, force a known RNG state, and call
    point_mutation.  Detect the BIRTH branch via its signature:

        log_q_rev == -log M   AND   log_q_fwd != 0

    (DEATH branch is the swap: log_q_fwd == -log M, log_q_rev != 0.)

    For a BIRTH, the new theta is stored on the mutated leaf -- but
    that leaf may be any leaf in the tree.  We don't try to identify
    it; instead we verify that log_q_fwd is consistent with SOME
    sample from N(0, tau^2) by checking the round-trip via the
    quadratic-form constant:

        log_q_fwd = -0.5*log(2*pi*tau^2) - 0.5*(theta/tau)^2

    -> theta^2 = -tau^2 * (2*log_q_fwd + log(2*pi*tau^2))

    The recovered theta^2 must match the squared value of the leaf
    that just became a parameter leaf.
    """
    print("[3.3b.2] birth-density analytic check (sweeping <=%d seeds)"
          % max_seeds)
    from real_functionC import _domain_fixer
    x, y = _make_xy(n=24, seed=2)
    domain = _domain_fixer(x)
    tau = 0.41
    log_M = math.log(len(_FUNCS))
    tol = 1e-10

    n_birth_checked = 0
    n_death_checked = 0

    for s in range(max_seeds):
        # Fresh tree per seed so the test is hermetic.
        tree = Tree(depth=2, functions=_FUNCS, x=domain,
                    Npars=1, value=1.0, T=1.0, g_tau=tau)

        # Capture pre-state of every leaf so we can identify which one
        # was mutated.  Use object identity (leaves are Node instances).
        pre_leaves = []
        _collect_leaves(tree.get_root(), pre_leaves)
        pre_state = {id(n): (n.is_param(), _safe_value(n)) for n in pre_leaves}

        gp_movesC.set_rng(random.Random(s), np.random.default_rng(s))
        lqf, lqr = point_mutation(tree, _FUNCS, _OPS)

        post_leaves = []
        _collect_leaves(tree.get_root() , post_leaves)

        # Branch detection.
        is_birth = (abs(lqr - (-log_M)) < tol) and (lqf != 0.0)
        is_death = (abs(lqf - (-log_M)) < tol) and (lqr != 0.0)

        if is_birth:
            # Find the leaf that flipped function -> parameter.
            theta = None
            for n in post_leaves:
                pre = pre_state.get(id(n))
                if pre is None:
                    continue
                was_param, _ = pre
                if (not was_param) and n.is_param():
                    theta = float(n.value)
                    break
            if theta is None:
                # Edge case: the mutated leaf is a new Node instance.
                # Skip but don't fail.
                continue

            expected_lqf = _log_g(theta, tau)
            assert math.isclose(lqf, expected_lqf, rel_tol=0.0, abs_tol=1e-10), (
                "[seed=%d] BIRTH lqf mismatch: got %.15g, expected %.15g "
                "(theta=%g, tau=%g)" % (s, lqf, expected_lqf, theta, tau)
            )
            n_birth_checked += 1

        elif is_death:
            # Find the leaf that flipped parameter -> function.
            theta = None
            for n in pre_leaves:
                if id(n) in {id(m) for m in post_leaves}:
                    was_param, val = pre_state[id(n)]
                    if was_param and (not n.is_param()):
                        theta = float(val)
                        break
            if theta is None:
                continue

            expected_lqr = _log_g(theta, tau)
            assert math.isclose(lqr, expected_lqr, rel_tol=0.0, abs_tol=1e-10), (
                "[seed=%d] DEATH lqr mismatch: got %.15g, expected %.15g "
                "(theta=%g, tau=%g)" % (s, lqr, expected_lqr, theta, tau)
            )
            n_death_checked += 1

        # Stop early once we've verified both branches several times.
        if n_birth_checked >= 5 and n_death_checked >= 5:
            break

    assert n_birth_checked > 0, \
        "no BIRTH branch hit in %d seeds; broaden search" % max_seeds
    print("[3.3b.2] PASS (births=%d, deaths=%d)"
          % (n_birth_checked, n_death_checked))


def _collect_leaves(node, out):
    if node is None:
        return
    if node.is_leaf():
        out.append(node)
        return
    _collect_leaves(node.get_left(), out)
    _collect_leaves(node.get_right(), out)


def _safe_value(node):
    try:
        return float(node.value)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# (c) Per-move proposal log-density checks
# ---------------------------------------------------------------------------
def _count_nodes_py(node):
    if node is None:
        return 0
    if node.is_leaf():
        return 1
    return 1 + _count_nodes_py(node.get_left()) + _count_nodes_py(node.get_right())


def _count_selectable_py(node):
    """Mirror of gp_movesC._count_selectable: total nodes minus parameter
    leaves. This is the size_T appearing in subtree_mutation's selection
    density."""
    if node is None:
        return 0
    if node.is_leaf():
        return 0 if node.is_param() else 1
    return 1 + _count_selectable_py(node.get_left()) + \
               _count_selectable_py(node.get_right())


def _count_leaves_py(node):
    """Number of leaf nodes in the subtree rooted at `node`."""
    if node is None:
        return 0
    if node.is_leaf():
        return 1
    return _count_leaves_py(node.get_left()) + _count_leaves_py(node.get_right())


def _count_internal_py(node):
    """Number of internal (non-leaf) nodes in the subtree rooted at `node`."""
    if node is None:
        return 0
    if node.is_leaf():
        return 0
    return 1 + _count_internal_py(node.get_left()) + \
               _count_internal_py(node.get_right())


def _seed_rngs(s):
    """Pin both module RNGs from a single integer seed."""
    gp_movesC.set_rng(random.Random(s), np.random.default_rng(s))


def _make_simple_tree(domain):
    """sin(x) + a0  -> 3 nodes, 1 internal, 1 fn leaf, 1 par leaf."""
    x_sym, a0 = sp.Symbol('x'), sp.Symbol('a0')
    return expr_to_tree(sp.sin(x_sym) + a0, domain, _FUNCS)


def _make_deeper_tree(domain):
    """sin(x) * (cos(x) + a0)  -> 5 nodes, 2 internal, 2 fn leaves, 1 par."""
    x_sym, a0 = sp.Symbol('x'), sp.Symbol('a0')
    return expr_to_tree(sp.sin(x_sym) * (sp.cos(x_sym) + a0), domain, _FUNCS)


def _make_no_param_tree(domain):
    """sin(x) + cos(x) -> 3 nodes, no parameter leaves. Used to force
    point_mutation onto its symmetric branches."""
    x_sym = sp.Symbol('x')
    return expr_to_tree(sp.sin(x_sym) + sp.cos(x_sym), domain, _FUNCS)


def _make_balanced_tree(domain):
    """(sin(x) + cos(x)) * (exp(x) + a0)  -> 7 nodes, 3 internal (root + two
    inner ops), 4 leaves.  Both non-root internal nodes have a sibling that
    is a perfect-binary height-1 subtree, so internal_hoist lands on-support
    (its reverse split is finite)."""
    x_sym, a0 = sp.Symbol('x'), sp.Symbol('a0')
    return expr_to_tree(
        (sp.sin(x_sym) + sp.cos(x_sym)) * (sp.exp(x_sym) + a0),
        domain, _FUNCS)


def test_3_3c_subtree_mutation_density(n_seeds=200):
    """subtree_mutation: lqr - lqf == log(size_T) - log(size_Tp) on support;
    lqr == -inf off support. The selection-density terms are the only ones
    that survive the cancellation under v1.5 parameter recycling."""
    print("[3.3c.subtree_mutation] selection-density identity")
    from real_functionC import _domain_fixer
    x, _ = _make_xy(n=24, seed=11)
    domain = _domain_fixer(x)
    tol = 1e-10

    n_on = 0
    n_off = 0
    for s in range(n_seeds):
        # Mix simple and deeper trees so we hit varied (size_T, size_Tp)
        # pairs and exercise both root-and non-root selections.
        tree = _make_simple_tree(domain) if (s % 2 == 0) else _make_deeper_tree(domain)
        size_T = _count_selectable_py(tree.get_root())

        _seed_rngs(s)
        lqf, lqr = subtree_mutation(tree, _FUNCS)

        size_Tp = _count_selectable_py(tree.get_root())

        # lqf is always finite: it's just -log(size_T).
        expected_lqf = -math.log(size_T)
        assert math.isclose(lqf, expected_lqf, rel_tol=0.0, abs_tol=tol), (
            "[seed=%d] subtree lqf mismatch: got %.15g, expected %.15g "
            "(size_T=%d)" % (s, lqf, expected_lqf, size_T)
        )

        if math.isinf(lqr):
            # Off-support: old subtree was not perfect-binary at d_max.
            assert lqr == -math.inf, "lqr must be -inf on off-support"
            n_off += 1
            continue

        expected_lqr = -math.log(size_Tp)
        assert math.isclose(lqr, expected_lqr, rel_tol=0.0, abs_tol=tol), (
            "[seed=%d] subtree lqr mismatch: got %.15g, expected %.15g "
            "(size_Tp=%d)" % (s, lqr, expected_lqr, size_Tp)
        )

        # Composite identity (the actual MH-relevant quantity).
        expected_diff = math.log(size_T) - math.log(size_Tp)
        assert math.isclose(lqr - lqf, expected_diff,
                            rel_tol=0.0, abs_tol=tol), (
            "[seed=%d] subtree lqr-lqf mismatch: got %.15g, expected %.15g"
            % (s, lqr - lqf, expected_diff)
        )
        n_on += 1

    assert n_on > 0, "no on-support hit in %d seeds" % n_seeds
    print("[3.3c.subtree_mutation] PASS  (on=%d, off=%d)" % (n_on, n_off))


def test_3_3c_add_node_density(n_seeds=200):
    """add_node: paired reverse move is leaf_hoist, whose selection density is
    uniform over the LEAVES of T' (every leaf is non-root).  The added leaf L
    is a leaf of T', so lqr == -log n_leaves(T')."""
    print("[3.3c.add_node] reverse selection density")
    from real_functionC import _domain_fixer
    x, _ = _make_xy(n=24, seed=12)
    domain = _domain_fixer(x)
    tol = 1e-10

    n_checked = 0
    for s in range(n_seeds):
        tree = _make_simple_tree(domain)
        n_before = _count_nodes_py(tree.get_root())

        _seed_rngs(s)
        lqf, lqr = add_node(tree, _FUNCS, _OPS)

        n_after = _count_nodes_py(tree.get_root())
        # add_node grows the tree by at least one internal node.
        assert n_after > n_before, \
            "add_node failed to grow: %d -> %d (seed=%d)" % (n_before, n_after, s)
        assert math.isfinite(lqf), "add_node lqf must be finite"
        assert math.isfinite(lqr), "add_node lqr must be finite (always on-support)"

        n_leaves_after = _count_leaves_py(tree.get_root())
        expected_lqr = -math.log(n_leaves_after)
        assert math.isclose(lqr, expected_lqr, rel_tol=0.0, abs_tol=tol), (
            "[seed=%d] add_node lqr mismatch: got %.15g, expected %.15g "
            "(n_leaves(T')=%d)" % (s, lqr, expected_lqr, n_leaves_after)
        )
        n_checked += 1

    print("[3.3c.add_node] PASS  (%d/%d checked)" % (n_checked, n_seeds))


def test_3_3c_split_density(n_seeds=200):
    """split: paired reverse move is internal_hoist, which selects uniformly
    over the NON-ROOT internal nodes of T'.  The split node u is internal and
    non-root in T', so lqr == -log(n_int(T') - 1)."""
    print("[3.3c.split] reverse selection density")
    from real_functionC import _domain_fixer
    x, _ = _make_xy(n=24, seed=13)
    domain = _domain_fixer(x)
    tol = 1e-10

    n_checked = 0
    for s in range(n_seeds):
        # split needs >=1 internal node to have an internal target;
        # the deeper fixture guarantees that.
        tree = _make_deeper_tree(domain)
        n_before = _count_nodes_py(tree.get_root())

        _seed_rngs(s)
        lqf, lqr = split(tree, _FUNCS, _OPS)

        n_after = _count_nodes_py(tree.get_root())
        assert n_after > n_before, \
            "split failed to grow: %d -> %d (seed=%d)" % (n_before, n_after, s)
        assert math.isfinite(lqf), "split lqf must be finite"
        assert math.isfinite(lqr), "split lqr must be finite"

        n_int_after = _count_internal_py(tree.get_root())
        expected_lqr = -math.log(n_int_after - 1)
        assert math.isclose(lqr, expected_lqr, rel_tol=0.0, abs_tol=tol), (
            "[seed=%d] split lqr mismatch: got %.15g, expected %.15g "
            "(n_int(T')-1=%d)" % (s, lqr, expected_lqr, n_int_after - 1)
        )
        n_checked += 1

    print("[3.3c.split] PASS  (%d/%d checked)" % (n_checked, n_seeds))


def test_3_3c_leaf_hoist_density(n_seeds=300, tol=1e-10):
    """leaf_hoist: forward move picks uniformly over the LEAVES of T (every
    leaf is non-root), so lqf == -log n_leaves(T).  lqr may be -inf when the
    dropped sibling is not a valid `add_node` reverse (off-support)."""
    print("[3.3c.leaf_hoist] forward selection density + off-support handling")
    from real_functionC import _domain_fixer
    x, _ = _make_xy(n=24, seed=14)
    domain = _domain_fixer(x)

    n_on = 0
    n_off = 0
    for s in range(n_seeds):
        tree = _make_deeper_tree(domain)
        n_before = _count_nodes_py(tree.get_root())
        n_leaves_before = _count_leaves_py(tree.get_root())

        _seed_rngs(s)
        lqf, lqr = leaf_hoist(tree, _FUNCS, _OPS)

        n_after = _count_nodes_py(tree.get_root())
        assert n_after < n_before, \
            "leaf_hoist failed to shrink: %d -> %d (seed=%d)" % (n_before, n_after, s)
        assert math.isfinite(lqf), "leaf_hoist lqf must be finite"

        expected_lqf = -math.log(n_leaves_before)
        assert math.isclose(lqf, expected_lqf, rel_tol=0.0, abs_tol=tol), (
            "[seed=%d] leaf_hoist lqf mismatch: got %.15g, expected %.15g "
            "(n_leaves(T)=%d)" % (s, lqf, expected_lqf, n_leaves_before)
        )
        if math.isinf(lqr):
            assert lqr == -math.inf, "leaf_hoist off-support lqr must be -inf"
            n_off += 1
        else:
            n_on += 1

    # The depth-2 fixture exercises both branches: lifting the leaf whose
    # sibling is a height-1 subtree is on-support; lifting a leaf whose
    # sibling is a bare leaf is off-support.
    assert n_on > 0, "leaf_hoist never landed on-support in %d seeds" % n_seeds
    print("[3.3c.leaf_hoist] PASS  (on=%d, off=%d)" % (n_on, n_off))


def test_3_3c_internal_hoist_density(n_seeds=300, tol=1e-10):
    """internal_hoist: forward move picks uniformly over the NON-ROOT
    internal nodes of T, so lqf == -log(n_int(T) - 1).  lqr may be -inf when
    the dropped sibling is not a valid `split` reverse (off-support)."""
    print("[3.3c.internal_hoist] forward selection density + off-support handling")
    from real_functionC import _domain_fixer
    x, _ = _make_xy(n=24, seed=14)
    domain = _domain_fixer(x)

    n_on = 0
    n_off = 0
    for s in range(n_seeds):
        # The balanced fixture has two non-root internal nodes, each with a
        # height-1 perfect-binary sibling, so internal_hoist always fires and
        # lands on-support.
        tree = _make_balanced_tree(domain)
        n_before = _count_nodes_py(tree.get_root())
        n_int_before = _count_internal_py(tree.get_root())

        _seed_rngs(s)
        lqf, lqr = internal_hoist(tree, _FUNCS, _OPS)

        n_after = _count_nodes_py(tree.get_root())
        assert n_after < n_before, \
            "internal_hoist failed to shrink: %d -> %d (seed=%d)" % (n_before, n_after, s)
        assert math.isfinite(lqf), "internal_hoist lqf must be finite"

        expected_lqf = -math.log(n_int_before - 1)
        assert math.isclose(lqf, expected_lqf, rel_tol=0.0, abs_tol=tol), (
            "[seed=%d] internal_hoist lqf mismatch: got %.15g, expected %.15g "
            "(n_int(T)-1=%d)" % (s, lqf, expected_lqf, n_int_before - 1)
        )
        if math.isinf(lqr):
            n_off += 1
        else:
            n_on += 1

    assert n_on > 0, "internal_hoist never landed on-support in %d seeds" % n_seeds
    print("[3.3c.internal_hoist] PASS  (on=%d, off=%d)" % (n_on, n_off))



def test_3_3c_point_mutation_symmetric(n_seeds=200, tol=1e-12):
    """point_mutation: every (lqf, lqr) returned by the move falls into
    exactly one of three regimes:

      (i)   symmetric branch          -> (0, 0) exactly,
      (ii)  BIRTH (function -> param) -> lqf = log N(theta; 0, tau^2),
                                          lqr = -log M,
      (iii) DEATH (param -> function) -> lqf = -log M,
                                          lqr = log N(theta; 0, tau^2).

    Branch (i) covers internal-op resample, function-leaf re-pick (do_cross
    miss), and param-leaf identity (do_cross miss).  Branches (ii) and (iii)
    are the BIRTH/DEATH cases already validated for absolute density in
    test_3_3b_birth_density; here we only need to verify that the (0, 0)
    *symmetric* path is reachable and that no fourth regime exists.
    """
    print("[3.3c.point_mutation] symmetric branch reachability + regime exhaustion")
    from real_functionC import _domain_fixer
    x, _ = _make_xy(n=24, seed=15)
    domain = _domain_fixer(x)

    M = float(len(_FUNCS))
    log_M = math.log(M)
    tau = 0.5

    n_sym = 0
    n_birth = 0
    n_death = 0
    n_other = 0

    # Mix three fixtures so we exercise all three branches at least once:
    fixtures = [_make_simple_tree, _make_deeper_tree, _make_no_param_tree]

    for s in range(n_seeds):
        mk = fixtures[s % len(fixtures)]
        tree = mk(domain)
        tree.set_g_tau(tau)

        _seed_rngs(s)
        lqf, lqr = point_mutation(tree, _FUNCS, _OPS)

        if abs(lqf) < tol and abs(lqr) < tol:
            n_sym += 1
            continue

        # Non-symmetric: must be exactly one of BIRTH/DEATH.
        is_birth = (abs(lqr - (-log_M)) < 1e-10) and (lqf < 0.0) and math.isfinite(lqf)
        is_death = (abs(lqf - (-log_M)) < 1e-10) and (lqr < 0.0) and math.isfinite(lqr)
        if is_birth:
            n_birth += 1
        elif is_death:
            n_death += 1
        else:
            n_other += 1

    assert n_other == 0, \
        "[3.3c.point_mutation] %d unclassified outputs in %d seeds " \
        "(expected only sym/birth/death)" % (n_other, n_seeds)
    assert n_sym > 0, \
        "no symmetric (0,0) branch hit in %d seeds; check _KIND_CROSS_PROB" \
        % n_seeds
    # Don't hard-require BIRTH/DEATH counts here -- (b) covers their
    # density correctness and we use mixed fixtures, so distribution
    # is data-dependent.  Just print for diagnostic value.
    print("[3.3c.point_mutation] PASS  (sym=%d, birth=%d, death=%d, "
          "other=%d) over %d seeds" %
          (n_sym, n_birth, n_death, n_other, n_seeds))


# ---------------------------------------------------------------------------
# (d) BIC scorer unit test
# ---------------------------------------------------------------------------
#
# Reference formula (from real_functionC._log_marginal_bic and
# function_treeC._shortcut_bic):
#
#     log_marginal_bic = -0.5 * N * (log(2*pi*MSE) + 1)  -  0.5 * k * log(N)
#
# where N = len(y), MSE = optimized mean-squared residual, k = #free params.
#
# With sigma fixed (scalar):
#     log_marginal_bic = -0.5 * N * MSE / sigma^2  -  0.5 * k * log(N)
#
# (Note: the optimizer rescales the loss by 1/sigma^2 BEFORE this term, so
# the cached MSE returned to _shortcut_bic is already in sigma units.)
# ---------------------------------------------------------------------------
def _make_linear_xy(n=64, slope=2.0, intercept=3.0, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(-2.0, 2.0, n, dtype=np.float64)
    y = (slope * x + intercept).astype(np.float64)
    y = y + rng.normal(0.0, noise, size=n).astype(np.float64)
    return x, y


def _build_linear_tree(domain):
    """y_hat = a0 * x + a1   (k = 2)."""
    x_sym = sp.Symbol('x')
    a0, a1 = sp.Symbol('a0'), sp.Symbol('a1')
    return expr_to_tree(a0 * x_sym + a1, domain, _FUNCS)


def _build_quadratic_tree(domain):
    """y_hat = a0 * x**2 + a1 * x + a2   (k = 3).  Linear data is a
    nested submodel, so the optimum quadratic MSE <= optimum linear MSE."""
    x_sym = sp.Symbol('x')
    a0, a1, a2 = sp.Symbol('a0'), sp.Symbol('a1'), sp.Symbol('a2')
    return expr_to_tree(a0 * x_sym**2 + a1 * x_sym + a2, domain, _FUNCS)


def _count_ets_py(node):
    """Pure-Python mirror of gp_movesC._collect_ets count: every leaf is a
    0-ET; every internal node whose BOTH children are leaves is a 2-ET; nested
    ETs are all counted."""
    if node is None:
        return 0
    if node.is_leaf():
        return 1
    left = node.get_left()
    right = node.get_right()
    c = 0
    if (left is not None and right is not None
            and left.is_leaf() and right.is_leaf()):
        c = 1
    return c + _count_ets_py(left) + _count_ets_py(right)


def _param_values_py(node, out):
    """Collect parameter-leaf values in the subtree rooted at `node`."""
    if node is None:
        return out
    if node.is_leaf():
        if node.is_param():
            out.append(float(node.value))
        return out
    _param_values_py(node.get_left(), out)
    _param_values_py(node.get_right(), out)
    return out


def test_3_3c_etr_density(n_seeds=400, tol=1e-9):
    """etr_mutation (Phase 2.11): exact reconstruction of BOTH proposal
    densities from a SINGLE-LEAF start.

    A single-leaf tree has exactly one elementary tree (the root leaf), so it
    is always the selected ET (Omega_i = 1, o_i = 0) and the WHOLE tree is
    replaced -- making every density component observable from the after-tree:

        N0 = M + 1 ; N2 = O * N0 * N0 ; s(a) = N0 if a == 0 else N2
        log_q_fwd = -log 1       - log 2 - log s(o_f) + sum_new log g(theta_new)
        log_q_rev = -log Omega_f - log 2 - log s(0)   + 0   (old leaf: no param)

    o_f and Omega_f are read from the after-tree, theta_new from its parameter
    leaves. Both replacement arities (o_f in {0, 2}) must occur.
    """
    print("[3.3c.etr] single-leaf exact density reconstruction")
    from real_functionC import _domain_fixer
    x, _ = _make_xy(n=24, seed=21)
    domain = _domain_fixer(x)

    M = len(_FUNCS)
    O = len(_OPS)
    N0 = M + 1
    N2 = O * N0 * N0
    x_sym = sp.Symbol('x')

    n_leaf_out = 0   # o_f == 0
    n_2et_out = 0    # o_f == 2
    for s in range(n_seeds):
        # Fresh single FUNCTION leaf: o_i = 0, Omega_i = 1, no old parameter.
        tree = expr_to_tree(sp.sin(x_sym), domain, _FUNCS)
        assert tree.get_root().is_leaf(), "fixture must be a single leaf"
        if hasattr(tree, "set_g_tau"):
            tree.set_g_tau(0.7)
        g_tau = tree.g_tau

        _seed_rngs(s)
        lqf, lqr = gp_movesC.etr_mutation(tree, _FUNCS, _OPS)

        root = tree.get_root()
        o_f = 0 if root.is_leaf() else 2
        omega_f = _count_ets_py(root)
        thetas_new = _param_values_py(root, [])
        log_g_new = sum(_log_g(t, g_tau) for t in thetas_new)

        s_f = N0 if o_f == 0 else N2
        expected_lqf = -math.log(1.0) - math.log(2.0) - math.log(s_f) + log_g_new
        expected_lqr = -math.log(omega_f) - math.log(2.0) - math.log(N0)

        assert math.isfinite(lqf) and math.isfinite(lqr), \
            "[seed=%d] etr densities must be finite on-support" % s
        assert math.isclose(lqf, expected_lqf, rel_tol=0.0, abs_tol=tol), (
            "[seed=%d] etr lqf mismatch: got %.15g expected %.15g "
            "(o_f=%d, s_f=%d, log_g_new=%.6g)"
            % (s, lqf, expected_lqf, o_f, s_f, log_g_new))
        assert math.isclose(lqr, expected_lqr, rel_tol=0.0, abs_tol=tol), (
            "[seed=%d] etr lqr mismatch: got %.15g expected %.15g "
            "(omega_f=%d)" % (s, lqr, expected_lqr, omega_f))
        if o_f == 0:
            n_leaf_out += 1
        else:
            n_2et_out += 1

    assert n_leaf_out > 0 and n_2et_out > 0, (
        "etr density test did not exercise BOTH replacement arities "
        "(leaf=%d, 2et=%d)" % (n_leaf_out, n_2et_out))
    print("[3.3c.etr] PASS  reconstructed lqf/lqr over %d seeds "
          "(o_f=0: %d, o_f=2: %d)" % (n_seeds, n_leaf_out, n_2et_out))


def test_3_3c_etr_offsupport(n_seeds=300):
    """etr_mutation off-support: a 0-ET -> 2-ET replacement that would exceed
    the depth cap is force-rejected with lqr == -inf (never a finite-but-wrong
    density). Uses a temporarily lowered _MAX_TREE_DEPTH (save/restore)."""
    print("[3.3c.etr] depth-cap force-reject -> lqr == -inf")
    from real_functionC import _domain_fixer
    x, _ = _make_xy(n=24, seed=22)
    domain = _domain_fixer(x)

    _set_depth = getattr(gp_movesC, "set_max_tree_depth", None)
    _get_depth = getattr(gp_movesC, "get_max_tree_depth", None)
    if _set_depth is None or _get_depth is None:
        print("[3.3c.etr] SKIP off-support (build predates set_max_tree_depth)")
        return
    _old = _get_depth()
    n_off = 0
    n_on = 0
    try:
        _set_depth(2)   # start tree is depth 2 (at cap); any leaf->2ET busts it
        for s in range(n_seeds):
            tree = _make_balanced_tree(domain)   # every leaf sits at the cap
            _seed_rngs(s)
            lqf, lqr = gp_movesC.etr_mutation(tree, _FUNCS, _OPS)
            if math.isinf(lqr):
                assert lqr == -math.inf, "off-support lqr must be exactly -inf"
                n_off += 1
            else:
                assert math.isfinite(lqr), "on-support lqr must be finite"
                n_on += 1
    finally:
        _set_depth(_old)
    assert n_off > 0, "no off-support (depth-cap) rejections were triggered"
    print("[3.3c.etr] PASS  (off=%d, on=%d)" % (n_off, n_on))


def _build_cubic_tree(domain):
    """y_hat = a0 * x**3 + a1 * x**2 + a2 * x + a3   (k = 4)."""
    x_sym = sp.Symbol('x')
    a0, a1, a2, a3 = (sp.Symbol('a0'), sp.Symbol('a1'),
                      sp.Symbol('a2'), sp.Symbol('a3'))
    return expr_to_tree(a0 * x_sym**3 + a1 * x_sym**2 + a2 * x_sym + a3,
                        domain, _FUNCS)


def test_3_3d_bic_finite_on_linear():
    """log_marginal_bic is finite on a well-posed (model, data) pair."""
    print("[3.3d.finite] log_marginal_bic on y=2x+3 + noise")
    from real_functionC import _domain_fixer
    x, y = _make_linear_xy(n=64, slope=2.0, intercept=3.0, noise=0.05, seed=10)
    domain = _domain_fixer(x)

    tree = _build_linear_tree(domain)
    # Default sigma path (sigma=1.0 scalar OR sigma=None) -- both must
    # produce a finite scalar.
    score_sigma_default = tree.log_marginal_bic(y)
    assert np.isfinite(score_sigma_default), \
        "default-sigma log_marginal_bic non-finite: %r" % score_sigma_default

    tree2 = _build_linear_tree(domain)
    tree2.set_sigma(None)
    score_sigma_none = tree2.log_marginal_bic(y)
    assert np.isfinite(score_sigma_none), \
        "sigma=None log_marginal_bic non-finite: %r" % score_sigma_none

    print("[3.3d.finite] PASS  (sigma=1.0: %.6g, sigma=None: %.6g)"
          % (score_sigma_default, score_sigma_none))


def test_3_3d_bic_penalises_complexity():
    """More parameters at (approximately) equal MSE -> lower log_evidence.

    Linear data is nested inside the quadratic and cubic families, so the
    quadratic optimum MSE is <= the linear optimum MSE, and similarly for
    cubic vs quadratic.  Any improvement is bounded by the noise budget
    (the linear model is already the true generator), so the
    -0.5*log(N) per-extra-parameter penalty dominates and BIC must
    strictly decrease as k grows.
    """
    print("[3.3d.penalty] complexity penalty: linear vs quadratic vs cubic")
    from real_functionC import _domain_fixer
    x, y = _make_linear_xy(n=128, slope=2.0, intercept=3.0, noise=0.05, seed=11)
    domain = _domain_fixer(x)
    N = len(y)

    t_lin = _build_linear_tree(domain)
    t_quad = _build_quadratic_tree(domain)
    t_cub = _build_cubic_tree(domain)

    s_lin = t_lin.log_marginal_bic(y)
    s_quad = t_quad.log_marginal_bic(y)
    s_cub = t_cub.log_marginal_bic(y)
    for label, s in (("linear", s_lin), ("quadratic", s_quad), ("cubic", s_cub)):
        assert np.isfinite(s), "%s BIC non-finite: %r" % (label, s)

    # Strict ordering: BIC(linear) > BIC(quadratic) > BIC(cubic).
    # We allow a tiny slack for optimizer noise (worst-case the per-step
    # penalty is 0.5*log(N) ~ 2.4 nats for N=128, so 1e-3 slack is safe).
    slack = 1e-3
    assert s_lin > s_quad - slack, (
        "expected BIC(lin) > BIC(quad): %.6g vs %.6g" % (s_lin, s_quad)
    )
    assert s_quad > s_cub - slack, (
        "expected BIC(quad) > BIC(cub): %.6g vs %.6g" % (s_quad, s_cub)
    )

    # The strict inequalities should hold at the level of the penalty,
    # 0.5*log(N) per added parameter, modulo small MSE improvement.
    penalty = 0.5 * math.log(N)
    drop_lq = s_lin - s_quad
    drop_qc = s_quad - s_cub
    print("[3.3d.penalty] PASS  (lin=%.4g, quad=%.4g, cub=%.4g; "
          "drop_lq=%.4g, drop_qc=%.4g, 0.5*log(N)=%.4g)"
          % (s_lin, s_quad, s_cub, drop_lq, drop_qc, penalty))


def test_3_3d_bic_sigma_modes_differ(tol=1e-6):
    """sigma=None vs sigma=fixed scalar produce different but finite
    scores. The two code paths use distinct formulas:

      sigma=None  : -0.5*N*(log(2*pi*MSE) + 1) - 0.5*k*log(N)
      sigma=s>0   : -0.5*N*(MSE/s^2)           - 0.5*k*log(N)
                    (after the optimizer's MSE has been divided by s^2,
                     via _shortcut_bic).

    Two distinct sigma values must in turn give two distinct scores.
    """
    print("[3.3d.sigma] sigma=None vs sigma=fixed produce different scores")
    from real_functionC import _domain_fixer
    x, y = _make_linear_xy(n=96, slope=2.0, intercept=3.0, noise=0.10, seed=12)
    domain = _domain_fixer(x)

    t_none = _build_linear_tree(domain)
    t_none.set_sigma(None)
    s_none = t_none.log_marginal_bic(y)

    t_s1 = _build_linear_tree(domain)
    t_s1.set_sigma(0.10)
    s_s1 = t_s1.log_marginal_bic(y)

    t_s2 = _build_linear_tree(domain)
    t_s2.set_sigma(0.20)
    s_s2 = t_s2.log_marginal_bic(y)
    print(t_s1.get_sigma(), t_s2.get_sigma())
    for label, s in (("sigma=None", s_none), ("sigma=0.10", s_s1),
                     ("sigma=0.20", s_s2)):
        assert np.isfinite(s), "%s BIC non-finite: %r" % (label, s)
    print(s_none, s_s1, s_s2)
    assert abs(s_none - s_s1) > tol, \
        "sigma=None and sigma=0.10 produced identical scores: %.15g" % s_none
    assert abs(s_s1 - s_s2) > tol, \
        "sigma=0.10 and sigma=0.20 produced identical scores: %.15g" % s_s1

    print("[3.3d.sigma] PASS  (sigma=None: %.6g, sigma=0.10: %.6g, "
          "sigma=0.20: %.6g)" % (s_none, s_s1, s_s2))


def test_3_3d_bic_formula_consistency(tol=1e-6):
    """Reconstruct the BIC value from (N, k, MSE) and compare with what
    Tree.log_marginal_bic returns. This pins the formula -- if anyone
    rewrites the scorer with a different convention (e.g. a different
    sign or a +log term), the equality below will catch it.
    """
    print("[3.3d.formula] reconstruct BIC from (N, k, MSE)")
    from real_functionC import _domain_fixer
    x, y = _make_linear_xy(n=80, slope=2.0, intercept=3.0, noise=0.07, seed=13)
    domain = _domain_fixer(x)
    N = len(y)

    # sigma=None branch: use root._log_marginal_bic directly via the
    # high-level entry point, then recompute the formula by hand.
    tree = _build_linear_tree(domain)
    tree.set_sigma(None)
    score = tree.log_marginal_bic(y)
    k = len(tree.get_param_dicts()[1])
    assert k == 2, "linear tree should have k=2 params, got %d" % k

    # Reproduce the MSE from the optimized tree.  log_marginal_bic
    # leaves the tree at its optimum, so name_values now hold the best fit.
    y_pred = tree.get_root().get_fun().get_eval(domain, tree.get_param_dicts()[1]).astype(np.float64)
    mse = float(np.mean((y - y_pred) ** 2))
    assert math.isfinite(mse) and mse > 0.0, "bad MSE: %r" % mse

    expected = (-0.5 * N * (math.log(2.0 * math.pi * mse) + 1.0)
                - 0.5 * k * math.log(N))
    print(expected, mse, score)
    assert math.isclose(score, expected, rel_tol=tol, abs_tol=tol), (
        "BIC formula mismatch: got %.10g, expected %.10g (N=%d, k=%d, MSE=%.6g)"
        % (score, expected, N, k, mse)
    )
    print("[3.3d.formula] PASS  (score=%.6g, hand-computed=%.6g, MSE=%.6g)"
          % (score, expected, mse))


# ---------------------------------------------------------------------------
# (e) Metropolis-Hastings acceptance rule unit test
# ---------------------------------------------------------------------------
#
# The structural-move acceptance rule is implemented inline inside
# Grove._propose_* (groveC.pyx). It is not exposed as a Python-callable
# function, so we mirror it here as a reference and check the three
# properties listed in PLAN.md (3.3(e)).
#
# Reference rule (lifted verbatim from groveC.pyx ~ lines 580-625):
#
#   1. If lqr is -inf (or any non-finite): hard reject (upstream guard).
#   2. Compute  dtarget = (new_post + new_prior) - (old_post + old_prior),
#               q_corr  = lqr - lqf.
#   3. If T <= 0  (greedy):
#        - dtarget > 0                       -> log_alpha = 0     (accept)
#        - dtarget == 0  and q_corr >= 0     -> log_alpha = 0     (accept)
#        - else                              -> log_alpha = -inf  (reject)
#      If T > 0:
#        log_alpha = dtarget / T + q_corr
#   4. Decision:
#        log_alpha >= 0       -> accept
#        log_alpha == -inf    -> reject
#        else                 -> sample u in (0,1); accept iff log(u) < log_alpha.
# ---------------------------------------------------------------------------
def _mh_decide(dtarget, lqf, lqr, T, u):
    """Pure-Python mirror of Grove's MH structural-move rule.

    Returns
    -------
    (accepted, log_alpha) : (bool, float)
        log_alpha is the saturated MH log-acceptance ratio used to decide.
    """
    # Upstream guard inside Grove._propose_*: if lqr is not finite, the
    # structural step short-circuits to reject before MH is even called.
    if not math.isfinite(lqr):
        return False, -math.inf

    q_corr = lqr - lqf
    if T <= 0.0:
        if dtarget > 0.0:
            log_alpha = 0.0
        elif dtarget == 0.0 and q_corr >= 0.0:
            log_alpha = 0.0
        else:
            log_alpha = -math.inf
    else:
        log_alpha = dtarget / T + q_corr

    if log_alpha >= 0.0:
        return True, log_alpha
    if log_alpha == -math.inf:
        return False, log_alpha
    accepted = (u <= 0.0) or (math.log(u) < log_alpha)
    return accepted, log_alpha


def test_3_3e_mh_always_accept_when_target_dominates(n_u=50):
    """dtarget >> 0 and q_corr >= 0 -> log_alpha >= 0 -> always accepted,
    independently of u in [0, 1).  Covers the 'free lunch' regime where
    both target and Jacobian point the same way."""
    print("[3.3e.always_accept] dtarget >> 0, q_corr >= 0")
    rng = np.random.default_rng(31)
    us = rng.uniform(0.0, 1.0, size=n_u)

    dtarget = 1.0e6      # arbitrarily large positive target shift
    for T in (0.0, 1.0, 5.0):
        for q_corr_value in (0.0, 0.5, 10.0):
            lqf = 0.0
            lqr = q_corr_value  # so lqr - lqf = q_corr_value >= 0
            for u in us:
                accepted, la = _mh_decide(dtarget, lqf, lqr, T, float(u))
                assert accepted, (
                    "expected accept at T=%g, q_corr=%g, u=%g; got "
                    "log_alpha=%.6g" % (T, q_corr_value, u, la)
                )
                assert la >= 0.0, "log_alpha must saturate >= 0; got %.6g" % la
    print("[3.3e.always_accept] PASS  (T in {0, 1, 5}, q_corr in {0, 0.5, 10})")


def test_3_3e_mh_always_reject_when_lqr_minus_inf(n_u=50):
    """lqr == -inf is the upstream off-support signal: the structural
    step rejects before the MH ratio is even formed.  The condition must
    hold for every (T, dtarget, lqf, u)."""
    print("[3.3e.always_reject] lqr == -inf -> hard reject")
    rng = np.random.default_rng(32)
    us = rng.uniform(0.0, 1.0, size=n_u)
    lqr = -math.inf

    for T in (0.0, 0.5, 1.0, 5.0):
        for dtarget in (-1e6, -1.0, 0.0, 1.0, 1e6):
            for lqf in (-3.0, 0.0, 4.5):
                for u in us:
                    accepted, la = _mh_decide(dtarget, lqf, lqr, T, float(u))
                    assert not accepted, (
                        "expected reject at T=%g, dtarget=%g, lqf=%g, u=%g"
                        % (T, dtarget, lqf, u)
                    )
                    assert la == -math.inf, \
                        "log_alpha must be -inf; got %.6g" % la
    print("[3.3e.always_reject] PASS  (any T, any dtarget, any lqf)")


def test_3_3e_mh_greedy_at_zero_temperature(n_u=50):
    """T == 0 (greedy): accept iff dtarget > 0, independent of u.
    Tie-breaker: dtarget == 0 accepts iff q_corr >= 0.
    This is the ablation knob used to exercise the deterministic limit
    of the kernel."""
    print("[3.3e.greedy] T == 0 greedy regime")
    rng = np.random.default_rng(33)
    us = rng.uniform(0.0, 1.0, size=n_u)

    # Non-tie regime: q_corr is irrelevant.
    for dtarget in (-5.0, -1e-9, 1e-9, 5.0):
        expected = dtarget > 0.0
        for q_corr_value in (-10.0, -1.0, 0.0, 1.0, 10.0):
            lqf = 0.0
            lqr = q_corr_value
            for u in us:
                accepted, la = _mh_decide(dtarget, lqf, lqr, 0.0, float(u))
                assert accepted == expected, (
                    "T=0 greedy decision wrong at dtarget=%g, q_corr=%g: "
                    "got %s, expected %s" % (dtarget, q_corr_value,
                                             accepted, expected)
                )

    # Tie regime: dtarget == 0 -> accept iff q_corr >= 0.
    for q_corr_value in (-1.0, -1e-12, 0.0, 1e-12, 1.0):
        expected = q_corr_value >= 0.0
        lqf = 0.0
        lqr = q_corr_value
        for u in us:
            accepted, la = _mh_decide(0.0, lqf, lqr, 0.0, float(u))
            assert accepted == expected, (
                "T=0 tie decision wrong at q_corr=%g: got %s, expected %s"
                % (q_corr_value, accepted, expected)
            )
    print("[3.3e.greedy] PASS")


def test_3_3e_mh_random_branch_consistency(n_seeds=2000, tol=0.05):
    """At T > 0 with finite dtarget and q_corr giving log_alpha < 0, the
    accept rate over uniform u must approach exp(log_alpha).  This pins
    the random branch of the rule (the actual MCMC kernel needs this to
    obey detailed balance)."""
    print("[3.3e.stochastic] empirical accept rate matches exp(log_alpha)")
    rng = np.random.default_rng(34)

    # Choose (T, dtarget, q_corr) so log_alpha is a known negative number.
    T = 1.0
    dtarget = -1.0
    lqf = 0.0
    lqr = -0.5             # q_corr = -0.5
    expected_log_alpha = dtarget / T + (lqr - lqf)  # -1.5
    expected_rate = math.exp(expected_log_alpha)    # ~0.2231

    accepted = 0
    for _ in range(n_seeds):
        u = float(rng.uniform(0.0, 1.0))
        ok, la = _mh_decide(dtarget, lqf, lqr, T, u)
        assert math.isclose(la, expected_log_alpha, rel_tol=0.0, abs_tol=1e-12)
        if ok:
            accepted += 1

    empirical_rate = accepted / n_seeds
    assert abs(empirical_rate - expected_rate) < tol, (
        "stochastic accept rate off: got %.4f, expected %.4f (tol=%.3f)"
        % (empirical_rate, expected_rate, tol)
    )
    print("[3.3e.stochastic] PASS  (rate=%.4f, expected=%.4f)"
          % (empirical_rate, expected_rate))


# ---------------------------------------------------------------------------
# (f) Grove integration test
# ---------------------------------------------------------------------------
#
# Hooks all earlier 3.3 layers together: per-move densities (c), BIC scorer
# (d), and the MH rule (e) must compose into a Grove kernel that:
#   - Runs without raising on a realistic target (y = sin(x) + noise).
#   - Lands in a sensible acceptance band (PLAN.md: [0.05, 0.95]).
#   - Exercises every *structural* move type at least once
#     (parameter_refresh fires only with pr_prob=0.02 per step, so it
#      may legitimately miss inside 200 steps -- we report it but do
#      not assert).
#   - Has a finite log_evidence on the returned best_tree.
#   - Is fully deterministic across two runs from the same seed
#     (this re-uses the (a) check on a longer trajectory).
# ---------------------------------------------------------------------------
def _grove_run_state(g):
    """Snapshot of grove state used for cross-run equality."""
    tops = [t.tree_get_description() for t in g.trees]
    props = dict(g.move_proposals)
    accs = dict(g.move_accepts)
    posts = np.asarray(g.root_posteriors, dtype=np.float64).copy()
    return tops, props, accs, posts


def _make_sin_grove_xy(n=64, noise=0.05, seed=0):
    """y = sin(x) + small noise on x in [-2, 2]."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-2.0, 2.0, n, dtype=np.float64)
    y = np.sin(x).astype(np.float64) + rng.normal(0.0, noise, size=n).astype(np.float64)
    return x, y


def test_3_3f_grove_integration(n_steps=200, seed=20260610):
    """End-to-end Grove run on y = sin(x) + noise."""
    print("[3.3f.integration] Grove on y=sin(x), n_steps=%d, seed=%d"
          % (n_steps, seed))

    x, y = _make_sin_grove_xy(n=64, noise=0.05, seed=seed)

    # size=4 trees, depth=2 -- matches the (a) determinism fixture so the
    # cross-run equality check is on identical territory.
    g = _make_grove(x, y, seed=seed, g_tau=None, size=4, depth=2)
    g.run(y, n_steps=n_steps)

    # --- 1. Acceptance rate band -------------------------------------------
    rate = float(g.acceptance_rate())
    assert 0.05 <= rate <= 0.95, (
        "acceptance rate out of PLAN-specified band [0.05, 0.95]: %.4f "
        "(proposals=%d, accepts=%d)"
        % (rate, int(g.proposal_count), int(g.acceptance_count))
    )

    # --- 2. Every structural move was proposed at least once ---------------
    stats = g.move_stats()
    structural = ["point_mutation", "leaf_hoist", "internal_hoist", "add_node",
                  "split", "subtree_mutation", "subtree_crossover"]
    for name in structural:
        assert name in stats, "move_stats missing %r" % name
        assert stats[name]["proposed"] > 0, (
            "structural move %r had zero proposals in %d steps; check "
            "_choose_move probabilities" % (name, n_steps)
        )

    # parameter_refresh is informational: pr_prob=0.02 per step means
    # E[count] ~ 4 over 200 steps, so absence is not a failure.
    pr_proposed = stats.get("parameter_refresh", {}).get("proposed", 0)

    # --- 3. best_tree() has a finite log_evidence --------------------------
    best_post = float(g.root_posteriors[g.best_index()])
    assert math.isfinite(best_post), \
        "best_tree() root posterior is not finite: %r" % best_post
    best = g.best_tree()
    assert best is not None, "best_tree() returned None"
    assert best.is_valid(g.get_domain()), "best_tree() is not numerically valid"

    # --- 4. Determinism over the longer trajectory -------------------------
    g2 = _make_grove(x, y, seed=seed, g_tau=None, size=4, depth=2)
    g2.run(y, n_steps=n_steps)

    t1, p1, a1, post1 = _grove_run_state(g)
    t2, p2, a2, post2 = _grove_run_state(g2)
    assert t1 == t2, \
        "tree topologies differ across same-seed runs:\n  g1=%s\n  g2=%s" % (t1, t2)
    assert p1 == p2, "move_proposals differ across same-seed runs: %r vs %r" % (p1, p2)
    assert a1 == a2, "move_accepts differ across same-seed runs: %r vs %r" % (a1, a2)
    assert np.allclose(post1, post2, atol=1e-12, equal_nan=True), \
        "root_posteriors differ across same-seed runs:\n  %s\n  %s" % (post1, post2)

    print("[3.3f.integration] PASS  (rate=%.3f, best_post=%.3f, "
          "PR_proposed=%d, structural_proposals=%s)"
          % (rate, best_post, pr_proposed,
             {n: stats[n]["proposed"] for n in structural}))


# ---------------------------------------------------------------------------
# 3.3(g) Polya prior unit test
# ---------------------------------------------------------------------------
# These checks pin down three contracts of `PolyaTree`:
#   1) `log_prior_from_tree` matches the analytic Dirichlet-Multinomial
#      predictive value for a hand-built tree on a freshly-initialised
#      Polya tree (all alphas = alpha_0, no children yet -> every step
#      contributes -log(n_total)).
#   2) Driving accepted MCMC moves through a Grove causes the Polya state
#      to evolve (`polya.t` advances, alphas leave the symmetric prior),
#      and `log_prior_from_tree` returns a value different from what a
#      freshly-initialised Polya would give on the same tree.
#   3) With `auto_register=True`, encountering a primitive that was not
#      pre-registered grows the Polya vocabulary in-place and yields a
#      finite log_prior. With `auto_register=False`, the same call raises.
# ---------------------------------------------------------------------------

_POLYA_OPS_FIXED = ["+", "-", "*", "/", "o", "**"]  # PolyaTree hard-codes these


def _polya_path_from_tree_py(polya, root_node):
    """Pure-Python pre-order DFS yielding the same (kind, idx) path as
    PolyaTree._extract_recursive, without registering unknown leaves.

    Used by the analytic test to predict the expected log_prior. Mirrors
    polya_treeC.pyx semantics: unknown ops raise (we never call this with
    unknown ops); unknown primitives are skipped (this helper is only used
    when the vocab is known to cover the tree).
    """
    out = []
    op_idx = polya._op_to_idx
    prim_idx = polya._prim_to_idx

    def visit(node):
        if node is None:
            return
        if node.is_leaf():
            desc = node.get_fun().get_description()
            j = prim_idx.get(desc, -1)
            if j >= 0:
                out.append(("prim", j))
            return
        op = node.get_op()
        i = op_idx.get(op, -1)
        if i >= 0:
            out.append(("op", i))
        visit(node.get_left())
        visit(node.get_right())

    visit(root_node)
    return out


def test_3_3g_polya_log_prior_analytic():
    """Fresh PolyaTree + hand-built tree: log_prior matches the analytic
    Dirichlet-Multinomial predictive.

    On a fresh tree, every node has alphas = alpha_0 * ones(n_total), so
    each visited step contributes log(alpha_0 / (n_total * alpha_0))
    = -log(n_total). Unvisited subregions (no children created yet) also
    fall back to -log(n_total). Hence log_prior = -|path| * log(n_total).
    """
    domain = np.linspace(0.001, 10.0, 64).astype(np.float64)
    functions = ["sin(x)", "cos(x)", "x"]
    polya = PolyaTree(_POLYA_OPS_FIXED, functions, alpha_0=1.0,
                      auto_register=False, strict_ops=True)

    # Tree:  sin(x) + cos(x)
    #   pre-order path: [('op', '+'), ('prim', 'sin(x)'), ('prim', 'cos(x)')]
    x_sym = sp.Symbol('x')
    tree = expr_to_tree(sp.sin(x_sym) + sp.cos(x_sym), domain, functions)

    path = polya.extract_path(tree.get_root())
    expected_path = _polya_path_from_tree_py(polya, tree.get_root())
    assert path == expected_path, (
        "extract_path mismatch:\n  C: %r\n  Py: %r" % (path, expected_path))

    # Spot-check: the path has length 3 ('+' / 'sin(x)' / 'cos(x)').
    assert len(path) == 3, "expected 3-step path, got %r" % (path,)

    n_total = polya.n_ops + polya.n_prims  # 6 + 3 = 9
    expected_lp = -len(path) * math.log(n_total)
    got_lp = polya.log_prior_from_tree(tree.get_root())
    assert math.isfinite(got_lp), "log_prior_from_tree is not finite: %r" % got_lp
    assert abs(got_lp - expected_lp) < 1e-12, (
        "log_prior_from_tree mismatch: got %.12g, expected %.12g (= -%d * log %d)"
        % (got_lp, expected_lp, len(path), n_total))

    # log_prior is READ-ONLY: calling it twice gives the same value, and
    # polya.t stays at 0 (no updates triggered).
    got_lp2 = polya.log_prior_from_tree(tree.get_root())
    assert got_lp == got_lp2, "log_prior_from_tree is not deterministic"
    assert polya.t == 0, "log_prior_from_tree must not advance polya.t"

    print("[3.3g.analytic] PASS  log_prior=%.6f  expected=%.6f  n_total=%d"
          % (got_lp, expected_lp, n_total))


def test_3_3g_polya_update_changes_prior(n_steps=40, seed=20260610):
    """Drive a short Grove run; verify update_path mutates Polya state.

    Contract: every accepted structural MH step calls polya.update_path(),
    which (a) reinforces the visited path's alphas with gamma_t=1/(1+t),
    and (b) increments polya.t. Therefore the post-run log_prior of a
    surviving tree must differ from the prior a freshly-initialised Polya
    would assign to that same tree.
    """
    x, y = _make_xy(n=64, seed=seed)

    functions = ["sin(x)", "cos(x)", "x"]
    operations = ["+", "-", "*", "/"]

    polya = PolyaTree(_POLYA_OPS_FIXED, functions, alpha_0=1.0,
                      auto_register=True, strict_ops=True)

    random.seed(seed)
    np.random.seed(seed)
    g = Grove(x, y, T=1.0, size=2, depth=2, funcs=functions,
              operations=operations, Npars=2, init_value=1.0,
              base_temp=1.0, use_gradients=False, burn_in=False,
              use_mpi=0, use_spawn=False, lambda_reg=0.0,
              polya=polya, seed=seed)

    g.run(y, n_steps=n_steps)

    # Some accepted move must have called update_path. On y=sin(x)+noise
    # with size=2 / depth=2 trees this is essentially guaranteed in 40
    # steps; if it ever became flaky we'd raise the step count.
    assert polya.t > 0, (
        "Polya.t did not advance over %d steps; update_path was never called"
        % n_steps)

    # Alpha mass on at least one root slot must have moved off alpha_0=1.
    root_alphas = np.asarray(polya.root.alphas, dtype=np.float64)
    assert np.any(np.abs(root_alphas - 1.0) > 1e-12), (
        "Polya root alphas unchanged after %d updates: %r" % (polya.t, root_alphas))

    # Compare log_prior on a surviving tree against a fresh Polya: they
    # must differ, since the trained Polya has non-uniform alphas along
    # the visited subtree.
    survivor = g.best_tree()
    fresh = PolyaTree(_POLYA_OPS_FIXED, functions, alpha_0=1.0,
                      auto_register=True, strict_ops=True)
    lp_trained = polya.log_prior_from_tree(survivor.get_root())
    lp_fresh = fresh.log_prior_from_tree(survivor.get_root())
    assert math.isfinite(lp_trained) and math.isfinite(lp_fresh), \
        "log_prior must be finite after update (got trained=%r, fresh=%r)" \
        % (lp_trained, lp_fresh)
    assert abs(lp_trained - lp_fresh) > 1e-9, (
        "log_prior did not change after update_path: trained=%.12g fresh=%.12g"
        % (lp_trained, lp_fresh))

    print("[3.3g.update] PASS  polya.t=%d  lp_trained=%.4f  lp_fresh=%.4f  "
          "delta=%+.4f" % (polya.t, lp_trained, lp_fresh, lp_trained - lp_fresh))


def test_3_3g_polya_auto_register_unknown_leaf():
    """auto_register=True: encountering a primitive outside the initial
    vocabulary grows the vocab, returns a finite log_prior, and does NOT
    raise. auto_register=False: same call raises KeyError.
    """
    domain = np.linspace(0.001, 10.0, 32).astype(np.float64)
    functions = ["sin(x)", "cos(x)", "x"]

    # Build a tree that uses cos(x); we'll create a Polya with primitives
    # = ['x'] only so cos(x) and sin(x) are both unknown.
    x_sym = sp.Symbol('x')
    tree = expr_to_tree(sp.sin(x_sym) + sp.cos(x_sym), domain, functions)
    root = tree.get_root()

    # --- auto_register=True: silent vocab growth -------------------------
    polya_auto = PolyaTree(_POLYA_OPS_FIXED, ["x"], alpha_0=1.0,
                           auto_register=True, strict_ops=True)
    n_prims_before = polya_auto.n_prims
    root_n_choices_before = polya_auto.root.n_choices
    assert n_prims_before == 1 and "sin(x)" not in polya_auto.primitives, \
        "fixture broken: expected vocab=['x']"

    lp = polya_auto.log_prior_from_tree(root)
    assert math.isfinite(lp), "auto-register path produced non-finite log_prior: %r" % lp

    # Both sin(x) and cos(x) should have been registered.
    assert "sin(x)" in polya_auto.primitives, \
        "sin(x) was not auto-registered: %r" % polya_auto.primitives
    assert "cos(x)" in polya_auto.primitives, \
        "cos(x) was not auto-registered: %r" % polya_auto.primitives
    assert polya_auto.n_prims == n_prims_before + 2, (
        "n_prims should have grown by 2, got %d -> %d"
        % (n_prims_before, polya_auto.n_prims))

    # _grow_all_nodes must have extended every Dirichlet, so root.alphas
    # now has +2 slots, all initialised to alpha_0.
    assert polya_auto.root.n_choices == root_n_choices_before + 2, (
        "root.n_choices not grown: %d -> %d"
        % (root_n_choices_before, polya_auto.root.n_choices))
    assert polya_auto.root.alphas.shape[0] == polya_auto.root.n_choices, \
        "alphas length out of sync with n_choices"

    # --- auto_register=False: must raise KeyError ------------------------
    polya_strict = PolyaTree(_POLYA_OPS_FIXED, ["x"], alpha_0=1.0,
                             auto_register=False, strict_ops=True)
    raised = False
    try:
        polya_strict.log_prior_from_tree(root)
    except KeyError:
        raised = True
    assert raised, "auto_register=False should raise KeyError on unknown leaf"

    print("[3.3g.auto_register] PASS  vocab %d -> %d, lp=%.4f"
          % (n_prims_before, polya_auto.n_prims, lp))


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # OpenMP / BLAS threading affects bit-equality on some routines.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    test_3_3a_determinism()
    test_3_3b_g_tau_plumbing()
    test_3_3b_birth_density()
    test_3_3c_subtree_mutation_density()
    test_3_3c_add_node_density()
    test_3_3c_split_density()
    test_3_3c_leaf_hoist_density()
    test_3_3c_internal_hoist_density()
    test_3_3c_point_mutation_symmetric()
    test_3_3c_etr_density()
    test_3_3c_etr_offsupport()
    test_3_3d_bic_finite_on_linear()
    test_3_3d_bic_penalises_complexity()
    test_3_3d_bic_sigma_modes_differ()
    test_3_3d_bic_formula_consistency()
    test_3_3e_mh_always_accept_when_target_dominates()
    test_3_3e_mh_always_reject_when_lqr_minus_inf()
    test_3_3e_mh_greedy_at_zero_temperature()
    test_3_3e_mh_random_branch_consistency()
    test_3_3f_grove_integration()
    test_3_3g_polya_log_prior_analytic()
    test_3_3g_polya_update_changes_prior()
    test_3_3g_polya_auto_register_unknown_leaf()
    print("\nAll Phase 3.3 (a)+(b)+(c)+(d)+(e)+(f)+(g) tests passed.")
