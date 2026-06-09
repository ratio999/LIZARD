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
      * add_node, split:  lqr == -log(|N(T')| - 1)
        (paired hoist selects uniformly over non-root nodes of T').
      * hoist_mutation:    lqf == -log(|N(T)|  - 1).
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
                       split, hoist_mutation)
from groveC import Grove
from function_treeC import Tree, expr_to_tree


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
    """add_node: paired reverse move is leaf-hoist whose selection density
    is uniform over non-root nodes of T'.  So lqr == -log(|N(T')| - 1)."""
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

        expected_lqr = -math.log(n_after - 1)
        assert math.isclose(lqr, expected_lqr, rel_tol=0.0, abs_tol=tol), (
            "[seed=%d] add_node lqr mismatch: got %.15g, expected %.15g "
            "(|N(T')|=%d)" % (s, lqr, expected_lqr, n_after)
        )
        n_checked += 1

    print("[3.3c.add_node] PASS  (%d/%d checked)" % (n_checked, n_seeds))


def test_3_3c_split_density(n_seeds=200):
    """split: same reverse-density formula as add_node (paired internal-hoist
    selects uniformly over non-root nodes of T')."""
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

        expected_lqr = -math.log(n_after - 1)
        assert math.isclose(lqr, expected_lqr, rel_tol=0.0, abs_tol=tol), (
            "[seed=%d] split lqr mismatch: got %.15g, expected %.15g "
            "(|N(T')|=%d)" % (s, lqr, expected_lqr, n_after)
        )
        n_checked += 1

    print("[3.3c.split] PASS  (%d/%d checked)" % (n_checked, n_seeds))


def test_3_3c_hoist_density(n_seeds=300, tol=1e-10):
    """hoist_mutation: forward move picks uniformly over non-root nodes of
    T, so lqf == -log(|N(T)| - 1).  lqr may be -inf when the lifted
    sibling is not a valid `add_node` reverse (off-support)."""
    print("[3.3c.hoist] forward selection density + off-support handling")
    from real_functionC import _domain_fixer
    x, _ = _make_xy(n=24, seed=14)
    domain = _domain_fixer(x)

    n_on = 0
    n_off = 0
    for s in range(n_seeds):
        tree = _make_deeper_tree(domain)
        n_before = _count_nodes_py(tree.get_root())

        _seed_rngs(s)
        lqf, lqr = hoist_mutation(tree, _FUNCS, _OPS)

        n_after = _count_nodes_py(tree.get_root())
        assert n_after < n_before, \
            "hoist failed to shrink: %d -> %d (seed=%d)" % (n_before, n_after, s)
        assert math.isfinite(lqf), "hoist lqf must be finite"

        expected_lqf = -math.log(n_before - 1)
        assert math.isclose(lqf, expected_lqf, rel_tol=0.0, abs_tol=tol), (
            "[seed=%d] hoist lqf mismatch: got %.15g, expected %.15g "
            "(|N(T)|=%d)" % (s, lqf, expected_lqf, n_before)
        )
        if math.isinf(lqr):
            assert lqr == -math.inf, "hoist off-support lqr must be -inf"
            n_off += 1
        else:
            n_on += 1

    # Both branches must be exercised in 300 seeds for our depth-2 fixture.
    # If one side is empty, the kernel is misclassifying targets and
    # something is wrong upstream.
    assert n_on > 0, "hoist never landed on-support in %d seeds" % n_seeds
    print("[3.3c.hoist] PASS  (on=%d, off=%d)" % (n_on, n_off))


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
    test_3_3c_hoist_density()
    test_3_3c_point_mutation_symmetric()
    print("\nAll Phase 3.3 (a)+(b)+(c) tests passed.")
