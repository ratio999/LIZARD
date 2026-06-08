"""
Phase 3.3 tests.

(a) Determinism smoke test: same seed + same g_tau -> identical Grove
    trajectory (per-tree topology, move counters, root posteriors).

(b) Birth-density g unit test: verify g_tau plumbing (Grove -> Tree),
    default = std(y), and that point_mutation's BIRTH branch emits
    log_q_fwd = log N(theta_new; 0, g_tau^2) and log_q_rev = -log M
    (forward/reverse densities per Green 1995 RJMCMC).

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

import gp_movesC
from gp_movesC import point_mutation
from groveC import Grove
from function_treeC import Tree


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
if __name__ == "__main__":
    # OpenMP / BLAS threading affects bit-equality on some routines.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    test_3_3a_determinism()
    test_3_3b_g_tau_plumbing()
    test_3_3b_birth_density()
    print("\nAll Phase 3.3 (a)+(b) tests passed.")
