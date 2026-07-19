"""Phase FD test suite — flexible multivariate domains.

Companion to flexible_domain_plan.md §5 (tests FD-a … FD-k). Plain script,
no pytest — same conventions as test_phase23.py. Run:

    python test_phaseFD.py            # full suite (serial, no spawn)
    mpiexec -n 1 python test_phaseFD.py --mpi   # adds the live spawn test

The variable registry is module-global state: every test that installs a
non-legacy registry restores ('x',) in a finally block, so tests are order-
independent and the legacy default is always left behind.
"""
import sys
import time
import pickle

import numpy as np
import sympy as sp

from real_functionC import (RealFunctionC, AtomSpec, set_variables,
                            get_variables, n_variables, is_variable_name,
                            instantiate_atom, _domain_fixer)
from function_treeC import Tree, node_from_skel, expr_to_tree
from groveC import Grove, Forest, _tree_to_wire, _tree_from_wire
from polya_treeC import PolyaTree


LEGACY = ('x',)
OPS4 = ["+", "-", "*", "/"]
OPS6 = ["+", "-", "*", "/", "o", "**"]
FUNCS2 = ["x0", "x1", "sin(x0)", "sin(x1)"]
FUNCS1 = ["sin(x)", "cos(x)", "x"]


def _make_X2(n=64):
    """(2, n) C-contiguous domain: x0 in [-2, 2], x1 in [0.5, 3]."""
    x0 = np.linspace(-2.0, 2.0, n, dtype=np.float64)
    x1 = np.linspace(0.5, 3.0, n, dtype=np.float64)
    return np.ascontiguousarray(np.vstack([x0, x1]))


def _grove2(X, y, seed, funcs=None, size=4, depth=2, n_steps=0, **kw):
    g = Grove(X, y, T=1.0, size=size, depth=depth,
              funcs=(FUNCS2 if funcs is None else funcs),
              Npars=1, init_value=1.0, operations=OPS4, base_temp=1.0,
              use_gradients=False, seed=seed, g_tau=0.7, **kw)
    if n_steps:
        g.run(y, n_steps)
    return g


def _assert_raises(exc, fn, *args, **kw):
    try:
        fn(*args, **kw)
    except exc:
        return
    raise AssertionError("%s not raised by %r" % (exc.__name__, fn))


# ---------------------------------------------------------------------------
# FD-a  Registry & classification
# ---------------------------------------------------------------------------
def test_FD_a_registry():
    print("[FD-a] registry & classification")
    assert get_variables() == LEGACY, "default registry must be ('x',)"
    assert n_variables() == 1
    try:
        set_variables(('x0', 'x1'))
        assert get_variables() == ('x0', 'x1')
        assert n_variables() == 2
        assert is_variable_name('x0') and is_variable_name('x1')
        assert not is_variable_name('x') and not is_variable_name('a0')

        # Registration validation.
        _assert_raises(ValueError, set_variables, ('y',))          # observations
        _assert_raises(ValueError, set_variables, ('a0',))         # 'a' band
        _assert_raises(ValueError, set_variables, ('u', 'b3'))     # 'b' band
        _assert_raises(ValueError, set_variables, ('sin',))        # sympy func
        _assert_raises(ValueError, set_variables, ('pi',))         # constant
        _assert_raises(ValueError, set_variables, ('u', 'u'))      # duplicate
        _assert_raises(ValueError, set_variables, ())              # empty
        assert get_variables() == ('x0', 'x1'), \
            "failed registration must not clobber the registry"

        X = _make_X2()
        rf = RealFunctionC.from_string('a0*x0 + sin(x1)', X)
        varz = {str(s) for s in rf.get_variables()}
        parz = {str(s) for s in rf.get_params()}
        assert varz == {'x0', 'x1'}, varz
        assert parz == {'a0'}, parz

        # THE catastrophe assert (plan §5 FD-a): a registered 'x0' leaf is
        # NOT a parameter — pre-fix classification made it one, and the
        # optimizer would tune the independent variable as a constant.
        leaf = RealFunctionC.from_string('x0', X)
        assert not leaf.is_param(), "variable leaf classified as parameter!"
        pleaf = RealFunctionC.from_string('a0', X)
        assert pleaf.is_param(), "parameter leaf must stay a parameter"
    finally:
        set_variables(LEGACY)

    # Legacy classification untouched.
    x1d = np.linspace(-2.0, 2.0, 32)
    rf = RealFunctionC.from_string('a0*sin(x)', x1d)
    assert {str(s) for s in rf.get_variables()} == {'x'}
    assert {str(s) for s in rf.get_params()} == {'a0'}
    print("      OK")


# ---------------------------------------------------------------------------
# FD-b  Naming guards
# ---------------------------------------------------------------------------
def test_FD_b_naming_guards():
    print("[FD-b] naming guards")
    X = _make_X2()
    try:
        set_variables(('x0', 'x1'))
        spec = AtomSpec('A2', 'c0*sin(c1*x1)')
        # Fresh-name guard: a registered variable may never label a slot.
        _assert_raises(ValueError, instantiate_atom, spec, X, ['x0', 'b1'])
        rf = instantiate_atom(spec, X, ['b0', 'b1'])
        assert rf.atom_symbol == 'A2'

        # Integration guard: an 80-step d=2 chain (births included) never
        # mints a parameter named like a variable. name_values travels on
        # the wire tuple (Python-visible), descriptions via sympify.
        y = (2.0 * X[0] + np.sin(X[1])).astype(np.float64)
        g = _grove2(X, y, seed=4242, n_steps=80)
        for t in g.trees:
            wire = _tree_to_wire(t, 0.0)
            nv = wire[1]
            bad = set(nv) & {'x0', 'x1'}
            assert not bad, "parameter minted with a variable name: %r" % bad
    finally:
        set_variables(LEGACY)
    print("      OK")


# ---------------------------------------------------------------------------
# FD-c  Shapes & evaluate
# ---------------------------------------------------------------------------
def test_FD_c_shapes():
    print("[FD-c] shapes & evaluate")
    n = 64
    X = _make_X2(n)
    try:
        set_variables(('x0', 'x1'))
        # Coordinate rows come back exactly.
        assert np.array_equal(
            RealFunctionC.from_string('x0', X).get_eval(X), X[0])
        assert np.array_equal(
            RealFunctionC.from_string('x1', X).get_eval(X), X[1])
        # Parameter leaf: (n,) fill — the full_like bug class (plan §7 R1).
        out = RealFunctionC.from_string('a0', X).get_eval(X, {'a0': 3.25})
        assert out.shape == (n,), out.shape
        assert np.all(out == 3.25)
        # Compound vs numpy reference.
        rf = RealFunctionC.from_string('sin(x0)*x1 + a0', X)
        ref = np.sin(X[0]) * X[1] + 1.5
        assert np.allclose(rf.get_eval(X, {'a0': 1.5}), ref)
        # Non-finite predictions keep the (n,) contract.
        bad = RealFunctionC.from_string('log(x0)', X).get_eval(X)
        assert bad.shape == (n,)
        assert np.isnan(bad[:5]).any() or np.isinf(bad[:5]).any() \
            or not np.isfinite(bad).all()
        # is_defined mask is (n,).
        mask = RealFunctionC.from_string('log(x0)', X).is_defined(X)
        assert np.asarray(mask).shape == (n,)

        # _domain_fixer: in-place zero scrub, shape preserved.
        Z = np.ascontiguousarray(
            np.vstack([np.array([0.0, 1.0, 2.0]),
                       np.array([3.0, 0.0, 5.0])]))
        Zf = _domain_fixer(Z)
        assert Zf.shape == (2, 3)
        assert Zf[0, 0] == 1e-10 and Zf[1, 1] == 1e-10 and Zf[1, 2] == 5.0

        # Construction-time strictness (plan §6 Q1/Q2 adopted).
        y = np.zeros(n)
        _assert_raises(ValueError, Grove, X.T, y, 1.0, 2, 2, FUNCS2,
                       operations=OPS4, variables=('x0', 'x1'))
        _assert_raises(ValueError, Grove, X[None, :, :], y, 1.0, 2, 2,
                       FUNCS2, operations=OPS4)
        _assert_raises(ValueError, Grove, X, y, 1.0, 2, 2, FUNCS2)  # ops None
        _assert_raises(ValueError, Grove, X, y, 1.0, 2, 2, FUNCS2,
                       operations=OPS6)                             # 'o' at d>1
        _assert_raises(ValueError, Grove, X, y, 1.0, 2, 2,
                       ["x0", "sin(x2)"], operations=OPS4)          # vocab typo
        # Adversarial-review regressions: a (1, n) single-variable domain
        # must be rejected (it would ride the d=1 paths as a 2-D array),
        # and variables must not be a bare string.
        _assert_raises(ValueError, Grove,
                       np.ascontiguousarray(X[:1, :]), y, 1.0, 2, 2,
                       ["x0", "sin(x0)"], operations=OPS4)
        _assert_raises(ValueError, Grove, X, y, 1.0, 2, 2, FUNCS2,
                       operations=OPS4, variables='x0x1')
        # is_defined except path: mis-shaped input yields a defined,
        # correctly-sized all-False mask (was an out-of-bounds C read).
        m = RealFunctionC.from_string('x0 + x1', X).is_defined(
            np.array([1.0]))
        m = np.asarray(m)
        assert m.shape == (1,) and not m.any(), m
    finally:
        set_variables(LEGACY)

    # d=1 values bit-equal to the direct numpy computation.
    x1d = np.linspace(-2.0, 2.0, n)
    assert np.array_equal(
        RealFunctionC.from_string('sin(x)', x1d).get_eval(x1d), np.sin(x1d))
    print("      OK")


# ---------------------------------------------------------------------------
# FD-d  BIC scorer at d=2
# ---------------------------------------------------------------------------
def _tree_from_skel2(skel, X, funcs, sigma=1.0):
    root = node_from_skel(skel, X, None)
    return Tree(0, list(funcs), X, root, 0, 1.0, 1.0, 0.01, sigma)


def test_FD_d_bic():
    print("[FD-d] BIC scorer at d=2")
    n = 64
    X = _make_X2(n)
    rng = np.random.default_rng(3)
    try:
        set_variables(('x0', 'x1'))
        y = (2.0 * X[0] + 3.0 * X[1]
             + rng.normal(0.0, 0.05, size=n)).astype(np.float64)

        full = ('O', '+',
                ('O', '*', ('L', 'a0'), ('L', 'x0')),
                ('O', '*', ('L', 'a1'), ('L', 'x1')))
        part = ('O', '*', ('L', 'a0'), ('L', 'x0'))

        t_full = _tree_from_skel2(full, X, FUNCS2)
        t_part = _tree_from_skel2(part, X, FUNCS2)
        lp_full = t_full.log_marginal_bic(y, True)
        lp_part = t_part.log_marginal_bic(y, True)
        assert np.isfinite(lp_full) and np.isfinite(lp_part)
        assert lp_full > lp_part, (lp_full, lp_part)

        # Fixed-σ branch differs from profiled-σ, both finite.
        t_fix = _tree_from_skel2(full, X, FUNCS2, sigma=0.05)
        lp_fix = t_fix.log_marginal_bic(y, True)
        assert np.isfinite(lp_fix) and lp_fix != lp_full

        # NaN-leak regression (plan §1.4): a tree non-finite on part of the
        # grid scores at penalty scale — never near a healthy competitor.
        t_bad = _tree_from_skel2(
            ('O', '*', ('L', 'a0'), ('L', 'log(x0)')), X,
            ["x0", "x1", "log(x0)"])
        lp_bad = t_bad.log_marginal_bic(y, True)
        assert lp_bad < -1.0e50, "partial-domain tree escaped the penalty"
    finally:
        set_variables(LEGACY)
    print("      OK  (full=%.2f part=%.2f fixed=%.2f penalty=%.2e)"
          % (lp_full, lp_part, lp_fix, lp_bad))


# ---------------------------------------------------------------------------
# FD-e  Moves & determinism at d=2
# ---------------------------------------------------------------------------
def test_FD_e_moves_determinism():
    print("[FD-e] moves & determinism at d=2")
    n = 48
    X = _make_X2(n)
    rng = np.random.default_rng(0)
    y = (np.sin(X[0]) + 0.5 * X[1]
         + rng.normal(0.0, 0.05, size=n)).astype(np.float64)
    try:
        set_variables(('x0', 'x1'))
        g1 = _grove2(X, y, seed=12345, n_steps=60)
        g2 = _grove2(X, y, seed=12345, n_steps=60)
        t1 = [t.tree_get_description() for t in g1.trees]
        t2 = [t.tree_get_description() for t in g2.trees]
        assert t1 == t2, "same seed must reproduce the d=2 trajectory"
        assert dict(g1.move_proposals) == dict(g2.move_proposals)
        assert dict(g1.move_accepts) == dict(g2.move_accepts)
        p1 = np.asarray(g1.root_posteriors, dtype=np.float64)
        p2 = np.asarray(g2.root_posteriors, dtype=np.float64)
        assert np.allclose(p1, p2, atol=1e-12, equal_nan=True)
        assert np.isfinite(p1).all(), "post-run posteriors must be finite"

        n_prop = sum(dict(g1.move_proposals).values())
        n_acc = sum(dict(g1.move_accepts).values())
        assert n_prop >= 60 and 0 < n_acc < n_prop

        g3 = _grove2(X, y, seed=999, n_steps=60)
        t3 = [t.tree_get_description() for t in g3.trees]
        assert t3 != t1, "different seed should explore differently"
    finally:
        set_variables(LEGACY)
    print("      OK  (accept rate %.2f)" % (n_acc / float(n_prop)))


# ---------------------------------------------------------------------------
# FD-f  Pólya prior at d=2
# ---------------------------------------------------------------------------
def test_FD_f_polya():
    print("[FD-f] Polya prior at d=2")
    n = 48
    X = _make_X2(n)
    rng = np.random.default_rng(1)
    y = (2.0 * X[0] + rng.normal(0.0, 0.05, size=n)).astype(np.float64)
    try:
        set_variables(('x0', 'x1'))
        polya = PolyaTree(OPS6, FUNCS2, alpha_0=1.0,
                          auto_register=True, strict_ops=True)
        skel = ('O', '+', ('L', 'sin(x0)'), ('L', 'x1'))
        t = _tree_from_skel2(skel, X, FUNCS2)
        lp0 = polya.log_prior_from_tree(t.get_root())
        assert np.isfinite(lp0)

        g = _grove2(X, y, seed=7, n_steps=40, polya=polya)
        assert polya.t > 0, "accepted moves must reinforce the urn"
        lp1 = polya.log_prior_from_tree(t.get_root())
        assert np.isfinite(lp1) and lp1 != lp0

        # Frozen-vocabulary Forest at d=2: vocabulary bit-stable through a
        # short EMC run, unknown symbols impossible by construction.
        temps = np.asarray([1.0, 2.0], dtype=np.float64)
        f = Forest(X, y, temps, 3, 2, FUNCS2, Npars=1, init_value=1.0,
                   operations=OPS4, use_gradients=False, seed=99,
                   freeze_vocab=True)
        n_prims0 = f.polya.n_prims
        f.compute_posteriors_all(y)
        for _ in range(5):
            f.run_all(y, 4)
            f.attempt_swaps(y)
            f.attempt_crossovers(y)
        assert f.polya.n_prims == n_prims0, "frozen vocabulary grew!"
        best = f.best_tree()
        assert best is not None
    finally:
        set_variables(LEGACY)
    print("      OK")


# ---------------------------------------------------------------------------
# FD-g  Serial end-to-end recovery at d=2 (the success criterion)
# ---------------------------------------------------------------------------
def test_FD_g_recovery():
    print("[FD-g] serial d=2 recovery: y = 2.5*x0 + sin(x1)")
    n = 80
    X = _make_X2(n)
    rng = np.random.default_rng(17)
    truth = 2.5 * X[0] + np.sin(X[1])
    y = (truth + rng.normal(0.0, 0.05, size=n)).astype(np.float64)

    try:
        set_variables(('x0', 'x1'))
        g = Grove(X, y, T=1.0, size=6, depth=2, funcs=FUNCS2,
                  Npars=2, init_value=1.0, operations=OPS4, base_temp=1.0,
                  use_gradients=False, seed=17, g_tau=1.0, sigma=0.05,
                  burn_in=True, use_mpi=0)
        if not g.trees:
            g.burn_in_depth2(y, True, True)
        g.run(y, 150)
        best = g.best_tree()
        desc = best.tree_get_description()
        yhat = np.asarray(best.evaluate(), dtype=np.float64)
        nrmse = float(np.sqrt(np.mean((yhat - truth) ** 2)) / np.std(truth))
        assert yhat.shape == (n,)
        assert nrmse < 0.05, "recovery failed: %s (nrmse %.4f)" % (desc, nrmse)
        syms = {str(s) for s in sp.sympify(desc).free_symbols}
        assert syms & {'x0', 'x1'} == {'x0', 'x1'}, \
            "winner should use both variables: %s" % desc

        # Irrelevant-variable control: x1 present in the vocabulary but the
        # data only depends on x0 — the run must stay healthy.
        y0 = (2.5 * X[0] + rng.normal(0.0, 0.05, size=n)).astype(np.float64)
        g0 = Grove(X, y0, T=1.0, size=4, depth=2, funcs=FUNCS2,
                   Npars=2, init_value=1.0, operations=OPS4, base_temp=1.0,
                   use_gradients=False, seed=71, g_tau=1.0, sigma=0.05,
                   burn_in=True, use_mpi=0)
        if not g0.trees:
            g0.burn_in_depth2(y0, True, True)
        g0.run(y0, 80)
        assert np.isfinite(
            np.asarray(g0.root_posteriors, dtype=np.float64)).all()
    finally:
        set_variables(LEGACY)
    print("      OK  (best: %s | nrmse %.5f)" % (desc, nrmse))


# ---------------------------------------------------------------------------
# FD-i  Wire round-trips
# ---------------------------------------------------------------------------
def test_FD_i_wire():
    print("[FD-i] wire round-trips")
    n = 48
    X = _make_X2(n)
    try:
        set_variables(('x0', 'x1'))
        skel = ('O', '+',
                ('O', '*', ('L', 'a0'), ('L', 'x0')),
                ('L', 'sin(x1)'))
        t = _tree_from_skel2(skel, X, FUNCS2)
        wire = _tree_to_wire(t, -12.5)
        # d>1 ALWAYS ships the skeleton payload (plan §0 D8).
        assert isinstance(wire[0], tuple) and wire[0][0] == 'skel3b', wire[0]
        t2 = _tree_from_wire(wire[0], wire[1], X, FUNCS2, 1.0, 1.0, None)
        assert t2 is not None
        assert t2.tree_get_description() == t.tree_get_description()

        # Atom-bearing d=2 skeleton round-trip + missing-registry reject.
        spec = AtomSpec('A2', 'c0*sin(c1*x1)')
        askel = ('O', '+', ('A', 'A2', ['b0', 'b1']), ('L', 'x0'))
        root = node_from_skel(askel, X, {'A2': spec})
        ta = Tree(0, FUNCS2, X, root)
        wa = _tree_to_wire(ta, -1.0)
        assert wa[0][0] == 'skel3b'
        ta2 = _tree_from_wire(wa[0], wa[1], X, FUNCS2, 1.0, 1.0,
                              {'A2': spec})
        assert ta2 is not None
        assert ta2.tree_get_description() == ta.tree_get_description()
        _assert_raises(KeyError, node_from_skel, askel, X, None)
    finally:
        set_variables(LEGACY)

    # Legacy d=1 atom-free payload is still the RAW sympy expr.
    x1d = np.linspace(-2.0, 2.0, n)
    t1 = expr_to_tree(sp.sympify('a0*x + sin(x)'), x1d, FUNCS1)
    w1 = _tree_to_wire(t1, -3.0)
    assert not (isinstance(w1[0], tuple)), \
        "legacy d=1 wire must stay the raw-expr payload"
    t1b = _tree_from_wire(w1[0], w1[1], x1d, FUNCS1, 1.0, 1.0, None)
    assert t1b is not None
    assert t1b.tree_get_description() == t1.tree_get_description()
    print("      OK")


# ---------------------------------------------------------------------------
# FD-k  Atoms at d=2
# ---------------------------------------------------------------------------
def test_FD_k_atoms():
    print("[FD-k] atoms at d=2")
    n = 48
    X = _make_X2(n)
    try:
        set_variables(('x0', 'x1'))
        spec = AtomSpec('A2', 'c0*sin(c1*x1)')
        # Registry-first template classification: x1 is the VARIABLE,
        # c0/c1 the slots (pre-fix, x1 matched the slot regex).
        assert spec.k == 2 and sorted(spec.slots) == ['c0', 'c1']
        _assert_raises(ValueError, AtomSpec, 'K', 'c0*c1')   # no variable
        spec2 = pickle.loads(pickle.dumps(spec))
        assert spec2.slots == spec.slots and spec2.k == spec.k

        rf = instantiate_atom(spec, X, ['b0', 'b1'])
        out = rf.get_eval(X, {'b0': 2.0, 'b1': 3.0})
        assert out.shape == (n,)
        assert np.allclose(out, 2.0 * np.sin(3.0 * X[1]))

        # Grove at d=2 with an atom: constructs (registry-aware re-derive)
        # and samples a few steps without error.
        rng = np.random.default_rng(5)
        y = (2.0 * np.sin(1.3 * X[1])
             + rng.normal(0.0, 0.05, size=n)).astype(np.float64)
        g = _grove2(X, y, seed=11, n_steps=40, atoms=[spec])
        assert np.isfinite(
            np.asarray(g.root_posteriors, dtype=np.float64)).all()
    finally:
        set_variables(LEGACY)
    print("      OK")


# ---------------------------------------------------------------------------
# FD-j  Informational timing (no gate)
# ---------------------------------------------------------------------------
def test_FD_j_timing():
    print("[FD-j] evaluate timing (informational)")
    n = 64
    x1d = np.linspace(-2.0, 2.0, n)
    rf1 = RealFunctionC.from_string('a0*sin(x) + x', x1d)
    t0 = time.time()
    for _ in range(2000):
        rf1.get_eval(x1d, {'a0': 1.1})
    d1 = time.time() - t0
    try:
        set_variables(('x0', 'x1'))
        X = _make_X2(n)
        rf2 = RealFunctionC.from_string('a0*sin(x0) + x1', X)
        t0 = time.time()
        for _ in range(2000):
            rf2.get_eval(X, {'a0': 1.1})
        d2 = time.time() - t0
    finally:
        set_variables(LEGACY)
    print("      d=1: %.3fs | d=2: %.3fs per 2000 evaluates" % (d1, d2))


# ---------------------------------------------------------------------------
# FD-i2  Live spawn at d=2 (only under mpiexec, opt-in via --mpi)
# ---------------------------------------------------------------------------
def test_FD_i2_spawn():
    print("[FD-i2] live run_mpi_spawn at d=2 (2 rungs, atom on board)")
    n = 60
    X = _make_X2(n)
    rng = np.random.default_rng(9)
    y = (2.0 * X[0] + np.sin(X[1])
         + rng.normal(0.0, 0.05, size=n)).astype(np.float64)
    try:
        set_variables(('x0', 'x1'))
        # Atoms + non-legacy registry over the spawn wire: regression for
        # the unpickle-before-set_variables deadlock (adversarial review) —
        # templates now cross as (name, template) pairs and are rebuilt
        # worker-side AFTER the registry install.
        spec = AtomSpec('Asin2', 'c0*sin(c1*x1)')
        temps = np.asarray([1.0, 2.0], dtype=np.float64)
        f = Forest(X, y, temps, 3, 2, FUNCS2, Npars=1, init_value=1.0,
                   operations=OPS4, use_gradients=False, seed=13,
                   freeze_vocab=True, atoms=[spec])
        stats = f.run_mpi_spawn(y, 4, 4, 13)
        assert isinstance(stats, dict)
        best = f.best_tree()
        assert best is not None
        desc = best.tree_get_description()
        syms = {str(s) for s in sp.sympify(desc).free_symbols}
        assert syms <= ({'x0', 'x1'}
                        | {s for s in syms if not is_variable_name(s)})
        posts = [np.asarray(g.root_posteriors, dtype=np.float64)
                 for g in f.groves]
        assert all(np.isfinite(p).all() for p in posts)
    finally:
        set_variables(LEGACY)
    print("      OK  (best: %s)" % desc)


def main():
    t0 = time.time()
    test_FD_a_registry()
    test_FD_b_naming_guards()
    test_FD_c_shapes()
    test_FD_d_bic()
    test_FD_e_moves_determinism()
    test_FD_f_polya()
    test_FD_g_recovery()
    test_FD_i_wire()
    test_FD_k_atoms()
    test_FD_j_timing()
    if '--mpi' in sys.argv:
        test_FD_i2_spawn()
    else:
        print("[FD-i2] skipped (pass --mpi under `mpiexec -n 1` to run)")
    assert get_variables() == LEGACY, "a test leaked a non-legacy registry"
    print("=" * 60)
    print("ALL PHASE-FD TESTS PASSED in %.1fs" % (time.time() - t0))


if __name__ == '__main__':
    main()
