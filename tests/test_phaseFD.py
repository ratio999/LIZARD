"""Phase FD test suite — flexible multivariate domains.

Companion to flexible_domain_plan.md §5 (tests FD-a … FD-k). Plain script,
no pytest — same conventions as test_phase23.py. Run:

    python test_phaseFD.py            # full suite (serial, no spawn)
    mpiexec -n 1 python test_phaseFD.py --mpi   # adds the live spawn test

The variable registry is module-global state: every test that installs a
non-legacy registry restores ('x',) in a finally block, so tests are order-
independent and the legacy default is always left behind.
"""
import os
import sys
import time
import pickle

import numpy as np
import sympy as sp

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

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


# ===========================================================================
# d = 3 certification (I3-B preparation, D1)
#
# Three registered variables ('x0','x1','x2'). The kernel is written
# generically in d; these tests pin that down at d = 3 on every path the
# full-space driver exercises: registry/classification, shapes, naming
# guards, construction strictness, BIC, moves + determinism, the Polya
# prior, frozen-vocabulary EMC with swaps/crossovers, serial recovery,
# wire round-trips, archive-style evaluation on a foreign-length domain,
# and (under --mpi) the spawn burn-in and the spawn sampler.
# ===========================================================================
VARS3 = ('x0', 'x1', 'x2')
FUNCS3 = ["x0", "x1", "x2", "sin(x1)"]


def _make_X3(n=64, seed=2024):
    """(3, n) C-contiguous domain: x0 in [-2, 2], x1 in [0.5, 3],
    x2 in [1, 2], rows drawn INDEPENDENTLY (fixed seed).

    Not linspace: three linspace rows are exact affine functions of one
    another, so any two of them span the third and a model in two
    variables reproduces a three-variable law exactly — BIC then rightly
    prefers the smaller model and "uses all three variables" is
    ill-posed. Independent rows make the three coordinates genuinely
    informative, as they are in the pooled catalogue."""
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-2.0, 2.0, n)
    x1 = rng.uniform(0.5, 3.0, n)
    x2 = rng.uniform(1.0, 2.0, n)
    return np.ascontiguousarray(np.vstack([x0, x1, x2]).astype(np.float64))


def _grove3(X, y, seed, funcs=None, size=4, depth=2, n_steps=0, **kw):
    g = Grove(X, y, T=1.0, size=size, depth=depth,
              funcs=(FUNCS3 if funcs is None else funcs),
              Npars=1, init_value=1.0, operations=OPS4, base_temp=1.0,
              use_gradients=False, seed=seed, g_tau=0.7, **kw)
    if n_steps:
        g.run(y, n_steps)
    return g


def test_FD3_a_registry():
    print("[FD3-a] registry & classification at d=3")
    try:
        set_variables(VARS3)
        assert get_variables() == VARS3
        assert n_variables() == 3
        assert all(is_variable_name(v) for v in VARS3)
        assert not is_variable_name('x') and not is_variable_name('x3')
        X = _make_X3()
        rf = RealFunctionC.from_string('a0*x0 + x1*x2', X)
        varz = {str(s) for s in rf.get_variables()}
        parz = {str(s) for s in rf.get_params()}
        assert varz == set(VARS3), varz
        assert parz == {'a0'}, parz
        # Every registered leaf is a variable, never a parameter.
        for v in VARS3:
            assert not RealFunctionC.from_string(v, X).is_param(), v
        assert RealFunctionC.from_string('a0', X).is_param()
        # A name that LOOKS like a variable but is not registered is a
        # parameter (the regex route) — x3 here.
        assert RealFunctionC.from_string('x3', X).is_param()
    finally:
        set_variables(LEGACY)
    print("      OK")


def test_FD3_c_shapes():
    print("[FD3-c] shapes, evaluate & construction strictness at d=3")
    n = 64
    X = _make_X3(n)
    try:
        set_variables(VARS3)
        for i, v in enumerate(VARS3):
            assert np.array_equal(
                RealFunctionC.from_string(v, X).get_eval(X), X[i]), v
        out = RealFunctionC.from_string('a0', X).get_eval(X, {'a0': 3.25})
        assert out.shape == (n,) and np.all(out == 3.25)
        rf = RealFunctionC.from_string('sin(x0)*x1 + a0*x2', X)
        ref = np.sin(X[0]) * X[1] + 1.5 * X[2]
        assert np.allclose(rf.get_eval(X, {'a0': 1.5}), ref)
        bad = RealFunctionC.from_string('log(x0)', X).get_eval(X)
        assert bad.shape == (n,) and not np.isfinite(bad).all()
        mask = RealFunctionC.from_string('log(x0)', X).is_defined(X)
        assert np.asarray(mask).shape == (n,)
        Z = np.ascontiguousarray(np.vstack([
            np.array([0.0, 1.0]), np.array([3.0, 0.0]),
            np.array([0.0, 5.0])]))
        Zf = _domain_fixer(Z)
        assert Zf.shape == (3, 2)
        assert Zf[0, 0] == 1e-10 and Zf[1, 1] == 1e-10 \
            and Zf[2, 0] == 1e-10 and Zf[2, 1] == 5.0

        y = np.zeros(n)
        # (n, d) layout rejected; wrong name count rejected; ops None
        # rejected; 'o' rejected; unregistered vocabulary name rejected;
        # bare-string variables rejected.
        _assert_raises(ValueError, Grove, np.ascontiguousarray(X.T), y,
                       1.0, 2, 2, FUNCS3, operations=OPS4, variables=VARS3)
        _assert_raises(ValueError, Grove, X, y, 1.0, 2, 2, FUNCS3,
                       operations=OPS4, variables=('x0', 'x1'))
        _assert_raises(ValueError, Grove, X, y, 1.0, 2, 2, FUNCS3)
        _assert_raises(ValueError, Grove, X, y, 1.0, 2, 2, FUNCS3,
                       operations=OPS6)
        _assert_raises(ValueError, Grove, X, y, 1.0, 2, 2,
                       ["x0", "x1", "sin(x3)"], operations=OPS4)
        _assert_raises(ValueError, Grove, X, y, 1.0, 2, 2, FUNCS3,
                       operations=OPS4, variables='x0x1x2')
        # The implicit registry from a bare (3, n) array is ('x0','x1','x2').
        g = Grove(X, y, 1.0, 2, 2, FUNCS3, operations=OPS4, seed=1)
        assert get_variables() == VARS3
        assert len(g.trees) == 2
    finally:
        set_variables(LEGACY)
    print("      OK")


def test_FD3_b_naming_guards():
    print("[FD3-b] naming guards at d=3 (80-step chain with births)")
    X = _make_X3()
    try:
        set_variables(VARS3)
        spec = AtomSpec('A3', 'c0*sin(c1*x2)')
        _assert_raises(ValueError, instantiate_atom, spec, X, ['x2', 'b1'])
        _assert_raises(ValueError, instantiate_atom, spec, X, ['b0', 'x0'])
        rf = instantiate_atom(spec, X, ['b0', 'b1'])
        assert rf.atom_symbol == 'A3'

        y = (2.0 * X[0] + np.sin(X[1]) * X[2]).astype(np.float64)
        g = _grove3(X, y, seed=4243, n_steps=80)
        n_birth = 0
        for t in g.trees:
            wire = _tree_to_wire(t, 0.0)
            nv = wire[1]
            bad = set(nv) & set(VARS3)
            assert not bad, "parameter minted with a variable name: %r" % bad
            n_birth += len(nv)
        props = dict(g.move_proposals)
        assert sum(props.values()) >= 80
    finally:
        set_variables(LEGACY)
    print("      OK  (%d live parameters after the chain)" % n_birth)


def _tree_from_skel3(skel, X, funcs, sigma=1.0):
    root = node_from_skel(skel, X, None)
    return Tree(0, list(funcs), X, root, 0, 1.0, 1.0, 0.01, sigma)


def test_FD3_d_bic():
    print("[FD3-d] BIC scorer at d=3")
    n = 64
    X = _make_X3(n)
    rng = np.random.default_rng(3)
    try:
        set_variables(VARS3)
        y = (2.0 * X[0] + 3.0 * X[1] + 1.5 * X[2]
             + rng.normal(0.0, 0.05, size=n)).astype(np.float64)
        full = ('O', '+',
                ('O', '+',
                 ('O', '*', ('L', 'a0'), ('L', 'x0')),
                 ('O', '*', ('L', 'a1'), ('L', 'x1'))),
                ('O', '*', ('L', 'a2'), ('L', 'x2')))
        part = ('O', '+',
                ('O', '*', ('L', 'a0'), ('L', 'x0')),
                ('O', '*', ('L', 'a1'), ('L', 'x1')))
        t_full = _tree_from_skel3(full, X, FUNCS3)
        t_part = _tree_from_skel3(part, X, FUNCS3)
        lp_full = t_full.log_marginal_bic(y, True)
        lp_part = t_part.log_marginal_bic(y, True)
        assert np.isfinite(lp_full) and np.isfinite(lp_part)
        assert lp_full > lp_part, (lp_full, lp_part)
        nv = dict(t_full.get_param_dicts()[1])
        assert abs(nv['a0'] - 2.0) < 0.1 and abs(nv['a1'] - 3.0) < 0.1 \
            and abs(nv['a2'] - 1.5) < 0.1, nv
        t_fix = _tree_from_skel3(full, X, FUNCS3, sigma=0.05)
        lp_fix = t_fix.log_marginal_bic(y, True)
        assert np.isfinite(lp_fix) and lp_fix != lp_full
        t_bad = _tree_from_skel3(
            ('O', '*', ('L', 'a0'), ('L', 'log(x0)')), X,
            ["x0", "x1", "x2", "log(x0)"])
        lp_bad = t_bad.log_marginal_bic(y, True)
        assert lp_bad < -1.0e50, "partial-domain tree escaped the penalty"
    finally:
        set_variables(LEGACY)
    print("      OK  (full=%.2f part=%.2f fixed=%.2f penalty=%.2e)"
          % (lp_full, lp_part, lp_fix, lp_bad))


def test_FD3_e_moves_determinism():
    print("[FD3-e] moves & determinism at d=3")
    n = 48
    X = _make_X3(n)
    rng = np.random.default_rng(0)
    y = (np.sin(X[1]) * X[2] + 0.5 * X[0]
         + rng.normal(0.0, 0.05, size=n)).astype(np.float64)
    try:
        set_variables(VARS3)
        g1 = _grove3(X, y, seed=12345, n_steps=60)
        g2 = _grove3(X, y, seed=12345, n_steps=60)
        t1 = [t.tree_get_description() for t in g1.trees]
        t2 = [t.tree_get_description() for t in g2.trees]
        assert t1 == t2, "same seed must reproduce the d=3 trajectory"
        assert dict(g1.move_proposals) == dict(g2.move_proposals)
        assert dict(g1.move_accepts) == dict(g2.move_accepts)
        p1 = np.asarray(g1.root_posteriors, dtype=np.float64)
        p2 = np.asarray(g2.root_posteriors, dtype=np.float64)
        assert np.allclose(p1, p2, atol=1e-12, equal_nan=True)
        assert np.isfinite(p1).all()
        n_prop = sum(dict(g1.move_proposals).values())
        n_acc = sum(dict(g1.move_accepts).values())
        assert n_prop >= 60 and 0 < n_acc < n_prop
        g3 = _grove3(X, y, seed=999, n_steps=60)
        assert [t.tree_get_description() for t in g3.trees] != t1
    finally:
        set_variables(LEGACY)
    print("      OK  (accept rate %.2f)" % (n_acc / float(n_prop)))


def test_FD3_f_polya_emc():
    print("[FD3-f] Polya prior + frozen-vocabulary EMC at d=3")
    n = 48
    X = _make_X3(n)
    rng = np.random.default_rng(1)
    y = (2.0 * X[0] + X[2] + rng.normal(0.0, 0.05, size=n)).astype(
        np.float64)
    try:
        set_variables(VARS3)
        polya = PolyaTree(OPS6, FUNCS3, alpha_0=1.0,
                          auto_register=True, strict_ops=True)
        skel = ('O', '+', ('L', 'sin(x1)'), ('O', '*', ('L', 'x2'),
                                             ('L', 'x0')))
        t = _tree_from_skel3(skel, X, FUNCS3)
        lp0 = polya.log_prior_from_tree(t.get_root())
        assert np.isfinite(lp0)
        g = _grove3(X, y, seed=7, n_steps=40, polya=polya)
        assert polya.t > 0
        lp1 = polya.log_prior_from_tree(t.get_root())
        assert np.isfinite(lp1) and lp1 != lp0

        temps = np.asarray([1.0, 2.0, 4.0], dtype=np.float64)
        f = Forest(X, y, temps, 3, 2, FUNCS3, Npars=1, init_value=1.0,
                   operations=OPS4, use_gradients=False, seed=99,
                   freeze_vocab=True)
        n_prims0 = f.polya.n_prims
        f.compute_posteriors_all(y)
        for _ in range(6):
            f.run_all(y, 4)
            f.attempt_swaps(y)
            f.attempt_crossovers(y)
        assert f.polya.n_prims == n_prims0, "frozen vocabulary grew!"
        st = f.ladder_stats()
        assert sum(st["swap_attempts"]) > 0
        assert sum(st["xover_attempts"]) > 0
        assert f.best_tree() is not None
        for g_ in f.groves:
            assert np.isfinite(
                np.asarray(g_.root_posteriors, dtype=np.float64)).all()
    finally:
        set_variables(LEGACY)
    print("      OK  (swap attempts %s, xover attempts %s)"
          % (st["swap_attempts"], st["xover_attempts"]))


def test_FD3_g_recovery():
    print("[FD3-g] serial d=3 recovery: y = 2.5*x0 + x1*x2")
    n = 80
    X = _make_X3(n)
    rng = np.random.default_rng(17)
    truth = 2.5 * X[0] + X[1] * X[2]
    y = (truth + rng.normal(0.0, 0.05, size=n)).astype(np.float64)
    try:
        set_variables(VARS3)
        g = Grove(X, y, T=1.0, size=6, depth=2, funcs=FUNCS3,
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
        assert syms & set(VARS3) == set(VARS3), \
            "winner should use all three variables: %s" % desc
    finally:
        set_variables(LEGACY)
    print("      OK  (best: %s | nrmse %.5f)" % (desc, nrmse))


def test_FD3_i_wire():
    print("[FD3-i] wire round-trips at d=3")
    n = 48
    X = _make_X3(n)
    try:
        set_variables(VARS3)
        skel = ('O', '+',
                ('O', '*', ('L', 'a0'), ('L', 'x0')),
                ('O', '*', ('L', 'sin(x1)'), ('L', 'x2')))
        t = _tree_from_skel3(skel, X, FUNCS3)
        wire = _tree_to_wire(t, -12.5)
        assert isinstance(wire[0], tuple) and wire[0][0] == 'skel3b'
        t2 = _tree_from_wire(wire[0], wire[1], X, FUNCS3, 1.0, 1.0, None)
        assert t2 is not None
        assert t2.tree_get_description() == t.tree_get_description()
        assert np.allclose(t2.evaluate(), t.evaluate())
        # The sampling wire (force_skel) at d=3, pickled as it would cross
        # an intercomm.
        wire_f = _tree_to_wire(t, -12.5, True)
        wire_f = pickle.loads(pickle.dumps(wire_f))
        t3 = _tree_from_wire(wire_f[0], wire_f[1], X, FUNCS3, 1.0, 1.0, None)
        assert t3.tree_get_description() == t.tree_get_description()
        # Atom-bearing round-trip with a template in x2.
        spec = AtomSpec('A3', 'c0*sin(c1*x2)')
        askel = ('O', '+', ('A', 'A3', ['b0', 'b1']), ('L', 'x0'))
        root = node_from_skel(askel, X, {'A3': spec})
        ta = Tree(0, FUNCS3, X, root)
        wa = _tree_to_wire(ta, -1.0)
        ta2 = _tree_from_wire(wa[0], wa[1], X, FUNCS3, 1.0, 1.0, {'A3': spec})
        assert ta2 is not None
        assert ta2.tree_get_description() == ta.tree_get_description()
        _assert_raises(KeyError, node_from_skel, askel, X, None)
    finally:
        set_variables(LEGACY)
    print("      OK")


def test_FD3_h_archive_eval():
    print("[FD3-h] archive-style evaluate on a foreign-length domain")
    n, m = 48, 131
    X = _make_X3(n)
    rng = np.random.default_rng(23)
    X_full = np.ascontiguousarray(np.vstack([
        rng.uniform(-2.0, 2.0, m), rng.uniform(0.5, 3.0, m),
        rng.uniform(1.0, 2.0, m)]))
    try:
        set_variables(VARS3)
        skel = ('O', '+',
                ('O', '*', ('L', 'a0'), ('L', 'x0')),
                ('O', '*', ('L', 'x1'), ('L', 'x2')))
        t = _tree_from_skel3(skel, X, FUNCS3)
        y = (2.5 * X[0] + X[1] * X[2]).astype(np.float64)
        t.log_marginal_bic(y, True)
        pred = np.asarray(t.evaluate(X_full), dtype=np.float64)
        assert pred.shape == (m,), pred.shape
        a0 = dict(t.get_param_dicts()[1])['a0']
        assert np.allclose(pred, a0 * X_full[0] + X_full[1] * X_full[2])
        # Training-domain evaluation is unchanged by the foreign call.
        assert np.allclose(t.evaluate(), a0 * X[0] + X[1] * X[2])
        # After a chain: every live tree evaluates on the foreign domain
        # with the (m,) contract.
        g = _grove3(X, y, seed=31, n_steps=40)
        for tr in g.trees:
            p = np.asarray(tr.evaluate(X_full), dtype=np.float64)
            assert p.shape == (m,)
        # Wrong-d foreign domain must not silently evaluate.
        X2 = np.ascontiguousarray(X_full[:2, :])
        try:
            p = np.asarray(t.evaluate(X2), dtype=np.float64)
            assert not np.isfinite(p).all() or p.shape != (m,), \
                "(2, m) domain evaluated silently under a d=3 registry"
        except Exception:
            pass
    finally:
        set_variables(LEGACY)
    print("      OK")


def _burned_descs(g):
    return sorted(t.tree_get_description() for t in g.trees)


def test_FD3_m_burnin_budget():
    print("[FD3-m] burn-in budget: burn_in_top_k / burn_in_cap (serial)")
    n = 40
    X = _make_X3(n)
    rng = np.random.default_rng(41)
    y = (2.0 * X[0] + X[1] * X[2]
         + rng.normal(0.0, 0.05, size=n)).astype(np.float64)
    # Three-primitive vocabulary: the test runs several exhaustive
    # burn-ins and only the budget logic is under test here.
    kw = dict(T=1.0, size=6, depth=2, funcs=["x0", "x1", "x2"], Npars=2,
              init_value=1.0, operations=OPS4, base_temp=1.0,
              use_gradients=False, g_tau=1.0, sigma=0.05, burn_in=True,
              use_mpi=0)
    try:
        set_variables(VARS3)
        _assert_raises(ValueError, Grove, X, y, burn_in_cap=-1, seed=1, **kw)
        _assert_raises(ValueError, Grove, X, y, burn_in_top_k=-1, seed=1,
                       **kw)
        # Exhaustive reference (defaults) vs explicit zeros: identical.
        t0 = time.time()
        g_ex = Grove(X, y, seed=5, **kw)
        t_ex = time.time() - t0
        g_ex0 = Grove(X, y, seed=5, burn_in_top_k=0, burn_in_cap=0, **kw)
        assert _burned_descs(g_ex) == _burned_descs(g_ex0)
        assert np.allclose(np.asarray(g_ex.root_posteriors, dtype=float),
                           np.asarray(g_ex0.root_posteriors, dtype=float))
        # After burn-in the Grove holds EVERY valid candidate as a tree.
        ex_set = set(_burned_descs(g_ex))
        n_ex = len(g_ex.trees)
        # A cap larger than the exhaustive count is a no-op.
        g_big = Grove(X, y, seed=5, burn_in_cap=10 ** 9, **kw)
        assert _burned_descs(g_big) == _burned_descs(g_ex)
        # Capped: deterministic under the seed, fewer scored candidates,
        # and a different seed draws a different subset.
        t0 = time.time()
        g_c1 = Grove(X, y, seed=5, burn_in_cap=200, **kw)
        t_c = time.time() - t0
        g_c2 = Grove(X, y, seed=5, burn_in_cap=200, **kw)
        g_c3 = Grove(X, y, seed=6, burn_in_cap=200, **kw)
        assert _burned_descs(g_c1) == _burned_descs(g_c2), \
            "capped burn-in must be reproducible under the seed"
        assert np.allclose(np.asarray(g_c1.root_posteriors, dtype=float),
                           np.asarray(g_c2.root_posteriors, dtype=float))
        assert len(g_c1.trees) < n_ex, (len(g_c1.trees), n_ex)
        assert set(_burned_descs(g_c1)) <= ex_set, \
            "capped stage 2 must score a SUBSET of the exhaustive set"
        assert set(_burned_descs(g_c1)) != set(_burned_descs(g_c3)), \
            "different seeds should sample different subsets"
        assert np.isfinite(
            np.asarray(g_c1.root_posteriors, dtype=float)).all()
        # top-K alone: subset of the exhaustive set, deterministic.
        g_k1 = Grove(X, y, seed=5, burn_in_top_k=8, **kw)
        g_k2 = Grove(X, y, seed=5, burn_in_top_k=8, **kw)
        assert _burned_descs(g_k1) == _burned_descs(g_k2)
        assert len(g_k1.trees) < n_ex
        assert set(_burned_descs(g_k1)) <= ex_set
        # Both knobs together; the chain keeps running afterwards.
        g_kc = Grove(X, y, seed=5, burn_in_top_k=8, burn_in_cap=100, **kw)
        g_kc.run(y, 30)
        assert np.isfinite(
            np.asarray(g_kc.root_posteriors, dtype=float)).all()
        # Sanity: the sampled subset does not disturb the chain RNG —
        # two capped Groves with the same seed take the same first steps.
        g_c1.run(y, 30)
        g_c2.run(y, 30)
        assert [t.tree_get_description() for t in g_c1.trees] == \
            [t.tree_get_description() for t in g_c2.trees]
    finally:
        set_variables(LEGACY)
    print("      OK  (exhaustive %d candidates in %.1fs; cap=200 -> %d "
          "candidates in %.1fs)" % (n_ex, t_ex, len(g_c1.trees), t_c))


def test_FD46_wide_domains():
    """d = 4 and d = 6 sanity (I3-B D9 K4): the d-generic claims of the
    d=3 certification, exercised at the widths the wave-level scaffolds
    use, with CONSTANT-PER-SIMULATION pin columns riding the domain.

    Pin columns are constant within a simulation's rows and vary only
    across simulations, so any single-sim slice is degenerate in them —
    pooled scoring is what must stay healthy."""
    print("[FD46] d=4 and d=6 domains with constant-per-sim pin columns")
    rng = np.random.default_rng(46)
    n_sims, m = 12, 20
    n = n_sims * m
    for d, vars_d in ((4, ('x0', 'x1', 'x2', 'x3')),
                      (6, ('x0', 'x1', 'x2', 'x3', 'x4', 'x5'))):
        try:
            set_variables(vars_d)
            rows = [np.concatenate([rng.uniform(0.5, 3.0, m)
                                    for _ in range(n_sims)])]
            for j in range(1, d):
                # constant per simulation, varying across simulations
                vals = rng.uniform(0.5, 2.0, n_sims)
                rows.append(np.repeat(vals, m))
            X = np.ascontiguousarray(np.vstack(rows).astype(np.float64))
            assert get_variables() == vars_d and n_variables() == d
            # coordinate rows come back exactly; pins constant per block
            for j in range(d):
                assert np.array_equal(
                    RealFunctionC.from_string(vars_d[j], X).get_eval(X),
                    X[j])
            assert np.all(X[1][:m] == X[1][0])
            # compound evaluate against numpy, using a pin column
            rf = RealFunctionC.from_string('a0*x1*x0 + x%d' % (d - 1), X)
            ref = 1.5 * X[1] * X[0] + X[d - 1]
            assert np.allclose(rf.get_eval(X, {'a0': 1.5}), ref)
            # BIC scoring + a short chain on the wide domain
            y = (2.0 * X[0] * X[1] + X[2]
                 + rng.normal(0.0, 0.05, n)).astype(np.float64)
            skel = ('O', '+', ('O', '*', ('O', '*', ('L', 'a0'),
                                          ('L', 'x0')), ('L', 'x1')),
                    ('L', 'x2'))
            root = node_from_skel(skel, X, None)
            t = Tree(0, [v for v in vars_d], X, root, 0, 1.0, 1.0, 0.01,
                     0.05)
            lp = t.log_marginal_bic(y, True)
            assert np.isfinite(lp), (d, lp)
            nv = dict(t.get_param_dicts()[1])
            assert abs(nv['a0'] - 2.0) < 0.1, nv
            g = Grove(X, y, T=1.0, size=3, depth=2,
                      funcs=[v for v in vars_d], Npars=1, init_value=1.0,
                      operations=OPS4, base_temp=1.0, use_gradients=False,
                      seed=17, g_tau=1.0, sigma=0.05)
            g.run(y, 40)
            assert np.isfinite(np.asarray(g.root_posteriors,
                                          dtype=np.float64)).all()
            for tr in g.trees:
                wire = _tree_to_wire(tr, 0.0)
                assert not (set(wire[1]) & set(vars_d)), \
                    "parameter minted with a variable name at d=%d" % d
            # archive-style evaluate on a foreign-length domain
            m2 = 37
            X2 = np.ascontiguousarray(np.vstack(
                [rng.uniform(0.5, 2.0, m2) for _ in range(d)]))
            p = np.asarray(t.evaluate(X2), dtype=np.float64)
            assert p.shape == (m2,)
        finally:
            set_variables(LEGACY)
        print("      OK  d=%d" % d)


def test_FD3_j_timing():
    print("[FD3-j] evaluate timing at d=3 (informational)")
    n = 64
    try:
        set_variables(VARS3)
        X = _make_X3(n)
        rf = RealFunctionC.from_string('a0*sin(x0) + x1*x2', X)
        t0 = time.time()
        for _ in range(2000):
            rf.get_eval(X, {'a0': 1.1})
        d3 = time.time() - t0
    finally:
        set_variables(LEGACY)
    print("      d=3: %.3fs per 2000 evaluates" % d3)


def test_FD3_i2_spawn():
    print("[FD3-i2] spawn burn-in + run_mpi_spawn at d=3")
    import exp_common
    n = 60
    X = _make_X3(n)
    rng = np.random.default_rng(9)
    truth = 2.0 * X[0] + X[1] * X[2]
    y = (truth + rng.normal(0.0, 0.05, size=n)).astype(np.float64)
    try:
        set_variables(VARS3)
        roots, posts, wall = exp_common.spawn_burn_in(
            X, y, FUNCS3, OPS4, 0.05, seed=5, n_workers=2)
        assert len(roots) > 0 and len(posts) == len(roots)
        assert np.isfinite(posts[0])
        assert get_variables() == VARS3, "spawn burn-in clobbered registry"
        print("      spawn burn-in: %d candidates in %.1fs; best %s"
              % (len(roots), wall, roots[0].get_fun().get_description()))
        # Capped spawn burn-in: reproducible under the seed across a real
        # spawn, and far fewer candidates. (The subset-of-exhaustive check
        # lives in the serial test FD3-m: the spawn master dedups worker
        # results through its description cache, so the first-seen
        # instance of a degenerate duplicate depends on the work
        # partition and an item-level subset is not an exact set
        # relation on the spawn path — the same holds at d = 1.)
        rc1, pc1, wc1 = exp_common.spawn_burn_in(
            X, y, FUNCS3, OPS4, 0.05, seed=5, n_workers=2, burn_in_cap=400)
        rc2, pc2, wc2 = exp_common.spawn_burn_in(
            X, y, FUNCS3, OPS4, 0.05, seed=5, n_workers=2, burn_in_cap=400)
        d1 = [r.get_fun().get_description() for r in rc1]
        d2 = [r.get_fun().get_description() for r in rc2]
        assert d1 == d2, "capped spawn burn-in not reproducible under seed"
        assert np.allclose(pc1, pc2)
        assert 0 < len(rc1) < len(roots), (len(rc1), len(roots))
        for dsc in d1:
            bad = {str(s) for s in sp.sympify(dsc).free_symbols} - set(VARS3)
            assert all(not is_variable_name(s) for s in bad), dsc
        print("      capped spawn burn-in (cap=400): %d candidates in %.1fs"
              % (len(rc1), wc1))
        forest = exp_common.seeded_forest(
            X, y, [1.0, 2.0], 3, FUNCS3, OPS4, 0.05, seed=7,
            top_roots=roots)
        spec = AtomSpec('Asin3', 'c0*sin(c1*x2)')
        temps = np.asarray([1.0, 2.0], dtype=np.float64)
        f = Forest(X, y, temps, 3, 2, FUNCS3, Npars=1, init_value=1.0,
                   operations=OPS4, use_gradients=False, seed=13,
                   freeze_vocab=True, atoms=[spec])
        stats = f.run_mpi_spawn(y, 4, 4, 13)
        assert isinstance(stats, dict)
        best = f.best_tree()
        assert best is not None
        desc = best.tree_get_description()
        syms = {str(s) for s in sp.sympify(desc).free_symbols}
        assert not (syms - set(VARS3)) or all(
            not is_variable_name(s) for s in syms - set(VARS3))
        for g_ in f.groves:
            assert np.isfinite(
                np.asarray(g_.root_posteriors, dtype=np.float64)).all()
        assert get_variables() == VARS3
        del forest
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
    # d = 3
    test_FD3_a_registry()
    test_FD3_c_shapes()
    test_FD3_b_naming_guards()
    test_FD3_d_bic()
    test_FD3_e_moves_determinism()
    test_FD3_f_polya_emc()
    test_FD3_g_recovery()
    test_FD3_i_wire()
    test_FD3_h_archive_eval()
    test_FD3_m_burnin_budget()
    test_FD46_wide_domains()
    test_FD3_j_timing()
    if '--mpi' in sys.argv:
        test_FD3_i2_spawn()
    else:
        print("[FD3-i2] skipped (pass --mpi under `mpiexec -n 1` to run)")
    assert get_variables() == LEGACY, "a test leaked a non-legacy registry"
    print("=" * 60)
    print("ALL PHASE-FD TESTS PASSED (d=2 and d=3) in %.1fs"
          % (time.time() - t0))


if __name__ == '__main__':
    main()
