"""test_i3b_prep.py -- I3-B preparation: pooled dataset utilities
(i3b_pool), Polya pricing helper (i3b_polya_pricing) and the driver's
robustness options. Plain script, no pytest:

    python test_i3b_prep.py

The d = 3 kernel certification lives in test_phaseFD.py (FD3-*); this
file covers the Python-side deliverables D3, D4 and D5.
"""
import math
import os
import sys
import tempfile
import time

import numpy as np
import sympy as sp

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_SRC = os.path.join(os.path.dirname(_HERE), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import real_functionC
import i3b_pool
import i3b_polya_pricing as pp
from function_treeC import node_from_skel, Tree, tree_to_skel

LEGACY = ('x',)


def _assert_raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return
    raise AssertionError("%s not raised" % exc.__name__)


# ---------------------------------------------------------------------------
# D4: pooled dataset
# ---------------------------------------------------------------------------
def test_pool_build_and_folds():
    print("[P-a] build_pooled: layout, per-sim constants, folds by sim")
    sims = i3b_pool.synthetic_sims(n_sims=12, m=50, seed=1)
    pool = i3b_pool.build_pooled(sims)
    X, y, s = pool["X"], pool["y"], pool["sigma"]
    assert X.shape == (3, 600) and X.flags["C_CONTIGUOUS"]
    assert X.dtype == np.float64 and y.shape == (600,) and s.shape == (600,)
    # x1, x2 constant within a simulation and equal to the declared values
    for sid in pool["sims"]:
        rows = pool["sid"] == sid
        assert rows.sum() == 50
        assert np.all(X[1, rows] == pool["coords"][sid][0])
        assert np.all(X[2, rows] == pool["coords"][sid][1])
    # folds: deterministic, a property of the id, constant within a sim
    for sid in pool["sims"]:
        rows = pool["sid"] == sid
        f = set(pool["fold"][rows].tolist())
        assert len(f) == 1 and f == {i3b_pool.fold_of_sid(sid)}
    assert i3b_pool.fold_of_sid("SIM0003") == i3b_pool.fold_of_sid("SIM0003")
    assert 0 <= i3b_pool.fold_of_sid("anything", 5) < 5
    # the hash is stable across processes: pin three values
    pinned = {s: i3b_pool.fold_of_sid(s) for s in ("SIM0000", "SIM0001",
                                                    "SXS:BBH:0305")}
    assert pinned == {s: int(__import__("hashlib").sha1(s.encode())
                             .hexdigest()[:8], 16) % 5 for s in pinned}
    # validation
    bad = dict(sims[0])
    bad["sigma"] = 0.0
    _assert_raises(ValueError, i3b_pool.build_pooled, [bad])
    dup = [sims[0], sims[0]]
    _assert_raises(ValueError, i3b_pool.build_pooled, dup)
    print("      OK  (folds: %s)" % np.bincount(pool["fold"]).tolist())


def test_pool_thinning():
    print("[P-b] thinning: N_t per sim, deterministic, sim-local")
    sims = i3b_pool.synthetic_sims(n_sims=20, m=80, seed=2)
    pool = i3b_pool.build_pooled(sims)
    t1 = i3b_pool.thin_pooled(pool, n_t=15, seed=7)
    t2 = i3b_pool.thin_pooled(pool, n_t=15, seed=7)
    t3 = i3b_pool.thin_pooled(pool, n_t=15, seed=8)
    assert t1["y"].shape == (300,) and t1["X"].shape == (3, 300)
    assert np.array_equal(t1["X"], t2["X"]) and np.array_equal(t1["y"],
                                                               t2["y"])
    assert not np.array_equal(t1["y"], t3["y"])
    for sid in pool["sims"]:
        assert (t1["sid"] == sid).sum() == 15
    # a simulation's rows do not depend on which OTHER sims are present
    sub = i3b_pool.thin_pooled(pool, n_t=15, seed=7,
                               sims_subset=pool["sims"][:5])
    assert sorted(set(sub["sid"].tolist())) == sorted(pool["sims"][:5])
    for sid in pool["sims"][:5]:
        assert np.array_equal(sub["y"][sub["sid"] == sid],
                              t1["y"][t1["sid"] == sid])
    # short simulations keep all rows
    short = i3b_pool.thin_pooled(pool, n_t=500, seed=1)
    assert short["y"].shape == pool["y"].shape
    # stratified sim subset: deterministic, covers the x1 range
    pick1 = i3b_pool.select_sims_stratified(pool, 10, seed=3)
    pick2 = i3b_pool.select_sims_stratified(pool, 10, seed=3)
    assert pick1 == pick2 and len(pick1) == 10
    x1s = sorted(pool["coords"][s][0] for s in pick1)
    allx = sorted(pool["coords"][s][0] for s in pool["sims"])
    assert x1s[0] <= allx[4] and x1s[-1] >= allx[-5], "subset not spread"
    assert i3b_pool.select_sims_stratified(pool, 99, seed=3) == \
        sorted(pool["sims"])
    print("      OK")


def test_pool_split_and_io():
    print("[P-c] by-simulation split, train-first order, save/load")
    sims = i3b_pool.synthetic_sims(n_sims=15, m=30, seed=5)
    pool = i3b_pool.build_pooled(sims)
    ordered, n_tr = i3b_pool.split_train_first(pool, held_fold=0)
    assert np.all(ordered["fold"][:n_tr] != 0)
    assert np.all(ordered["fold"][n_tr:] == 0)
    assert n_tr + (pool["fold"] == 0).sum() == pool["y"].shape[0]
    # no simulation straddles the split
    tr_sims = set(ordered["sid"][:n_tr].tolist())
    ho_sims = set(ordered["sid"][n_tr:].tolist())
    assert not (tr_sims & ho_sims), "a simulation leaked across the split"
    assert ordered["X"].flags["C_CONTIGUOUS"]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "pool.npz")
        i3b_pool.save_pooled(path, ordered, extra_meta={"note": "test"})
        back = i3b_pool.load_pooled(path)
    assert np.array_equal(back["X"], ordered["X"])
    assert np.array_equal(back["y"], ordered["y"])
    assert np.array_equal(back["sigma"], ordered["sigma"])
    assert back["sid"].tolist() == ordered["sid"].tolist()
    assert back["fold"].tolist() == ordered["fold"].tolist()
    assert back["coords"] == {s: tuple(v) for s, v in ordered["coords"].items()}
    assert back["meta"]["note"] == "test"
    print("      OK  (train %d / held-out %d rows)"
          % (n_tr, pool["y"].shape[0] - n_tr))


# ---------------------------------------------------------------------------
# D5: robustness helpers
# ---------------------------------------------------------------------------
def test_robustness_helpers():
    print("[P-d] sigma floor + winsorisation")
    rng = np.random.default_rng(0)
    y = rng.normal(0.0, 1.0, 2000)
    assert abs(i3b_pool.robust_spread(y) - 1.0) < 0.08
    sigma = np.full(2000, 1e-6)
    eff, floor, n_lift = i3b_pool.sigma_with_floor(sigma, y, 0.02)
    assert abs(floor - 0.02 * i3b_pool.robust_spread(y)) < 1e-12
    assert np.all(eff >= floor) and n_lift == 2000
    big = np.full(2000, 5.0)
    eff2, _, n2 = i3b_pool.sigma_with_floor(big, y, 0.02)
    assert np.array_equal(eff2, big) and n2 == 0
    # an outlier does not move the floor (IQR-based)
    y_out = y.copy()
    y_out[0] = 1e6
    assert abs(i3b_pool.robust_spread(y_out) - i3b_pool.robust_spread(y)) \
        < 0.01
    w = i3b_pool.winsorise(y_out, 1.0)
    assert w.max() <= np.percentile(y_out, 99.0) + 1e-12
    assert np.array_equal(i3b_pool.winsorise(y_out, 0.0), y_out)
    print("      OK")


# ---------------------------------------------------------------------------
# D3: Polya pricing
# ---------------------------------------------------------------------------
def test_pricing():
    print("[P-e] Polya pricing at d=3: kernel == closed form, table, "
          "unexpressible forms")
    funcs = pp.vocabulary_i3b()
    assert len(funcs) == 11 and "log(x0)" not in funcs
    nt = pp.n_total_cells(funcs)
    assert nt == 6 + 11 + 1
    try:
        polya = pp.make_t0_polya(funcs)
        assert polya.n_prims == 12 and polya.unify_params
        # kernel price of hand-built skeletons equals -N log(n_total)
        cases = {
            ('L', 'x0'): 1,
            ('O', '*', ('L', 'a0'), ('L', 'x1')): 3,
            ('O', '+', ('O', '*', ('L', 'a0'), ('L', 'exp(x1)')),
             ('O', '/', ('L', 'x0'), ('L', 'tanh(x2)'))): 7,
        }
        for skel, n in cases.items():
            lp, nn = pp.price_skel(skel, funcs, polya=polya)
            assert nn == n
            assert abs(lp + n * math.log(nt)) < 1e-9, (skel, lp)
        # parameter names do not matter (psi_const cell)
        lp_a, _ = pp.price_skel(('O', '*', ('L', 'a0'), ('L', 'x1')), funcs,
                                polya=polya)
        lp_b, _ = pp.price_skel(('O', '*', ('L', 'b7'), ('L', 'x1')), funcs,
                                polya=polya)
        assert lp_a == lp_b
        # expression strings: canonical skeletons
        lp, n, sk = pp.price_expr("2*x1 + a0*tanh(x2)", funcs, polya=polya)
        assert n == 7 and sk[0] == 'O' and sk[1] == '+'
        lp, n, sk = pp.price_expr("x0 - x1/x2", funcs, polya=polya)
        assert n == 5 and sk[1] == '-' and sk[3][1] == '/'
        lp, n, sk = pp.price_expr("x1**3", funcs, polya=polya)
        assert n == 5
        lp, n, sk = pp.price_expr("-x0 + 3", funcs, polya=polya)
        assert n == 3
        lp, n, sk = pp.price_expr("a0*x0*x1 + a1*x0*x2 + a2", funcs,
                                  polya=polya)
        assert n == 13
        # the machine's own tree prices identically through the skeleton
        real_functionC.set_variables(pp.VARIABLES_I3B)
        X = pp._dummy_domain(pp.VARIABLES_I3B)
        root = node_from_skel(sk, X, None)
        tr = Tree(0, list(funcs), X, root)
        assert abs(polya.log_prior_from_tree(tr.get_root()) - lp) < 1e-12
        sk2 = tree_to_skel(tr.get_root())
        lp2, n2 = pp.price_skel(sk2, funcs, polya=polya)
        assert n2 == n and abs(lp2 - lp) < 1e-12
        # JSON round trip of skeletons
        assert pp.skel_from_json(pp.skel_to_json(sk)) == sk
        # unexpressible
        for ex in ("exp(-x0*x1)", "sqrt(x1)", "tanh(a0*x2)", "x1**0.5"):
            _assert_raises(pp.NotExpressible, pp.price_expr, ex, funcs,
                           polya=polya)
        tab = pp.pricing_table(funcs)
        assert tab["n_total_cells"] == nt
        assert abs(tab["per_node_log_cost"] - math.log(nt)) < 1e-6
        assert tab["ladder_by_nodes"][7] == round(-7 * math.log(nt), 6)
        assert len(tab["unexpressible_examples"]) == 4
    finally:
        real_functionC.set_variables(LEGACY)
    print("      OK  (n_total %d, %.3f nats per node)" % (nt, math.log(nt)))


def main():
    t0 = time.time()
    test_pool_build_and_folds()
    test_pool_thinning()
    test_pool_split_and_io()
    test_robustness_helpers()
    test_pricing()
    assert real_functionC.get_variables() == LEGACY
    print("=" * 60)
    print("ALL I3B-PREP TESTS PASSED in %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
