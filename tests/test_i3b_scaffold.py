"""test_i3b_scaffold.py -- acceptance tests for the wave-level scaffold
work (I3-B D9): the channel templates' symbolic identity against the
science-side fitters (K5), the driver's building blocks, and the K7
synthetic recovery go/no-go. Plain script:

    PYTHONHASHSEED=0 python test_i3b_scaffold.py [--fast]

--fast skips the K7 chain (the slow, decisive part; run it for the
acceptance verdict). The multi-slot kernel machinery itself is
certified in test_phaseFX.py (FX2-*); this file certifies the D9
APPLICATION layer.
"""
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_SRC = os.path.join(os.path.dirname(_HERE), "src")
_EXP = os.path.join(os.path.dirname(_HERE), "experiments")
for _p in (_EXP, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import sympy as sp

import real_functionC
import function_treeC
import exp_common
import i3b_polya_pricing as pricing
import exp_mr_i3b_scaffold as drv

N_PASS = 0
N_FAIL = 0


def check(label, cond, extra=""):
    global N_PASS, N_FAIL
    if cond:
        N_PASS += 1
        print("  [PASS] %s" % label, flush=True)
    else:
        N_FAIL += 1
        print("  [FAIL] %s %s" % (label, extra), flush=True)


def main():
    fast = "--fast" in sys.argv
    t00 = time.time()
    exp_common.set_caps()

    # ---------------------------------------------------------------- S-a
    print("[S-a] template algebra == the science-side fitters (K5)")
    # AMP: TS3c with the FULL 8-knob map of i4_ts3cphys_fit at the
    # deployed conventions (amp_kn8: t1 = 0, r3 = 1, t3 = 0, f -> 1),
    # built INDEPENDENTLY of the driver's reduced form.
    Ka, Kb, Kc, Kd = [sp.Symbol(s) for s in drv.SLOT_SYMS]
    x0, x1, x2 = sp.symbols("x0 x1 x2")
    x3, x4, x5 = sp.symbols("x3 x4 x5")
    s1 = sp.Integer(1)
    # knobs_to_coeffs with kn = (r1, r2, r3=1, t1=0, t2=tau2, t3=0,
    # s, f=1), A = (2*x1, x1, x1), Aref = x5, signs = (s1, *):
    full = {
        sp.Symbol('c0'): 2 * x1 * Ka,
        sp.Symbol('c1'): -(2 * x1 * Ka) * 0,
        sp.Symbol('c2'): s1 * 1 * (Kd * x5),
        sp.Symbol('c3'): x1 * Kb,
        sp.Symbol('c4'): -(x1 * Kb) * Kc,
        sp.Symbol('c5'): s1 * (1 - 1) * (Kd * x5),
        sp.Symbol('c6'): x1 * 1,
        sp.Symbol('c7'): -(x1 * 1) * 0,
        sp.Symbol('c98'): x3, sp.Symbol('c99'): x4,
        sp.Symbol('x'): x0}
    ts3c_full = sp.sympify(drv.TPL_TS3C).subs(full, simultaneous=True)
    amp_drv = sp.sympify(drv.channel_template("amp", 1.0))
    diff = sp.simplify(ts3c_full - amp_drv)
    check("S-a1 amp reduction: driver template == TS3c at the deployed "
          "conventions", diff == 0 or bool(ts3c_full.equals(amp_drv)),
          str(diff)[:120])
    # the order's explicitly written reduced form, typed independently:
    c0, c2, c3, c4 = (2 * x1 * Ka, Kd * x5, x1 * Kb, -x1 * Kb * Kc)
    order_form = (x3 - c2 * sp.tanh(c4)
                  + ((x4 - c2 * c3 * (1 - sp.tanh(c4) ** 2)) / c0)
                  * sp.tanh(c0 * x0)
                  + c2 * sp.tanh(c3 * x0 + c4))
    check("S-a2 amp reduction matches the work order's written form",
          bool(order_form.equals(amp_drv)))
    # PHASE: the order's phi(t), typed independently.
    W, g = Kc, Kd
    r1t, r2t = Ka, Kb
    phi_order = (-x3 * (1 + W)
                 / (g * W * 2 * x1 * r1t + (1 - g) * W * 2 * x2 * r2t)
                 * sp.log((1 + g * W * sp.exp(-2 * x1 * r1t * x0)
                           + (1 - g) * W * sp.exp(-2 * x2 * r2t * x0))
                          / (1 + W)))
    ph_drv = sp.sympify(drv.channel_template("phase"))
    check("S-a3 phase template matches the work order's phi(t)",
          bool(phi_order.equals(ph_drv)))

    # ---------------------------------------------------------------- S-b
    print("[S-b] specs, masks, seeding, pool validation")
    try:
        real_functionC.set_variables(drv.VARS["amp"])
        spec_a = drv.build_spec("amp", 1.0)
        check("S-b1 amp spec: 4 slots, masks on all, x2 ABSENT from the "
              "frozen shape (it enters only through slots)",
              spec_a.n_slots == 4
              and all(spec_a.slot_variables[s] == drv.SLOT_MASK
                      for s in drv.SLOT_SYMS)
              and "x2" not in spec_a.template_variables
              and set(spec_a.template_variables)
              <= {"x0", "x1", "x3", "x4", "x5"},
              spec_a.template_variables)
        real_functionC.set_variables(drv.VARS["phase"])
        spec_p = drv.build_spec("phase")
        check("S-b2 phase spec: 4 slots, template vars x0..x3",
              spec_p.n_slots == 4
              and sorted(spec_p.template_variables)
              == ["x0", "x1", "x2", "x3"])
        # seeded scaffold evaluates == direct lambdify at start values
        pool = drv.synthetic_pool("phase", n_sims=6, m=30, seed=3)
        X = pool["X"]
        root = drv.seed_root(spec_p, X, drv.START_DEFAULTS["phase"],
                             "phase")
        tr = function_treeC.Tree(0, ["x1", "x2"], X, node=root)
        tr.tree_update(tr.get_root())
        nv = dict(tr.get_param_dicts()[1])
        got = np.asarray(tr.evaluate(), dtype=np.float64)
        fn = sp.lambdify(
            [sp.Symbol(v) for v in drv.VARS["phase"]]
            + [sp.Symbol(s) for s in drv.SLOT_SYMS],
            sp.sympify(spec_p.template), "numpy")
        sv = drv.START_DEFAULTS["phase"]
        ref = np.asarray(fn(*(list(X) + [sv[k] for k in
                                         drv.KNOBS["phase"]])),
                         dtype=np.float64)
        check("S-b3 seeded scaffold evaluates exactly (start values)",
              np.allclose(got, ref),
              float(np.max(np.abs(got - ref))))
        check("S-b4 k = 4 at seeding (one constant per knob)",
              len(nv) == 4, nv)
        drv.validate_pool(pool["X"], pool["sid"])
        check("S-b5 synthetic pool passes the pin contract", True)
        Xbad = pool["X"].copy()
        Xbad[1, 0] += 0.5
        raised = False
        try:
            drv.validate_pool(np.ascontiguousarray(Xbad), pool["sid"])
        except SystemExit:
            raised = True
        check("S-b6 a varying pin column is refused", raised)
    finally:
        real_functionC.set_variables(("x",))

    # ---------------------------------------------------------------- S-c
    if fast:
        print("[S-c] SKIPPED (--fast): K7 synthetic recovery")
    else:
        print("[S-c] K7 synthetic recovery: planted knob laws")
        try:
            real_functionC.set_variables(("x0", "x1", "x2"))
            toy = real_functionC.FlexAtomSpec(
                "Toy", "A*tanh(x0) + B", slot_symbols=["A", "B"],
                allow_variables=True,
                slot_variables={"A": ("x1", "x2"), "B": ("x1", "x2")})
            rng = np.random.default_rng(4)
            n_sims, m = 30, 40
            cols, ys, sids = [], [], []
            for i in range(n_sims):
                a1 = rng.uniform(0.6, 1.6)
                a2 = rng.uniform(0.6, 1.6)
                t = np.sort(rng.uniform(0.0, 4.0, m))
                A_true = 0.5 + a1 * a2
                B_true = np.tanh(a2)
                cols.append(np.vstack([t, np.full(m, a1),
                                       np.full(m, a2)]))
                ys.append(A_true * np.tanh(t) + B_true)
                sids.extend(["T%03d" % i] * m)
            X = np.ascontiguousarray(np.hstack(cols).astype(np.float64))
            y = np.concatenate(ys)
            sd = 0.02 * float(np.std(y))
            y = np.ascontiguousarray(y + rng.normal(0.0, sd, y.size))
            funcs = ["x1", "x2", "tanh(x1)", "tanh(x2)"]
            slots = [function_treeC.node_from_skel(('L', 'k0'), X, None),
                     function_treeC.node_from_skel(('L', 'k1'), X, None)]
            root = function_treeC.make_flex_node(toy, slots, [])
            forest = exp_common.seeded_forest(
                X, y, [1.0, 1.5, 2.25, 4.0], 4, funcs,
                ["+", "-", "*", "/"], sd, 21, [root], flex_atoms=[toy],
                flex_root_lock=True)
            forest.run_all(y, 60)
            t0 = time.time()
            for rnd in range(300):
                forest.run_all(y, 6)
                for _ in range(4):
                    forest.attempt_swaps(y)
                forest.attempt_crossovers(y)
            best = forest.best_tree()
            fx = function_treeC.collect_flex_nodes(best.get_root(),
                                                   None)[0]
            nv = dict(best.get_param_dicts()[1])
            descA = fx.flex_slots[0].get_fun().get_description()
            descB = fx.flex_slots[1].get_fun().get_description()
            print("      best slots after %.0fs: A=%s  B=%s"
                  % (time.time() - t0, descA, descB), flush=True)
            # numeric law check on a fresh per-sim grid
            g1 = rng.uniform(0.6, 1.6, 200)
            g2 = rng.uniform(0.6, 1.6, 200)
            G = np.ascontiguousarray(np.vstack(
                [np.zeros(200), g1, g2]))
            A_hat = np.asarray(fx.flex_slots[0].get_fun().get_eval(G, nv),
                               dtype=np.float64)
            B_hat = np.asarray(fx.flex_slots[1].get_fun().get_eval(G, nv),
                               dtype=np.float64)
            errA = float(np.sqrt(np.mean((A_hat - (0.5 + g1 * g2)) ** 2))
                         / np.std(0.5 + g1 * g2))
            errB = float(np.sqrt(np.mean((B_hat - np.tanh(g2)) ** 2))
                         / max(np.std(np.tanh(g2)), 1e-12))
            # The work order's acceptance criterion: the planted laws
            # are recovered (numerically), OR the found scaffold beats
            # the PLANTED tree's joint evidence on the same data — the
            # Bayesian reading: the data did not compel the exact form.
            slt = [function_treeC.node_from_skel(
                ('O', '+', ('L', 'k0'),
                 ('O', '*', ('L', 'x1'), ('L', 'x2'))), X, None),
                function_treeC.node_from_skel(('L', 'tanh(x2)'), X, None)]
            rt_true = function_treeC.make_flex_node(toy, slt, [])
            t_true = function_treeC.Tree(0, list(funcs), X, node=rt_true)
            t_true.set_sigma(sd)
            t_true.tree_update(t_true.get_root())
            post_true = float(t_true.log_marginal_bic(y, True))
            post_best = float(max(
                p for gv in forest.groves for p in gv.root_posteriors
                if p is not None and np.isfinite(p)))
            print("      planted-tree evidence %.3f vs found best %.3f; "
                  "law errs A=%.4f B=%.4f" % (post_true, post_best,
                                              errA, errB), flush=True)
            check("S-c1 planted law A recovered, or found evidence beats "
                  "the planted tree", errA < 0.05
                  or post_best >= post_true,
                  "err=%.4f post_best=%.1f post_true=%.1f A=%s"
                  % (errA, post_best, post_true, descA))
            check("S-c2 planted law B recovered, or found evidence beats "
                  "the planted tree", errB < 0.05
                  or post_best >= post_true,
                  "err=%.4f post_best=%.1f post_true=%.1f B=%s"
                  % (errB, post_best, post_true, descB))
            # masks proven across the FINAL POPULATIONS of every rung
            bad = 0
            for gv in forest.groves:
                for tcur in gv.trees:
                    for f2 in function_treeC.collect_flex_nodes(
                            tcur.get_root(), None):
                        for sl in f2.flex_slots:
                            vs = {str(s) for lf in sl.get_leaves(None)
                                  for s in lf.get_fun().get_variables()}
                            bad += int("x0" in vs)
            check("S-c3 no slot anywhere in the ladder ever contains "
                  "x0 (t)", bad == 0, bad)
            check("S-c4 every tree kept its scaffold",
                  all(len(function_treeC.collect_flex_nodes(
                      tcur.get_root(), None)) == 1
                      for gv in forest.groves for tcur in gv.trees))
            check("S-c5 root lock held: every tree IS the bare scaffold",
                  all(tcur.get_root().is_flex()
                      for gv in forest.groves for tcur in gv.trees))
        finally:
            real_functionC.set_variables(("x",))

    print("=" * 66)
    print("test_i3b_scaffold: %d passed, %d failed (%.0fs)"
          % (N_PASS, N_FAIL, time.time() - t00))
    return 1 if N_FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
