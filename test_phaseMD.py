"""test_phaseMD.py -- certification of Phase MD multi-dataset joint
evidence (2026-07-31, PLAN MR.4).

WHAT THIS PINS
--------------
ONE shared skeleton tau scored on J datasets D_j = (x_j, y_j, sigma_j),
each with its OWN independently optimized theta_j:

    log post(tau) ~ log pi_polya(tau) + REDUCE_j [ log L_j(theta_hat_j)
                                                   - (k/2) log n_j ]

REDUCE = 'sum' (DEFAULT: the joint evidence), 'min' / 'softmin' opt-in
(baseline-shifted fit terms, complexity kept summed -- D4/D5/D6).
Prior / tilt / moves / reinforcement untouched; datasets=None is
bit-identical legacy (certified separately by the historical suites).
v1 refusals: burn_in (D2), parameter_refresh (D1), run_mpi_spawn (D8).

CHECKS
------
  [MD-a] J=1 equivalence: datasets=[(x,y,s)] reproduces the legacy
         single-dataset posterior/BIC/theta trajectory exactly;
  [MD-b] duplicate-dataset identity: datasets=[D,D] gives
         theta_hat_1 == theta_hat_2 and joint == 2 x single, exactly;
  [MD-c] synthetic recovery: one skeleton, two theta_true; the joint
         run recovers it and both theta_hat_j; a D1-only distractor
         loses the joint comparison;
  [MD-d] strictest validity: finite on D1, non-finite on D2 =>
         impossible-candidate score, and NOT cached;
  [MD-e] cache: joint value shape, reference-theta install on hit,
         penalty-exclusion;
  [MD-f] sigma branches: per-dataset sigma-array analytic pin and the
         profiled (sigma=None) k+1-per-dataset penalty;
  [MD-g] joint x transparent atoms (Phase FX): scaffolded skeleton
         across two datasets, per-dataset internal thetas;
  [MD-h] reduction unit: sum/min/softmin (+baselines, beta limits),
         penalty helper k vs k+1, error paths;
  [MD-i] refusals + contract: burn_in, PR, spawn, reference-dataset
         equality, Forest pass-through + serial EMC smoke.

RUN
---
  $ENV/bin/python test_phaseMD.py          (no MPI needed)
"""
import os
import sys
import copy
import random

os.environ.setdefault("OMP_NUM_THREADS", "1")
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import sympy as sp

import real_functionC
import function_treeC
import gp_movesC
import groveC
from groveC import Grove, Forest

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


def expect_raises(label, exc, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc:
        check(label, True)
        return
    except Exception as e:
        check(label, False, "wrong exception %r" % (e,))
        return
    check(label, False, "no exception raised")


def make_tree(expr_str, domain, funcs, values=None, sigma=1.0, g_tau=1.0):
    tree = function_treeC.expr_to_tree(sp.sympify(expr_str),
                                       np.ascontiguousarray(domain,
                                                            dtype=np.float64),
                                       list(funcs))
    if values:
        by_name = {p.get_fun().get_description(): p
                   for p in tree.get_params()}
        for nm, v in values.items():
            by_name[nm].set_value(float(v))
        tree.tree_update(tree.get_root())
    tree.set_sigma(sigma)
    tree.set_g_tau(g_tau)
    return tree


# ---------------------------------------------------------------- fixtures
RNG = np.random.default_rng(17)
X1 = np.linspace(0.5, 3.0, 60)
X2 = np.linspace(0.8, 3.5, 45)               # different length AND range
FUNCS = ["tanh(x)", "sin(x)", "log(x)", "x"]
OPS4 = ["+", "-", "*", "/"]

# One skeleton, two different truths (the MD-c science shape).
TH1 = (2.0, 1.5)                              # y = c0*tanh(c1*x)
TH2 = (-1.2, 0.7)
S1 = np.full(X1.size, 0.05)
S2 = np.full(X2.size, 0.04)
Y1 = TH1[0] * np.tanh(TH1[1] * X1) + RNG.normal(0, 1, X1.size) * S1
Y2 = TH2[0] * np.tanh(TH2[1] * X2) + RNG.normal(0, 1, X2.size) * S2
DSETS = [(X1, Y1, S1), (X2, Y2, S2)]


def main():
    gp_movesC.set_max_tree_depth(4)
    gp_movesC.set_subtree_rebuild_max_height(2)

    # ---------------------------------------------------------------- MD-h
    print("[MD-h] reduction unit + penalty helper")
    check("MD-h1 sum reduction",
          groveC.reduce_joint([-10.0, -20.0], [1.0, 2.0], "sum") == -30.0)
    # min: L = S + P -> [-9, -18]; min - sum(P) = -18 - 3 = -21
    check("MD-h2 min reduction (complexity stays summed)",
          groveC.reduce_joint([-10.0, -20.0], [1.0, 2.0], "min") == -21.0)
    # min with baselines: shift dataset 2 so IT is no longer worst
    v = groveC.reduce_joint([-10.0, -20.0], [1.0, 2.0], "min", 8.0,
                            [0.0, -12.0])
    check("MD-h3 baselines shift the argmin", v == -9.0 - 3.0, repr(v))
    big = groveC.reduce_joint([-10.0, -20.0], [1.0, 2.0], "softmin", 200.0)
    sml = groveC.reduce_joint([-10.0, -20.0], [1.0, 2.0], "softmin", 1e-6)
    mean_fit = (-9.0 - 18.0) / 2.0 - 3.0
    # NB the NORMALIZED softmin approaches min + log(J)/beta from above
    # (the 1/J normalization is what makes beta->0 the mean), so the
    # large-beta pin includes that exact offset.
    check("MD-h4 softmin: beta->inf ~ min + log(J)/beta, beta->0 ~ mean",
          abs(big - (-21.0 + np.log(2.0) / 200.0)) < 1e-9
          and abs(sml - mean_fit) < 1e-3,
          "big=%r sml=%r" % (big, sml))
    check("MD-h5 non-finite dataset score => -inf",
          groveC.reduce_joint([-10.0, float("-inf")], [1.0, 1.0], "sum")
          == float("-inf"))
    expect_raises("MD-h6 unknown mode raises", ValueError,
                  groveC.reduce_joint, [-1.0], [0.5], "median")
    n = 60
    check("MD-h7 penalty: k with sigma, k+1 profiled",
          groveC.md_dataset_penalty(3, n, 1.0) == 0.5 * 3 * np.log(n)
          and groveC.md_dataset_penalty(3, n, None)
          == 0.5 * 4 * np.log(n))

    # ---------------------------------------------------------------- MD-a
    print("[MD-a] J=1 equivalence with the legacy path")
    def leg_grove(seed):
        return Grove(X1, Y1, 1.0, 4, 2, list(FUNCS), sigma=S1.copy(),
                     Npars=1, operations=list(OPS4), use_gradients=False,
                     seed=seed, g_tau=1.0)

    def j1_grove(seed):
        return Grove(X1, Y1, 1.0, 4, 2, list(FUNCS), sigma=S1.copy(),
                     Npars=1, operations=list(OPS4), use_gradients=False,
                     seed=seed, g_tau=1.0, datasets=[(X1, Y1, S1)])

    gl, gj = leg_grove(23), j1_grove(23)
    gl.compute_posteriors(Y1, False)
    gj.compute_posteriors(Y1, False)
    pl = np.asarray(gl.root_posteriors, dtype=np.float64)
    pj = np.asarray(gj.root_posteriors, dtype=np.float64)
    check("MD-a1 initial posteriors identical",
          np.array_equal(pl, pj), "%s vs %s" % (pl, pj))
    # Exact trajectory equivalence needs SYMMETRIC cache states: legacy
    # compute_posteriors bypasses the BIC cache while the joint one
    # populates it (by design -- a hit saves J fits; documented in PLAN
    # MR.4). So compare FRESH groves whose initial scoring goes through
    # run()'s lazy fill, which caches identically on both paths.
    # NB legacy PR fires with pr_prob=0.02 while joint forces 0.0 --
    # equalize BEFORE stepping so the RNG streams stay aligned.
    gl2, gj2 = leg_grove(23), j1_grove(23)
    gl2.pr_prob = 0.0
    gl2.run(Y1, 150)
    gj2.run(Y1, 150)
    check("MD-a2 150-step trajectory identical (best description)",
          gl2.best_tree().tree_get_description()
          == gj2.best_tree().tree_get_description(),
          "%r vs %r" % (gl2.best_tree().tree_get_description(),
                        gj2.best_tree().tree_get_description()))
    check("MD-a3 posts identical after run (symmetric cache states)",
          np.array_equal(np.asarray(gl2.root_posteriors),
                         np.asarray(gj2.root_posteriors)),
          "%s vs %s" % (gl2.root_posteriors, gj2.root_posteriors))
    check("MD-a4 joint ctor forces pr_prob = 0 (D1)", gj.pr_prob == 0.0)

    # ---------------------------------------------------------------- MD-b
    print("[MD-b] duplicate-dataset identity")
    seedt = make_tree("a0*tanh(a1*x)", X1, FUNCS, sigma=S1.copy())
    gdup = Grove(X1, Y1, 1.0, 1, 2, list(FUNCS), sigma=S1.copy(),
                 node_list=[copy.deepcopy(seedt).get_root()],
                 operations=list(OPS4), use_gradients=False, seed=5,
                 g_tau=1.0, datasets=[(X1, Y1, S1), (X1, Y1, S1)])
    gdup.compute_posteriors(Y1, False)
    joint2 = float(gdup.root_posteriors[0])
    gsin = Grove(X1, Y1, 1.0, 1, 2, list(FUNCS), sigma=S1.copy(),
                 node_list=[copy.deepcopy(seedt).get_root()],
                 operations=list(OPS4), use_gradients=False, seed=5,
                 g_tau=1.0, datasets=[(X1, Y1, S1)])
    gsin.compute_posteriors(Y1, False)
    single = float(gsin.root_posteriors[0])
    check("MD-b1 joint(D,D) == 2 x single(D) exactly",
          joint2 == 2.0 * single, "%r vs 2*%r" % (joint2, single))
    ths = gdup.joint_thetas(gdup.trees[0])
    check("MD-b2 theta_hat_1 == theta_hat_2 exactly",
          ths is not None and len(ths) == 2 and ths[0] == ths[1],
          repr(ths))

    # ---------------------------------------------------------------- MD-c
    print("[MD-c] synthetic recovery: one skeleton, two truths")
    true_t = make_tree("a0*tanh(a1*x)", X1, FUNCS, sigma=S1.copy())
    # Distractor tuned to D1 only: a line matches D1's near-saturated
    # tanh range poorly on D2's different theta.
    dist_t = make_tree("a0*x + a1", X1, FUNCS, sigma=S1.copy())
    gc2 = Grove(X1, Y1, 1.0, 2, 2, list(FUNCS), sigma=S1.copy(),
                node_list=[true_t.get_root(), dist_t.get_root()],
                operations=list(OPS4), use_gradients=False, seed=7,
                g_tau=1.0, datasets=DSETS)
    gc2.compute_posteriors(Y1, False)
    p_true = float(gc2.root_posteriors[0])
    p_dist = float(gc2.root_posteriors[1])
    check("MD-c1 true skeleton beats the distractor on joint evidence",
          np.isfinite(p_true) and p_true > p_dist,
          "true=%r dist=%r" % (p_true, p_dist))
    ths = gc2.joint_thetas(gc2.trees[0])
    # (review 2026-07-31 finding 10: the original assertion was vacuous
    # — `A and B or True`. Real pin: the two datasets were generated
    # from DIFFERENT theta_true, so the fitted per-dataset value sets
    # must differ.)
    check("MD-c2 per-dataset theta_hat_j differ (different theta_true)",
          ths is not None and len(ths) == 2
          and sorted(ths[0].values()) != sorted(ths[1].values()),
          repr(ths))
    # exact functional check: tree with theta_j must track dataset j
    tt = gc2.trees[0]
    ok_fit = True
    for j, (xj, yj, sj, thj) in enumerate(
            [(X1, Y1, S1, ths[0]), (X2, Y2, S2, ths[1])]):
        gp = {p.get_fun().get_description(): p for p in tt.get_params()}
        for nm, v in thj.items():
            if nm in gp:
                gp[nm].set_value(float(v))
        tt.tree_update(tt.get_root())
        pred = np.asarray(tt.evaluate(np.ascontiguousarray(xj)),
                          dtype=np.float64)
        chi2 = float(np.mean(((pred - yj) / sj) ** 2))
        if chi2 > 3.0:
            ok_fit = False
    check("MD-c3 per-dataset theta_hat_j track their own datasets "
          "(chi2/dof < 3 on both)", ok_fit)
    # short joint EMC run: the true skeleton must remain the best tree
    gc2.run(Y1, 120)
    best_desc = gc2.best_tree().tree_get_description()
    check("MD-c4 after 120 joint steps the best tree still carries tanh",
          "tanh" in best_desc, best_desc)

    # ---------------------------------------------------------------- MD-d
    print("[MD-d] strictest-wins validity")
    XN = np.linspace(-2.0, -0.5, 30)          # log(x) invalid here
    YN = np.abs(XN)
    SN = np.full(XN.size, 0.1)
    logt = make_tree("a0*log(x)", X1, FUNCS, sigma=S1.copy())
    gd = Grove(X1, Y1, 1.0, 1, 2, list(FUNCS), sigma=S1.copy(),
               node_list=[logt.get_root()], operations=list(OPS4),
               use_gradients=False, seed=9, g_tau=1.0,
               datasets=[(X1, Y1, S1), (XN, YN, SN)])
    gd.compute_posteriors(Y1, False)
    p_bad = float(gd.root_posteriors[0])
    check("MD-d1 finite on D1 but invalid on D2 => impossible-candidate "
          "score", p_bad <= -1.0e290, repr(p_bad))
    check("MD-d2 the invalid candidate is NOT cached",
          gd.trees[0].tree_get_description() not in gd._bic_cache)

    # ---------------------------------------------------------------- MD-e
    print("[MD-e] cache semantics")
    ge = Grove(X1, Y1, 1.0, 1, 2, list(FUNCS), sigma=S1.copy(),
               node_list=[make_tree("a0*tanh(a1*x)", X1, FUNCS).get_root()],
               operations=list(OPS4), use_gradients=False, seed=11,
               g_tau=1.0, datasets=DSETS)
    ge.compute_posteriors(Y1, False)
    key = ge.trees[0].tree_get_description()
    check("MD-e1 cache entry exists with the joint value shape",
          key in ge._bic_cache and len(ge._bic_cache[key]) == 2
          and isinstance(ge._bic_cache[key][1], list)
          and len(ge._bic_cache[key][1]) == 2)
    ref_nv = dict(ge._bic_cache[key][1][0])
    # perturb the live theta, re-score -> cache hit must restore the
    # REFERENCE (dataset 0) theta
    for p in ge.trees[0].get_params():
        p.set_value(99.0)
    ge.trees[0].tree_update(ge.trees[0].get_root())
    ge.root_posteriors[0] = np.nan
    ge.compute_posteriors(Y1, False)
    now_nv = ge.trees[0].get_param_dicts()[1]
    check("MD-e2 cache hit installs the reference theta (dataset 0)",
          all(abs(now_nv[k] - ref_nv[k]) < 1e-15 for k in ref_nv),
          "%r vs %r" % (now_nv, ref_nv))

    # ---------------------------------------------------------------- MD-f
    print("[MD-f] sigma branches per dataset")
    # (i) sigma-array: joint == sum of analytic per-dataset BICs at
    # theta_hat (computed via score_dataset on fresh trees)
    tf1 = make_tree("a0*tanh(a1*x)", X1, FUNCS, sigma=S1.copy())
    post1, nv1 = tf1.score_dataset(np.ascontiguousarray(X1),
                                   np.ascontiguousarray(Y1), S1.copy())
    tf2 = make_tree("a0*tanh(a1*x)", X1, FUNCS, sigma=S1.copy())
    post2, nv2 = tf2.score_dataset(np.ascontiguousarray(X2),
                                   np.ascontiguousarray(Y2), S2.copy())
    gf = Grove(X1, Y1, 1.0, 1, 2, list(FUNCS), sigma=S1.copy(),
               node_list=[make_tree("a0*tanh(a1*x)", X1, FUNCS).get_root()],
               operations=list(OPS4), use_gradients=False, seed=13,
               g_tau=1.0, datasets=DSETS)
    gf.compute_posteriors(Y1, False)
    joint = float(gf.root_posteriors[0])
    check("MD-f1 sigma-array joint == sum of independent per-dataset "
          "scores", abs(joint - (post1 + post2))
          < 1e-9 * max(1.0, abs(joint)),
          "joint=%r sum=%r" % (joint, post1 + post2))
    # (ii) profiled sigma (None): k+1 penalty per dataset
    tp = make_tree("a0*tanh(a1*x)", X1, FUNCS, sigma=1.0)
    pp1, nvp1 = tp.score_dataset(np.ascontiguousarray(X1),
                                 np.ascontiguousarray(Y1), None)
    r1 = Y1 - np.asarray(tp.evaluate(X1), dtype=np.float64)
    mse1 = float(np.mean(r1 * r1))
    exp1 = (-0.5 * X1.size * (np.log(2 * np.pi * mse1) + 1.0)
            - 0.5 * (len(nvp1) + 1) * np.log(X1.size))
    check("MD-f2 profiled branch: analytic k+1 pin on the explicit-x "
          "entry", abs(pp1 - exp1) < 1e-6 * max(1.0, abs(exp1)),
          "post=%r exp=%r" % (pp1, exp1))

    # ---------------------------------------------------------------- MD-g
    print("[MD-g] joint x transparent atoms (Phase FX)")
    spec = real_functionC.FlexAtomSpec("G", "c0*tanh(s)")
    slot = make_tree("a0*x", X1, FUNCS, values={"a0": 1.0})
    fnode = function_treeC.make_flex_node(spec, slot.get_root(), ["c0"],
                                          None)
    ftree = function_treeC.Tree(0, list(FUNCS),
                                np.ascontiguousarray(X1), node=fnode)
    ftree.set_sigma(S1.copy())
    ftree.set_g_tau(1.0)
    ftree.tree_update(ftree.get_root())
    gg = Grove(X1, Y1, 1.0, 1, 2, list(FUNCS), sigma=S1.copy(),
               node_list=[ftree.get_root()], operations=list(OPS4),
               use_gradients=False, seed=19, g_tau=1.0,
               datasets=DSETS, flex_atoms=[spec])
    gg.compute_posteriors(Y1, False)
    pg = float(gg.root_posteriors[0])
    thg = gg.joint_thetas(gg.trees[0])
    check("MD-g1 scaffolded skeleton scores finite jointly",
          np.isfinite(pg) and pg > -1.0e90, repr(pg))
    check("MD-g2 per-dataset thetas cover the flex internal c0 and "
          "DIFFER across datasets (theta_true differ)",
          thg is not None and "c0" in thg[0] and "c0" in thg[1]
          and abs(thg[0]["c0"] - thg[1]["c0"]) > 0.1,
          repr(thg))
    gg.run(Y1, 80)
    n_fx = len(function_treeC.collect_flex_nodes(
        gg.trees[0].get_root(), None))
    check("MD-g3 scaffold immortal under the joint kernel", n_fx == 1)

    # ---------------------------------------------------------------- MD-i
    print("[MD-i] refusals + contract + Forest")
    expect_raises("MD-i1 burn_in + datasets refused (D2)", ValueError,
                  Grove, X1, Y1, 1.0, 2, 2, list(FUNCS), burn_in=True,
                  datasets=DSETS)
    expect_raises("MD-i2 positional x must equal datasets[0][0]",
                  ValueError, Grove, X2, Y2, 1.0, 2, 2, list(FUNCS),
                  datasets=DSETS)
    expect_raises("MD-i3 bad combine mode refused", ValueError,
                  Grove, X1, Y1, 1.0, 2, 2, list(FUNCS), datasets=DSETS,
                  md_combine="median")
    expect_raises("MD-i4 baselines length mismatch refused", ValueError,
                  Grove, X1, Y1, 1.0, 2, 2, list(FUNCS), datasets=DSETS,
                  md_baselines=[0.0])
    expect_raises("MD-i5 dataset x/y length mismatch refused", ValueError,
                  Grove, X1, Y1, 1.0, 2, 2, list(FUNCS),
                  datasets=[(X1, Y1, S1), (X2, Y1, S2)])
    fo = Forest(X1, Y1, np.array([1.0, 2.0]), 2, 2, list(FUNCS),
                sigma=S1.copy(), operations=list(OPS4), seed=29,
                g_tau=1.0, datasets=DSETS)
    fo.compute_posteriors_all(Y1)
    fo.run_all(Y1, 25)
    fo.attempt_swaps(Y1)
    fo.attempt_crossovers(Y1)
    posts_ok = all(np.isfinite(p) or p <= -1.0e290
                   for g in fo.groves
                   for p in np.asarray(g.root_posteriors))
    check("MD-i6 Forest serial EMC smoke under joint scoring "
          "(run/swap/crossover)", posts_ok)
    expect_raises("MD-i7 run_mpi_spawn refuses datasets (D8)",
                  NotImplementedError, fo.run_mpi_spawn, Y1, 1, 1)
    # PR hard guard: re-enable by hand -> the PR branch must raise
    gpr = Grove(X1, Y1, 1.0, 2, 2, list(FUNCS), sigma=S1.copy(),
                Npars=1, operations=list(OPS4), use_gradients=False,
                seed=31, g_tau=1.0, datasets=[(X1, Y1, S1)])
    gpr.compute_posteriors(Y1, False)
    gpr.pr_prob = 1.0                       # force the PR branch
    try:
        gpr.run(Y1, 3)
        check("MD-i8 hand-re-enabled PR raises under datasets (D1)",
              False, "no exception")
    except ValueError:
        check("MD-i8 hand-re-enabled PR raises under datasets (D1)", True)

    # min/softmin end-to-end: the Grove's reduced energy must equal a
    # MANUAL reduction of independent score_dataset fits (review finding
    # 10: finiteness alone was too weak a pin).
    import warnings as _w
    m1 = make_tree("a0*tanh(a1*x)", X1, FUNCS, sigma=S1.copy())
    mp1, mn1 = m1.score_dataset(np.ascontiguousarray(X1),
                                np.ascontiguousarray(Y1), S1.copy())
    m2 = make_tree("a0*tanh(a1*x)", X1, FUNCS, sigma=S1.copy())
    mp2, mn2 = m2.score_dataset(np.ascontiguousarray(X2),
                                np.ascontiguousarray(Y2), S2.copy())
    pens = [groveC.md_dataset_penalty(len(mn1), X1.size, S1),
            groveC.md_dataset_penalty(len(mn2), X2.size, S2)]
    for mode in ("min", "softmin"):
        manual = groveC.reduce_joint([mp1, mp2], pens, mode, 8.0, None)
        with _w.catch_warnings():
            _w.simplefilter("ignore")     # unbaselined-min advisory
            gm = Grove(X1, Y1, 1.0, 1, 2, list(FUNCS), sigma=S1.copy(),
                       node_list=[make_tree("a0*tanh(a1*x)", X1,
                                            FUNCS).get_root()],
                       operations=list(OPS4), use_gradients=False,
                       seed=37, g_tau=1.0, datasets=DSETS,
                       md_combine=mode)
        gm.compute_posteriors(Y1, False)
        got = float(gm.root_posteriors[0])
        check("MD-i9 %s mode == manual reduction of independent fits"
              % mode, abs(got - manual) < 1e-9 * max(1.0, abs(manual)),
              "got=%r manual=%r" % (got, manual))
    # (review findings 2/3: refusal coverage beyond the ctor flag)
    expect_raises("MD-i10 size==0 burn-in idiom refused with datasets",
                  ValueError, Grove, X1, Y1, 1.0, 0, 2, list(FUNCS),
                  datasets=[(X1, Y1, S1)])
    gref = Grove(X1, Y1, 1.0, 1, 2, list(FUNCS), sigma=S1.copy(),
                 node_list=[make_tree("a0*x + a1", X1, FUNCS).get_root()],
                 operations=list(OPS4), use_gradients=False, seed=47,
                 g_tau=1.0, datasets=[(X1, Y1, S1)])
    expect_raises("MD-i11 burn_in_combinations refused post-ctor",
                  ValueError, gref.burn_in_combinations, Y1)
    expect_raises("MD-i12 burn_in_depth2 refused post-ctor",
                  ValueError, gref.burn_in_depth2, Y1)

    # ---------------------------------------------------------------- MD-j
    print("[MD-j] review-fix regressions (2026-07-31 adversarial review)")
    # j1 (finding 1): RANDOM-INIT joint Groves must seed root_posteriors
    # at the JOINT scale. Duplicate-dataset identity extended to the
    # ctor path: same seed => same initial trees => joint(D,D) init
    # posts == 2 x joint(D) init posts, exactly.
    gj1 = Grove(X1, Y1, 1.0, 3, 2, list(FUNCS), sigma=S1.copy(), Npars=1,
                operations=list(OPS4), use_gradients=False, seed=41,
                g_tau=1.0, datasets=[(X1, Y1, S1)])
    gj2 = Grove(X1, Y1, 1.0, 3, 2, list(FUNCS), sigma=S1.copy(), Npars=1,
                operations=list(OPS4), use_gradients=False, seed=41,
                g_tau=1.0, datasets=[(X1, Y1, S1), (X1, Y1, S1)])
    pj1 = [float(p) for p in gj1.root_posteriors]
    pj2 = [float(p) for p in gj2.root_posteriors]
    check("MD-j1 random-init ctor posts are JOINT-scale "
          "(joint(D,D) == 2 x joint(D) at init, exactly)",
          all(pj2[i] == 2.0 * pj1[i] for i in range(3)),
          "%s vs %s" % (pj2, pj1))
    # j2 (finding 2): joint_thetas hands back COPIES — mutating them must
    # not corrupt the cache. (The ctor pops its own cache entries to keep
    # J=1 exactness, so repopulate via compute_posteriors first.)
    gj2.compute_posteriors(Y1, False)
    ths_a = gj2.joint_thetas(gj2.trees[0])
    if ths_a is not None and len(ths_a) > 0 and len(ths_a[0]) > 0:
        k0 = list(ths_a[0].keys())[0]
        ths_a[0][k0] = 12345.0
        ths_b = gj2.joint_thetas(gj2.trees[0])
        check("MD-j2 joint_thetas returns deep copies (cache immune to "
              "caller mutation)", ths_b[0][k0] != 12345.0, repr(ths_b[0]))
    else:
        check("MD-j2 joint_thetas returns deep copies (cache immune to "
              "caller mutation)", ths_a is not None, repr(ths_a))
    # j3 (finding 4): dataset x with an EXACT ZERO (e.g. t=0 at the
    # aligned peak) is scrubbed like the legacy domain, so 1/x-bearing
    # candidates stay in-support on every dataset — and the CALLER's
    # array is never mutated.
    XZ = np.linspace(-1.0, 1.0, 21)          # contains exact 0.0
    XZ_user = XZ.copy()
    YZ = 2.0 / (XZ_user + 3.0)
    SZ = np.full(21, 0.1)
    inv_t = make_tree("a0/x", X1, FUNCS, sigma=S1.copy())
    gz = Grove(X1, Y1, 1.0, 1, 2, list(FUNCS), sigma=S1.copy(),
               node_list=[inv_t.get_root()], operations=list(OPS4),
               use_gradients=False, seed=43, g_tau=1.0,
               datasets=[(X1, Y1, S1), (XZ, YZ, SZ)])
    gz.compute_posteriors(Y1, False)
    pz = float(gz.root_posteriors[0])
    check("MD-j3a zero-bearing dataset x is scrubbed (1/x stays "
          "in-support jointly)", np.isfinite(pz) and pz > -1.0e290,
          repr(pz))
    check("MD-j3b caller's dataset array is NOT mutated by the scrub",
          np.array_equal(XZ, XZ_user) and 0.0 in XZ,
          repr(XZ[np.argmin(np.abs(XZ))]))

    # ------------------------------------------------------------- summary
    print("=" * 66)
    print("test_phaseMD: %d passed, %d failed" % (N_PASS, N_FAIL))
    return 1 if N_FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
