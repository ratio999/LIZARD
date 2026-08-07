"""test_sigma_array.py -- certification of the per-point-sigma (sigma-array)
scoring/optimize path (sigma-array fix, 2026-07-28).

WHAT THIS PINS
--------------
The sigma-array route existed in the kernel since Phase 2.7(a) but was
unreachable: Tree._update_parameters / _update_parameters_and_fitness
passed (input_pars, method_str, use_gradients, self.sigma) into
Node.optimize_sigma(..., input_pars, SIGMA, method_str, use_gradients) --
the str landed in the typed sigma buffer slot and raised TypeError
(flagged at FD review, flexible_domain_plan section 9, "never exercised
by suites"). Additionally gp_movesC.parameter_refresh took `double sigma`
and crashed on the first PR draw of any Grove configured with an array.
Both are fixed; parameter_refresh gained an ADDITIVE per-point branch
(Tree._log_likelihood_at_sigma_arr); the scalar paths are byte-untouched
and remain certified by the historical suites + fingerprints.

CHECKS
------
  [SA-a] regression: array-sigma optimize completes at BOTH fixed call
         sites and lands on the analytic weighted-least-squares optimum;
  [SA-b] analytic scorer identity (optimize=False):
         post == -0.5*sum(r_i^2/sigma_i^2) - 0.5*k*log(N);
  [SA-c] constant array == scalar: optimize=False exact, optimize=True
         same Powell landing (posts + theta-hat agree);
  [SA-d] zero-likelihood semantics on the array path: a tree with any
         non-finite prediction scores at penalty scale (< -1e50) --
         the PLAN 4.2b NaN-leak contract holds on the new branch;
  [SA-e] parameter_refresh: (i) no TypeError under array sigma,
         (ii) scalar vs constant-array chains are step-identical under
         pinned RNG (same accepts, same theta trajectory);
  [SA-f] Grove EMC smoke with a heteroscedastic sigma ramp (d=1):
         finite posts, sane acceptance, PR fired, same-seed determinism;
  [SA-g] d=2 (flexible domain) x sigma-array Grove smoke;
  [SA-h] Forest smoke: swaps + trans-grove crossovers under array sigma.

RUN
---
  $ENV/bin/python test_sigma_array.py          (no MPI needed)
"""
import os
import sys
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


def make_tree(expr_str, domain, funcs, values=None):
    """expr -> Tree with pinned parameter values (name -> value)."""
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
    return tree


# ---------------------------------------------------------------- fixtures
RNG = np.random.default_rng(7)
X = np.linspace(0.5, 3.0, 60)
SIG_RAMP = np.linspace(0.05, 0.4, 60)          # heteroscedastic ramp
Y = 2.0 * X + 0.5 + RNG.normal(0.0, 1.0, X.size) * SIG_RAMP
FUNCS = ["sin(x)", "x"]
OPS4 = ["+", "-", "*", "/"]

# Analytic weighted least squares for y ~ a0*x + a1 with weights 1/sigma^2.
_W = 1.0 / SIG_RAMP ** 2
_A = np.vstack([X, np.ones_like(X)]).T
_AW = _A * _W[:, None]
THETA_WLS = np.linalg.solve(_A.T @ _AW, _AW.T @ Y)   # [a0, a1]


def main():
    gp_movesC.set_max_tree_depth(4)
    gp_movesC.set_subtree_rebuild_max_height(2)

    # ---------------------------------------------------------------- SA-a
    print("[SA-a] array-sigma optimize path (both fixed call sites)")
    # Site 1: update_parameters -> _update_parameters
    t1 = make_tree("a0*x + a1", X, FUNCS)
    t1.set_sigma(SIG_RAMP.copy())
    t1.set_g_tau(1.0)
    try:
        t1.update_parameters(X, Y)
        nv = t1.get_param_dicts()[1]
        check("SA-a1 update_parameters runs (was TypeError)", True)
        # NB expr_to_tree REASSIGNS parameter names by encounter order in
        # sympy's canonical arg ordering ("a0*x + a1" -> tree a0 =
        # intercept, a1 = slope), so compare role-agnostically: the
        # fitted PREDICTION must match the analytic WLS prediction, and
        # the fitted values as a (sorted) set must match the WLS pair.
        pred = np.asarray(t1.evaluate(X), dtype=np.float64)
        pred_wls = _A @ THETA_WLS
        check("SA-a2 Powell lands on the WLS optimum",
              float(np.max(np.abs(pred - pred_wls))) < 2e-2
              and np.all(np.abs(np.sort(list(nv.values()))
                                - np.sort(THETA_WLS)) < 2e-2),
              "nv=%s wls=%s" % (nv, THETA_WLS))
    except TypeError as e:
        check("SA-a1 update_parameters runs (was TypeError)", False, repr(e))
    # Site 2: log_marginal_bic(optimize=True) -> _update_parameters_and_fitness
    t2 = make_tree("a0*x + a1", X, FUNCS)
    t2.set_sigma(SIG_RAMP.copy())
    t2.set_g_tau(1.0)
    try:
        post2 = t2.log_marginal_bic(Y, True, False, "Powell")
        check("SA-a3 log_marginal_bic(optimize=True) finite", np.isfinite(post2),
              "post=%r" % post2)
        # Fast path returns the SAME quantity as the analytic formula at
        # the fitted theta (mse from result.fun is already /sigma^2).
        # Residuals via tree.evaluate: role-agnostic w.r.t. the
        # encounter-order parameter naming (see SA-a2 note).
        nv2 = t2.get_param_dicts()[1]
        r = Y - np.asarray(t2.evaluate(X), dtype=np.float64)
        expected2 = -0.5 * np.sum(r * r / SIG_RAMP ** 2) \
                    - 0.5 * len(nv2) * np.log(X.size)
        check("SA-a4 fast-path post == analytic at theta-hat",
              abs(post2 - expected2) < 1e-6 * max(1.0, abs(expected2)),
              "post=%r expected=%r" % (post2, expected2))
    except TypeError as e:
        check("SA-a3 log_marginal_bic(optimize=True) finite", False, repr(e))

    # ---------------------------------------------------------------- SA-b
    print("[SA-b] analytic scorer identity (optimize=False)")
    t3 = make_tree("a0*x", X, FUNCS, values={"a0": 1.7})
    t3.set_sigma(SIG_RAMP.copy())
    post3 = t3.log_marginal_bic(Y, False, False, "Powell")
    r3 = Y - 1.7 * X
    expected3 = -0.5 * np.sum(r3 * r3 / SIG_RAMP ** 2) - 0.5 * 1 * np.log(X.size)
    check("SA-b1 slow-path post == -0.5*sum(r^2/s^2) - 0.5*k*logN",
          abs(post3 - expected3) < 1e-9 * abs(expected3),
          "post=%r expected=%r" % (post3, expected3))
    # list input must coerce, not TypeError (boundary hardening)
    t3l = make_tree("a0*x", X, FUNCS, values={"a0": 1.7})
    t3l.set_sigma([float(s) for s in SIG_RAMP])
    post3l = t3l.log_marginal_bic(Y, False, False, "Powell")
    check("SA-b2 list sigma coerces to the same score",
          abs(post3l - post3) < 1e-12 * abs(post3),
          "post=%r post_list=%r" % (post3, post3l))

    # ---------------------------------------------------------------- SA-c
    print("[SA-c] constant array == scalar consistency")
    S0 = 0.2
    SIG_CONST = np.full(X.size, S0)
    ta = make_tree("a0*x + a1", X, FUNCS)
    ts = make_tree("a0*x + a1", X, FUNCS)
    ta.set_sigma(SIG_CONST.copy())
    ts.set_sigma(S0)
    pa0 = ta.log_marginal_bic(Y, False, False, "Powell")
    ps0 = ts.log_marginal_bic(Y, False, False, "Powell")
    check("SA-c1 optimize=False: const-array == scalar (exact)",
          abs(pa0 - ps0) < 1e-10 * abs(ps0), "arr=%r sc=%r" % (pa0, ps0))
    pa1 = ta.log_marginal_bic(Y, True, False, "Powell")
    ps1 = ts.log_marginal_bic(Y, True, False, "Powell")
    nva = ta.get_param_dicts()[1]
    nvs = ts.get_param_dicts()[1]
    check("SA-c2 optimize=True: posts agree",
          abs(pa1 - ps1) < 1e-5 * abs(ps1), "arr=%r sc=%r" % (pa1, ps1))
    check("SA-c3 optimize=True: theta-hat agrees",
          all(abs(nva[k] - nvs[k]) < 1e-3 * max(1.0, abs(nvs[k]))
              for k in nvs), "arr=%s sc=%s" % (nva, nvs))

    # ---------------------------------------------------------------- SA-d
    print("[SA-d] zero-likelihood semantics on the array path")
    XN = np.linspace(-2.0, 2.0, 50)          # negatives -> x**0.5 is NaN
    tn = make_tree("x**0.5", XN, ["x"])      # 0.5 becomes param a0=0.5
    tn.set_sigma(np.full(XN.size, 0.1))
    postn = tn.log_marginal_bic(np.ones(XN.size), False, False, "Powell")
    check("SA-d1 partial-NaN prediction scores at penalty scale",
          postn < -1e50, "post=%r" % postn)

    # ---------------------------------------------------------------- SA-e
    print("[SA-e] parameter_refresh under array sigma")
    tp = make_tree("a0*x + a1", X, FUNCS)
    tp.set_sigma(SIG_RAMP.copy())
    tp.set_g_tau(1.0)
    names = gp_movesC.pr_coord_names(tp)
    am = gp_movesC.AMState(names)
    gp_movesC.set_rng(random.Random(99), np.random.default_rng(99))
    try:
        out = gp_movesC.parameter_refresh(tp, Y, SIG_RAMP.copy(), am, 1.0)
        check("SA-e1 PR runs under array sigma (was TypeError)",
              isinstance(out, tuple) and out[0] in (True, False))
    except TypeError as e:
        check("SA-e1 PR runs under array sigma (was TypeError)", False, repr(e))
    # (ii) scalar chain vs constant-array chain: step-identical under
    # pinned RNG (logL values equal to fp assoc; same draws -> same path).
    NSTEP = 25

    def run_pr_chain(sigma_obj):
        t = make_tree("a0*x + a1", X, FUNCS)
        t.set_sigma(0.2 if not hasattr(sigma_obj, "__len__") else
                    np.asarray(sigma_obj, dtype=np.float64))
        t.set_g_tau(1.0)
        a = gp_movesC.AMState(gp_movesC.pr_coord_names(t))
        gp_movesC.set_rng(random.Random(123), np.random.default_rng(123))
        accepts = []
        for _ in range(NSTEP):
            acc, _l1, _l2 = gp_movesC.parameter_refresh(t, Y, sigma_obj, a, 1.0)
            accepts.append(bool(acc))
        return accepts, t.get_param_dicts()[1]

    acc_s, nv_s = run_pr_chain(0.2)
    acc_a, nv_a = run_pr_chain(np.full(X.size, 0.2))
    check("SA-e2 scalar vs const-array PR chains: same accepts",
          acc_s == acc_a, "%s vs %s" % (acc_s, acc_a))
    check("SA-e3 scalar vs const-array PR chains: same theta",
          all(abs(nv_s[k] - nv_a[k]) < 1e-9 for k in nv_s),
          "%s vs %s" % (nv_s, nv_a))

    # SA-e4: VALUE-LEVEL pin of the heteroscedastic likelihood. The
    # constant-array checks above cannot distinguish a wrong per-point
    # weight (r^2/s vs r^2/s^2, dropped weights, nansum leniency): the
    # log-det cancels in the ratio and a constant array degenerates to
    # the scalar case. Here we replicate the ENTIRE PR decision sequence
    # independently in numpy -- AMState cold-start AND warm proposals,
    # Robbins-Monro updates, accept draws, and the per-point Gaussian
    # logL under a genuinely heteroscedastic sigma (x8 ramp) -- and
    # demand the kernel trajectory match step-for-step. Any weighting
    # error changes some alpha by orders of magnitude and diverges the
    # trajectory within a few steps.
    NSTEP4 = 12
    tpin = make_tree("a0*x + a1", X, FUNCS)
    tpin.set_g_tau(1.0)
    names4 = list(gp_movesC.pr_coord_names(tpin))
    K4 = len(names4)
    am4 = gp_movesC.AMState(names4)
    gp_movesC.set_rng(random.Random(77), np.random.default_rng(77))
    py77 = random.Random(77)
    np77 = np.random.default_rng(77)
    rf_root = tpin.get_root().get_fun()
    theta_m = np.array([tpin.get_param_dicts()[1][k] for k in names4])
    # AMState replica state (defaults from gp_movesC)
    mu_m = np.zeros(K4)
    C_m = np.eye(K4) * 0.01           # rel^2
    loglam_m = 0.0
    nvis_m = 0
    ALPHA_STAR = 0.234 if K4 > 1 else 0.44
    KAPPA, REL, EPS, JIT = 0.6, 0.1, 1e-6, 1e-8

    def logL_pp(th):
        pred = np.asarray(rf_root.get_eval(
            X, {names4[i]: float(th[i]) for i in range(K4)}),
            dtype=np.float64)
        r = Y - pred
        return (-0.5 * np.sum(np.log(2.0 * np.pi * SIG_RAMP ** 2))
                - 0.5 * np.sum(r * r / SIG_RAMP ** 2))

    acc_kernel, acc_mine = [], []
    for _ in range(NSTEP4):
        acc, _x1, _x2 = gp_movesC.parameter_refresh(
            tpin, Y, SIG_RAMP.copy(), am4, 1.0)
        acc_kernel.append(bool(acc))
        # --- replica of one PR step ---
        if nvis_m < K4 + 1:
            scale = np.maximum(EPS, np.abs(theta_m) * REL)
            th_new = theta_m + np77.normal(0.0, scale)
        else:
            cov = np.exp(2.0 * loglam_m) * (C_m + JIT * np.eye(K4))
            th_new = theta_m + np.linalg.cholesky(cov) @ \
                np77.standard_normal(K4)
        la = logL_pp(th_new) - logL_pp(theta_m)
        if not np.isfinite(logL_pp(th_new)):
            la = -np.inf
        alpha = 1.0 if la >= 0.0 else (0.0 if la <= -700.0
                                       else float(np.exp(la)))
        u = py77.random()
        mine = u < alpha
        acc_mine.append(mine)
        th_after = th_new if mine else theta_m
        if mine:
            theta_m = th_new
        g = (nvis_m + 1) ** (-KAPPA)
        loglam_m += g * (alpha - ALPHA_STAR)
        d_old = th_after - mu_m
        mu_m = mu_m + g * d_old
        d_new = th_after - mu_m
        C_m = C_m + g * (np.outer(d_old, d_new) - C_m)
        C_m = 0.5 * (C_m + C_m.T)
        nvis_m += 1
    nv4 = tpin.get_param_dicts()[1]
    check("SA-e4 heteroscedastic PR: kernel accepts == numpy replica",
          acc_kernel == acc_mine, "%s vs %s" % (acc_kernel, acc_mine))
    check("SA-e5 heteroscedastic PR: kernel theta == numpy replica",
          all(abs(nv4[names4[i]] - theta_m[i]) < 1e-9 for i in range(K4)),
          "%s vs %s" % (nv4, theta_m))

    # SA-e6: new guards through the public boundary -- a sigma with a
    # non-positive entry or a wrong length must make PR a deterministic
    # reject (both states score -inf), never a crash or an accept.
    bad1 = SIG_RAMP.copy(); bad1[7] = 0.0
    bad2 = SIG_RAMP[:-3].copy()
    tg = make_tree("a0*x + a1", X, FUNCS)
    tg.set_g_tau(1.0)
    amg = gp_movesC.AMState(gp_movesC.pr_coord_names(tg))
    gp_movesC.set_rng(random.Random(5), np.random.default_rng(5))
    rej1 = [gp_movesC.parameter_refresh(tg, Y, bad1, amg, 1.0)[0]
            for _ in range(3)]
    rej2 = [gp_movesC.parameter_refresh(tg, Y, bad2, amg, 1.0)[0]
            for _ in range(3)]
    check("SA-e6 sigma<=0 / wrong-length sigma => deterministic reject",
          not any(rej1) and not any(rej2), "%s %s" % (rej1, rej2))

    # ---------------------------------------------------------------- SA-f
    print("[SA-f] Grove EMC smoke, heteroscedastic sigma (d=1)")

    def run_grove(seed):
        g = Grove(X.copy(), Y.copy(), 1.0, 4, 2, list(FUNCS),
                  sigma=SIG_RAMP.copy(), Npars=1, operations=list(OPS4),
                  use_gradients=False, seed=seed, g_tau=1.0)
        g.pr_prob = 0.15          # make PR fire often enough to certify
        g.run(Y, 300)
        return g

    g1 = run_grove(11)
    posts = np.asarray(g1.root_posteriors, dtype=np.float64)
    st = g1.move_stats()
    check("SA-f1 all posts finite", bool(np.all(np.isfinite(posts))),
          "%s" % posts)
    acc_rate = g1.acceptance_rate()
    check("SA-f2 acceptance in (0.01, 0.99)", 0.01 < acc_rate < 0.99,
          "rate=%r" % acc_rate)
    # ACCEPTED, not proposed: 'proposed' is incremented by the Grove
    # BEFORE parameter_refresh runs, so it cannot fail even if PR were a
    # permanent silent no-op; >=1 acceptance proves the array-sigma PR
    # branch genuinely executes end-to-end.
    check("SA-f3 parameter_refresh ACCEPTED under array sigma",
          st["parameter_refresh"]["accepted"] >= 1, "%s" % st)
    # sigma-identity pin: the ramp must actually live on the trees (a
    # kernel that silently dropped sigma would pass every smoke above).
    check("SA-f3b trees carry the sigma ramp",
          bool(np.allclose(np.asarray(g1.trees[0].get_sigma(),
                                      dtype=np.float64), SIG_RAMP)))
    # value pin: the stored best post must equal the analytic weighted
    # formula recomputed from the tree itself.
    bi = g1.best_index()
    bt1 = g1.trees[bi]
    rb = Y - np.asarray(bt1.evaluate(X), dtype=np.float64)
    kb = len(bt1.get_param_dicts()[1])
    exp_b = -0.5 * np.sum(rb * rb / SIG_RAMP ** 2) \
        - 0.5 * kb * np.log(X.size)
    check("SA-f3c stored best post == analytic weighted BIC",
          abs(float(posts[bi]) - exp_b) < 1e-6 * max(1.0, abs(exp_b)),
          "stored=%r analytic=%r" % (float(posts[bi]), exp_b))
    g2 = run_grove(11)
    d1 = [t.tree_get_description() for t in g1.trees]
    d2 = [t.tree_get_description() for t in g2.trees]
    check("SA-f4 same-seed determinism (descriptions)", d1 == d2,
          "%s vs %s" % (d1, d2))
    check("SA-f5 same-seed determinism (posts)",
          bool(np.allclose(posts,
                           np.asarray(g2.root_posteriors, dtype=np.float64),
                           rtol=0, atol=1e-12, equal_nan=True)))

    # ---------------------------------------------------------------- SA-g
    print("[SA-g] d=2 flexible domain x sigma-array smoke")
    try:
        n2 = 64
        rng2 = np.random.default_rng(3)
        X2 = np.vstack([np.linspace(-2.0, 2.0, n2),
                        np.linspace(0.5, 3.0, n2)])
        X2 = np.ascontiguousarray(X2, dtype=np.float64)
        SIG2 = np.linspace(0.05, 0.3, n2)
        Y2 = 2.5 * X2[0] + np.sin(X2[1]) + rng2.normal(0, 1, n2) * SIG2
        g2d = Grove(X2, Y2, 1.0, 4, 2, ["x0", "x1", "sin(x1)"],
                    sigma=SIG2, Npars=1, operations=list(OPS4),
                    use_gradients=False, seed=17, g_tau=1.0,
                    variables=("x0", "x1"))
        g2d.run(Y2, 80)
        p2 = np.asarray(g2d.root_posteriors, dtype=np.float64)
        check("SA-g1 d=2 array-sigma Grove: finite posts",
              bool(np.all(np.isfinite(p2))), "%s" % p2)
    finally:
        real_functionC.set_variables(("x",))
    check("SA-g2 registry restored to legacy",
          real_functionC.get_variables() == ("x",))

    # ---------------------------------------------------------------- SA-h
    print("[SA-h] Forest smoke: swaps/crossovers under array sigma")
    f = Forest(X.copy(), Y.copy(), np.array([1.0, 1.6]), 3, 2, list(FUNCS),
               sigma=SIG_RAMP.copy(), Npars=1, operations=list(OPS4),
               use_gradients=False, seed=5)
    f.compute_posteriors_all(Y)
    for _ in range(8):
        f.run_all(Y, 3)
        f.attempt_swaps(Y)
        f.attempt_crossovers(Y)
    ok_posts = all(np.all(np.isfinite(np.asarray(g.root_posteriors,
                                                 dtype=np.float64)))
                   for g in f.groves)
    check("SA-h1 Forest posts finite through swaps+crossovers", ok_posts)
    ls = f.ladder_stats()
    check("SA-h2 ladder monitors sane",
          ls["swap_attempts"][0] >= 1 and ls["xover_attempts"][0] >= 1,
          "%s" % ls)
    bt = f.best_tree()
    check("SA-h3 best_tree available", bt is not None,
          "" if bt is None else bt.tree_get_description())

    print("\n[sigma-array] PASS=%d FAIL=%d" % (N_PASS, N_FAIL), flush=True)
    return 1 if N_FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
