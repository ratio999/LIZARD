"""test_phasePP.py -- certification of Phase PP pinned parameters in
opaque atoms (2026-08-06, PROMPT_DEV_PINNED_PARAMS.md; PLAN MR.5).

WHAT THIS PINS (pun intended)
-----------------------------
`AtomSpec(name, template, pinned={placeholder: float})` fixes a strict
SUBSET of a template's placeholders to physics-supplied constants. The
values are SUBSTITUTED into the template expression at spec
construction, so downstream they are numeric literals inside the
compiled leaf function: structurally incapable of appearing in
get_params / name_values, the optimizer vector, the BIC k, theta dicts,
or rename maps. Identity: the canonical vocabulary symbol becomes
`name[c1=v1,c2=v2]` (sorted keys, shortest round-trip float repr) — a
pinned atom is a BASIC FUNCTION whose complexity is charged through the
Polya prior (its own cell), never through k, and different pin values
canonicalize to different symbols (distinct cells / registry keys /
burn-in tokens / wire payloads). `expr_to_tree(..., pin_numbers=True)`
renders numeric literals as constant leaves (default False = legacy).
Everything is opt-in; unpinned paths are bit-identical (wire configs
keep the historical (name, template) pairs).

CHECKS (brief T-P1..T-P6 mapping noted per section)
------
  [PP-a] spec construction, identity, validation, pickling  (T-P2a)
  [PP-b] leaf integration: k excludes pins, optimizer cannot move them
         (bitwise prediction check), analytic BIC pin        (T-P1)
  [PP-c] descriptions faithful + skel/wire round trip        (T-P2)
  [PP-d] long move chain: pins never renamed/freed/perturbed (T-P4)
  [PP-e] burn-in with a pinned vocabulary (serial ctor path); spawn
         sampling wire under --mpi (mpiexec -n 1)            (T-P3)
  [PP-f] datasets= (J=2): per-dataset theta excludes pins; joint
         finite; pins identical across datasets              (T-P5)
  [PP-g] expr_to_tree pin_numbers True/False                 (T-P6)
  [PP-h] Polya: extended symbol is its own cell; coverage

RUN
---
  $ENV/bin/python test_phasePP.py            (serial checks)
  $MPI -n 1 $ENV/bin/python test_phasePP.py --mpi   (adds the spawn check)
"""
import os
import sys
import copy
import pickle
import random

os.environ.setdefault("OMP_NUM_THREADS", "1")
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_SRC = os.path.join(os.path.dirname(_HERE), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np
import sympy as sp

import real_functionC
import function_treeC
import gp_movesC
import polya_treeC
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


def atom_tree(spec, names, theta, domain, funcs, sigma=1.0):
    """One pinned-atom leaf as a full Tree, thetas installed."""
    rf = real_functionC.instantiate_atom(spec,
                                         np.ascontiguousarray(domain,
                                                              dtype=np.float64),
                                         list(names))
    node = function_treeC.Node(rf)
    node.theta = dict(theta)
    tree = function_treeC.Tree(0, list(funcs),
                               np.ascontiguousarray(domain,
                                                    dtype=np.float64),
                               node=node)
    tree.set_sigma(sigma)
    tree.set_g_tau(1.0)
    tree.tree_update(tree.get_root())
    return tree


# ---------------------------------------------------------------- fixtures
RNG = np.random.default_rng(29)
X = np.linspace(0.5, 3.0, 60)
FUNCS = ["sin(x)", "x"]
OPS4 = ["+", "-", "*", "/"]
D1V, W1V = 0.1731, 0.0806                    # the P-MOT style constants
PIN = {"c1": D1V, "c2": W1V}
TPL = "c0*exp(-c1*x)*sin(c2*x + c3)"


def ref_eval(x, c0, c3):
    return c0 * np.exp(-D1V * x) * np.sin(W1V * x + c3)


def main():
    gp_movesC.set_max_tree_depth(4)
    gp_movesC.set_subtree_rebuild_max_height(2)
    run_mpi = "--mpi" in sys.argv

    # ---------------------------------------------------------------- PP-a
    print("[PP-a] spec construction / identity / validation / pickling")
    spec = real_functionC.AtomSpec("Abeat", TPL, PIN)
    check("PP-a1 slots exclude pins; k = free count",
          spec.slots == ["c0", "c3"] and spec.k == 2,
          "slots=%r k=%r" % (spec.slots, spec.k))
    check("PP-a2 canonical symbol extends the name deterministically",
          spec.name == "Abeat[c1=0.1731,c2=0.0806]"
          and spec.base_name == "Abeat",
          repr(spec.name))
    e_str = str(spec.expr)
    check("PP-a3 pins are LITERALS in the spec expr (no c1/c2 symbols)",
          "0.1731" in e_str and "0.0806" in e_str
          and "c1" not in e_str and "c2" not in e_str, e_str)
    spec_other = real_functionC.AtomSpec("Abeat", TPL, {"c1": 0.2, "c2": W1V})
    check("PP-a4 different pin values => different canonical symbols",
          spec_other.name != spec.name, repr(spec_other.name))
    legacy = real_functionC.AtomSpec("Aleg", "c0*sin(c1*x)")
    check("PP-a5 unpinned spec unchanged (name==base_name, pinned None, "
          "legacy 2-tuple reduce)",
          legacy.name == "Aleg" and legacy.base_name == "Aleg"
          and legacy.pinned is None
          and legacy.__reduce__() == (real_functionC.AtomSpec,
                                      ("Aleg", "c0*sin(c1*x)")))
    rt = pickle.loads(pickle.dumps(spec))
    check("PP-a6 pickle round trip preserves identity + pins + k",
          rt.name == spec.name and rt.base_name == "Abeat"
          and rt.pinned == {"c1": D1V, "c2": W1V} and rt.k == 2
          and str(rt.expr) == str(spec.expr))
    expect_raises("PP-a7 pinned key not a placeholder rejected", ValueError,
                  real_functionC.AtomSpec, "B", TPL, {"c9": 1.0})
    expect_raises("PP-a8 non-finite pin rejected", ValueError,
                  real_functionC.AtomSpec, "B", TPL, {"c1": float("nan")})
    expect_raises("PP-a9 pinning EVERY placeholder rejected", ValueError,
                  real_functionC.AtomSpec, "B", "c0*sin(c1*x)",
                  {"c0": 1.0, "c1": 2.0})
    expect_raises("PP-a10 variable-annihilating pin rejected", ValueError,
                  real_functionC.AtomSpec, "B", "c0*sin(c1*x)", {"c1": 0.0})
    expect_raises("PP-a11 bad pin key pattern rejected", ValueError,
                  real_functionC.AtomSpec, "B", TPL, {"sigma": 1.0})
    # Review 2026-08-06 fixes:
    expect_raises("PP-a12 pin annihilating a FREE slot rejected "
                  "(zero pin would break the skel wire arity)",
                  ValueError, real_functionC.AtomSpec, "B",
                  "c0*sin(x) + c1*c2", {"c2": 0.0})
    expect_raises("PP-a13 brackets in a base name rejected "
                  "(extended-symbol spoofing)", ValueError,
                  real_functionC.AtomSpec, "A[c1=0.5]", TPL)
    hp = 0.05819273645182937            # needs all 17 digits
    spec_h = real_functionC.AtomSpec("Ah", "c0*exp(-c1*x)", {"c1": hp})
    th = atom_tree(spec_h, ["a0"], {"a0": 1.0}, X, FUNCS)
    ev_h = np.asarray(th.evaluate(X), dtype=np.float64)
    check("PP-a14 17-digit pin survives the string round trip EXACTLY "
          "(bitwise vs numpy at the same double)",
          np.array_equal(ev_h, np.exp(-hp * X)),
          "maxerr=%r" % float(np.max(np.abs(ev_h - np.exp(-hp * X)))))

    # ---------------------------------------------------------------- PP-b
    print("[PP-b] leaf integration: k, optimizer, BIC (T-P1)")
    t = atom_tree(spec, ["a0", "a1"], {"a0": 2.0, "a1": 0.7}, X, FUNCS,
                  sigma=0.05)
    nv = t.get_param_dicts()[1]
    check("PP-b1 name_values = free internals only (pins excluded from k)",
          sorted(nv.keys()) == ["a0", "a1"], repr(nv))
    leaf = t.get_leaves()[0]
    check("PP-b2 leaf_k / leaf_param_names see only free internals",
          leaf.leaf_k() == 2
          and sorted(leaf.leaf_param_names()) == ["a0", "a1"])
    pred = np.asarray(t.evaluate(X), dtype=np.float64)
    check("PP-b3 evaluation matches the pinned composition exactly",
          float(np.max(np.abs(pred - ref_eval(X, 2.0, 0.7)))) < 1e-12,
          "err=%r" % float(np.max(np.abs(pred - ref_eval(X, 2.0, 0.7)))))
    # T-P1 core: fit c0,c3 on synthetic data; the PIN must be bitwise
    # inert — predictions at a FIXED theta are identical before/after the
    # optimizer ran (a moved pin would change the compiled map itself).
    y = ref_eval(X, 1.7, 0.4) + RNG.normal(0, 0.01, X.size)
    # T-P1 faithful reading (review 2026-08-06 F3): the bitwise check
    # must inspect the tree the optimizer ACTUALLY TOUCHED — capture its
    # description and its fixed-theta prediction BEFORE the fit, then
    # re-install the same theta after the fit and demand bitwise
    # equality (a moved pin changes the compiled map itself).
    def install(tree, th):
        lf0 = tree.get_leaves()[0]
        for nm, v in th.items():
            lf0.theta[nm] = float(v)
        tree.tree_update(tree.get_root())
    desc_before = t.tree_get_description()
    install(t, {"a0": 1.0, "a1": 0.0})
    probe_before = np.asarray(t.evaluate(X), dtype=np.float64).copy()
    install(t, {"a0": 2.0, "a1": 0.7})
    t.update_parameters(np.ascontiguousarray(X), np.ascontiguousarray(y),
                        "Powell", False)
    nv_fit = dict(t.get_param_dicts()[1])
    check("PP-b4 optimizer fits ONLY free internals (recovers c0, c3; "
          "key set exactly the free names)",
          sorted(nv_fit.keys()) == ["a0", "a1"]
          and abs(nv_fit["a0"] - 1.7) < 0.2
          and abs(nv_fit["a1"] - 0.4) < 0.2,
          repr(nv_fit))
    install(t, {"a0": 1.0, "a1": 0.0})
    probe_after = np.asarray(t.evaluate(X), dtype=np.float64)
    check("PP-b5 pins bitwise inert ON THE FITTED TREE (description "
          "unchanged; fixed-theta predictions bitwise identical)",
          t.tree_get_description() == desc_before
          and np.array_equal(probe_before, probe_after))
    sig = np.full(X.size, 0.05)
    t2 = atom_tree(spec, ["a0", "a1"], {"a0": 1.7, "a1": 0.4}, X, FUNCS,
                   sigma=sig.copy())
    post = t2.log_marginal_bic(np.ascontiguousarray(y), False)
    r = y - np.asarray(t2.evaluate(X), dtype=np.float64)
    expected = -0.5 * np.sum(r * r / sig ** 2) - 0.5 * 2 * np.log(X.size)
    check("PP-b6 BIC k == 2 (pins uncharged; analytic pin)",
          abs(post - expected) < 1e-6 * max(1.0, abs(expected)),
          "post=%r expected=%r" % (post, expected))

    # ---------------------------------------------------------------- PP-c
    print("[PP-c] faithful descriptions + wire round trip (T-P2)")
    desc = t2.tree_get_description()
    check("PP-c1 description renders the pinned values numerically",
          "0.1731" in desc and "0.0806" in desc, desc)
    d_other = atom_tree(spec_other, ["a0", "a1"], {"a0": 1.7, "a1": 0.4},
                        X, FUNCS).tree_get_description()
    check("PP-c2 different pins => different canonical descriptions",
          d_other != desc, "%r vs %r" % (d_other, desc))
    payload, nvals, _p = groveC._tree_to_wire(t2, -1.0, True)
    check("PP-c3 wire payload is skel3b with the EXTENDED atom symbol",
          payload[0] == "skel3b" and spec.name in repr(payload),
          repr(payload)[:120])
    trt = groveC._tree_from_wire(payload, nvals,
                                 np.ascontiguousarray(X, dtype=np.float64),
                                 list(FUNCS), 1.0, 1.0, {spec.name: spec})
    check("PP-c4 wire round trip: description + predictions identical",
          trt is not None
          and trt.tree_get_description() == desc
          and float(np.max(np.abs(
              np.asarray(trt.evaluate(X)) - np.asarray(t2.evaluate(X)))))
          == 0.0)
    t_none = groveC._tree_from_wire(payload, nvals,
                                    np.ascontiguousarray(X,
                                                         dtype=np.float64),
                                    list(FUNCS), 1.0, 1.0,
                                    {spec_other.name: spec_other})
    check("PP-c5 wrong-pin registry cannot silently substitute "
          "(rebuild rejects)", t_none is None)
    # Review 2026-08-06 F6: the DANGEROUS path is a same-key registry
    # holding a wrong-pin spec — the desc-fidelity gate must reject it.
    t_forged = groveC._tree_from_wire(payload, nvals,
                                      np.ascontiguousarray(
                                          X, dtype=np.float64),
                                      list(FUNCS), 1.0, 1.0,
                                      {spec.name: spec_other})
    check("PP-c6 same-key/wrong-spec registry rejected by the "
          "description-fidelity gate", t_forged is None)

    # ---------------------------------------------------------------- PP-d
    print("[PP-d] long move chain: pins survive every move (T-P4)")
    nodes = []
    for _ in range(4):
        tt = atom_tree(spec, ["a0", "a1"], {"a0": 1.7, "a1": 0.4}, X,
                       FUNCS, sigma=0.05)
        nodes.append(tt.get_root())
    gd = Grove(X, y, 1.0, 4, 2, list(FUNCS), sigma=0.05,
               node_list=nodes, operations=list(OPS4),
               use_gradients=False, seed=7, g_tau=1.0, atoms=[spec])
    gd.compute_posteriors(y, False)
    gd.run(y, 250)
    # Review 2026-08-06 F4: the oracle is EXACT-literal extraction plus
    # registry identity — substring matching on "0.1731" would accept a
    # drifted 0.17312, and a rebuild that reset atom_symbol to the base
    # name (the REPORT §16.2 class) would keep the desc while breaking
    # wire identity.
    pins_ok = True
    n_atomic = 0
    for tt in gd.trees:
        for lf in tt.get_leaves():
            if lf.is_atomic():
                n_atomic += 1
                d = lf.get_fun().get_description()
                lits = set(abs(float(f)) for f in
                           sp.sympify(d).atoms(sp.Float))
                if not ({D1V, W1V} <= lits):
                    pins_ok = False
                if lf.get_fun().atom_symbol != spec.name:
                    pins_ok = False
                if lf.leaf_k() != 2:
                    pins_ok = False
                for nm in lf.leaf_param_names():
                    if nm in ("c1", "c2"):
                        pins_ok = False
    check("PP-d1 250-step chain (births/deaths/renames/crossover): every "
          "surviving atomic leaf carries BOTH pin literals exactly, the "
          "extended atom_symbol, k=2, and no pin-name leak",
          pins_ok, "atomic leaves seen: %d" % n_atomic)
    check("PP-d2 chain touched atomic leaves at all (test not vacuous)",
          n_atomic > 0, "n_atomic=%d" % n_atomic)
    stats = gd.move_stats()
    check("PP-d2b the chain proposed crossovers and PR (rename passes "
          "and coordinate machinery exercised)",
          stats["subtree_crossover"]["proposed"] > 0
          and stats["parameter_refresh"]["proposed"] > 0,
          repr({k: v["proposed"] for k, v in stats.items()}))
    # Deterministic backstop for the rename passes (F5): direct
    # rename_leaf_params on a pinned leaf.
    rfp = real_functionC.instantiate_atom(
        spec, np.ascontiguousarray(X, dtype=np.float64), ["a0", "a1"])
    rfr = real_functionC.rename_leaf_params(rfp, {"a0": "b7", "a1": "b8"})
    # NB abs(): sympy folds sign into the literal (-c1*x with c1 pinned
    # becomes Mul(Float(-0.1731), x)); the pin's magnitude is the
    # invariant, the sign belongs to the surrounding structure.
    d_r = rfr.get_description()
    lits_r = set(abs(float(f)) for f in sp.sympify(d_r).atoms(sp.Float))
    check("PP-d3 rename_leaf_params: literals bit-exact, atom_symbol "
          "unchanged, theta0 remapped to the new names",
          ({D1V, W1V} <= lits_r)
          and rfr.atom_symbol == spec.name
          and sorted(rfr.theta0.keys()) == ["b7", "b8"],
          "desc=%r theta0=%r" % (d_r, rfr.theta0))

    # ---------------------------------------------------------------- PP-e
    print("[PP-e] burn-in vocabulary + spawn wire (T-P3)")
    gb = Grove(X, y, 1.0, 6, 2, list(FUNCS), sigma=0.05, Npars=1,
               operations=list(OPS4), use_gradients=False, seed=11,
               g_tau=1.0, burn_in=True, atoms=[spec])
    pins_ok = True
    n_atomic = 0
    for tt in gb.trees:
        if tt is None:
            continue
        for lf in tt.get_leaves():
            if lf.is_atomic():
                n_atomic += 1
                d = lf.get_fun().get_description()
                if "0.1731" not in d or "0.0806" not in d:
                    pins_ok = False
    check("PP-e1 serial burn-in enumerates the pinned atom via its "
          "extended token; pins intact in the seeded population",
          pins_ok and n_atomic > 0, "atomic leaves: %d" % n_atomic)
    if run_mpi:
        fo = Forest(X, y, np.array([1.0, 2.0]), 3, 2, list(FUNCS),
                    sigma=0.05, operations=list(OPS4), seed=13,
                    g_tau=1.0, atoms=[spec], n_procs=2)
        out = fo.run_mpi_spawn(np.ascontiguousarray(y), 2, 5, seed=13)
        pins_ok = True
        n_atomic = 0
        for g in fo.groves:
            for tt in g.trees:
                for lf in tt.get_leaves():
                    if lf.is_atomic():
                        n_atomic += 1
                        d = lf.get_fun().get_description()
                        if "0.1731" not in d or "0.0806" not in d:
                            pins_ok = False
        check("PP-e2 spawn sampling (config triple wire + worker rebuild "
              "+ skel round trips): pins intact in every returned tree",
              pins_ok and n_atomic > 0,
              "atomic leaves: %d" % n_atomic)
    else:
        print("  [INFO] PP-e2 skipped (pass --mpi under `mpiexec -n 1` "
              "to run the spawn check)")

    # ---------------------------------------------------------------- PP-f
    print("[PP-f] datasets= (J=2) with pinned atoms (T-P5)")
    X2 = np.linspace(0.8, 3.5, 45)
    y1 = ref_eval(X, 1.7, 0.4) + RNG.normal(0, 0.01, X.size)
    y2 = ref_eval(X2, -0.9, 1.1) + RNG.normal(0, 0.01, X2.size)
    s1 = np.full(X.size, 0.01)
    s2 = np.full(X2.size, 0.01)
    tf = atom_tree(spec, ["a0", "a1"], {"a0": 1.0, "a1": 0.0}, X, FUNCS,
                   sigma=s1.copy())
    gf = Grove(X, y1, 1.0, 1, 2, list(FUNCS), sigma=s1.copy(),
               node_list=[tf.get_root()], operations=list(OPS4),
               use_gradients=False, seed=17, g_tau=1.0, atoms=[spec],
               datasets=[(X, y1, s1), (X2, y2, s2)])
    gf.compute_posteriors(y1, False)
    pj = float(gf.root_posteriors[0])
    ths = gf.joint_thetas(gf.trees[0])
    check("PP-f1 joint post finite with a pinned atom",
          np.isfinite(pj) and pj > -1.0e90, repr(pj))
    check("PP-f2 per-dataset theta covers ONLY free internals and "
          "DIFFERS across datasets (different truths)",
          ths is not None
          and sorted(ths[0].keys()) == ["a0", "a1"]
          and sorted(ths[1].keys()) == ["a0", "a1"]
          and abs(ths[0]["a0"] - ths[1]["a0"]) > 0.5,
          repr(ths))
    check("PP-f3 pins identical across datasets by construction "
          "(one compiled map; desc carries the same literals)",
          "0.1731" in gf.trees[0].tree_get_description())
    # Review 2026-08-06 F9: analytic joint pin — the joint post must be
    # the SUM of per-dataset (-0.5*chi2_j - 0.5*k*log n_j) at the
    # per-dataset thetas, with k = 2 (the FREE internals; a k = 4
    # regression, i.e. pins charged, shifts each term by log(n_j)).
    manual = 0.0
    for (xj, yj, sj), thj in zip([(X, y1, s1), (X2, y2, s2)], ths):
        tj = atom_tree(spec, ["a0", "a1"], thj, X, FUNCS)
        pr = np.asarray(tj.evaluate(np.ascontiguousarray(xj)),
                        dtype=np.float64)
        manual += (-0.5 * float(np.sum(((yj - pr) / sj) ** 2))
                   - 0.5 * 2 * np.log(xj.size))
    check("PP-f4 joint post == analytic sum with k=2 per dataset "
          "(pins uncharged under MD)",
          abs(pj - manual) < 1e-6 * max(1.0, abs(manual)),
          "joint=%r manual=%r" % (pj, manual))

    # ---------------------------------------------------------------- PP-g
    print("[PP-g] expr_to_tree pin_numbers flag (T-P6)")
    e = sp.sympify("0.1731*sin(x)")
    tp = function_treeC.expr_to_tree(e, np.ascontiguousarray(X), list(FUNCS),
                                     pin_numbers=True)
    check("PP-g1 pin_numbers=True: literal is a CONSTANT (k == 0)",
          len(tp.get_param_dicts()[1]) == 0,
          repr(tp.get_param_dicts()[1]))
    check("PP-g2 pin_numbers=True: evaluation exact",
          float(np.max(np.abs(np.asarray(tp.evaluate(X))
                              - 0.1731 * np.sin(X)))) < 1e-12)
    tl = function_treeC.expr_to_tree(e, np.ascontiguousarray(X), list(FUNCS))
    nvl = tl.get_param_dicts()[1]
    check("PP-g3 default (legacy): literal becomes ONE free parameter "
          "holding the value",
          len(nvl) == 1 and abs(list(nvl.values())[0] - 0.1731) < 1e-12,
          repr(nvl))
    tq = function_treeC.expr_to_tree(sp.sympify("a0*sin(x) + 0.5"),
                                     np.ascontiguousarray(X), list(FUNCS),
                                     pin_numbers=True)
    nvq = tq.get_param_dicts()[1]
    by_q = {p.get_fun().get_description(): p for p in tq.get_params()}
    for nm in nvq:
        by_q[nm].set_value(2.0)
    tq.tree_update(tq.get_root())
    evq = np.asarray(tq.evaluate(X), dtype=np.float64)
    check("PP-g4 pin_numbers: the free param stays free (fittable) while "
          "the literal stays pinned (eval = 2*sin + 0.5 exactly)",
          len(nvq) == 1
          and float(np.max(np.abs(evq - (2.0 * np.sin(X) + 0.5))))
          < 1e-12,
          repr(nvq))
    # Review 2026-08-06 F2: deepcopy of a pin_numbers tree — the
    # pinned_const flag and the literal's constant_value must survive
    # RealFunctionC.__copy__/__deepcopy__ (Grove deepcopies on EVERY
    # proposal; a dropped flag re-classifies the constant as a fitted
    # parameter and evaluates it as 1.0).
    tp5 = function_treeC.expr_to_tree(e, np.ascontiguousarray(X),
                                      list(FUNCS), pin_numbers=True)
    dc5 = copy.deepcopy(tp5)
    check("PP-g5 deepcopy keeps the constant pinned (k still 0; "
          "evaluation exact on the copy)",
          len(dc5.get_param_dicts()[1]) == 0
          and float(np.max(np.abs(np.asarray(dc5.evaluate(X))
                                  - 0.1731 * np.sin(X)))) < 1e-12,
          repr(dc5.get_param_dicts()[1]))
    tg6 = function_treeC.expr_to_tree(sp.sympify("2*sin(x)"),
                                      np.ascontiguousarray(X),
                                      list(FUNCS), pin_numbers=True)
    check("PP-g6 Integer literals pin too (2*sin(x): k=0, eval exact)",
          len(tg6.get_param_dicts()[1]) == 0
          and float(np.max(np.abs(np.asarray(tg6.evaluate(X))
                                  - 2.0 * np.sin(X)))) < 1e-12)
    # Review 2026-08-06 F4: pin_numbers trees are SCORING-ONLY — seeding
    # one into a Grove must refuse loudly (constant leaves sit outside
    # the move vocabulary; evolving them breaks detailed balance).
    tp7 = function_treeC.expr_to_tree(e, np.ascontiguousarray(X),
                                      list(FUNCS), pin_numbers=True)
    expect_raises("PP-g7 Grove refuses pin_numbers seeds "
                  "(scoring-only contract enforced)", ValueError,
                  Grove, X, y, 1.0, 1, 2, list(FUNCS),
                  node_list=[tp7.get_root()], operations=list(OPS4),
                  seed=3)

    # ---------------------------------------------------------------- PP-h
    print("[PP-h] Polya: extended symbol is its own basic-function cell")
    pt = polya_treeC.PolyaTree(OPS4, ["x", "sin(x)", "a0"], 1.0, False, True)
    i1 = pt.add_primitive(spec.name, None)
    i2 = pt.add_primitive(spec_other.name, None)
    check("PP-h1 two pin configurations register as two distinct cells",
          i1 != i2, "%r %r" % (i1, i2))
    ta = atom_tree(spec, ["a0", "a1"], {"a0": 1.0, "a1": 0.0}, X, FUNCS)
    try:
        pt.assert_covers(ta.get_root())
        path = pt.extract_path(ta.get_root())
        check("PP-h2 coverage + path extraction through the extended cell",
              isinstance(path, list) and len(path) >= 1
              and np.isfinite(pt.log_prior(path)))
    except Exception as ex:
        check("PP-h2 coverage + path extraction through the extended cell",
              False, repr(ex))

    # ---------------------------------------------------------------- PP-i
    print("[PP-i] pinned atoms x flex scaffolds (both features on)")
    fspec = real_functionC.FlexAtomSpec("G", "c0*tanh(s)")
    expect_raises("PP-i1 flex name colliding with a pinned atom's BASE "
                  "name is refused (config-confusion trap)", ValueError,
                  Grove, X, y, 1.0, 2, 2, list(FUNCS),
                  atoms=[spec],
                  flex_atoms=[real_functionC.FlexAtomSpec("Abeat",
                                                          "c0*tanh(s)")])
    slot = make_tree("a0*x", X, FUNCS, values={"a0": 1.0})
    fnode = function_treeC.make_flex_node(fspec, slot.get_root(), ["c0"],
                                          None)
    ft = function_treeC.Tree(0, list(FUNCS),
                             np.ascontiguousarray(X, dtype=np.float64),
                             node=fnode)
    ft.set_sigma(0.05)
    ft.set_g_tau(1.0)
    ft.tree_update(ft.get_root())
    at = atom_tree(spec, ["a0", "a1"], {"a0": 1.7, "a1": 0.4}, X, FUNCS,
                   sigma=0.05)
    gi = Grove(X, y, 1.0, 2, 2, list(FUNCS), sigma=0.05,
               node_list=[at.get_root(), ft.get_root()],
               operations=list(OPS4), use_gradients=False, seed=23,
               g_tau=1.0, atoms=[spec], flex_atoms=[fspec])
    gi.compute_posteriors(y, False)
    gi.run(y, 120)
    ok_i = True
    n_at = 0
    n_fx = 0
    for tt in gi.trees:
        n_fx += len(function_treeC.collect_flex_nodes(tt.get_root(),
                                                      None))
        for lf in tt.get_leaves():
            if lf.is_atomic():
                n_at += 1
                lits = set(abs(float(f)) for f in
                           sp.sympify(
                               lf.get_fun().get_description()).atoms(
                                   sp.Float))
                if not ({D1V, W1V} <= lits):
                    ok_i = False
    check("PP-i2 combined chain: pins exact in every atomic leaf, "
          "scaffold immortal, both features coexist",
          ok_i and n_fx >= 1, "atomic=%d flex=%d" % (n_at, n_fx))

    # ------------------------------------------------------------- summary
    print("=" * 66)
    print("test_phasePP: %d passed, %d failed" % (N_PASS, N_FAIL))
    return 1 if N_FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
