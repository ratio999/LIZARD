"""test_phaseFX.py -- certification of Phase FX transparent atoms v1
(2026-07-29).

WHAT THIS PINS
--------------
A transparent atom ("flex scaffold") is a template expression, e.g.
W = c0*tanh(c1*s + c2), whose internal structure is INSPECTABLE (the c_i
are real optimizer/PR coordinates) and whose slot `s` holds a live,
searchable subtree. v1 contract (PLAN.md Phase MR.2):

  * entry by SEEDING only (make_flex_node / node_list / 'F' wire tag);
  * scaffolds are IMMORTAL under the outer kernel: every outer move that
    would delete/replace/rebuild a flex node (or select inside its slot)
    force-rejects; ops may be added above a scaffold (split) and removed
    again (internal_hoist) as long as no flex node is destroyed;
  * slots evolve ONLY via the dedicated flex_slot_mutation move (uniform
    scaffold pick x uniform inner move from the 7-set; the correction
    reduces exactly to the inner move's (lqf, lqr));
  * identity = UNEXPANDED sympy Function head W(c0, ..., slot_expr);
    evaluation = numeric composition of the compiled template over the
    slot's output (NaN-fill on failure -> zero-likelihood semantics);
  * every internal theta counts 1 toward the BIC k, exactly like a plain
    parameter; PR routes flex trees through name_values coordinates;
  * Polya: ONE template cell, then the slot path continues in the child
    subtrie; registration invariance via Forest pre-registration;
  * feature strictly opt-in: flex_atoms=None Groves/Forests are
    bit-identical legacy (certified separately by the historical suites).

CHECKS
------
  [FX-a] FlexAtomSpec validation, compilation, pickling;
  [FX-b] make_flex_node + numeric composition exactness + NaN policy;
  [FX-c] Tree integration: params/k/BIC identity, deepcopy isolation;
  [FX-d] outer-kernel immortality guards (per-move force-rejects) and
         non-over-blocking on the flex-free part of a mixed tree;
  [FX-e] flex_slot_mutation unit: invariance, motion, name discipline,
         determinism, no-op convention on flex-free trees;
  [FX-f] Polya integration: template cell + slot subtrie path, coverage,
         KeyError without registration, auto_register path;
  [FX-g] wire: 'F' skeleton round trip, _tree_to_wire forces skel3b,
         _tree_from_wire rebuild + theta install, missing-spec reject;
  [FX-h] Grove/Forest integration: ctor validation, 9-vector move probs,
         conditional stats key, N-step immortality + determinism, Forest
         smoke (swaps/crossovers) + run_mpi_spawn refusal;
  [FX-i] end-to-end recovery: y = 2*tanh(1.5x - 1) + noise from a seeded
         c0*tanh(s) scaffold; slot grows the needed affine/extra structure,
         NRMSE small, scaffold
         still present.

RUN
---
  $ENV/bin/python test_phaseFX.py          (no MPI needed)
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
    except exc as e:
        check(label, True)
        return e
    except Exception as e:
        check(label, False, "wrong exception %r" % (e,))
        return None
    check(label, False, "no exception raised")
    return None


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


def make_flex_tree(spec, slot_expr, names, domain, funcs, slot_values=None,
                   theta=None, sigma=1.0, g_tau=1.0):
    """Scaffold `spec` over slot `slot_expr` as a full outer Tree.

    NB expr_to_tree canonicalizes parameter names into the 'a' family
    (encounter order), so slot exprs here use canonical a-names -- the
    realistic seeding pattern; internal thetas live in the 'c' family and
    can never collide."""
    slot_tree = make_tree(slot_expr, domain, funcs, values=slot_values)
    fnode = function_treeC.make_flex_node(spec, slot_tree.get_root(),
                                          list(names), theta)
    tree = function_treeC.Tree(0, list(funcs),
                               np.ascontiguousarray(domain,
                                                    dtype=np.float64),
                               node=fnode)
    tree.set_sigma(sigma)
    tree.set_g_tau(g_tau)
    tree.tree_update(tree.get_root())
    return tree


def n_flex(tree):
    return len(function_treeC.collect_flex_nodes(tree.get_root(), None))


# ---------------------------------------------------------------- fixtures
RNG = np.random.default_rng(11)
X = np.linspace(0.5, 3.0, 60)
FUNCS = ["tanh(x)", "sin(x)", "log(x)", "x"]
OPS4 = ["+", "-", "*", "/"]
NEG_INF = float("-inf")


def main():
    gp_movesC.set_max_tree_depth(4)
    gp_movesC.set_subtree_rebuild_max_height(2)

    # ---------------------------------------------------------------- FX-a
    print("[FX-a] FlexAtomSpec validation / compilation / pickling")
    spec3 = real_functionC.FlexAtomSpec("W", "c0*tanh(c1*s + c2)")
    check("FX-a1 good spec: slots sorted, k, fields",
          spec3.name == "W" and spec3.k == 3
          and spec3.slots == ["c0", "c1", "c2"]
          and spec3.slot_symbol == "s",
          "slots=%r k=%r" % (spec3.slots, spec3.k))
    s_arr = 0.3 * X
    got = np.asarray(spec3.compiled(s_arr, 2.0, 1.5, -1.0), dtype=np.float64)
    want = 2.0 * np.tanh(1.5 * s_arr - 1.0)
    check("FX-a2 compiled template == numpy composition",
          float(np.max(np.abs(got - want))) < 1e-12)
    expect_raises("FX-a3 name 'a3' (parameter pattern) rejected", ValueError,
                  real_functionC.FlexAtomSpec, "a3", "c0*tanh(s)")
    expect_raises("FX-a4 name 'x' (registered variable) rejected", ValueError,
                  real_functionC.FlexAtomSpec, "x", "c0*tanh(s)")
    expect_raises("FX-a5 name 'sin' (known sympy head) rejected", ValueError,
                  real_functionC.FlexAtomSpec, "sin", "c0*tanh(s)")
    expect_raises("FX-a6 stray free symbol in template rejected", ValueError,
                  real_functionC.FlexAtomSpec, "W2", "c0*tanh(z)")
    expect_raises("FX-a7 slot-free template rejected", ValueError,
                  real_functionC.FlexAtomSpec, "W2", "c0*c1")
    expect_raises("FX-a8 registered variable inside template rejected",
                  ValueError, real_functionC.FlexAtomSpec, "W2", "c0*tanh(x*s)")
    expect_raises("FX-a9 slot_symbol 'x' rejected", ValueError,
                  real_functionC.FlexAtomSpec, "W2", "c0*tanh(s)", "x")
    expect_raises("FX-a10 slot_symbol 'a0' rejected", ValueError,
                  real_functionC.FlexAtomSpec, "W2", "a0*tanh(s)", "a0")
    spec_b = real_functionC.FlexAtomSpec("V", "b2*s + b0")
    check("FX-a11 slot names sorted by (letter, index)",
          spec_b.slots == ["b0", "b2"], repr(spec_b.slots))
    spec_rt = pickle.loads(pickle.dumps(spec3))
    got_rt = np.asarray(spec_rt.compiled(s_arr, 2.0, 1.5, -1.0),
                        dtype=np.float64)
    check("FX-a12 pickle round trip (fields + compiled)",
          spec_rt.name == "W" and spec_rt.slots == spec3.slots
          and spec_rt.template == spec3.template
          and float(np.max(np.abs(got_rt - want))) < 1e-12)

    # ---------------------------------------------------------------- FX-b
    print("[FX-b] make_flex_node + numeric composition")
    tfx = make_flex_tree(spec3, "a0*x", ["c0", "c1", "c2"], X, FUNCS,
                         slot_values={"a0": 0.7},
                         theta={"c0": 2.0, "c1": 1.5, "c2": -1.0})
    root = tfx.get_root()
    check("FX-b1 node flags: is_flex / not leaf / not constant",
          bool(root.is_flex()) and not bool(root.is_leaf())
          and not bool(root.is_constant()))
    pred = np.asarray(tfx.evaluate(X), dtype=np.float64)
    want_b = 2.0 * np.tanh(1.5 * (0.7 * X) - 1.0)
    check("FX-b2 evaluation == numeric composition",
          float(np.max(np.abs(pred - want_b))) < 1e-12,
          "max err %r" % float(np.max(np.abs(pred - want_b))))
    desc = tfx.tree_get_description()
    check("FX-b3 description keeps the UNEXPANDED head",
          desc.startswith("W(") and "tanh" not in desc, repr(desc))
    tfx_again = make_flex_tree(spec3, "a0*x", ["c0", "c1", "c2"], X, FUNCS,
                               slot_values={"a0": 0.7},
                               theta={"c0": 2.0, "c1": 1.5, "c2": -1.0})
    check("FX-b4 description deterministic across rebuilds",
          desc == tfx_again.tree_get_description())
    leaf_descs = sorted(lf.get_fun().get_description()
                        for lf in tfx.get_leaves())
    check("FX-b5 slot leaves ARE tree leaves (params visible)",
          leaf_descs == ["a0", "x"], repr(leaf_descs))
    check("FX-b6 default theta is the 1.0 reference",
          all(v == 1.0 for v in function_treeC.make_flex_node(
              spec3, make_tree("a0*x", X, FUNCS).get_root(),
              ["c0", "c1", "c2"], None).theta.values()))
    expect_raises("FX-b7 wrong internal-name count rejected", ValueError,
                  function_treeC.make_flex_node, spec3,
                  make_tree("a0*x", X, FUNCS).get_root(), ["c0", "c1"], None)
    expect_raises("FX-b8 variable name as internal name rejected", ValueError,
                  function_treeC.make_flex_node, spec3,
                  make_tree("a0*x", X, FUNCS).get_root(),
                  ["c0", "c1", "x"], None)
    expect_raises("FX-b9 flex nesting rejected", ValueError,
                  function_treeC.make_flex_node, spec3,
                  tfx_again.get_root(), ["d0", "d1", "d2"], None)
    # NaN policy: slot log(a0*x) with b0 < 0 -> slot NaN -> composition
    # NaN-fills -> zero-likelihood semantics on the scoring path.
    tnan = make_flex_tree(spec3, "log(a0*x)", ["c0", "c1", "c2"], X, FUNCS,
                          slot_values={"a0": -1.0}, sigma=0.1)
    ynan = np.asarray(want_b, dtype=np.float64)
    post_nan = tnan.log_marginal_bic(ynan, False)
    check("FX-b10 NaN slot => penalty-scale posterior (no NaN leak)",
          np.isfinite(post_nan) and post_nan < -1e50, repr(post_nan))

    # ---------------------------------------------------------------- FX-c
    print("[FX-c] Tree integration: k, BIC identity, deepcopy isolation")
    nv = tfx.get_param_dicts()[1]
    check("FX-c1 name_values = slot params + internal thetas",
          sorted(nv.keys()) == ["a0", "c0", "c1", "c2"]
          and abs(nv["a0"] - 0.7) < 1e-15 and abs(nv["c0"] - 2.0) < 1e-15,
          repr(nv))
    sig_arr = np.linspace(0.05, 0.4, X.size)
    tsc = make_flex_tree(spec3, "a0*x", ["c0", "c1", "c2"], X, FUNCS,
                         slot_values={"a0": 0.7},
                         theta={"c0": 2.0, "c1": 1.5, "c2": -1.0},
                         sigma=sig_arr.copy())
    ysc = want_b + RNG.normal(0.0, 1.0, X.size) * sig_arr
    post = tsc.log_marginal_bic(ysc, False)
    r = ysc - np.asarray(tsc.evaluate(X), dtype=np.float64)
    expected = (-0.5 * np.sum(r * r / sig_arr ** 2)
                - 0.5 * 4 * np.log(X.size))
    check("FX-c2 BIC k counts every internal theta (analytic pin, k=4)",
          abs(post - expected) < 1e-6 * max(1.0, abs(expected)),
          "post=%r expected=%r" % (post, expected))
    dc = copy.deepcopy(tfx)
    check("FX-c3 deepcopy preserves flex (count + description + eval)",
          n_flex(dc) == 1 and dc.tree_get_description() == desc
          and float(np.max(np.abs(np.asarray(dc.evaluate(X)) - pred))) == 0.0)
    dc_fx = function_treeC.collect_flex_nodes(dc.get_root(), None)[0]
    dc_fx.theta["c0"] = 5.0
    dc_fx.update_func()
    dc.tree_update(dc.get_root())
    pred_orig_after = np.asarray(tfx.evaluate(X), dtype=np.float64)
    check("FX-c4 deepcopy isolation: mutating the copy's theta leaves the "
          "original untouched",
          float(np.max(np.abs(pred_orig_after - pred))) == 0.0
          and float(np.max(np.abs(np.asarray(dc.evaluate(X)) - pred))) > 0.1)
    # Optimizer write-back: Powell must reach the flex thetas by NAME and
    # persist them into node.theta (evaluation must match name_values).
    topt = make_flex_tree(spec3, "a0*x", ["c0", "c1", "c2"], X, FUNCS,
                          slot_values={"a0": 1.0}, sigma=0.05)
    post_opt = topt.log_marginal_bic(want_b.copy(), True, False, "Powell")
    nv_opt = topt.get_param_dicts()[1]
    pred_opt = np.asarray(topt.evaluate(X), dtype=np.float64)
    manual = (nv_opt["c0"]
              * np.tanh(nv_opt["c1"] * (nv_opt["a0"] * X) + nv_opt["c2"]))
    check("FX-c5 optimize writes thetas back (eval == manual composition)",
          np.isfinite(post_opt)
          and float(np.max(np.abs(pred_opt - manual))) < 1e-10,
          "err=%r" % float(np.max(np.abs(pred_opt - manual))))
    check("FX-c6 optimize actually fits (residual shrinks)",
          float(np.max(np.abs(pred_opt - want_b))) < 0.15,
          "max resid %r" % float(np.max(np.abs(pred_opt - want_b))))

    # ---------------------------------------------------------------- FX-d
    print("[FX-d] outer-kernel immortality guards")
    def fresh_flex():
        return make_flex_tree(spec3, "a0*x", ["c0", "c1", "c2"], X, FUNCS,
                              slot_values={"a0": 0.7})

    gp_movesC.set_rng(random.Random(31), np.random.default_rng(31))
    lq = gp_movesC.add_node(fresh_flex(), FUNCS, OPS4, False, None)
    check("FX-d1 add_node force-rejects on a flex-bearing tree",
          lq[1] == NEG_INF, repr(lq))
    lq = gp_movesC.leaf_hoist(fresh_flex(), FUNCS, OPS4, None)
    check("FX-d2 leaf_hoist force-rejects on a flex-bearing tree",
          lq[1] == NEG_INF, repr(lq))
    lq = gp_movesC.point_mutation(fresh_flex(), FUNCS, OPS4, None, None)
    check("FX-d3 point_mutation on the scaffold force-rejects",
          lq[1] == NEG_INF, repr(lq))
    lq = gp_movesC.subtree_mutation(fresh_flex(), FUNCS, OPS4)
    check("FX-d4 subtree_mutation force-rejects (subtree contains flex)",
          lq[1] == NEG_INF, repr(lq))
    ta = fresh_flex()
    tb = make_tree("a0*x + a1", X, FUNCS)
    lq = gp_movesC.subtree_crossover(ta, tb)
    check("FX-d5 crossover force-rejects when a flex subtree is selected",
          lq[1] == NEG_INF, repr(lq))
    # split may legally ADD an op node around the scaffold (n_flex and the
    # slot are untouched); internal_hoist may only remove it when no flex
    # node would be destroyed.
    tsp = fresh_flex()
    desc_before = tsp.tree_get_description()
    gp_movesC.set_rng(random.Random(32), np.random.default_rng(32))
    lq = gp_movesC.split(tsp, FUNCS, OPS4, None)
    check("FX-d6 split above the scaffold is LEGAL (flex intact)",
          np.isfinite(lq[0]) and np.isfinite(lq[1]) and n_flex(tsp) == 1
          and desc_before in tsp.tree_get_description(),
          "lq=%r n_flex=%d desc=%r" % (lq, n_flex(tsp),
                                       tsp.tree_get_description()))
    hoist_ok = True
    hoist_hit = False
    for sd in range(40):
        th = copy.deepcopy(tsp)
        gp_movesC.set_rng(random.Random(100 + sd),
                          np.random.default_rng(100 + sd))
        lqh = gp_movesC.internal_hoist(th, FUNCS, OPS4, None)
        if np.isfinite(lqh[1]):
            hoist_hit = True
            if n_flex(th) != 1:
                hoist_ok = False
    check("FX-d7 internal_hoist never destroys the scaffold "
          "(finite proposals keep n_flex == 1)",
          hoist_ok and hoist_hit, "hit=%s ok=%s" % (hoist_hit, hoist_ok))
    tetr = fresh_flex()
    desc_etr = tetr.tree_get_description()
    lq = gp_movesC.etr_mutation(tetr, FUNCS, OPS4, None)
    check("FX-d8 etr_mutation on a flex-only tree: no ETs => symmetric "
          "no-op (legacy omega_i==0 convention), tree untouched",
          lq == (0.0, 0.0) and tetr.tree_get_description() == desc_etr
          and n_flex(tetr) == 1, repr(lq))
    # Non-over-blocking: a mixed tree's flex-free side must stay mutable.
    finite_seen = 0
    flex_kept = True
    for sd in range(30):
        tm = fresh_flex()
        # Right side is parameter-free: the slot's canonical 'a0' must not
        # be re-minted on the outer side (seeder name discipline).
        troot = function_treeC.Node(None, tm.get_root(),
                                    make_tree("sin(x)*x", X,
                                              FUNCS).get_root(), "+")
        tmix = function_treeC.Tree(0, list(FUNCS),
                                   np.ascontiguousarray(X, dtype=np.float64),
                                   node=troot)
        tmix.tree_update(tmix.get_root())
        gp_movesC.set_rng(random.Random(200 + sd),
                          np.random.default_rng(200 + sd))
        lqm = gp_movesC.point_mutation(tmix, FUNCS, OPS4, None, None)
        if np.isfinite(lqm[1]):
            finite_seen += 1
        if n_flex(tmix) != 1:
            flex_kept = False
    check("FX-d9 mixed tree: flex-free side stays mutable, flex preserved",
          finite_seen > 0 and flex_kept,
          "finite=%d kept=%s" % (finite_seen, flex_kept))

    # ---------------------------------------------------------------- FX-e
    print("[FX-e] flex_slot_mutation unit")
    tsl = fresh_flex()
    # Pin BOTH module RNGs (the FX-e6 lesson): subtree_mutation rebuilds
    # draw from function_treeC's stream — unpinned, the finite-proposal
    # count varies across processes (observed 14..40 on identical
    # binaries; the 2026-07-31 MD cross-check caught it at 14).
    gp_movesC.set_rng(random.Random(41), np.random.default_rng(41))
    function_treeC.set_rng(random.Random(42), np.random.default_rng(42))
    finite_props = 0
    invariant_ok = True
    slot_descs = set()
    for _ in range(200):
        lqs = gp_movesC.flex_slot_mutation(tsl, FUNCS, OPS4)
        if np.isfinite(lqs[0]) and np.isfinite(lqs[1]):
            finite_props += 1
        if n_flex(tsl) != 1:
            invariant_ok = False
        slot_descs.add(function_treeC.collect_flex_nodes(
            tsl.get_root(), None)[0].get_left().get_fun().get_description())
    check("FX-e1 200 slot moves: n_flex invariant, finite proposals occur",
          invariant_ok and finite_props > 20,
          "finite=%d invariant=%s" % (finite_props, invariant_ok))
    check("FX-e2 the slot actually moves (structure diversity)",
          len(slot_descs) > 3, "distinct slots: %d" % len(slot_descs))
    desc_now = tsl.tree_get_description()
    rebuilt = function_treeC.Tree(0, list(FUNCS),
                                  np.ascontiguousarray(X, dtype=np.float64),
                                  node=tsl.get_root())
    check("FX-e3 tree stays self-consistent after the chain "
          "(description == rebuilt description)",
          desc_now == rebuilt.tree_get_description(),
          "%r vs %r" % (desc_now, rebuilt.tree_get_description()))
    names_list = [p.get_fun().get_description() for p in tsl.get_params()]
    plain_names = [nm for nm in names_list]
    check("FX-e4 name discipline: no duplicate coordinate names",
          len(plain_names) == len(set(plain_names)), repr(plain_names))
    tplain = make_tree("a0*x + a1", X, FUNCS)
    desc_plain = tplain.tree_get_description()
    lqs = gp_movesC.flex_slot_mutation(tplain, FUNCS, OPS4)
    check("FX-e5 flex-free tree: hard reject (0, -inf), tree untouched "
          "(a (0,0) no-op would be a guaranteed accept that re-reinforces "
          "the incumbent's Polya path)",
          lqs[0] == 0.0 and lqs[1] == NEG_INF
          and tplain.tree_get_description() == desc_plain, repr(lqs))
    # Outer depth cap: the slot move must force-reject any proposal that
    # would push the OUTER tree (scaffold + slot) past the bounded
    # support -- the inner caps see only the slot-as-a-tree.
    gp_movesC.set_max_tree_depth(3)
    try:
        deep_hits = 0
        cap_ok = True
        for sd in range(60):
            td = make_flex_tree(spec3, "a0*x + a1", ["c0", "c1", "c2"],
                                X, FUNCS)
            gp_movesC.set_rng(random.Random(500 + sd),
                              np.random.default_rng(500 + sd))
            function_treeC.set_rng(random.Random(501 + sd),
                                   np.random.default_rng(501 + sd))
            lqd = gp_movesC.flex_slot_mutation(td, FUNCS, OPS4)
            depth_after = td.get_root().depth()
            if np.isfinite(lqd[1]):
                deep_hits += 1
                if depth_after > 3:
                    cap_ok = False
        check("FX-e7 outer depth cap enforced on the slot move "
              "(no finite proposal exceeds the bounded support)",
              cap_ok and deep_hits > 0,
              "finite=%d cap_ok=%s" % (deep_hits, cap_ok))
    finally:
        gp_movesC.set_max_tree_depth(4)
    da, db = fresh_flex(), fresh_flex()
    for tree_i in (da, db):
        # Pin BOTH module RNGs: subtree_mutation rebuilds draw from
        # function_treeC's stream (build_tree), the rest from gp_movesC's.
        gp_movesC.set_rng(random.Random(77), np.random.default_rng(77))
        function_treeC.set_rng(random.Random(78), np.random.default_rng(78))
        for _ in range(60):
            gp_movesC.flex_slot_mutation(tree_i, FUNCS, OPS4)
    check("FX-e6 pinned-RNG determinism of the slot chain",
          da.tree_get_description() == db.tree_get_description(),
          "%r vs %r" % (da.tree_get_description(),
                        db.tree_get_description()))

    # ---------------------------------------------------------------- FX-f
    print("[FX-f] Polya integration")
    tfp = make_flex_tree(spec3, "a0*x", ["c0", "c1", "c2"], X, FUNCS,
                         slot_values={"a0": 0.7})
    pt = polya_treeC.PolyaTree(OPS4, ["x", "a0", "W"], 1.0, False, True)
    path = pt.extract_path(tfp.get_root())
    lp = pt.log_prior(path)
    check("FX-f1 registered head: path extracts, prior finite",
          isinstance(path, list) and len(path) >= 2 and np.isfinite(lp),
          "path=%r lp=%r" % (path, lp))
    try:
        pt.assert_covers(tfp.get_root())
        check("FX-f2 assert_covers passes with the head registered", True)
    except Exception as e:
        check("FX-f2 assert_covers passes with the head registered", False,
              repr(e))
    pt_miss = polya_treeC.PolyaTree(OPS4, ["x", "a0"], 1.0, False, True)
    expect_raises("FX-f3 unregistered head: extract_path raises KeyError",
                  KeyError, pt_miss.extract_path, tfp.get_root())
    expect_raises("FX-f4 unregistered head: assert_covers raises KeyError",
                  KeyError, pt_miss.assert_covers, tfp.get_root())
    pt_auto = polya_treeC.PolyaTree(OPS4, ["x", "a0"], 1.0, True, True)
    path_a = pt_auto.extract_path(tfp.get_root())
    check("FX-f5 auto_register path: head registers on first contact",
          np.isfinite(pt_auto.log_prior(path_a)))
    # The slot path continues in the child subtrie: a slot CHANGE must
    # change the path while the template cell stays.
    tfp2 = make_flex_tree(spec3, "a0*x + a1", ["c0", "c1", "c2"], X, FUNCS)
    pt2 = polya_treeC.PolyaTree(OPS4, ["x", "a0", "a1", "W"], 1.0, False,
                                True)
    p_a = pt2.extract_path(tfp.get_root())
    p_b = pt2.extract_path(tfp2.get_root())
    check("FX-f6 slot structure is part of the path (paths differ)",
          p_a != p_b, "p_a=%r p_b=%r" % (p_a, p_b))

    # ---------------------------------------------------------------- FX-g
    print("[FX-g] wire ('F' skeleton) round trip")
    twr = make_flex_tree(spec3, "a0*x", ["c0", "c1", "c2"], X, FUNCS,
                         slot_values={"a0": 0.7},
                         theta={"c0": 2.0, "c1": 1.5, "c2": -1.0})
    skel = function_treeC.tree_to_skel(twr.get_root())
    check("FX-g1 skeleton carries the 'F' tag + names + slot skel",
          isinstance(skel, tuple) and skel[0] == "F" and skel[1] == "W"
          and tuple(skel[2]) == ("c0", "c1", "c2"), repr(skel))
    payload, nvals, post_w = groveC._tree_to_wire(twr, -12.5, False)
    check("FX-g2 _tree_to_wire FORCES the skel3b payload for flex trees",
          isinstance(payload, tuple) and payload[0] == "skel3b",
          repr(type(payload)))
    trt = groveC._tree_from_wire(payload, nvals,
                                 np.ascontiguousarray(X, dtype=np.float64),
                                 list(FUNCS), 1.0, 1.0, {"W": spec3})
    check("FX-g3 round trip: description survives",
          trt is not None
          and trt.tree_get_description() == twr.tree_get_description(),
          "None" if trt is None else trt.tree_get_description())
    if trt is not None:
        e_orig = np.asarray(twr.evaluate(X), dtype=np.float64)
        e_rt = np.asarray(trt.evaluate(X), dtype=np.float64)
        check("FX-g4 round trip: thetas travel in name_values (eval equal)",
              float(np.max(np.abs(e_orig - e_rt))) < 1e-12,
              "err=%r" % float(np.max(np.abs(e_orig - e_rt))))
    else:
        check("FX-g4 round trip: thetas travel in name_values (eval equal)",
              False, "rebuild returned None")
    t_none = groveC._tree_from_wire(payload, nvals,
                                    np.ascontiguousarray(X,
                                                         dtype=np.float64),
                                    list(FUNCS), 1.0, 1.0, {})
    check("FX-g5 missing template spec => reject (None), not a crash",
          t_none is None)

    # ---------------------------------------------------------------- FX-h
    print("[FX-h] Grove / Forest integration")
    Y_H = 2.0 * np.tanh(1.5 * X - 1.0) + RNG.normal(0.0, 0.05, X.size)
    expect_raises("FX-h1 Grove rejects non-FlexAtomSpec entries", TypeError,
                  Grove, X, Y_H, 1.0, 2, 2, list(FUNCS),
                  flex_atoms=["c0*tanh(s)"])
    expect_raises("FX-h2 Grove rejects adapt_moves + flex_atoms", ValueError,
                  Grove, X, Y_H, 1.0, 2, 2, list(FUNCS), adapt_moves=True,
                  flex_atoms=[spec3])
    expect_raises("FX-h2b Grove rejects burn_in + flex_atoms", ValueError,
                  Grove, X, Y_H, 1.0, 2, 2, list(FUNCS), burn_in=True,
                  flex_atoms=[spec3])
    expect_raises("FX-h3 Grove rejects a flex/opaque atom name collision",
                  ValueError, Grove, X, Y_H, 1.0, 2, 2, list(FUNCS),
                  atoms=[real_functionC.AtomSpec("W", "a0*sin(a1*x)")],
                  flex_atoms=[spec3])
    g_leg = Grove(X, Y_H, 1.0, 2, 2, list(FUNCS), operations=list(OPS4),
                  seed=5)
    g_fx = Grove(X, Y_H, 1.0, 2, 2, list(FUNCS), operations=list(OPS4),
                 seed=5, flex_atoms=[spec3])
    p8 = np.asarray(g_leg.get_move_probs(), dtype=np.float64)
    p9 = np.asarray(g_fx.get_move_probs(), dtype=np.float64)
    check("FX-h4 legacy 8-vector untouched; flex appends a 9th entry",
          p8.size == 8 and p9.size == 9
          and abs(float(p9.sum()) - 1.0) < 1e-12
          and float(np.max(np.abs(p9[:8] - 0.80 * p8))) < 1e-12
          and abs(float(p9[8]) - 0.20) < 1e-12,
          "p8=%r p9=%r" % (p8, p9))
    check("FX-h5 pair equality survives the flex scaling",
          abs(float(p9[1] - p9[3])) < 1e-15
          and abs(float(p9[2] - p9[4])) < 1e-15)
    check("FX-h6 stats key present IFF the feature is on",
          "flex_slot_mutation" not in g_leg.move_stats()
          and "flex_slot_mutation" in g_fx.move_stats())
    # N-step immortality + determinism on a seeded flex population.
    def flex_grove(seed):
        nodes = []
        for _ in range(4):
            tseed = make_flex_tree(spec3, "a0*x", ["c0", "c1", "c2"],
                                   X, FUNCS, slot_values={"a0": 1.0})
            nodes.append(tseed.get_root())
        g = Grove(X, Y_H, 1.0, 4, 2, list(FUNCS), sigma=0.05,
                  node_list=nodes, operations=list(OPS4), seed=seed,
                  g_tau=float(np.std(Y_H)), flex_atoms=[spec3])
        # node_list seeding leaves root_posteriors unscored (legacy
        # contract): prime them exactly like the exp drivers do.
        g.compute_posteriors(Y_H, False)
        return g

    g1 = flex_grove(13)
    g1.run(Y_H, 300)
    counts = [n_flex(t) for t in g1.trees]
    stats1 = g1.move_stats()
    check("FX-h7 300 steps: every tree keeps exactly ONE scaffold",
          counts == [1, 1, 1, 1], repr(counts))
    check("FX-h8 the slot move actually fires",
          stats1["flex_slot_mutation"]["proposed"] > 0, repr(stats1))
    g2 = flex_grove(13)
    g2.run(Y_H, 300)
    check("FX-h9 same-seed determinism (best description + move stats)",
          g1.best_tree().tree_get_description()
          == g2.best_tree().tree_get_description()
          and stats1 == g2.move_stats())
    # Forest: shared-Polya head registration + serial EMC smoke + spawn
    # refusal.
    def flex_forest():
        node_lists = []
        for _g in range(2):
            nodes = []
            for _ in range(3):
                tseed = make_flex_tree(spec3, "a0*x", ["c0", "c1", "c2"],
                                       X, FUNCS, slot_values={"a0": 1.0})
                nodes.append(tseed.get_root())
            node_lists.append(nodes)
        return Forest(X, Y_H, np.array([1.0, 2.0]), 3, 2, list(FUNCS),
                      sigma=0.05, node_lists=node_lists,
                      operations=list(OPS4), seed=29,
                      g_tau=float(np.std(Y_H)), flex_atoms=[spec3])

    fo = flex_forest()
    fo.compute_posteriors_all(Y_H)
    try:
        fo.polya.assert_covers(fo.groves[0].trees[0].get_root())
        check("FX-h10 Forest pre-registers the template head in the shared "
              "Polya", True)
    except Exception as e:
        check("FX-h10 Forest pre-registers the template head in the shared "
              "Polya", False, repr(e))
    fo.run_all(Y_H, 60)
    fo.attempt_swaps(Y_H)
    fo.attempt_crossovers(Y_H)
    fcounts = [n_flex(t) for g in fo.groves for t in g.trees]
    check("FX-h11 Forest smoke (run/swap/crossover): scaffolds preserved "
          "everywhere", fcounts == [1] * 6, repr(fcounts))
    expect_raises("FX-h12 run_mpi_spawn refuses flex in v1",
                  NotImplementedError, fo.run_mpi_spawn, Y_H, 1, 1)
    expect_raises("FX-h13 Forest-level adapt_moves + flex fails fast",
                  ValueError, Forest, X, Y_H, np.array([1.0, 2.0]), 2, 2,
                  list(FUNCS), operations=list(OPS4), adapt_moves=True,
                  flex_atoms=[spec3])

    # ---------------------------------------------------------------- FX-i
    print("[FX-i] end-to-end recovery through the scaffold")
    rng_i = np.random.default_rng(3)
    y_clean = 2.0 * np.tanh(1.5 * X - 1.0)
    y_i = y_clean + rng_i.normal(0.0, 0.02, X.size)
    spec1 = real_functionC.FlexAtomSpec("G", "c0*tanh(s)")
    nodes = []
    for _ in range(6):
        tseed = make_flex_tree(spec1, "a0*x", ["c0"], X, FUNCS,
                               slot_values={"a0": 1.0})
        nodes.append(tseed.get_root())
    g_e2e = Grove(X, y_i, 1.0, 6, 2, list(FUNCS), sigma=0.02,
                  node_list=nodes, operations=list(OPS4), seed=101,
                  g_tau=float(np.std(y_i)), flex_atoms=[spec1])
    g_e2e.compute_posteriors(y_i, False)
    g_e2e.run(y_i, 1500)
    best = g_e2e.best_tree()
    pred_best = np.asarray(best.evaluate(X), dtype=np.float64)
    nrmse = float(np.sqrt(np.mean((pred_best - y_clean) ** 2))
                  / np.std(y_clean))
    check("FX-i1 recovered y = 2*tanh(1.5x - 1) through the slot "
          "(NRMSE < 0.05)", nrmse < 0.05,
          "nrmse=%r best=%r" % (nrmse, best.tree_get_description()))
    check("FX-i2 the scaffold is still in the best tree", n_flex(best) == 1,
          best.tree_get_description())
    print("      best: %s  (nrmse %.3g)"
          % (best.tree_get_description(), nrmse), flush=True)

    # ------------------------------------------------------------- summary
    print("=" * 66)
    print("test_phaseFX: %d passed, %d failed" % (N_PASS, N_FAIL))
    return 1 if N_FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
