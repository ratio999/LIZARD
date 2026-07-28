# test_phase212_wire.py
"""
Phase 2.12 spawn-sampling WIRE-FIDELITY regression (2026-07-24).

The defect (PROMPT_DEV_WIRE_FIDELITY.md; PLAN.md 4.4 dev plan)
--------------------------------------------------------------
The sampling wire used to ship atom-free trees as RAW SYMPY
expressions. Sympy canonicalizes on construction (x*(x*x) -> x**3,
x + x -> 2*x, term reordering), so whenever the sampling vocabulary
could not re-express the folded form (the depth-3 / Feynman suites
exclude '**'), the receiver's round-trip fidelity gate
(`_tree_from_wire`: rebuilt description == str(expr)) REJECTED the
rebuild and the caller kept its stale tree. Measured effect: after
`Forest.run_mpi_spawn` most master slots still held their INITIAL
trees (4/6 and 6/6 observed) although the workers sampled the full
budget — every downstream readout (best_tree, ...) inherited the
staleness. test_phase2mpi [2.12] never caught it because it asserted
only valid+finite+explored, never that the master population actually
CHANGED.

The fix (groveC._tree_to_wire force_skel=True on the sampling wire)
-------------------------------------------------------------------
`run_mpi_spawn` + `mpi_worker._sampler_*` now ship EVERY tree as the
('skel3b', skeleton, description) structural payload (Phase 2.10
Stage 3b format): shape-exact, sympy-free, gate compares against the
SHIPPED description. The burn-in spawn wire keeps the legacy dual
format (default force_skel=False) — certified bit-identical.

What this file asserts
----------------------
  [W1] the three folding-prone structures from the defect report
       round-trip through the forced-skeleton wire (they were all
       <REJECTED> on the legacy payload), shape-exact, pickle-safe,
       with parameter values installed from name_values;
  [W2] negative control: the SAME trees on the legacy (default)
       payload still fold and still get rejected — documents that the
       defect mechanism was real and that the legacy path (burn-in
       wire, [2.10p] contract) is untouched;
  [W3] end-to-end: after run_mpi_spawn over the folding-prone
       vocabulary the master population REFRESHES (slots differ from
       their initial descriptions) — the assertion whose absence let
       the defect stay latent. Also re-asserts [2.12](a)-style
       validity (slots valid + posteriors finite).

Run:  mpiexec -n 1 python test_phase212_wire.py     (W3 spawns workers)
Additive: new file, new name — no existing suite touched.
"""

import pickle
import sys

import numpy as np

import function_treeC as ftc
import groveC

X = np.linspace(1.0, 2.5, 60)
Y = X ** 2
FUNCS = ["sin(x)", "x"]          # house format: applied strings
OPS = ["+", "-", "*"]            # '**'-free on purpose: folding-prone

N_PASS = 0
N_FAIL = 0


def check(label, ok, detail=""):
    global N_PASS, N_FAIL
    tag = "PASS" if ok else "FAIL"
    if ok:
        N_PASS += 1
    else:
        N_FAIL += 1
    msg = "[%s] %s" % (tag, label)
    if detail:
        msg += " — %s" % detail
    print(msg, flush=True)


def build_tree(skel):
    """Exact-shape tree from a skeleton tuple (the only way to hold a
    structure like x*(x*x) in memory: sympy would fold a string parse)."""
    dom = np.ascontiguousarray(X, dtype=np.float64)
    root = ftc.node_from_skel(skel, dom, None)
    t = ftc.Tree(0, list(FUNCS), X, node=root)
    t.tree_update(t.get_root())
    return t


# The three payloads from the defect report's minimal repro, as
# skeletons (tag 'L' = plain leaf, 'O' = operator node):
S_X3 = ('O', '*', ('L', 'x'), ('O', '*', ('L', 'x'), ('L', 'x')))
S_A_X3 = ('O', '*', ('L', 'a0'), ('O', '*', ('L', 'x'), ('L', 'x')))
S_MIX = ('O', '+',
         ('O', '*', ('L', 'x'), ('L', 'x')),
         ('O', '*', ('L', 'a0'), ('O', '*', ('L', 'x'), ('L', 'x'))))
SKELS = [("x*(x*x)", S_X3), ("a0*(x*x)", S_A_X3), ("x*x + a0*(x*x)", S_MIX)]


def test_w1_forced_skeleton_roundtrip():
    print("[W1] forced-skeleton wire: folding-prone structures round-trip")
    for name, skel in SKELS:
        t = build_tree(skel)
        desc0 = t.tree_get_description()
        wire = groveC._tree_to_wire(t, -7.5, True)
        check("W1 %s: payload is skel3b" % name,
              isinstance(wire[0], tuple) and wire[0][0] == 'skel3b')
        check("W1 %s: shipped description == source description" % name,
              wire[0][2] == desc0, "%r vs %r" % (wire[0][2], desc0))
        # ship it exactly like mpi4py would
        w2 = pickle.loads(pickle.dumps(wire))
        rb = groveC._tree_from_wire(w2[0], w2[1], X, list(FUNCS),
                                    1.0, 1.0, None)
        check("W1 %s: rebuild NOT rejected" % name, rb is not None)
        if rb is None:
            continue
        check("W1 %s: description round-trips exactly" % name,
              rb.tree_get_description() == desc0,
              "%r vs %r" % (rb.tree_get_description(), desc0))
        check("W1 %s: skeleton (exact shape) round-trips" % name,
              ftc.tree_to_skel(rb.get_root()) == skel)
    # parameter values install from name_values on the skeleton path
    t = build_tree(S_A_X3)
    wire = groveC._tree_to_wire(t, -7.5, True)
    nv = dict(wire[1])
    check("W1 param: a0 travels in name_values", 'a0' in nv,
          "name_values=%r" % (nv,))
    nv['a0'] = 2.5
    rb = groveC._tree_from_wire(wire[0], nv, X, list(FUNCS), 1.0, 1.0, None)
    got = None if rb is None else dict(rb.get_param_dicts()[1]).get('a0')
    check("W1 param: value installs on rebuild (a0=2.5)", got == 2.5,
          "got %r" % (got,))


def test_w2_legacy_payload_still_folds():
    print("[W2] negative control: legacy (default) payload still folds "
          "and is still rejected")
    for name, skel in SKELS:
        t = build_tree(skel)
        w = groveC._tree_to_wire(t, -7.5)      # DEFAULT: legacy dual format
        check("W2 %s: default payload stays legacy (raw expr)" % name,
              not isinstance(w[0], tuple),
              "payload type %s" % type(w[0]).__name__)
        rb = groveC._tree_from_wire(w[0], w[1], X, list(FUNCS),
                                    1.0, 1.0, None)
        check("W2 %s: legacy rebuild rejected (canonical form leaves "
              "the '**'-free vocabulary)" % name, rb is None,
              "" if rb is None else "unexpectedly rebuilt %r"
              % rb.tree_get_description())


def test_w3_master_population_refreshes():
    # NB coverage: W3 exercises the forced-skeleton wire at the
    # init_trees producer and the worker-sampler round producers; the
    # two CROSSOVER-CHILD producers (groveC run_mpi_spawn) only fire on
    # an accepted trans-grove crossover, which this seed may not
    # trigger — that force_skel=True call-site is covered structurally
    # by W1 (which drives _tree_to_wire(t, ., True) directly). Run
    # under the launcher: `mpiexec -n 1 python test_phase212_wire.py`.
    print("[W3] end-to-end: run_mpi_spawn over the folding-prone "
          "vocabulary refreshes the master population")
    F = groveC.Forest(X, Y, np.array([1.0, 1.6]), 6, 3, list(FUNCS),
                      sigma=1.0, Npars=2, operations=list(OPS),
                      use_gradients=False, seed=7)
    F.compute_posteriors_all(Y)
    init = [[t.tree_get_description() for t in g.trees] for g in F.groves]
    try:
        stats = F.run_mpi_spawn(Y, 20, 20, seed=5,
                                max_tree_depth=4,
                                subtree_rebuild_max_height=2)
    except Exception as e:
        # Fail LOUD but keep the suite summary intact (e.g. when run as
        # plain `python` without the mpiexec launcher, singleton
        # MPI_Comm_spawn may raise). Mirrors the test_phase44 T7 guard.
        check("W3 run_mpi_spawn completes (needs 'mpiexec -n 1')",
              False, "raised %r" % (e,))
        return
    check("W3 run_mpi_spawn returns a stats dict", isinstance(stats, dict),
          "stats=%r" % (stats,))
    final = [[t.tree_get_description() for t in g.trees] for g in F.groves]
    # [2.12](a)-style validity: every slot rebuilt, valid, finite post.
    ok_valid = True
    for g in F.groves:
        for s in range(g.size):
            if (g.trees[s] is None or not g.trees[s].is_valid(X)
                    or not np.isfinite(g.root_posteriors[s])):
                ok_valid = False
    check("W3 all master slots valid + posteriors finite", ok_valid)
    changed_g0 = sum(1 for a, b in zip(init[0], final[0]) if a != b)
    changed_all = sum(1 for gi, gf in zip(init, final)
                      for a, b in zip(gi, gf) if a != b)
    # Pre-fix measurement on this exact fixture family: 4/6 and 6/6
    # slots STALE (changed 2/6 and 0/6). The fix must clearly separate.
    check("W3 REFRESH cold grove: >= 3/6 slots differ from initial",
          changed_g0 >= 3, "changed %d/6; init %s -> final %s"
          % (changed_g0, init[0], final[0]))
    check("W3 REFRESH total: >= 6/12 slots differ from initial",
          changed_all >= 6, "changed %d/12" % changed_all)
    print("      [diag] stats: %r" % (stats,), flush=True)
    print("      [diag] final cold-grove population: %s" % (final[0],),
          flush=True)


def main():
    test_w1_forced_skeleton_roundtrip()
    test_w2_legacy_payload_still_folds()
    test_w3_master_population_refreshes()
    print()
    print("%d/%d checks passed." % (N_PASS, N_PASS + N_FAIL))
    sys.exit(0 if N_FAIL == 0 else 1)


if __name__ == "__main__":
    main()
