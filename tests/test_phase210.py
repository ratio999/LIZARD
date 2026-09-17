"""Phase 2.10 tests — Stage 1 (DP completion of the Pólya prior) and
Stage 2 (opaque atomic-leaf object + move-density cardinalities).

Runnable as a plain script (no pytest):

    python test_phase210.py

Stage 1 tests (a)-(g) are the THEORY_2_13.md §5 lemmas in executable form,
plus regression guards for the legacy mode:

  (a) legacy regression: remainder_mass=0 (default) keeps the historical
      behaviour exactly — append-alpha_0 growth, alpha_rem == 0 everywhere,
      unvisited default -log(n_total);
  (b) conservation (Lemma 5.2): in DP mode, registration carves mass from
      the remainder at EVERY node — alpha_total + alpha_rem is invariant
      per node, existing slots untouched, new slot == the carved mass;
  (c) registration invariance (Theorem 5.3): registering {psi1, psi2} in
      either order yields identical log-priors for identical symbol
      sequences (registrations commute);
  (d) unvisited-region predictive: in DP mode a custom-mass primitive is
      scored with the base-measure predictive log(m) - log(base_reg_total)
      in unvisited regions (NOT the symmetric -log(n_total));
  (e) fresh-node inheritance: nodes created lazily AFTER registrations
      carry the current base_alphas and rem_base;
  (f) remainder exhaustion + legacy custom-mass both raise ValueError;
  (g) get_state/set_state round-trips the DP fields, and a stripped
      (pre-2.10 style) snapshot restores as a legacy tree.

The urn REINFORCEMENT dynamics are intentionally untouched by Stage 1 and
stay covered by test_phase23 3.3(g).

Stage 2 tests (h)-(n) are DESIGN_2_10_PRIMITIVES.md §9 items 2-3 in
executable form:

  (h) AtomSpec validation + instantiate_atom: k, slot order, fresh-name
      instantiation, atom_symbol, compiled evaluation, theta0 fallback;
  (i) tree bookkeeping: name_values absorbs opaque internal names (plain
      params first), penalty counts k_psi, BIC penalizes the total
      continuous dimension, the optimizer writes theta back into the leaf;
  (j) deepcopy isolation: a copy's theta never aliases the original's;
  (k) Pólya vocabulary: extract_path registers the TEMPLATE symbol once,
      never the per-instantiation description (registration invariance of
      the opaque cell);
  (l) move densities with scripted RNGs: point_mutation birth/death factors
      -log(1+A) / -log M, the A=0 force-rejects (point_mutation death, ETR,
      subtree_mutation), ETR's N0 = M + 1 + A, and the add_node/leaf_hoist
      time-reversal identity with an opaque leaf inside the built subtree;
  (m) crossover-path label hygiene: rename_prefix + compaction rename
      opaque internal names (expr and theta together), no collisions;
  (n) parameter_refresh treats theta_psi as ordinary coordinates once the
      AMState dimension matches name_values.

The Stage 3 grove/MPI wiring (AMState construction from name_values,
vocabulary freeze, wire format) is intentionally NOT covered here.
"""

import math
import os
import sys
from copy import deepcopy

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from polya_treeC import PolyaTree

_PRIMS = ["x", "sin(x)"]
_OPS_IGNORED = ["+", "-", "*", "/", "o", "**"]   # PolyaTree pins its own 6


def _mk(remainder=0.0, alpha_0=1.0):
    return PolyaTree(_OPS_IGNORED, list(_PRIMS), alpha_0=alpha_0,
                     auto_register=False, strict_ops=True,
                     remainder_mass=remainder)


def _path(polya, seq):
    """Build a (kind, idx) construction path from symbol names, using THIS
    instance's index maps — indices may differ between registration orders,
    symbol sequences may not."""
    out = []
    for kind, name in seq:
        if kind == "op":
            out.append(("op", polya.operations.index(name)))
        else:
            out.append(("prim", polya.primitives.index(name)))
    return out


def _walk_nodes(node, out):
    out.append(node)
    for child in node.children.values():
        _walk_nodes(child, out)
    return out


# Deterministic reinforcement history used by several tests: two accepted
# paths driven through the public replay API (update_path is cdef-only).
_HISTORY_SYMS = (
    (("op", "+"), ("prim", "x"), ("prim", "sin(x)")),
    (("op", "*"), ("op", "+"), ("prim", "x"), ("prim", "x"), ("prim", "sin(x)")),
)


def _replay_history(polya):
    polya.replay_paths([_path(polya, seq) for seq in _HISTORY_SYMS])


def test_a_legacy_regression():
    print("[2.10a] legacy mode (remainder_mass=0) is the historical prior")
    p = _mk(remainder=0.0)
    assert not p.dp_mode and p.rem_base == 0.0
    n_total0 = p.n_ops + p.n_prims

    _replay_history(p)
    for node in _walk_nodes(p.root, []):
        assert node.alpha_rem == 0.0, "legacy nodes must carry alpha_rem == 0"

    # Unvisited default is the symmetric -log(n_total).
    lp = p.log_prior(_path(p, (("op", "+"), ("op", "+"),
                               ("prim", "x"), ("prim", "x"), ("prim", "x"))))
    assert math.isfinite(lp)

    # Legacy growth: append alpha_0; total mass GROWS by alpha_0 per node.
    before = {id(n): n.alpha_total for n in _walk_nodes(p.root, [])}
    p.add_primitive("exp(x)")
    for node in _walk_nodes(p.root, []):
        assert math.isclose(node.alpha_total, before[id(node)] + 1.0,
                            rel_tol=0.0, abs_tol=1e-12)
        assert node.alpha_rem == 0.0
        assert node.alphas[node.n_choices - 1] == 1.0
    # New symmetric unvisited default reflects the grown vocabulary.
    n_total1 = p.n_ops + p.n_prims
    assert n_total1 == n_total0 + 1
    print("[2.10a] PASS")


def test_b_conservation():
    print("[2.10b] Lemma 5.2: carving conserves alpha_total + alpha_rem per node")
    p = _mk(remainder=4.0)
    assert p.dp_mode
    _replay_history(p)          # reinforced, multi-node trie

    nodes = _walk_nodes(p.root, [])
    before = {id(n): (n.alphas.copy(), n.alpha_total, n.alpha_rem)
              for n in nodes}
    mass = 0.25
    p.add_primitive("exp(x)", mass)

    for n in _walk_nodes(p.root, []):
        a0, t0, r0 = before[id(n)]
        # conservation of the node's total base+reinforced mass
        assert math.isclose(n.alpha_total + n.alpha_rem, t0 + r0,
                            rel_tol=0.0, abs_tol=1e-12), \
            "alpha_total + alpha_rem not conserved by registration"
        # carve arithmetic
        assert math.isclose(n.alpha_total, t0 + mass, rel_tol=0.0, abs_tol=1e-12)
        assert math.isclose(n.alpha_rem, r0 - mass, rel_tol=0.0, abs_tol=1e-12)
        # existing slots untouched, new slot == mass
        assert (n.alphas[:len(a0)] == a0).all(), "existing slots perturbed"
        assert n.alphas[n.n_choices - 1] == mass
    assert math.isclose(p.rem_base, 4.0 - mass, rel_tol=0.0, abs_tol=1e-12)
    print("[2.10b] PASS  (%d nodes checked)" % len(nodes))


def test_c_registration_order_invariance():
    print("[2.10c] Theorem 5.3: registrations commute")
    # Identical construction + identical reinforcement history, then
    # register {psi1: 0.25, psi2: 0.5} in the two orders. Dyadic masses keep
    # the check bit-exact; mathematically it is exact for any masses.
    p1 = _mk(remainder=4.0)
    p2 = _mk(remainder=4.0)
    _replay_history(p1)
    _replay_history(p2)
    p1.add_primitive("exp(x)", 0.25)
    p1.add_primitive("log(x)", 0.5)
    p2.add_primitive("log(x)", 0.5)
    p2.add_primitive("exp(x)", 0.25)

    probes = (
        (("op", "+"), ("prim", "x"), ("prim", "sin(x)")),
        (("op", "*"), ("prim", "exp(x)"), ("prim", "log(x)")),
        (("op", "+"), ("op", "*"), ("prim", "log(x)"), ("prim", "x"),
         ("prim", "exp(x)")),
    )
    for seq in probes:
        l1 = p1.log_prior(_path(p1, seq))
        l2 = p2.log_prior(_path(p2, seq))
        assert math.isclose(l1, l2, rel_tol=1e-12, abs_tol=1e-12), (
            "registration order changed the prior on %r: %.17g vs %.17g"
            % (seq, l1, l2))
    assert math.isclose(p1.rem_base, p2.rem_base, rel_tol=0.0, abs_tol=1e-15)
    print("[2.10c] PASS  (%d probe sequences)" % len(probes))


def test_d_unvisited_base_predictive():
    print("[2.10d] DP unvisited-region predictive uses the base masses")
    p = _mk(remainder=4.0)
    mass = 0.25
    p.add_primitive("exp(x)", mass)
    # Path: root (visited) -> one unvisited step choosing the custom-mass
    # primitive. base_reg_total = (6 ops + 2 prims) * 1.0 + 0.25 = 8.25.
    seq = (("op", "+"), ("prim", "exp(x)"))
    lp = p.log_prior(_path(p, seq))
    n_reg = p.n_ops + p.n_prims
    step1 = math.log(1.0) - math.log(p.root.alpha_total)      # root: 1/8.25
    step2 = math.log(mass) - math.log(p.base_reg_total)       # unvisited
    expected = step1 + step2
    assert math.isclose(lp, expected, rel_tol=0.0, abs_tol=1e-12), (
        "unvisited DP predictive mismatch: got %.17g expected %.17g "
        "(legacy formula would give %.17g)"
        % (lp, expected, step1 - math.log(n_reg)))
    print("[2.10d] PASS")


def test_e_fresh_node_inheritance():
    print("[2.10e] lazily created nodes inherit current base masses")
    p = _mk(remainder=4.0)
    p.add_primitive("exp(x)", 0.25)
    # Reinforce a path AFTER registration -> creates fresh children.
    p.replay_paths([_path(p, (("op", "-"), ("prim", "exp(x)"),
                              ("prim", "x")))])
    # The child under ('op', idx('-')) was created post-registration.
    key = ("op", p.operations.index("-"))
    child = p.root.children[key]
    assert child.n_choices == p.n_ops + p.n_prims
    # Its base initialisation matched base_alphas (up to the reinforcement
    # its own path added: exactly one +gamma on one slot). Check alpha_rem
    # and the untouched slots instead of the reinforced one.
    assert math.isclose(child.alpha_rem, p.rem_base, rel_tol=0.0,
                        abs_tol=1e-15)
    assert child.alphas[child.n_choices - 1] >= 0.25 - 1e-15, \
        "fresh node's custom-mass slot missing its base mass"
    print("[2.10e] PASS")


def test_f_error_paths():
    print("[2.10f] remainder exhaustion + legacy custom mass raise")
    p = _mk(remainder=0.5)
    raised = False
    try:
        p.add_primitive("exp(x)", 0.75)      # > remainder
    except ValueError:
        raised = True
    assert raised, "over-carving must raise ValueError"

    q = _mk(remainder=0.0)
    raised = False
    try:
        q.add_primitive("exp(x)", 0.25)      # custom mass in legacy mode
    except ValueError:
        raised = True
    assert raised, "custom mass in legacy mode must raise ValueError"
    print("[2.10f] PASS")


def _strip_dp_fields(state):
    """Simulate a pre-2.10 snapshot: remove every DP-era key recursively."""
    state = dict(state)
    for k in ("dp_mode", "rem_base", "base_alphas"):
        state.pop(k, None)

    def strip_node(s):
        s = dict(s)
        s.pop("alpha_rem", None)
        s["children"] = {k: strip_node(c) for k, c in s["children"].items()}
        return s

    state["root"] = strip_node(state["root"])
    return state


def test_g_state_roundtrip():
    print("[2.10g] get_state/set_state round-trips DP fields; legacy "
          "snapshots restore")
    p = _mk(remainder=4.0)
    _replay_history(p)
    p.add_primitive("exp(x)", 0.25)
    probe = (("op", "+"), ("prim", "exp(x)"), ("prim", "x"))
    lp_ref = p.log_prior(_path(p, probe))

    q = _mk(remainder=0.0)
    q.set_state(p.get_state())
    assert q.dp_mode and math.isclose(q.rem_base, p.rem_base,
                                      rel_tol=0.0, abs_tol=1e-15)
    lp_q = q.log_prior(_path(q, probe))
    assert math.isclose(lp_q, lp_ref, rel_tol=0.0, abs_tol=1e-15), \
        "DP state did not round-trip"

    # Pre-2.10-style snapshot (DP keys stripped) restores as a legacy tree.
    r = _mk(remainder=0.0)
    r.set_state(_strip_dp_fields(p.get_state()))
    assert not r.dp_mode and r.rem_base == 0.0
    assert math.isfinite(r.log_prior(_path(r, probe)))
    print("[2.10g] PASS")


# ===========================================================================
# Stage 2 — opaque atomic leaves (DESIGN_2_10_PRIMITIVES.md §9 items 2-3)
# ===========================================================================

import real_functionC as rfc
import function_treeC as ftc
import gp_movesC as gpm

_X = np.linspace(0.1, 2.0, 40)
_FUNCS = ["sin(x)", "x", "exp(x)"]          # M = 3
_OPS = ["+", "-", "*", "/", "**", "o"]      # O = 6


def _spec():
    return rfc.AtomSpec("Asin", "c0*sin(c1*x)")


class _ScriptRNG:
    """random.Random stand-in with scripted outputs (choice pops an INDEX
    into the sequence it is offered). Raises on unexpected/exhausted draws
    so every test also asserts its exact RNG consumption."""

    def __init__(self, randranges=(), randoms=(), choices=(), gausses=()):
        self._rr = list(randranges)
        self._rd = list(randoms)
        self._ch = list(choices)
        self._g = list(gausses)

    def randrange(self, n):
        assert self._rr, "unexpected randrange(%r)" % n
        v = self._rr.pop(0)
        assert 0 <= v < n, (v, n)
        return v

    def random(self):
        assert self._rd, "unexpected random()"
        return self._rd.pop(0)

    def choice(self, seq):
        assert self._ch, "unexpected choice(%r)" % (seq,)
        return seq[self._ch.pop(0)]

    def gauss(self, mu, sigma):
        assert self._g, "unexpected gauss()"
        return self._g.pop(0)

    def sample(self, population, k):
        raise AssertionError("unexpected sample()")

    def drained(self):
        return not (self._rr or self._rd or self._ch or self._g)


class _NpStub:
    """np.random stand-in for the few numpy draws the fixtures reach."""

    def uniform(self, a, b):
        return 0.0

    def normal(self, loc, scale):
        return np.asarray(scale) * 0.0 + 0.01

    def standard_normal(self, n):
        return np.zeros(n)


def _install(gp_rng, fc_rng=None):
    """Install scripted RNGs; returns the originals for the finally block."""
    gp_orig = gpm.get_rng()
    fc_orig = ftc.get_rng()
    gpm.set_rng(gp_rng, _NpStub())
    if fc_rng is not None:
        ftc.set_rng(fc_rng, _NpStub())
    return gp_orig, fc_orig


def _restore(gp_orig, fc_orig):
    gpm.set_rng(gp_orig[0], gp_orig[1])
    ftc.set_rng(fc_orig[0], fc_orig[1])


def _leaf_tree(expr_str):
    """Single-leaf tree from a plain expression string."""
    node = ftc.Node(rfc.RealFunctionC.from_string(expr_str, _X,
                                                  compute_grad=False))
    return ftc.Tree(0, list(_FUNCS), _X, node=node)


def _atom_leaf_tree(names, sigma=1.0):
    """Single-leaf tree whose root is an opaque Asin leaf."""
    node = ftc.Node(rfc.instantiate_atom(_spec(), _X, list(names)))
    return ftc.Tree(0, list(_FUNCS), _X, node=node, sigma=sigma), node


def _atom_pair_tree(op, other_expr, atom_names, sigma=1.0):
    """Tree op(atom_leaf, other_leaf)."""
    atom_node = ftc.Node(rfc.instantiate_atom(_spec(), _X, list(atom_names)))
    other_node = ftc.Node(rfc.RealFunctionC.from_string(
        other_expr, _X, compute_grad=False))
    root = ftc.Node(None, atom_node, other_node, op)
    return ftc.Tree(1, list(_FUNCS), _X, node=root, sigma=sigma), atom_node, other_node


def test_h_atomspec_object():
    print("[2.10h] AtomSpec + instantiate_atom: the Psi_k object")
    spec = _spec()
    assert spec.k == 2 and spec.slots == ["c0", "c1"]
    rf = rfc.instantiate_atom(spec, _X, ["a5", "a3"])
    assert rf.atom_symbol == "Asin"
    assert rf.theta0 == {"a5": 1.0, "a3": 1.0}
    # compiled evaluation with the fresh names
    y = rf.get_eval(_X, {"a5": 2.0, "a3": 3.0})
    assert np.allclose(y, 2.0 * np.sin(3.0 * _X))
    # theta0 fallback: params-free and partial-params probes are defined
    assert np.allclose(rf.get_eval(_X), np.sin(_X))
    assert np.allclose(rf.get_eval(_X, {"a5": 2.0}), 2.0 * np.sin(_X))
    # the unified taxonomy predicates
    node = ftc.Node(rf)
    assert node.is_atomic() and (not node.is_param()) and node.is_leaf()
    assert node.leaf_k() == 2
    assert node.leaf_param_names() == ["a5", "a3"]
    assert node.vocab_symbol() == "Asin"
    assert node.theta == {"a5": 1.0, "a3": 1.0}      # auto-init from theta0
    # validation errors
    for bad in (("c", "c0*c1"),        # no x -> that IS the parameter leaf
                ("f", "sin(x)"),       # k = 0 -> belongs in `functions`
                ("s", "c0*sin(w*x)")): # stray symbol
        raised = False
        try:
            rfc.AtomSpec(*bad)
        except ValueError:
            raised = True
        assert raised, "AtomSpec(%r, %r) must raise" % bad
    raised = False
    try:
        rfc.instantiate_atom(spec, _X, ["a1"])       # wrong arity
    except ValueError:
        raised = True
    assert raised
    print("[2.10h] PASS")


def test_i_tree_bookkeeping_and_bic():
    print("[2.10i] name_values / penalty / BIC absorb k_psi; optimizer round-trip")
    tree, atom_node, par_node_other = _atom_pair_tree("+", "a0", ["a1", "a2"],
                                                      sigma=None)
    assert par_node_other.is_param() and par_node_other.leaf_k() == 1
    pv, nv = tree.get_param_dicts()
    # plain parameters first, then the opaque internals, in leaf order
    assert list(nv.keys()) == ["a0", "a1", "a2"]
    assert nv == {"a0": 1.0, "a1": 1.0, "a2": 1.0}
    assert np.allclose(tree.evaluate(), 1.0 + np.sin(_X))
    # theta edits propagate through tree_update into name_values
    atom_node.theta["a1"] = 2.0
    atom_node.theta["a2"] = 3.0
    tree.tree_update(tree.get_root())
    pv, nv = tree.get_param_dicts()
    assert nv["a1"] == 2.0 and nv["a2"] == 3.0
    assert np.allclose(tree.evaluate(), 1.0 + 2.0 * np.sin(3.0 * _X))
    # BIC (sigma profiled): penalty on the TOTAL continuous dimension
    # k(T) + sum k_psi + 1 = 1 + 2 + 1 (DESIGN §5).
    y_data = tree.evaluate() + 0.05 * np.sin(7.0 * _X)
    lp = tree.log_marginal_bic(y_data, optimize=False)
    mse = float(np.nanmean((y_data - tree.evaluate()) ** 2))
    n = y_data.shape[0]
    expected = (-0.5 * n * (math.log(2.0 * math.pi * mse) + 1.0)
                - 0.5 * (3 + 1) * math.log(n))
    assert math.isclose(lp, expected, rel_tol=1e-12), (lp, expected)
    # the optimizer covers theta_psi (gate is name_values, not tree.params)
    # and writes the optimum back INTO the leaf's theta
    lp_opt = tree.log_marginal_bic(y_data, optimize=True)
    assert math.isfinite(lp_opt) and lp_opt >= lp - 1e-9
    pv, nv = tree.get_param_dicts()
    for name in ("a1", "a2"):
        assert nv[name] == atom_node.theta[name], "leaf theta out of sync"
    # atomic-ONLY tree (zero plain parameter Nodes) still optimizes
    atree, anode = _atom_leaf_tree(["a0", "a1"], sigma=None)
    y2 = 1.7 * np.sin(1.1 * _X) + 0.02 * _X
    lp2_frozen = atree.log_marginal_bic(y2, optimize=False)
    lp2 = atree.log_marginal_bic(y2, optimize=True)
    assert math.isfinite(lp2) and lp2 >= lp2_frozen - 1e-9
    assert anode.theta != {"a0": 1.0, "a1": 1.0}, \
        "optimizer never touched the opaque theta"
    print("[2.10i] PASS")


def test_j_deepcopy_isolation():
    print("[2.10j] deepcopy isolates theta and preserves the atom identity")
    tree, atom_node, _ = _atom_pair_tree("+", "a0", ["a1", "a2"])
    atom_node.theta["a1"] = 4.0
    tree.tree_update(tree.get_root())
    t2 = deepcopy(tree)
    atomics2 = [lf for lf in t2.get_leaves() if lf.is_atomic()]
    assert len(atomics2) == 1
    lf2 = atomics2[0]
    assert lf2.vocab_symbol() == "Asin"
    assert lf2.get_fun().atom_symbol == "Asin"
    assert lf2.theta == atom_node.theta and lf2.theta is not atom_node.theta
    lf2.theta["a1"] = -9.0
    t2.tree_update(t2.get_root())
    assert atom_node.theta["a1"] == 4.0, "copy aliased the original theta"
    _, nv1 = tree.get_param_dicts()
    _, nv2 = t2.get_param_dicts()
    assert nv2["a1"] == -9.0 and nv1["a1"] == 4.0
    print("[2.10j] PASS")


def test_k_polya_vocab_symbol():
    print("[2.10k] Pólya vocabulary: one cell per template, never per instantiation")
    tree, atom_node, _ = _atom_pair_tree("+", "a0", ["a1", "a2"])
    p = PolyaTree(_OPS_IGNORED, ["x", "sin(x)", "exp(x)", "a0"],
                  alpha_0=1.0, auto_register=True, strict_ops=True,
                  remainder_mass=0.0)
    n0 = len(p.primitives)
    p.extract_path(tree.get_root())
    assert "Asin" in p.primitives
    assert len(p.primitives) == n0 + 1
    inst_desc = atom_node.get_fun().get_description()
    assert inst_desc not in p.primitives, \
        "instantiated description leaked into the vocabulary"
    # a second instantiation with DIFFERENT fresh names hits the same cell
    node2 = ftc.Node(rfc.instantiate_atom(_spec(), _X, ["a7", "a9"]))
    p.extract_path(node2)
    assert len(p.primitives) == n0 + 1
    # coverage check accepts the opaque leaf via its template symbol
    p.assert_covers(tree.get_root())
    # and the prior of the atom-bearing path is finite
    assert math.isfinite(p.log_prior_from_tree(tree.get_root()))
    print("[2.10k] PASS")


def test_l_move_densities():
    print("[2.10l] move densities: M+1 -> M+1+A, force-rejects, time reversal")
    atoms = [_spec()]
    M, O, A = len(_FUNCS), len(_OPS), 1

    # --- l1: point_mutation BIRTH of an atom (function leaf -> opaque) ----
    tree = _leaf_tree("sin(x)")
    script = _ScriptRNG(randranges=[0, 1], randoms=[0.3], gausses=[0.7, -0.2])
    orig = _install(script)
    try:
        lqf, lqr = gpm.point_mutation(tree, _FUNCS, _OPS, None, atoms)
    finally:
        _restore(*orig)
    assert script.drained()
    assert math.isclose(lqf, -math.log(1.0 + A))
    assert math.isclose(lqr, -math.log(M))
    root = tree.get_root()
    # warm start = theta0 (unit reference) + gauss draw (Stage 3a law)
    assert root.is_atomic() and root.theta == {"a0": 1.7, "a1": 0.8}
    _, nv = tree.get_param_dicts()
    assert nv == {"a0": 1.7, "a1": 0.8}

    # --- l2a: DEATH without atoms in the kernel -> force-reject, no-op ----
    script = _ScriptRNG(randranges=[0], randoms=[0.3])
    orig = _install(script)
    try:
        lqf, lqr = gpm.point_mutation(tree, _FUNCS, _OPS, None, None)
    finally:
        _restore(*orig)
    assert script.drained()
    assert lqf == 0.0 and math.isinf(lqr) and lqr < 0
    assert tree.get_root().is_atomic(), "force-reject must not mutate"

    # --- l2b: DEATH with atoms -> -log M forward, -log(1+A) reverse -------
    script = _ScriptRNG(randranges=[0], randoms=[0.3], choices=[2])
    orig = _install(script)
    try:
        lqf, lqr = gpm.point_mutation(tree, _FUNCS, _OPS, None, atoms)
    finally:
        _restore(*orig)
    assert script.drained()
    assert math.isclose(lqf, -math.log(M))
    assert math.isclose(lqr, -math.log(1.0 + A))
    root = tree.get_root()
    assert (not root.is_atomic()) and root.theta is None
    assert root.get_fun().get_description() == "exp(x)"
    _, nv = tree.get_param_dicts()
    assert nv == {}

    # --- l3: ETR 0-ET -> 2-ET with an atom child; N0 = M + 1 + A ---------
    n0 = M + 1 + A
    n2 = O * n0 * n0
    tree = _leaf_tree("x")
    script = _ScriptRNG(randranges=[0, 4, 0], randoms=[0.7], choices=[0],
                        gausses=[0.4, 0.5])
    orig = _install(script)
    try:
        lqf, lqr = gpm.etr_mutation(tree, _FUNCS, _OPS, atoms)
    finally:
        _restore(*orig)
    assert script.drained()
    assert math.isclose(lqf, -math.log(1.0) - math.log(2.0) - math.log(n2))
    assert math.isclose(lqr, -math.log(3.0) - math.log(2.0) - math.log(n0))
    atomics = [lf for lf in tree.get_leaves() if lf.is_atomic()]
    # warm start = theta0 (1.0) + the scripted gauss draws
    assert len(atomics) == 1 and atomics[0].theta == {"a0": 1.4, "a1": 1.5}
    assert np.all(np.isfinite(tree.evaluate()))

    # --- l4: ETR on an opaque leaf WITHOUT atoms -> force-reject, no-op ---
    atree, _ = _atom_leaf_tree(["a0", "a1"])
    script = _ScriptRNG(randranges=[0])
    orig = _install(script)
    try:
        lqf, lqr = gpm.etr_mutation(atree, _FUNCS, _OPS, None)
    finally:
        _restore(*orig)
    assert script.drained()
    assert lqf == 0.0 and math.isinf(lqr) and lqr < 0
    assert atree.get_root().is_atomic()

    # --- l5: add_node builds S containing an atom; leaf_hoist reverses ----
    tree = _leaf_tree("x")
    gp_script = _ScriptRNG(randranges=[3, 1], randoms=[0.1, 0.6, 0.9, 0.9],
                           choices=[0], gausses=[0.3, -0.4])
    fc_script = _ScriptRNG(choices=[0, 1, 0])
    orig = _install(gp_script, fc_script)
    try:
        lqf_a, lqr_a = gpm.add_node(tree, _FUNCS, _OPS, False, atoms)
    finally:
        _restore(*orig)
    assert gp_script.drained() and fc_script.drained()
    rho, d_max = 0.5, 4
    z = (1.0 - rho ** d_max) / (1.0 - rho)
    logpmf1 = math.log((rho ** 0) / z)
    build = 2.0 * (math.log(1.0 - 0.3) - math.log(M + A)) - math.log(O)
    exp_fwd = (-math.log(1.0) - math.log(O) + logpmf1 - math.log(2.0) + build)
    exp_rev = -math.log(3.0)
    assert math.isclose(lqf_a, exp_fwd), (lqf_a, exp_fwd)
    assert math.isclose(lqr_a, exp_rev), (lqr_a, exp_rev)
    assert sum(1 for lf in tree.get_leaves() if lf.is_atomic()) == 1
    # reverse: lift the original leaf, dropping S — time-reversal identity
    gp_script = _ScriptRNG(randranges=[2])
    orig = _install(gp_script)
    try:
        lqf_h, lqr_h = gpm.leaf_hoist(tree, _FUNCS, _OPS, atoms)
    finally:
        _restore(*orig)
    assert gp_script.drained()
    assert math.isclose(lqf_h, exp_rev) and math.isclose(lqr_h, exp_fwd), \
        "add_node/leaf_hoist time-reversal identity broken"
    assert tree.get_root().is_leaf()
    assert tree.tree_get_description() == "x"

    # --- l5b: leaf_hoist dropping an atom-bearing S WITHOUT atoms -> -inf -
    # A bare atom sibling (d_S = 0) is already off-support, so build a
    # height-1 sibling: root = '+'( '+'(atom, sin(x)), x ) and lift x.
    inner_atom = ftc.Node(rfc.instantiate_atom(_spec(), _X, ["a0", "a1"]))
    inner_sin = ftc.Node(rfc.RealFunctionC.from_string("sin(x)", _X,
                                                       compute_grad=False))
    inner = ftc.Node(None, inner_atom, inner_sin, "+")
    outer_leaf = ftc.Node(rfc.RealFunctionC.from_string("x", _X,
                                                        compute_grad=False))
    root2 = ftc.Node(None, inner, outer_leaf, "+")
    tree2 = ftc.Tree(2, list(_FUNCS), _X, node=root2)
    gp_script = _ScriptRNG(randranges=[2])       # leaves: [atom, sin, x] -> x
    orig = _install(gp_script)
    try:
        lqf, lqr = gpm.leaf_hoist(tree2, _FUNCS, _OPS, None)
    finally:
        _restore(*orig)
    assert gp_script.drained()
    assert math.isinf(lqr) and lqr < 0, \
        "atom-bearing sibling must be off-support without atoms"

    # --- l6: subtree_mutation force-rejects atom-bearing subtrees ---------
    tree3, _, _ = _atom_pair_tree("+", "x", ["a0", "a1"])
    desc_before = tree3.tree_get_description()
    gp_script = _ScriptRNG(randranges=[0])
    orig = _install(gp_script)
    try:
        lqf, lqr = gpm.subtree_mutation(tree3, _FUNCS, _OPS)
    finally:
        _restore(*orig)
    assert gp_script.drained()
    assert lqf == 0.0 and math.isinf(lqr) and lqr < 0
    assert tree3.tree_get_description() == desc_before, \
        "force-reject must not mutate"
    print("[2.10l] PASS")


def test_m_crossover_label_hygiene():
    print("[2.10m] crossover renames opaque internals (expr + theta) safely")
    p1, atom1, _ = _atom_pair_tree("+", "x", ["a0", "a1"])
    p2, atom2, _ = _atom_pair_tree("*", "sin(x)", ["a0", "a1"])
    atom2.theta["a0"] = 5.0
    atom2.theta["a1"] = 6.0
    p2.tree_update(p2.get_root())
    script = _ScriptRNG(randranges=[0, 0])
    orig = _install(script)
    try:
        lqf, lqr = gpm.subtree_crossover(p1, p2)
    finally:
        _restore(*orig)
    assert script.drained()
    assert lqf == 0.0 and lqr == 0.0          # single internal node each
    for t, expected_vals in ((p1, [5.0, 6.0]), (p2, [1.0, 1.0])):
        _, nv = t.get_param_dicts()
        assert sorted(nv.keys()) == ["a0", "a1"], nv
        atomics = [lf for lf in t.get_leaves() if lf.is_atomic()]
        assert len(atomics) == 1
        lf = atomics[0]
        assert lf.vocab_symbol() == "Asin"
        assert sorted(lf.theta.keys()) == sorted(lf.leaf_param_names())
        assert set(lf.leaf_param_names()) == set(nv.keys())
        assert sorted(lf.theta.values()) == expected_vals, \
            "theta values lost in the rename chain"
        assert np.all(np.isfinite(t.evaluate()))
    print("[2.10m] PASS")


def test_n_parameter_refresh_atomic():
    print("[2.10n] parameter_refresh treats theta_psi as ordinary coordinates")
    tree, atom_node = _atom_leaf_tree(["a0", "a1"])
    atom_node.theta["a0"] = 0.5
    atom_node.theta["a1"] = 1.5
    tree.tree_update(tree.get_root())
    y = np.asarray(tree.evaluate(), dtype=np.float64)
    # dimension mismatch (grove still builds AMState from tree.params:
    # Stage 3 wiring) -> safe no-op
    am_bad = gpm.AMState(["a0"])
    script = _ScriptRNG()
    orig = _install(script)
    try:
        acc, _, _ = gpm.parameter_refresh(tree, y, 1.0, am_bad, 1.0)
    finally:
        _restore(*orig)
    assert script.drained()
    assert not acc
    assert atom_node.theta == {"a0": 0.5, "a1": 1.5}
    # matched dimension: cold-start diagonal proposal, forced accept
    am = gpm.AMState(["a0", "a1"])
    script = _ScriptRNG(randoms=[0.0])       # accept draw
    orig = _install(script)
    try:
        acc, _, _ = gpm.parameter_refresh(tree, y, 1.0, am, 1.0)
    finally:
        _restore(*orig)
    assert script.drained()
    assert acc
    # _NpStub.normal returns +0.01 per coordinate
    assert math.isclose(atom_node.theta["a0"], 0.51)
    assert math.isclose(atom_node.theta["a1"], 1.51)
    _, nv = tree.get_param_dicts()
    assert math.isclose(nv["a0"], 0.51) and math.isclose(nv["a1"], 1.51)
    assert am.n_visits == 1
    print("[2.10n] PASS")


def test_o_psi2_recovery(n_steps=600, seed=123):
    """Phase 2.10 Stage 3a end-to-end (DESIGN §9 item 5): a single-process
    Forest given the Asin = c0*sin(c1*x) template recovers y = A sin(w x).

    The kernel is restricted to the four safe operations (no compose, no
    pow), so a frequency-scaled sinusoid is UNREACHABLE by combining the
    plain vocabulary — the opaque atom is the only way to fit the signal.
    Recovery therefore demonstrates the full chain: Forest registers the
    template in the shared Pólya (one cell), Grove threads `atoms` into
    the kernel, a move births the opaque leaf, the BIC path optimizes
    theta_psi (name_values gate), and MH keeps it.
    """
    print("[2.10o] Stage 3a end-to-end: Psi_2 recovery of A*sin(w*x)")
    import groveC

    x = np.linspace(0.1, 6.0, 50)
    y = 2.3 * np.sin(1.3 * x)
    spec = _spec()
    funcs = ["sin(x)", "x"]
    ops = ["+", "-", "*", "/"]

    forest = groveC.Forest(x, y, np.array([1.0]), 4, 1, list(funcs),
                           sigma=1.0, Npars=1, operations=list(ops),
                           use_gradients=False, seed=seed,
                           atoms=[spec])
    # (1) the template symbol is registered ONCE, at construction
    assert forest.polya.primitives.count("Asin") == 1
    grove = forest.groves[0]
    assert grove.atoms is not None and len(grove.atoms) == 1

    forest.run_all(y, n_steps)

    # (2) no instantiated atom description ever leaked into the vocabulary
    assert forest.polya.primitives.count("Asin") == 1
    for prim in forest.polya.primitives:
        assert "*" not in prim, \
            "instantiated opaque description leaked into the Polya: %r" % prim

    # (3) the best tree carries the opaque leaf and fits the signal
    best = forest.best_tree()
    atomics = [lf for lf in best.get_leaves() if lf.is_atomic()]
    assert len(atomics) >= 1, (
        "best tree %r contains no opaque leaf after %d steps"
        % (best.tree_get_description(), n_steps))
    resid = y - np.asarray(best.evaluate(), dtype=np.float64)
    mse = float(np.nanmean(resid * resid))
    assert mse < 0.05, (
        "best tree %r fits poorly (mse %.4f) — theta_psi not optimized?"
        % (best.tree_get_description(), mse))

    # (Stage 3b lifted the run_mpi_spawn guard: atoms now cross the wire
    # as skeleton payloads — unit-tested in (p); the live-MPI end-to-end
    # needs an MPI-spawn-capable environment, see test_phase2mpi.)
    print("[2.10o] PASS  (best: %s, mse %.2e)"
          % (forest.best_tree().tree_get_description(), mse))


def test_p_wire_roundtrip():
    """Phase 2.10 Stage 3b: the MPI tree wire. Atom-bearing trees ship a
    ('skel3b', skeleton, description) structural payload that reconstructs
    opaque leaves as units and preserves the exact tree shape; atom-free
    trees keep the historical raw-sympy-expr payload. The whole wire (and
    AtomSpec, for the worker startup config) must survive pickle — that is
    what mpi4py does to it."""
    print("[2.10p] Stage 3b wire: skeleton payload round-trip")
    import pickle
    import groveC

    spec = _spec()
    specs = {"Asin": spec}

    # --- atom-bearing tree: '+'(atom(a1,a2), a0), with non-trivial values
    tree, atom_node, par_node = _atom_pair_tree("+", "a0", ["a1", "a2"])
    atom_node.theta["a1"] = 2.0
    atom_node.theta["a2"] = 3.0
    par_node.set_value(0.5)
    tree.tree_update(tree.get_root())

    wire = groveC._tree_to_wire(tree, -12.5)
    payload, nv, post = wire
    assert isinstance(payload, tuple) and payload[0] == "skel3b"
    assert post == -12.5
    assert nv == {"a0": 0.5, "a1": 2.0, "a2": 3.0}

    # ship it exactly like mpi4py would
    wire2 = pickle.loads(pickle.dumps(wire))
    t2 = groveC._tree_from_wire(wire2[0], wire2[1], _X, list(_FUNCS),
                                1.0, 1.5, specs)
    assert t2 is not None
    assert t2.tree_get_description() == tree.tree_get_description()
    atomics = [lf for lf in t2.get_leaves() if lf.is_atomic()]
    assert len(atomics) == 1 and atomics[0].vocab_symbol() == "Asin"
    assert atomics[0].theta == {"a1": 2.0, "a2": 3.0}
    _, nv2 = t2.get_param_dicts()
    assert nv2 == nv
    # exact-shape guarantee: the skeleton itself round-trips identically
    assert ftc.tree_to_skel(t2.get_root()) == ftc.tree_to_skel(tree.get_root())

    # missing template registry -> REJECT (None), never a crash
    assert groveC._tree_from_wire(wire2[0], wire2[1], _X, list(_FUNCS),
                                  1.0, 1.5, None) is None

    # --- atom-free tree keeps the HISTORICAL payload (raw sympy expr) ----
    plain = _leaf_tree("sin(x)")
    w3 = groveC._tree_to_wire(plain, 1.0)
    assert not isinstance(w3[0], tuple), "atom-free payload must stay legacy"
    t3 = groveC._tree_from_wire(w3[0], w3[1], _X, list(_FUNCS),
                                1.0, 1.5, None)
    assert t3 is not None and t3.tree_get_description() == "sin(x)"

    # --- AtomSpec pickles (rides the worker startup config) --------------
    s2 = pickle.loads(pickle.dumps(spec))
    assert (s2.name, s2.template, s2.k, list(s2.slots)) == \
           (spec.name, spec.template, spec.k, list(spec.slots))
    print("[2.10p] PASS")


def test_r_vocabulary_freeze():
    """Phase 2.10 Stage 3c: the FULL vocabulary freeze (DESIGN §9 item 4's
    deferred half, settling Phase 2.3).

    Mechanism: unify_params maps EVERY plain parameter leaf onto ONE
    canonical psi_const cell (DESIGN §1 — the parameter leaf is one atom of
    Psi_1; names are bound labels new_par can ratchet without bound, so no
    finite per-name pool can ever be frozen). The vocabulary
    {ops} + {function leaves} + {psi_const} + {atom templates} is then
    finite and CLOSED, auto_register turns off, and the prior is one fixed,
    renaming-invariant measure from t = 0.
    """
    print("[2.10r] Stage 3c: vocabulary freeze / psi_const unification")
    import groveC
    from polya_treeC import PARAM_CELL_TOKEN

    def _ptree(name):
        par = ftc.Node(rfc.RealFunctionC.from_string(name, _X,
                                                     compute_grad=False))
        oth = ftc.Node(rfc.RealFunctionC.from_string("sin(x)", _X,
                                                     compute_grad=False))
        root = ftc.Node(None, par, oth, "+")
        return ftc.Tree(1, list(_FUNCS), _X, node=root)

    # --- (1) alpha-equivalence: the prior is renaming-invariant ----------
    p = PolyaTree(_OPS_IGNORED, list(_FUNCS), alpha_0=1.0,
                  auto_register=False, strict_ops=True,
                  remainder_mass=0.0, unify_params=True)
    n_vocab = len(p.primitives)
    assert p.primitives.count(PARAM_CELL_TOKEN) == 1
    lp_a0 = p.log_prior_from_tree(_ptree("a0").get_root())
    lp_b7 = p.log_prior_from_tree(_ptree("b7").get_root())
    assert lp_a0 == lp_b7, "renamed trees got different priors"
    assert len(p.primitives) == n_vocab, "extraction grew a frozen vocab"
    # reinforcement lands on the SHARED psi_const cell
    p.replay_paths([p.extract_path(_ptree("a0").get_root())])
    lp_a0_2 = p.log_prior_from_tree(_ptree("a0").get_root())
    lp_b7_2 = p.log_prior_from_tree(_ptree("b7").get_root())
    assert lp_a0_2 == lp_b7_2 and lp_a0_2 > lp_a0

    # --- (2) closed vocabulary: unknown FUNCTION leaf -> guarded KeyError
    bad = ftc.Node(rfc.RealFunctionC.from_string("log(x)", _X,
                                                 compute_grad=False))
    raised = False
    try:
        p.extract_path(bad)
    except KeyError:
        raised = True
    assert raised, "frozen vocab must raise on unknown symbols"
    assert len(p.primitives) == n_vocab

    # --- (3) reserved-token guards ----------------------------------------
    raised = False
    try:
        PolyaTree(_OPS_IGNORED, [PARAM_CELL_TOKEN], alpha_0=1.0,
                  auto_register=False, strict_ops=True,
                  remainder_mass=0.0, unify_params=True)
    except ValueError:
        raised = True
    assert raised, "token in primitives must be rejected"
    raised = False
    try:
        p.add_primitive(PARAM_CELL_TOKEN)
    except ValueError:
        raised = True
    assert raised, "registering the reserved token must be rejected"

    # --- (4) atoms coexist: template cell is distinct from psi_const ------
    p.add_primitive("Asin")          # explicit registration stays allowed
    mixed, _, _ = _atom_pair_tree("+", "a0", ["a1", "a2"])
    mpath = p.extract_path(mixed.get_root())
    idx_param = p.primitives.index(PARAM_CELL_TOKEN)
    idx_asin = p.primitives.index("Asin")
    assert ("prim", idx_param) in mpath and ("prim", idx_asin) in mpath
    p.assert_covers(mixed.get_root())

    # --- (5) state round-trip (the MPI replica path) -----------------------
    q = PolyaTree(_OPS_IGNORED, ["x"], alpha_0=1.0, auto_register=True,
                  strict_ops=True, remainder_mass=0.0)
    q.set_state(p.get_state())
    assert q.unify_params and q._param_idx == p._param_idx
    assert (q.log_prior_from_tree(_ptree("a0").get_root())
            == p.log_prior_from_tree(_ptree("a0").get_root()))

    # --- (6) frozen Forest end-to-end: the vocabulary NEVER grows ----------
    x = np.linspace(0.1, 6.0, 50)
    y = 2.3 * np.sin(1.3 * x)
    forest = groveC.Forest(x, y, np.array([1.0]), 4, 1,
                           ["sin(x)", "x"], sigma=1.0, Npars=1,
                           operations=["+", "-", "*", "/"],
                           use_gradients=False, seed=321,
                           atoms=[_spec()], freeze_vocab=True)
    assert (not forest.polya.auto_register) and forest.polya.unify_params
    vocab0 = list(forest.polya.primitives)
    assert vocab0.count(PARAM_CELL_TOKEN) == 1 and vocab0.count("Asin") == 1
    forest.run_all(y, 300)
    assert list(forest.polya.primitives) == vocab0, \
        "vocabulary grew during a frozen run"
    g0 = forest.groves[0]
    assert np.isfinite(g0.root_posteriors[g0.best_index()])
    print("[2.10r] PASS  (frozen vocab: %d cells, unchanged after 300 steps)"
          % len(vocab0))


def test_s_burn_in_atom_seeding():
    """Phase 2.10 (final item): the serial burn-in seeds opaque templates.

    Templates enter the enumeration as '@atom:<symbol>' tokens, so stage 1
    scores the pure atom as a depth-0 candidate and every depth-1 pairing
    with it. With funcs=["x"] and ops=["+"] nothing in the plain vocabulary
    can represent a frequency-scaled sinusoid, so the burn-in can only
    reach a good fit through the seeded atom — recovery proves the seeding
    end-to-end (construction, naming discipline, scoring, installation).
    """
    print("[2.10s] burn-in seeding of atom-bearing depth-0/1 candidates")
    import groveC

    x = np.linspace(0.1, 6.0, 50)
    y = 2.3 * np.sin(1.3 * x)
    grove = groveC.Grove(x, y, 1.0, 6, 1, ["x"], sigma=1.0,
                         operations=["+"], use_gradients=False,
                         burn_in=True, seed=11, atoms=[_spec()])
    assert grove.size > 0 and grove.trees, "burn-in produced no trees"

    # the best burned-in tree is atom-bearing and fits the signal only the
    # atom can represent
    best = grove.best_tree()
    atomics = [lf for lf in best.get_leaves() if lf.is_atomic()]
    assert len(atomics) >= 1, (
        "burn-in best tree %r carries no opaque leaf"
        % best.tree_get_description())
    resid = y - np.asarray(best.evaluate(), dtype=np.float64)
    mse = float(np.nanmean(resid * resid))
    assert mse < 1e-2, (
        "burn-in failed to fit through the seeded atom (best %r, mse %.4f)"
        % (best.tree_get_description(), mse))

    # naming discipline: wherever two opaque leaves share one tree, their
    # internal names are disjoint (shared names would couple their values)
    n_double = 0
    for t in grove.trees:
        leaves = [lf for lf in t.get_leaves() if lf.is_atomic()]
        if len(leaves) >= 2:
            n_double += 1
            names0 = set(leaves[0].leaf_param_names())
            names1 = set(leaves[1].leaf_param_names())
            assert not (names0 & names1), (
                "coupled atom internals in %r" % t.tree_get_description())
    assert n_double >= 1, "no atom-op-atom candidate was ever built"
    print("[2.10s] PASS  (best: %s, mse %.2e, %d atom-pair candidates)"
          % (best.tree_get_description(), mse, n_double))


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_a_legacy_regression()
    test_b_conservation()
    test_c_registration_order_invariance()
    test_d_unvisited_base_predictive()
    test_e_fresh_node_inheritance()
    test_f_error_paths()
    test_g_state_roundtrip()
    print("\nAll Phase 2.10 Stage 1 (DP completion) tests passed.\n")
    test_h_atomspec_object()
    test_i_tree_bookkeeping_and_bic()
    test_j_deepcopy_isolation()
    test_k_polya_vocab_symbol()
    test_l_move_densities()
    test_m_crossover_label_hygiene()
    test_n_parameter_refresh_atomic()
    print("\nAll Phase 2.10 Stage 2 (atomic-leaf object) tests passed.\n")
    test_o_psi2_recovery()
    test_p_wire_roundtrip()
    test_r_vocabulary_freeze()
    test_s_burn_in_atom_seeding()
    print("\nPhase 2.10 Stage 3a+3b+3c + burn-in seeding tests passed.")
