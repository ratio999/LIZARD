"""i3b_polya_pricing.py -- Polya structure-prior pricing for the I3-B
full-space vocabulary (I3-B preparation, D3).

What is priced
--------------
The machine's structure prior is the Polya-tree prior over CONSTRUCTION
PATHS. Under the frozen vocabulary the Forest uses (freeze_vocab=True:
unify_params, auto_register off, remainder_mass 0) the t = 0 measure is
symmetric: every decision node is a Dirichlet(alpha_0) over the same
n_total = n_ops + n_prims cells, where n_ops = 6 always (the PolyaTree
op cells are the fixed list +,-,*,/,o,** regardless of which ops the
Grove proposes with) and n_prims = P + 1 (the P declared function
leaves plus the single canonical parameter cell psi_const). A tree with
N nodes -- every operator node and every leaf is one construction step,
parameter leaves included -- therefore has

    log pi_0(T) = - N * log(6 + P + 1).

This module prices through the kernel (function_treeC.node_from_skel ->
PolyaTree.log_prior_from_tree on a fresh frozen PolyaTree) and checks
the closed form against it, so a future change in the kernel's measure
surfaces here as an assertion, not as a silently wrong table.

Exact vs canonical pricing
--------------------------
The prior is a function of the TREE, not of the expression string: the
same expression can be built as different trees (x0 + x0 vs 2*x0). For
a tree the machine actually held, price its skeleton (the I3-B archive
stores `skel` for this). For a bare expression string -- a hand-written
anchor, a deployed formula -- `expr_to_skel` builds the canonical
binary-tree reading of the sympy form: n-ary sums and products are
left-associated, a - b and a / b are '-' and '/' nodes, numbers and
unregistered symbols are parameter leaves, integer powers 2..4 are
repeated products, and a function application is a leaf only if it is a
declared vocabulary primitive. Anything else -- exp(x0*x1), sqrt, a
power of a compound -- is NOT expressible in this vocabulary without an
atom template or compose, and raises.

Run (WSL, mcmc_koopman, from gpu_ms):
    python i3b_polya_pricing.py                      # pricing table
    python i3b_polya_pricing.py "2*x1 + a0*tanh(x2)" # price expressions
"""
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import sympy as sp

import real_functionC
from function_treeC import node_from_skel
from polya_treeC import PolyaTree, PARAM_CELL_TOKEN

N_OPS_POLYA = 6          # the fixed PolyaTree op cells (+ - * / o **)
OPS_SAFE4 = ["+", "-", "*", "/"]
VARIABLES_I3B = ("x0", "x1", "x2")


def vocabulary_i3b(variables=VARIABLES_I3B, positive=("x1", "x2"),
                   with_log=True):
    """The declared I3-B vocabulary: per variable {x, exp(x), tanh(x)}
    and log(x) for the variables declared positive. x0 = t is NOT
    positive at the peak (t = 0), so log(x0) is excluded by default."""
    funcs = []
    for v in variables:
        funcs += [v, "exp(%s)" % v, "tanh(%s)" % v]
    if with_log:
        for v in variables:
            if v in positive:
                funcs.append("log(%s)" % v)
    return funcs


def make_t0_polya(funcs, alpha_0=1.0, atoms=None):
    """A fresh frozen PolyaTree, exactly as Forest(freeze_vocab=True)
    builds it (no reinforcement: t = 0, the symmetric measure). Atom
    templates, if any, are registered the way the Forest registers
    them (one cell per template symbol, positional mass argument)."""
    polya = PolyaTree(list(OPS_SAFE4), list(funcs), alpha_0=alpha_0,
                      auto_register=False, strict_ops=True,
                      remainder_mass=0.0, unify_params=True)
    if atoms:
        for a in atoms:
            polya.add_primitive(a.name, None)
    return polya


def n_total_cells(funcs, n_atoms=0):
    return N_OPS_POLYA + len(funcs) + 1 + int(n_atoms)


def closed_form_log_prior(n_nodes, funcs):
    return -float(n_nodes) * math.log(n_total_cells(funcs))


# ---------------------------------------------------------------------------
# expression -> canonical skeleton
# ---------------------------------------------------------------------------
class NotExpressible(ValueError):
    """The expression needs a construct outside the vocabulary (compose
    or an atom template)."""


def _chain(op, items):
    node = items[0]
    for it in items[1:]:
        node = ('O', op, node, it)
    return node


def expr_to_skel(expr, funcs, variables=VARIABLES_I3B):
    """Canonical skeleton of a sympy expression (or string) under the
    vocabulary `funcs`. Parameter leaves are named a0, a1, ... in order
    of appearance; every number and every non-variable symbol is one."""
    if isinstance(expr, str):
        expr = sp.sympify(expr)
    funcs = list(funcs)
    fset = set(funcs)
    vset = set(variables)
    counter = [0]

    def new_param():
        name = "a%d" % counter[0]
        counter[0] += 1
        return ('L', name)

    def rec(e):
        s = str(e)
        if s in fset:
            return ('L', s)
        if e.is_Number:
            return new_param()
        if isinstance(e, sp.Symbol):
            if s in vset:
                raise NotExpressible(
                    "variable %r is registered but not in the vocabulary "
                    "%r" % (s, funcs))
            return new_param()
        if isinstance(e, sp.Add):
            pos, neg = [], []
            for a in e.args:
                if isinstance(a, sp.Mul) and a.args[0] == -1:
                    neg.append(sp.Mul(*a.args[1:]))
                elif a.is_Number and a < 0:
                    neg.append(-a)
                else:
                    pos.append(a)
            if not pos:
                # -(b1 + b2 ...): a parameter times the sum
                return ('O', '*', new_param(), rec(sp.Add(*neg)))
            node = _chain('+', [rec(a) for a in pos])
            for b in neg:
                node = ('O', '-', node, rec(b))
            return node
        if isinstance(e, sp.Mul):
            num, den = [], []
            for a in e.args:
                if isinstance(a, sp.Pow) and a.args[1].is_Number \
                        and a.args[1] < 0:
                    den.append(sp.Pow(a.args[0], -a.args[1]))
                elif isinstance(a, sp.Rational) and a.q != 1 and a.p == 1:
                    den.append(sp.Integer(a.q))
                else:
                    num.append(a)
            if not num:
                num = [sp.Integer(1)]
            node = _chain('*', [rec(a) for a in num])
            for b in den:
                node = ('O', '/', node, rec(b))
            return node
        if isinstance(e, sp.Pow):
            base, ex = e.args
            if ex.is_Integer and 2 <= int(ex) <= 4:
                return _chain('*', [rec(base)] * int(ex))
            if ex.is_Integer and int(ex) == -1:
                return ('O', '/', new_param(), rec(base))
            raise NotExpressible(
                "power %s is not expressible with ops %r (no '**' at "
                "d > 1; integer powers 2..4 expand to products)"
                % (s, OPS_SAFE4))
        if isinstance(e, sp.Function):
            raise NotExpressible(
                "%s is a function of a compound argument: not a vocabulary "
                "leaf, and compose ('o') is d = 1-only. Use an AtomSpec "
                "template (e.g. AtomSpec('E', 'c0*%s') with the compound "
                "argument written in the registered variables) if this "
                "form is required." % (s, s))
        raise NotExpressible("cannot translate %s (%s)" % (s, type(e)))

    return rec(expr)


def skel_n_nodes(skel):
    if skel[0] == 'O':
        return 1 + skel_n_nodes(skel[2]) + skel_n_nodes(skel[3])
    return 1


def skel_to_json(skel):
    """JSON-safe form (lists) of a skeleton; inverse of skel_from_json."""
    if skel[0] == 'O':
        return ['O', skel[1], skel_to_json(skel[2]), skel_to_json(skel[3])]
    if skel[0] == 'A':
        return ['A', skel[1], list(skel[2])]
    return ['L', skel[1]]


def skel_from_json(obj):
    if obj[0] == 'O':
        return ('O', obj[1], skel_from_json(obj[2]), skel_from_json(obj[3]))
    if obj[0] == 'A':
        return ('A', obj[1], list(obj[2]))
    return ('L', obj[1])


# ---------------------------------------------------------------------------
# pricing
# ---------------------------------------------------------------------------
def _dummy_domain(variables):
    d = len(variables)
    X = np.linspace(0.5, 1.5, 8)
    if d == 1:
        return np.ascontiguousarray(X)
    return np.ascontiguousarray(np.vstack([X + 0.1 * i for i in range(d)]))


def price_skel(skel, funcs, variables=VARIABLES_I3B, polya=None,
               check_closed_form=True, atom_specs=None):
    """log pi_0 of a skeleton under the frozen t = 0 measure, computed
    through the kernel. Installs `variables` in the registry (the caller
    restores it if needed). atom_specs: {symbol: AtomSpec} for skeletons
    with opaque leaves (the closed form then counts the atom cells)."""
    real_functionC.set_variables(tuple(variables))
    if polya is None:
        polya = make_t0_polya(funcs, atoms=(list(atom_specs.values())
                                            if atom_specs else None))
    root = node_from_skel(skel, _dummy_domain(variables), atom_specs)
    lp = float(polya.log_prior_from_tree(root))
    n = skel_n_nodes(skel)
    if check_closed_form:
        cf = -float(n) * math.log(
            n_total_cells(funcs, len(atom_specs) if atom_specs else 0))
        if abs(lp - cf) > 1e-9 * max(1.0, abs(cf)):
            raise AssertionError(
                "kernel prior %.12g != closed form %.12g for %d nodes "
                "(n_total %d): the t = 0 measure is no longer symmetric"
                % (lp, cf, n, n_total_cells(
                    funcs, len(atom_specs) if atom_specs else 0)))
    return lp, n


def price_expr(expr, funcs, variables=VARIABLES_I3B, polya=None):
    """(log_prior, n_nodes, skeleton) of an expression string under the
    canonical reading. Raises NotExpressible for forms outside the
    vocabulary."""
    skel = expr_to_skel(expr, funcs, variables)
    lp, n = price_skel(skel, funcs, variables, polya)
    return lp, n, skel


def pricing_table(funcs, variables=VARIABLES_I3B, max_nodes=21):
    """The declared-vocabulary pricing table: cells, per-node cost, and
    the price ladder by node count, plus a few reference forms."""
    nt = n_total_cells(funcs)
    per_node = math.log(nt)
    polya = make_t0_polya(funcs)
    ladder = {n: round(closed_form_log_prior(n, funcs), 6)
              for n in range(1, max_nodes + 1)}
    refs = {}
    examples = [
        "x0", "a0", "a0*x0", "a0*x1 + a1", "2*x1", "a0*x1*x2",
        "a0*tanh(x2) + a1*x1", "a0*exp(x1)/x0 + a1", "x1/(a0 + tanh(x2))",
        "a0*x0*x1 + a1*x0*x2 + a2",
    ]
    for ex in examples:
        lp, n, sk = price_expr(ex, funcs, variables, polya)
        refs[ex] = dict(n_nodes=n, log_prior=round(lp, 6))
    unexpressible = []
    for ex in ["exp(-x0*x1)", "sqrt(x1)", "x1**0.5", "tanh(a0*x2)"]:
        try:
            price_expr(ex, funcs, variables, polya)
        except NotExpressible as err:
            unexpressible.append({"expr": ex, "reason": str(err)[:90]})
    return dict(
        variables=list(variables), funcs=list(funcs),
        n_ops_cells=N_OPS_POLYA, n_function_leaves=len(funcs),
        param_cell=PARAM_CELL_TOKEN, n_total_cells=nt,
        per_node_log_cost=round(per_node, 6),
        bic_equivalent_note=(
            "one extra node costs %.3f nats; one extra fitted parameter "
            "costs 0.5*log(n) nats of BIC, i.e. the structure prior "
            "equals BIC's per-parameter charge at n = exp(2*%.3f) = %.0f "
            "rows and is weaker above" % (per_node, per_node,
                                          math.exp(2 * per_node))),
        ladder_by_nodes=ladder, reference_forms=refs,
        unexpressible_examples=unexpressible)


def main(argv):
    funcs = vocabulary_i3b()
    if len(argv) > 1:
        polya = make_t0_polya(funcs)
        for ex in argv[1:]:
            try:
                lp, n, sk = price_expr(ex, funcs, VARIABLES_I3B, polya)
                print("%-40s  nodes=%2d  log_prior=%9.4f  skel=%s"
                      % (ex, n, lp, skel_to_json(sk)))
            except NotExpressible as err:
                print("%-40s  NOT EXPRESSIBLE: %s" % (ex, err))
        return
    tab = pricing_table(funcs)
    print(json.dumps(tab, indent=1))


if __name__ == "__main__":
    main(sys.argv)
