"""Structural-constraint builders for the Phase 4.4 prior tilt.

EXPERIMENT LAYER — nothing here is imported by the kernel. The kernel
side is one opt-in kwarg (Grove/Forest ``structural_prior=``, default
None = bit-identical legacy) that accepts a callable

    tree  ->  float   (a LOG-prior increment; encode -lambda*C here)

and adds it at exactly the six acceptance sums where the Polya
log-prior already enters (groveC._step, the within-grove crossover
pair, the swap and trans-grove-crossover U's, and the two spawn-master
mirrors). It NEVER enters mse, the BIC/evidence, root_posteriors, the
BIC cache, burn-in ranking, or the Powell objective.

The formal contract (why this wiring and not a "custom loss")
--------------------------------------------------------------
A constraint C(tau) that depends on the CANDIDATE and not on the data
y is a PRIOR statement. Implemented as a multiplicative tilt on the
structural prior,

    pi(tau)  propto  pi_polya(tau) * exp(-lambda * C(tau)),

nothing breaks: Bayes' theorem is intact, the evidence stays an
evidence under the declared prior, every MH ratio gains the one term
Delta(-lambda*C), BIC (a likelihood-side Laplace approximation) is
untouched, and bounded (finite) penalties preserve irreducibility of
the move graph. Whole-posterior tempering applies to the tilt like to
any prior factor, and replica swaps stay valid because the tilt sits
inside each chain's U.

The rejected wiring: pushing the same term into the mse/likelihood
(all a ``(y_true, y_pred)`` loss API can express) multiplies it by
n/(2 sigma^2) and lets tempering flatten it — constraint strength then
silently scales with sample size and noise, and the evidence loses its
meaning (the Gibbs-posterior trap). Same idea, two wirings; only the
prior-side one is sound.

Practical caveats (documented limits of v1)
-------------------------------------------
* Penalties are SOFT by design. Returning -inf (hard constraint) is
  possible but discouraged: it can disconnect the move graph. All
  builders here return bounded values, -lam*min(C, c_max); the kernel
  additionally floors non-finite returns at -1e90.
* Structure-scale constraints only. The kernel caches the tilt by
  tree description (the same key as the BIC cache), so a constraint
  evaluated through current parameter values is pinned at its
  first-seen theta. That matches the recommended use (asymptotic
  EXPONENTS are generically theta-independent); for genuinely
  theta-sensitive constraints pass structural_prior_cache=False and
  accept the plug-in caveat.
* Two ways to hand the tilt to the sampler. A RAW callable
  (structural_prior=) works in the serial EMC loop only: closures do
  not pickle, so run_mpi_spawn refuses them rather than letting the
  workers silently sample untilted. A DECLARATIVE SPEC
  (structural_prior_spec=, built with spec() below) works everywhere,
  including run_mpi_spawn: the picklable ('module', 'builder',
  kwargs) triple ships in the worker startup config and every process
  rebuilds the callable locally from code it already has (the
  AtomSpec pattern; this module must be importable in the workers —
  same directory/sys.path as the master). Under spawn, prefer
  constraints whose value is determined by the tree description alone
  — symbolic-resolvable, or theta_policy="reference" — because each
  process caches independently and a current-theta fallback can
  differ across replicas (see theta_policy below). Test-only hooks
  (on_eval) are not picklable and must not appear in spec kwargs.
* Burn-in ranking stays BIC-only (it already excludes the Polya
  prior); the tilt shapes the CHAIN, and the chain's population is
  where the constraint shows up. best_tree()/root_posteriors remain
  BIC-only by invariant.

Builders
--------
asymptotic_power(p, domain, ...) — "the function must behave like x^p
as x -> +infinity" (e.g. p=2 for 'diverges like x^2'). Symbolic-first:
sympy limit of log|f|/log x with parameters as generic real symbols
(theta-independent when it resolves); numeric fallback: log-log slope
between the two outermost of three probe points placed geometrically
beyond the data domain, at the tree's current theta. Any evaluation
failure earns the maximum penalty (evaluation failure = penalized,
the same spirit as the kernel's zero-likelihood semantics).
"""

import math

import numpy as np
import sympy


__all__ = [
    "asymptotic_power",
    "estimate_power_from_description",
    "tree_name_values",
    "spec",
]


def spec(builder, **kwargs):
    """Declarative, picklable tilt spec for THIS module's builders.

    Returns ('exp_constraints', builder, kwargs) — the triple
    groveC._resolve_structural_prior_spec understands. Pass it as
    Grove/Forest ``structural_prior_spec=``; unlike a raw callable it
    survives the run_mpi_spawn boundary (each worker rebuilds the
    callable locally). Example:

        structural_prior_spec=spec("asymptotic_power", p=2.0,
                                   domain=(1.0, 8.0), lam=25.0,
                                   theta_policy="reference")

    kwargs must be plain picklable data; do not put callables (e.g.
    the on_eval test hook) in a spec.
    """
    if not isinstance(builder, str):
        raise TypeError("spec: builder must be the builder NAME (str)")
    return ("exp_constraints", builder, dict(kwargs))


def tree_name_values(tree):
    """Current parameter dict of a kernel Tree (or duck-typed stand-in).

    Kernel trees expose it via get_param_dicts()[1] (the name -> value
    dict); test stand-ins may carry a plain .name_values attribute.
    Never raises; empty dict when nothing is available.
    """
    try:
        return dict(tree.get_param_dicts()[1])
    except Exception:
        pass
    try:
        nv = getattr(tree, "name_values", None)
        return dict(nv) if nv else {}
    except Exception:
        return {}


def _symbolic_power(desc, at="+inf"):
    """Leading power of the description as x -> +oo, parameters generic.

    Returns a float, +/-inf (super/sub-polynomial), or None when sympy
    cannot resolve it independently of the parameters (or errors).
    """
    try:
        x = sympy.Symbol("x", positive=True)
        expr = sympy.sympify(desc, locals={"x": x})
        # Generic real parameters: the leading POWER is
        # parameter-independent except on measure-zero cancellations.
        subs = {}
        for s in expr.free_symbols:
            if s.name != "x":
                subs[s] = sympy.Symbol(s.name, real=True, nonzero=True)
        expr = expr.subs(subs)
        if x not in expr.free_symbols:
            return 0.0  # constants: power 0
        lim = sympy.limit(sympy.log(sympy.Abs(expr)) / sympy.log(x),
                          x, sympy.oo)
        if lim is sympy.oo:
            return math.inf
        if lim is -sympy.oo:
            return -math.inf
        if lim.free_symbols:
            return None  # parameter-dependent: not structure-scale
        val = float(lim)
        return val
    except Exception:
        return None


def _numeric_power(desc, name_values, domain, probe_factors):
    """Log-log slope at current theta, two outermost probe points.

    Returns a float or None (non-finite / degenerate evaluations).
    """
    try:
        x = sympy.Symbol("x")
        expr = sympy.sympify(desc, locals={"x": x})
        nv = dict(name_values or {})
        expr = expr.subs({sympy.Symbol(k): float(v)
                          for k, v in nv.items()})
        free = [s for s in expr.free_symbols if s.name != "x"]
        if free:
            # Unvalued parameters left over: reference value 1.0 (the
            # same convention is_valid uses for its reference check).
            expr = expr.subs({s: 1.0 for s in free})
        f = sympy.lambdify(x, expr, modules=["numpy"])
        hi = float(max(abs(domain[0]), abs(domain[1]), 1.0))
        xs = np.array([hi * float(pf) for pf in probe_factors],
                      dtype=float)
        with np.errstate(all="ignore"):
            ys = np.asarray(f(xs), dtype=float)
        if ys.shape != xs.shape:
            ys = np.full_like(xs, float(ys))  # constant expressions
        ay = np.abs(ys)
        if not np.all(np.isfinite(ay)) or np.any(ay < 1e-290):
            return None
        # Outermost pair: least pre-asymptotic contamination.
        return float((math.log(ay[-1]) - math.log(ay[-2]))
                     / (math.log(xs[-1]) - math.log(xs[-2])))
    except Exception:
        return None


def estimate_power_from_description(desc, name_values, domain,
                                    probe_factors=(4.0, 16.0, 64.0),
                                    prefer_symbolic=True):
    """Estimated asymptotic power of `desc` as x -> +oo, or None.

    Symbolic-first (theta-independent when it resolves), numeric
    log-log fallback at the supplied parameter values. None means
    'could not be established' — builders translate that into the
    maximum penalty.
    """
    p_hat = None
    if prefer_symbolic and len(desc) <= 240:
        p_hat = _symbolic_power(desc)
    if p_hat is None:
        p_hat = _numeric_power(desc, name_values, domain, probe_factors)
    return p_hat


def asymptotic_power(p, domain, lam=25.0, c_max=10.0,
                     probe_factors=(4.0, 16.0, 64.0),
                     prefer_symbolic=True, theta_policy="current",
                     on_eval=None):
    """Builder: log-prior tilt enforcing 'behaves like x**p at +oo'.

    Parameters
    ----------
    p : float
        Target asymptotic power (p=2 encodes 'diverges like x^2').
    domain : (lo, hi)
        The DATA domain; numeric probes sit at hi*probe_factors,
        i.e. deliberately beyond the fitted range (that is where an
        asymptotic statement lives).
    lam : float
        Constraint strength in nats per unit exponent gap. The tilt is
        -lam * min(|p_hat - p|, c_max); with lam=25 a one-power
        mismatch costs 25 nats — decisive against typical BIC gaps —
        while staying finite (irreducibility preserved).
    c_max : float
        Cap on the exponent gap C (bounded-penalty contract). Also the
        charge when the power cannot be established at all.
    prefer_symbolic : bool
        Try the sympy limit first (theta-independent); fall back to
        the numeric probe.
    theta_policy : "current" | "reference"
        Parameter values the NUMERIC fallback probes at. "current"
        (default, v1 behaviour) reads the tree's fitted theta —
        informative, but cache-pinned at first sight and therefore
        replica-dependent under run_mpi_spawn. "reference" evaluates
        every parameter at 1.0 (the is_valid convention): the tilt
        becomes a pure function of the tree description, identical in
        every process — the recommended policy for spawn runs.
    on_eval : callable or None
        Optional hook, called as on_eval(desc, tilt) on every FRESH
        evaluation (test instrumentation: the kernel-side cache should
        make evaluations one-per-description).

    Returns
    -------
    callable tree -> float, suitable for Grove/Forest
    ``structural_prior=``. Never raises; never returns non-finite.
    """
    p = float(p)
    lo, hi = float(domain[0]), float(domain[1])
    if theta_policy not in ("current", "reference"):
        raise ValueError("asymptotic_power: theta_policy must be "
                         "'current' or 'reference'")

    def _tilt(tree):
        try:
            desc = tree.tree_get_description()
            nv = ({} if theta_policy == "reference"
                  else tree_name_values(tree))
            p_hat = estimate_power_from_description(
                desc, nv, (lo, hi),
                probe_factors=probe_factors,
                prefer_symbolic=prefer_symbolic)
            if p_hat is None or not math.isfinite(p_hat):
                # includes +/-inf estimates: super/sub-polynomial
                # candidates violate any finite-power statement
                # maximally.
                C = c_max
            else:
                C = min(abs(p_hat - p), c_max)
            out = -lam * C
        except Exception:
            out = -lam * c_max
        if on_eval is not None:
            try:
                on_eval(tree.tree_get_description(), out)
            except Exception:
                pass
        return out

    return _tilt
