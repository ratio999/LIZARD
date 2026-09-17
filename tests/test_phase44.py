"""Phase 4.4 feature test: the structural-constraint prior tilt.

Opt-in feature (Grove/Forest ``structural_prior=``, default None).
This suite checks, in order:

  T1  null equivalence — a tilt callable returning 0.0 leaves the
      sampled trajectory BIT-IDENTICAL to structural_prior=None
      (proves the six acceptance-site insertions inject exactly zero
      and consume no RNG when inactive);
  T2  estimator unit checks — exp_constraints power estimates on
      known descriptions (symbolic and numeric paths, inf handling);
  T3  steering — on y = x^2 (noiseless, sigma = 1), an
      asymptotic_power(3) tilt with lambda = 200 moves the chain
      population into x^3-behaved structures, and an agreeing
      asymptotic_power(2) tilt concentrates it at power 2. The
      no-tilt arm is the NEGATIVE CONTROL, and its observed behaviour
      is itself the feature's motivation: the data on [1, 2.5] do not
      pin behaviour beyond the domain (near-zero dressed limbs shift
      the asymptotic exponent at ~2 nats of BIC apiece), so its
      population powers SCATTER — the assertion is only that it is
      not already concentrated at the constraint target. The readout
      is the CHAIN population (root_posteriors/best_tree stay
      BIC-only by invariant — the tilt shapes what the chain retains,
      not the reported evidence);
  T4  cache discipline — the kernel evaluates the callable exactly
      once per unique description and stores exactly what it
      returned (_sp_cache == the on_eval log);
  T5  guards — non-callable rejected at construction; Forest forwards
      the callable to every Grove; run_mpi_spawn refuses a RAW
      callable (NotImplementedError — closures cannot cross the spawn
      boundary; unchanged in v2);
  T6  declarative specs (v2) — Grove resolves structural_prior_spec
      into a callable equivalent to the directly built one; spec +
      callable together rejected; malformed specs rejected;
      theta_policy="reference" makes the numeric fallback a pure
      function of the description (params at 1.0, the is_valid
      convention);
  T7  spawn end-to-end (v2 shipping + 2026-07-24 wire fix) —
      run_mpi_spawn ACCEPTS a declarative spec (ships to the workers,
      resolved locally, stats dict returned; T7), the master
      population REFRESHES (T7b — the assertion whose absence let the
      wire-starvation defect stay latent), and the steering is
      visible ON THE MASTER (T7c — the spawn mirror of serial T3b).
      Enabled by the sampling-wire fidelity fix: the wire now ships
      ('skel3b', skeleton, description) for ALL trees, so the folded
      x**k structures the p=3 tilt promotes survive the round-trip.
      (Historically the raw-sympy payload canonicalized x*x*x ->
      x**3, unrepresentable in the '**'-free vocabulary, and the
      fidelity gate discarded precisely the steered trees —
      PROMPT_DEV_WIRE_FIDELITY.md, RESOLVED; regression coverage in
      test_phase212_wire.py.)

Run (T7 spawns 2 workers):
  $ENV/bin/mpiexec -n 1 $ENV/bin/python test_phase44.py
"""

import sys

import numpy as np

import groveC
import exp_constraints as XC


RESULTS = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    line = "[%s] %s" % (tag, name)
    if detail:
        line += " — %s" % detail
    print(line, flush=True)
    RESULTS.append(bool(ok))


# ---------------------------------------------------------------- data
X_LO, X_HI = 1.0, 2.5
x = np.linspace(X_LO, X_HI, 60)
y = x ** 2                       # noiseless truth, diverges like x^2

FUNCS = ["sin(x)", "x"]          # house format: applied strings
OPS = ["+", "-", "*"]            # polynomial-only search: powers arise
                                 # structurally (x*x, x*(x*x)), so the
                                 # asymptotic exponent is a STRUCTURE
                                 # property — the v1 contract's home turf


def make_grove(sp=None, seed=44, sp_cache=True):
    return groveC.Grove(x, y, 1.0, 6, 3, list(FUNCS), sigma=1.0,
                        Npars=2, operations=list(OPS),
                        use_gradients=False, seed=seed,
                        structural_prior=sp,
                        structural_prior_cache=sp_cache)


def population_powers(grove):
    out = []
    for t in grove.trees:
        p = XC.estimate_power_from_description(
            t.tree_get_description(), XC.tree_name_values(t),
            (X_LO, X_HI))
        out.append(p)
    return out


def count_near(powers, target, tol=0.35):
    n = 0
    for p in powers:
        if p is not None and np.isfinite(p) and abs(p - target) <= tol:
            n += 1
    return n


class _FakeTree:
    """Duck-typed stand-in: the builder only reads these two members."""

    def __init__(self, desc, nv=None):
        self._d = desc
        self.name_values = dict(nv or {})

    def tree_get_description(self):
        return self._d


def main():
    # ---------------- T1: null equivalence --------------------------
    g_none = make_grove(sp=None, seed=44)
    g_zero = make_grove(sp=(lambda t: 0.0), seed=44)
    g_none.run(y, 150)
    g_zero.run(y, 150)
    d_none = [t.tree_get_description() for t in g_none.trees]
    d_zero = [t.tree_get_description() for t in g_zero.trees]
    p_none = [float(v) for v in g_none.root_posteriors]
    p_zero = [float(v) for v in g_zero.root_posteriors]
    check("T1 zero-tilt trajectory bit-identical to None",
          d_none == d_zero and p_none == p_zero,
          "descriptions %s, posts %s" % (
              "match" if d_none == d_zero else "DIFFER",
              "match" if p_none == p_zero else "DIFFER"))

    # ---------------- T2: estimator units ---------------------------
    est = XC.estimate_power_from_description
    e1 = est("a0*x**2", {"a0": 3.0}, (X_LO, X_HI))
    check("T2a power(a0*x**2) == 2",
          e1 is not None and abs(e1 - 2.0) < 1e-9, "got %r" % (e1,))
    e2 = est("(x*(x*x))", {}, (X_LO, X_HI))
    check("T2b power(x*(x*x)) == 3",
          e2 is not None and abs(e2 - 3.0) < 1e-9, "got %r" % (e2,))
    e3 = est("a0/x", {"a0": 2.0}, (X_LO, X_HI))
    check("T2c power(a0/x) == -1",
          e3 is not None and abs(e3 + 1.0) < 1e-9, "got %r" % (e3,))
    lam, cmax = 25.0, 10.0
    tilt = XC.asymptotic_power(2.0, (X_LO, X_HI), lam=lam, c_max=cmax)
    v_exp = tilt(_FakeTree("exp(x)"))
    check("T2d exp(x) (super-polynomial) earns the max penalty",
          abs(v_exp - (-lam * cmax)) < 1e-9, "got %r" % (v_exp,))
    v_ok = tilt(_FakeTree("a0*x**2", {"a0": 1.5}))
    check("T2e agreeing structure earns zero penalty",
          abs(v_ok) < 1e-9, "got %r" % (v_ok,))

    # ---------------- T3 + T4: steering + cache ---------------------
    N_STEPS = 500
    LAM = 200.0

    g0 = make_grove(sp=None, seed=91)
    g0.run(y, N_STEPS)
    pw0 = population_powers(g0)
    n3_ctrl = count_near(pw0, 3.0)
    check("T3a control (no tilt): asymptotics unpinned by data alone "
          "— population not already x^3-concentrated (n3 <= 1)",
          n3_ctrl <= 1,
          "powers %s" % ([None if p is None else round(p, 2)
                          for p in pw0],))

    log31 = []
    sp3 = XC.asymptotic_power(3.0, (X_LO, X_HI), lam=LAM, c_max=4.0,
                              on_eval=lambda d, v: log31.append((d, v)))
    g3 = make_grove(sp=sp3, seed=91)
    g3.run(y, N_STEPS)
    pw3 = population_powers(g3)
    m2, m3 = count_near(pw3, 2.0), count_near(pw3, 3.0)
    check("T3b wrong-power tilt (p=3, lam=200) steers the population "
          "to x^3 structures (n3 >= 4)",
          m3 >= 4 and m3 > m2, "powers %s" % (pw3,))

    sp2 = XC.asymptotic_power(2.0, (X_LO, X_HI), lam=LAM, c_max=4.0)
    g2 = make_grove(sp=sp2, seed=91)
    g2.run(y, N_STEPS)
    pw2 = population_powers(g2)
    k2 = count_near(pw2, 2.0)
    check("T3c agreeing tilt (p=2) is non-destructive (n2 >= 4)",
          k2 >= 4, "powers %s" % (pw2,))

    cache = dict(g3._sp_cache)
    n_eval, n_uniq = len(log31), len(set(d for d, _ in log31))
    ok_once = (n_eval == n_uniq == len(cache))
    ok_vals = all(d in cache and cache[d] == v for d, v in log31)
    check("T4a callable evaluated exactly once per unique description",
          ok_once, "%d evals, %d unique, %d cached"
          % (n_eval, n_uniq, len(cache)))
    check("T4b _sp_cache stores exactly the callable's returns",
          ok_vals)

    # ---------------- T5: guards ------------------------------------
    try:
        make_grove(sp=42)
        check("T5a non-callable structural_prior raises TypeError",
              False, "no exception raised")
    except TypeError:
        check("T5a non-callable structural_prior raises TypeError",
              True)

    cb = XC.asymptotic_power(2.0, (X_LO, X_HI), lam=10.0)
    F = groveC.Forest(x, y, np.array([1.0, 1.6]), 4, 3, list(FUNCS),
                      sigma=1.0, Npars=2, operations=list(OPS),
                      use_gradients=False, seed=7,
                      structural_prior=cb)
    fwd = all(g.structural_prior is cb for g in F.groves)
    check("T5b Forest forwards the callable to every Grove", fwd)
    try:
        F.run_mpi_spawn(y, 1, 1)
        check("T5c run_mpi_spawn + tilt raises NotImplementedError",
              False, "no exception raised")
    except NotImplementedError:
        check("T5c run_mpi_spawn + tilt raises NotImplementedError",
              True)
    except Exception as e:  # wrong type = still a failure, but say so
        check("T5c run_mpi_spawn + tilt raises NotImplementedError",
              False, "raised %r instead" % (e,))

    # ---------------- T6: declarative specs (v2) --------------------
    SPEC_KW = dict(p=3.0, domain=(X_LO, X_HI), lam=LAM, c_max=4.0)
    sp_spec = XC.spec("asymptotic_power", **SPEC_KW)
    g_spec = groveC.Grove(x, y, 1.0, 6, 3, list(FUNCS), sigma=1.0,
                          Npars=2, operations=list(OPS),
                          use_gradients=False, seed=44,
                          structural_prior_spec=sp_spec)
    ref_cb = XC.asymptotic_power(**SPEC_KW)
    descs = ["(x*(x*x))", "a0*x**2", "exp(x)", "sin(x)"]
    same = (callable(g_spec.structural_prior)
            and g_spec.structural_prior_spec == sp_spec
            and all(g_spec.structural_prior(_FakeTree(d))
                    == ref_cb(_FakeTree(d)) for d in descs))
    check("T6a Grove resolves a spec into the equivalent callable",
          same)
    try:
        groveC.Grove(x, y, 1.0, 6, 3, list(FUNCS), sigma=1.0,
                     Npars=2, operations=list(OPS),
                     use_gradients=False, seed=44,
                     structural_prior=(lambda t: 0.0),
                     structural_prior_spec=sp_spec)
        check("T6b spec + raw callable together raise TypeError",
              False, "no exception raised")
    except TypeError:
        check("T6b spec + raw callable together raise TypeError", True)
    bad = 0
    for bad_spec in [("exp_constraints", "asymptotic_power"),
                     ("exp_constraints", "no_such_builder", {}),
                     ("no_such_module_xyz", "asymptotic_power", {}),
                     42]:
        try:
            groveC.Grove(x, y, 1.0, 6, 3, list(FUNCS), sigma=1.0,
                         Npars=2, operations=list(OPS),
                         use_gradients=False, seed=44,
                         structural_prior_spec=bad_spec)
        except (TypeError, AttributeError, ImportError):
            bad += 1
    check("T6c malformed specs rejected at construction (4/4)",
          bad == 4, "%d/4 raised" % bad)
    ft = _FakeTree("x**a0", {"a0": 2.5})
    t_cur = XC.asymptotic_power(2.5, (X_LO, X_HI), lam=10.0,
                                theta_policy="current")(ft)
    t_ref = XC.asymptotic_power(1.0, (X_LO, X_HI), lam=10.0,
                                theta_policy="reference")(ft)
    t_ref_off = XC.asymptotic_power(2.5, (X_LO, X_HI), lam=10.0,
                                    theta_policy="reference")(ft)
    check("T6d theta_policy: current sees theta-hat (2.5), reference "
          "sees params at 1.0",
          abs(t_cur) < 1e-9 and abs(t_ref) < 1e-9
          and abs(t_ref_off - (-15.0)) < 1e-6,
          "cur %r ref %r ref_off %r" % (t_cur, t_ref, t_ref_off))

    # ---------------- T7: spawn end-to-end (v2 + wire fix) -----------
    # T7  : shipping contract (spec accepted under spawn, stats dict).
    # T7b : master population REFRESHES — the check whose absence let
    #       the wire-starvation defect stay latent (pre-fix footprint
    #       on this fixture family: 4/6 and 6/6 slots still initial).
    # T7c : steering visible ON THE MASTER — spawn mirror of serial
    #       T3b, enabled by the 2026-07-24 sampling-wire fidelity fix
    #       (skel3b payloads for ALL trees). Fixture = the
    #       folding-prone T3 vocabulary ON PURPOSE: it exercises
    #       exactly the x*(x*x) -> x**3 path the old wire discarded.
    F7 = groveC.Forest(x, y, np.array([1.0, 1.6]), 6, 2, list(FUNCS),
                       sigma=1.0, Npars=1, operations=list(OPS),
                       use_gradients=False, seed=7,
                       structural_prior_spec=XC.spec(
                           "asymptotic_power", p=3.0,
                           domain=(X_LO, X_HI), lam=LAM, c_max=4.0,
                           theta_policy="reference"))
    init7 = [t.tree_get_description() for t in F7.groves[0].trees]
    try:
        stats7 = F7.run_mpi_spawn(y, 30, 15, seed=5,
                                  max_tree_depth=3,
                                  subtree_rebuild_max_height=2)
        check("T7 spawn ACCEPTS a declarative spec and completes "
              "(returns stats dict)",
              isinstance(stats7, dict),
              "stats keys: %s" % (sorted(stats7.keys())
                                  if isinstance(stats7, dict)
                                  else stats7))
        final7 = [t.tree_get_description() for t in F7.groves[0].trees]
        n_stale = sum(1 for a, b in zip(init7, final7) if a == b)
        check("T7b master population REFRESHES (<= 2/6 slots still "
              "initial after the wire fix)", n_stale <= 2,
              "%d/6 stale; final %s" % (n_stale, final7))
        pw7 = population_powers(F7.groves[0])
        m2 = count_near(pw7, 2.0)
        m3 = count_near(pw7, 3.0)
        check("T7c spawn steering ON THE MASTER: p=3 spec tilt "
              "concentrates the cold population at x^3 "
              "(m3 >= 4 and m3 > m2)", m3 >= 4 and m3 > m2,
              "powers %s; m3=%d m2=%d"
              % ([None if p is None else round(p, 2) for p in pw7],
                 m3, m2))
    except Exception as e:
        check("T7 spawn end-to-end (spec + wire fix)",
              False, "raised %r" % (e,))

    n_pass = sum(RESULTS)
    print("\n%d/%d checks passed." % (n_pass, len(RESULTS)), flush=True)
    return 0 if n_pass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
