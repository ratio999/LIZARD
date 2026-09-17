"""feynman_exactness.py -- AI-Feynman-comparable exact-recovery grading
for the Feynman suites (optional post-hoc tool; the run pipeline and its
numeric `recovered` flag are untouched).

The criterion (agreed 2026-09-13): a winning tree is EXACT when equality
with the generating law holds MODULO THE PARAMETERS -- i.e. there exists
an assignment of the tree's parameter symbols making it identical to the
truth. Imperfectly estimated coefficients therefore never disqualify a
structurally correct tree. This is the thesis's "exact-structure
recovery" made machine-decidable, and it is the flavour of exactness the
AI Feynman papers report (an output counts as solved when its symbolic
form matches the target law).

How it is decided, per run record:
  1. Parse `best_tree` and `truth` with sympy; every free symbol not in
     the record's variable list is a parameter.
  2. REFIT the parameters against the NOISE-FREE truth on a dense mixed
     grid (linspace + uniform draws, fixed seed) by least squares,
     multi-started from the run's own fitted values plus jittered and
     random restarts. If the structure is right, this drives the maximum
     relative residual to numerical zero; a wrong-form approximant
     plateaus orders of magnitude higher.
  3. SNAP the refit constants to nearby exact values (rationals and
     pi-based constants via nsimplify, then rounded floats), substitute,
     and ask sympy whether the result equals the truth symbolically
     (`Expr.equals`, which falls back to high-precision numeric
     sampling).
  4. Grade:
       exact-symbolic  refit converged AND the snapped expression equals
                       the truth symbolically;
       exact-numeric   refit residual < 1e-8 relative, but no snapped
                       form passed the symbolic check (structure right,
                       constants not expressible in the snap basis);
       functional      not exact, but the run's held-out NRMSE passed
                       the suite's numeric criterion;
       miss            neither;
       ungradable      the description could not be parsed/evaluated.
     Records also carry `degenerate_params`: parameters the refit sent
     to 0, i.e. the tree matches the law only because a term vanishes.
     They still count as exact under the agreed criterion (equality
     modulo parameters), but the table shows them so the thesis can
     distinguish "the true parse" from "a superset that collapses".

Grades are written next to the input as <stem>_exactness.json and
printed as a table. Nothing in the input file is modified.

Usage (WSL, mcmc_koopman env, from gpu_ms):
    python feynman_exactness.py results/phase42_feynman.json
    python feynman_exactness.py <results.json> --tol-exact 1e-6

Multivariate records are supported: the grader reads `variables` (list)
and a per-variable `domain` ([[lo, hi], ...]) when present; absent, it
assumes the univariate convention (('x',), [lo, hi]).
"""
import argparse
import json
import os
import sys

import numpy as np
import sympy as sp
from scipy.optimize import least_squares

REL_TOL_EXACT_NUMERIC = 1e-8   # residual floor claiming exactness alone
REL_TOL_REFIT = 1e-6           # residual gate for trying the symbolic step
N_GRID = 400
N_RESTARTS = 6

# snap basis: plain rationals, then pi-flavoured constants
_SNAP_CONSTANTS = (sp.pi, sp.sqrt(sp.pi), sp.sqrt(2 * sp.pi), sp.E)


def _grid(domain, variables, seed=0, n=N_GRID):
    """Mixed evaluation grid: half linspace (or per-variable uniform for
    d > 1 -- independent draws, never collinear), half uniform draws."""
    rng = np.random.default_rng(seed)
    d = len(variables)
    if d == 1:
        lo, hi = float(domain[0]), float(domain[1])
        a = np.linspace(lo, hi, n // 2)
        b = rng.uniform(lo, hi, n - n // 2)
        return [np.concatenate([a, b])]
    doms = domain if hasattr(domain[0], "__len__") else [domain] * d
    return [rng.uniform(float(lo), float(hi), n) for lo, hi in doms]


def _lambdify(expr, var_syms, par_syms):
    f = sp.lambdify(list(var_syms) + list(par_syms), expr, "numpy")

    def call(cols, theta):
        with np.errstate(all="ignore"):
            try:
                out = f(*cols, *theta)
            except Exception:
                return None
            out = np.asarray(out, dtype=np.complex128)
            if np.iscomplexobj(out) and np.abs(out.imag).max(initial=0) > 1e-12:
                return None
            out = out.real.astype(np.float64)
            if out.shape == ():
                out = np.full(cols[0].shape, float(out))
            return out
    return call


def _snap_value(v, tol):
    """Candidate exact forms for a fitted float, nearest first."""
    cands = []
    for c in ({"rational": True},
              {"constants": _SNAP_CONSTANTS, "rational": False}):
        try:
            s = sp.nsimplify(sp.Float(v, 15), tolerance=tol, **c)
            if s.is_finite is not False and abs(float(s) - v) <= max(
                    tol, tol * abs(v)):
                cands.append(s)
        except Exception:
            pass
    cands.append(sp.Float(round(float(v), 6), 8))
    seen, out = set(), []
    for s in cands:
        k = sp.srepr(s)
        if k not in seen:
            seen.add(k)
            out.append(s)
    return out


def grade_record(rec, tol_exact=REL_TOL_REFIT, seed=0):
    variables = tuple(rec.get("variables") or ("x",))
    out = {"target": rec.get("target"), "seed": rec.get("seed"),
           "grade": None, "refit_rel_err": None, "snapped_expr": None,
           "degenerate_params": [], "note": ""}
    try:
        truth = sp.sympify(rec["truth"])
        desc = sp.sympify(rec["best_tree"])
    except Exception as e:
        out["grade"] = "ungradable"
        out["note"] = "parse failure: %s" % e
        return out
    var_syms = [sp.Symbol(v) for v in variables]
    par_syms = sorted((s for s in desc.free_symbols
                       if str(s) not in variables), key=str)
    stray = [s for s in truth.free_symbols if str(s) not in variables]
    if stray:
        out["grade"] = "ungradable"
        out["note"] = "truth has non-variable symbols %s" % stray
        return out

    cols = _grid(rec["domain"], variables, seed=seed)
    f_truth = _lambdify(truth, var_syms, [])
    t_vals = f_truth(cols, [])
    if t_vals is None or not np.all(np.isfinite(t_vals)):
        out["grade"] = "ungradable"
        out["note"] = "truth not finite on the grid"
        return out
    scale = max(float(np.max(np.abs(t_vals))), 1e-12)
    f_desc = _lambdify(desc, var_syms, par_syms)

    # ---- refit against the noise-free truth --------------------------------
    if par_syms:
        fitted = rec.get("best_params") or {}
        x0 = np.asarray([float(fitted.get(str(s), 1.0)) for s in par_syms])
        rng = np.random.default_rng(seed + 1)
        starts = [x0]
        for i in range(N_RESTARTS - 1):
            starts.append(x0 * rng.uniform(0.5, 1.5, x0.size)
                          + rng.normal(0.0, 0.1 + 0.1 * i, x0.size))

        def resid(theta):
            y = f_desc(cols, theta)
            if y is None:
                return np.full(t_vals.shape, 1e6)
            return np.nan_to_num((y - t_vals) / scale,
                                 nan=1e6, posinf=1e6, neginf=1e6)

        best_theta, best_err = None, np.inf
        for s0 in starts:
            try:
                r = least_squares(resid, s0, method="trf", xtol=1e-15,
                                  ftol=1e-15, gtol=1e-15, max_nfev=4000)
                err = float(np.max(np.abs(resid(r.x))))
                if err < best_err:
                    best_theta, best_err = np.asarray(r.x), err
            except Exception:
                continue
        theta = best_theta if best_theta is not None else x0
        rel_err = best_err
    else:
        theta = np.zeros(0)
        y = f_desc(cols, theta)
        rel_err = (float(np.max(np.abs((y - t_vals) / scale)))
                   if y is not None else np.inf)
    out["refit_rel_err"] = None if not np.isfinite(rel_err) else rel_err
    out["refit_params"] = {str(s): float(v)
                           for s, v in zip(par_syms, theta)}

    # ---- symbolic confirmation on snapped constants -------------------------
    exact_symbolic = False
    if rel_err < tol_exact:
        snap_tol = max(10.0 * rel_err, 1e-4)
        per_par = [_snap_value(float(v), snap_tol) for v in theta]
        # try the joint best-candidate substitution first, then one
        # variation per parameter (greedy; the space is tiny: k <= ~6)
        trials = [tuple(c[0] for c in per_par)]
        for i, cands in enumerate(per_par):
            for alt in cands[1:]:
                t = list(trials[0])
                t[i] = alt
                trials.append(tuple(t))
        for t in trials:
            try:
                sub = desc.subs(dict(zip(par_syms, t)))
                eq = sp.simplify(sub - truth) == 0 or bool(sub.equals(truth))
            except Exception:
                eq = False
            if eq:
                exact_symbolic = True
                out["snapped_expr"] = str(sub)
                out["degenerate_params"] = [
                    str(s) for s, v in zip(par_syms, t) if v == 0]
                break
        if not exact_symbolic and par_syms:
            '''Aliased parameters (e.g. a0*a1 acting as one constant)
            defeat per-parameter snapping: the refit lands anywhere on
            the a0*a1 = const manifold. Substitute the raw refit floats
            and snap the WHOLE expression instead -- nsimplify collapses
            the combined constants wherever they ended up in the tree.'''
            sub_raw = desc.subs({s: sp.Float(v, 15)
                                 for s, v in zip(par_syms, theta)})
            for kw in ({"rational": True},
                       {"constants": _SNAP_CONSTANTS, "rational": False}):
                try:
                    whole = sp.nsimplify(sub_raw, tolerance=snap_tol, **kw)
                    eq = (sp.simplify(whole - truth) == 0
                          or bool(whole.equals(truth)))
                except Exception:
                    eq = False
                if eq:
                    exact_symbolic = True
                    out["snapped_expr"] = str(whole)
                    out["degenerate_params"] = [
                        str(s) for s, v in zip(par_syms, theta)
                        if abs(v) < 1e-9]
                    break

    if exact_symbolic:
        out["grade"] = "exact-symbolic"
    elif rel_err < REL_TOL_EXACT_NUMERIC:
        out["grade"] = "exact-numeric"
        out["degenerate_params"] = [
            str(s) for s, v in zip(par_syms, theta) if abs(v) < 1e-9]
    elif rec.get("recovered"):
        out["grade"] = "functional"
    else:
        out["grade"] = "miss"
    return out


def grade_file(path, tol_exact=REL_TOL_REFIT):
    with open(path) as fp:
        data = json.load(fp)
    runs = data["runs"] if isinstance(data, dict) and "runs" in data else data
    graded = [grade_record(r, tol_exact=tol_exact) for r in runs]
    stem = os.path.splitext(path)[0]
    out_path = stem + "_exactness.json"
    with open(out_path, "w") as fp:
        json.dump({"criterion": "exact iff equality holds modulo the "
                                "parameters (refit + snap + sympy equals)",
                   "tol_exact": tol_exact,
                   "grades": graded}, fp, indent=1)
    return graded, out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_json")
    ap.add_argument("--tol-exact", type=float, default=REL_TOL_REFIT)
    args = ap.parse_args()
    graded, out_path = grade_file(args.results_json, args.tol_exact)
    n_ex = sum(g["grade"].startswith("exact") for g in graded)
    print("%-14s %-5s %-15s %-10s %s" % ("target", "seed", "grade",
                                         "refit err", "notes"))
    for g in graded:
        extra = ("degenerate:%s" % ",".join(g["degenerate_params"])
                 if g["degenerate_params"] else g["note"])
        err = ("%.2e" % g["refit_rel_err"]
               if g["refit_rel_err"] is not None else "-")
        print("%-14s %-5s %-15s %-10s %s" % (g["target"], g["seed"],
                                             g["grade"], err, extra))
    print("exact (either flavour): %d / %d   -> %s"
          % (n_ex, len(graded), out_path))


if __name__ == "__main__":
    main()
