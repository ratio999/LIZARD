"""test_feynman_exactness.py -- the exact-recovery grader must itself be
trustworthy before any thesis table uses it. Plain script, no pytest:

    python test_feynman_exactness.py

Covers: structurally exact trees with badly estimated coefficients
(the agreed criterion: equality modulo the parameters), algebraic
rearrangements, wrong-form approximants that pass the numeric NRMSE
gate but must NOT grade exact, vanishing-term degeneracies (exact, but
flagged), pi-flavoured constants, parameter-free descriptions,
multivariate records, and unparsable descriptions.
"""
import sys
import time

import numpy as np

from feynman_exactness import grade_record


def rec(best, truth, params=None, domain=(0.5, 3.0), variables=None,
        recovered=True, target="t", seed=0):
    r = {"target": target, "seed": seed, "best_tree": best, "truth": truth,
         "best_params": params or {}, "domain": list(domain),
         "recovered": recovered}
    if variables:
        r["variables"] = list(variables)
    return r


def check(name, r, want_grade, want_degenerate=None):
    g = grade_record(r)
    ok = g["grade"] == want_grade
    if ok and want_degenerate is not None:
        ok = (sorted(g["degenerate_params"]) == sorted(want_degenerate))
    status = "PASS" if ok else "FAIL"
    print("[%s] %-34s grade=%-15s err=%s %s"
          % (status, name, g["grade"],
             ("%.1e" % g["refit_rel_err"]) if g["refit_rel_err"] is not None
             else "-", g.get("snapped_expr") or g.get("note") or ""))
    if not ok:
        print("       wanted grade=%s degenerate=%s got degenerate=%s"
              % (want_grade, want_degenerate, g["degenerate_params"]))
    return ok


def main():
    t0 = time.time()
    oks = []

    # 1. The flux record as the real pipeline produced it: right parse,
    #    coefficient off at the third digit -> exact regardless.
    oks.append(check("flux, imperfect coefficient",
                     rec("a0/x**2", "2/x**2", {"a0": 1.9871}),
                     "exact-symbolic"))
    # 2. Coefficient WILDLY off: the criterion is modulo parameters.
    oks.append(check("flux, wild start",
                     rec("a0/x**2", "2/x**2", {"a0": 57.3}),
                     "exact-symbolic"))
    # 3. Algebraic rearrangement of velocity addition.
    oks.append(check("velocity addition, rearranged",
                     rec("(x + a0)/(a1*x + 1)", "(0.5 + x)/(1 + 0.5*x)",
                         {"a0": 0.62, "a1": 0.44}, domain=(0.0, 0.95)),
                     "exact-symbolic"))
    # 4. Scattering with both constants free.
    oks.append(check("scattering",
                     rec("a0 + a1*cos(x)", "1.2*(1 + 0.7*cos(x))",
                         {"a0": 1.0, "a1": 1.0}, domain=(0.0, 6.28)),
                     "exact-symbolic"))
    # 5. Wrong form that fits well numerically (rational approximant of
    #    the Gaussian): must NOT be exact.
    oks.append(check("gaussian vs rational approximant",
                     rec("a0/(a1 + x**2)**2", "exp(-x**2/2)/sqrt(2*pi)",
                         {"a0": 1.6, "a1": 2.0}, domain=(0.1, 3.0)),
                     "functional"))
    # 6. Same wrong form, run marked not recovered -> miss.
    oks.append(check("wrong form, not recovered",
                     rec("a0*cos(x)", "1.2*(1 + 0.7*cos(x))",
                         {"a0": 1.0}, domain=(0.0, 6.28), recovered=False),
                     "miss"))
    # 7. Vanishing-term degeneracy: exact only because a0 -> 0. Counts
    #    as exact under the criterion, flagged as degenerate.
    oks.append(check("superset with vanishing term",
                     rec("a0 + a1/x**2", "2/x**2",
                         {"a0": 0.01, "a1": 1.98}),
                     "exact-symbolic", want_degenerate=["a0"]))
    # 8. Planck law, exact parse with scale parameter.
    oks.append(check("planck",
                     rec("a0*x**3/(exp(x) - 1)", "x**3/(exp(x) - 1)",
                         {"a0": 1.02}, domain=(0.5, 10.0)),
                     "exact-symbolic"))
    # 9. Gaussian template: needs 1/sqrt(2*pi) -- snap or exact-numeric,
    #    either exact flavour is a pass.
    g9 = grade_record(rec("a0*exp(-a1*x**2)", "exp(-x**2/2)/sqrt(2*pi)",
                          {"a0": 0.4, "a1": 0.5}, domain=(0.1, 3.0)))
    ok9 = g9["grade"] in ("exact-symbolic", "exact-numeric")
    print("[%s] %-34s grade=%-15s (either exact flavour accepted)"
          % ("PASS" if ok9 else "FAIL", "gaussian, pi-flavoured constant",
             g9["grade"]))
    oks.append(ok9)
    # 10. Parameter-free description, exact.
    oks.append(check("no parameters, exact",
                     rec("2/x**2", "2/x**2", {}), "exact-symbolic"))
    # 11. Parameter-free description, wrong.
    oks.append(check("no parameters, wrong",
                     rec("3/x**2", "2/x**2", {}, recovered=False), "miss"))
    # 12. Multivariate: full inverse-square law at d=3.
    oks.append(check("multivariate flux d=3",
                     rec("a0*x0*x1/(x2*x2)", "x0*x1/x2**2",
                         {"a0": 1.1},
                         domain=[[0.5, 2.0], [0.5, 2.0], [0.5, 3.0]],
                         variables=("x0", "x1", "x2")),
                     "exact-symbolic"))
    # 13. Multivariate rearrangement: velocity addition in two variables.
    oks.append(check("multivariate velocity add d=2",
                     rec("(x0 + x1)/(a0 + x0*x1)", "(x0 + x1)/(1 + x0*x1)",
                         {"a0": 0.9}, domain=[[0.0, 0.9], [0.0, 0.9]],
                         variables=("x0", "x1")),
                     "exact-symbolic"))
    # 14. Multivariate wrong form passing numerically.
    oks.append(check("multivariate wrong form",
                     rec("a0*(x0 + x1)*(a1 - x0*x1)",
                         "(x0 + x1)/(1 + x0*x1)",
                         {"a0": 0.9, "a1": 1.0},
                         domain=[[0.0, 0.5], [0.0, 0.5]],
                         variables=("x0", "x1")),
                     "functional"))
    # 15. Unparsable description.
    oks.append(check("unparsable",
                     rec("zoo/(x", "2/x**2", {}), "ungradable"))
    # 16. Aliased parameters (a0*a1 one effective constant): still exact.
    oks.append(check("aliased parameters",
                     rec("a0*a1/x**2", "2/x**2", {"a0": 1.0, "a1": 1.0}),
                     "exact-symbolic"))

    n = sum(oks)
    print("=" * 60)
    print("feynman_exactness: %d/%d passed in %.1fs"
          % (n, len(oks), time.time() - t0))
    if n != len(oks):
        sys.exit(1)


if __name__ == "__main__":
    main()
