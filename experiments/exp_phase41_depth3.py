"""Phase 4.1 — depth-3 recovery suite: targets OUTSIDE the burn-in.

Motivation (RESULTS_PHASE4.md §4, C2): in the v2 suite 5/6 true
structures are depth <= 2, i.e. inside the shared burn-in's exhaustive
enumeration, so 11/12 credits the pipeline more than the MCMC kernel.
Here every true structure has tree height >= 3 — the burn-in CANNOT
produce it, so recovery measures the sampler's reach beyond its seeds:

  poly3     x^3 - 2x            nearby INSIDE mode x^3 (height 2) —
                                escape-from-a-seeded-mode test
  inv_quad  3x/(1 + x^2)        fully outside; peaked, non-monotone —
                                no height-2 form fits
  mixed     x*sin(x) + x/(1+x)  dominant part x*sin(x) IS a seed;
                                must grow the rational correction
                                (fitted const + x*sin(x): NRMSE ~0.09,
                                above threshold — correction required)

Protocol: identical to recovery v2 (sigma = known noise, ONE shared
spawn burn-in per (target, seed), same top-K seeding all rungs, ladder
T = [1, 1.22, 1.56, 2.17, 3.57, 10], depth caps 4/2) except the round
budget is doubled to 400 — the targets need genuine structural search.
Traces carry the 2026-07-13 additive fields (per-rung accepted moves,
unique structures, tree-depth/node-count lists) feeding fig7/fig8.

NEW NAMES THROUGHOUT (nothing from v1/v2 is overwritten).

Run (module dir, MPI-capable env — spawn burn-in!):
    mpiexec -n 1 python exp_phase41_depth3.py [--only=<target>]
Writes: results/phase41_depth3.json
"""

import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_SRC = os.path.join(os.path.dirname(_HERE), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

import gp_movesC

import exp_common

LADDER = [1.0 / b for b in (1.0, 0.82, 0.64, 0.46, 0.28, 0.10)]
BURN_IN_WORKERS = 2
SEEDS = (17, 71)
SIZE_PER_GROVE = 4
FUNCS = ["sin(x)", "cos(x)", "x"]
OPS = ["+", "-", "*", "/"]
NOISE_SD = 0.1
N_POINTS = 60
WARMUP_STEPS = 40
ROUNDS = 400
STEPS_PER_ROUND = 4
SWAPS_PER_ROUND = 4
MAX_TREE_DEPTH = 4
REBUILD_MAX_HEIGHT = 2
RECOVERY_NRMSE = 0.05
OUT_JSON = os.path.join(_HERE, "results", "phase41_depth3.json")

ONLY = None
for _a in sys.argv:
    if _a.startswith("--only="):
        ONLY = _a.split("=", 1)[1]
        OUT_JSON = OUT_JSON.replace(".json", "_%s.json" % ONLY)

# (name, truth-callable, truth string, domain)
# SNR check (lesson of RESULTS §2): signal sd vs noise 0.1 —
# poly3 ~ 11 (huge), inv_quad ~ 0.33 (SNR 3.3), mixed ~ 0.9 (SNR 9).
TARGETS = (
    ("poly3",    lambda x: x ** 3 - 2.0 * x,
     "x**3 - 2*x",              (0.5, 3.5)),
    ("inv_quad", lambda x: 3.0 * x / (1.0 + x * x),
     "3*x/(1 + x**2)",          (0.1, 6.0)),
    ("mixed",    lambda x: x * np.sin(x) + x / (1.0 + x),
     "x*sin(x) + x/(1 + x)",    (0.5, 3.5)),
)


def fitted_expression(desc, params, ndigits=4):
    """Substitute fitted parameter values into the description string."""
    try:
        import sympy as sp
        expr = sp.sympify(desc)
        # fixed display precision — raw Python floats re-expand to 15
        # digits through arithmetic like x/a0 -> (1/a0)*x
        subs = {sp.Symbol(k): sp.Float(round(float(v), ndigits), 6)
                for k, v in params.items()}
        return str(expr.subs(subs))
    except Exception:
        return "%s with %s" % (desc, {k: round(float(v), ndigits)
                                      for k, v in params.items()})


def run_target(name, f_true, truth_str, dom, seed):
    rng = np.random.default_rng(seed)
    x = np.linspace(dom[0], dom[1], N_POINTS).astype(np.float64)
    y = (f_true(x) + rng.normal(0.0, NOISE_SD, N_POINTS)).astype(np.float64)

    roots, posts, burn_wall = exp_common.spawn_burn_in(
        x, y, FUNCS, OPS, NOISE_SD, seed, None, BURN_IN_WORKERS)
    print("    %s seed %d: burn-in %d candidates in %.1f s; top: %s"
          % (name, seed, len(roots), burn_wall,
             roots[0].get_fun().get_description()), flush=True)
    # by construction every burn-in candidate has height <= 2
    seed_depth_max = max(r.depth() for r in roots[:SIZE_PER_GROVE])
    forest = exp_common.seeded_forest(x, y, LADDER, SIZE_PER_GROVE, FUNCS,
                                      OPS, NOISE_SD, seed, roots)

    t0 = time.perf_counter()
    forest.run_all(y, WARMUP_STEPS)
    traces = exp_common.emc_loop(forest, y, ROUNDS, STEPS_PER_ROUND,
                                 SWAPS_PER_ROUND, trace_every=5,
                                 label="%s seed %d" % (name, seed), t0=t0)
    wall = time.perf_counter() - t0

    best = forest.best_tree()
    g0 = forest.groves[0]
    best_post = float(g0.root_posteriors[g0.best_index()])

    yhat = np.asarray(best.evaluate(x), dtype=np.float64)
    mse_train = float(np.nanmean((y - yhat) ** 2))

    x_test = np.linspace(dom[0], dom[1], 200).astype(np.float64)
    f_test = f_true(x_test)
    yhat_test = np.asarray(best.evaluate(x_test), dtype=np.float64)
    resid = f_test - yhat_test
    nrmse = float(np.sqrt(np.nanmean(resid ** 2))
                  / max(np.sqrt(np.mean(f_test ** 2)), 1e-12))
    recovered = bool(nrmse < RECOVERY_NRMSE)

    params = dict(best.get_param_dicts()[1])
    best_depth = int(best.get_root().depth())
    out = {
        "target": name, "truth": truth_str, "seed": seed,
        "best_tree": best.tree_get_description(),
        "best_params": params,
        "expression_fitted": fitted_expression(best.tree_get_description(),
                                               params),
        "best_depth": best_depth,
        "seed_depth_max": int(seed_depth_max),
        "beyond_burn_in": bool(best_depth >= 3),
        "best_posterior": best_post,
        "mse_train": mse_train, "nrmse_test": nrmse,
        "recovered": recovered,
        "wall_seconds": wall, "burn_in_seconds": burn_wall,
        "burn_in_top": roots[0].get_fun().get_description(),
        "ladder_stats": forest.ladder_stats(),
        "traces": traces,
    }
    print("  %-9s seed %-3d %s  nrmse=%.4f  depth=%d%s  %.0fs  best: %s"
          % (name, seed, "RECOVERED" if recovered else "MISSED   ",
             nrmse, best_depth,
             " (beyond burn-in)" if out["beyond_burn_in"] else "",
             wall, out["best_tree"]), flush=True)
    return out


def main():
    gp_movesC.set_max_tree_depth(MAX_TREE_DEPTH)
    gp_movesC.set_subtree_rebuild_max_height(REBUILD_MAX_HEIGHT)

    print("Phase 4.1 DEPTH-3 suite — targets outside the burn-in "
          "enumeration; ladder %s" % ["%.2f" % t for t in LADDER])
    print("conditions: funcs=%s ops=%s noise=%.2f n=%d rounds=%d seeds=%s"
          % (FUNCS, OPS, NOISE_SD, N_POINTS, ROUNDS, list(SEEDS)), flush=True)

    targets = TARGETS
    if ONLY:
        targets = tuple(t for t in TARGETS if t[0] == ONLY)
        print("single-target rerun (--only=%s)" % ONLY, flush=True)

    runs = []
    for name, f_true, truth_str, dom in targets:
        for seed in SEEDS:
            runs.append(run_target(name, f_true, truth_str, dom, seed))

    n_rec = sum(1 for r in runs if r["recovered"])
    n_beyond = sum(1 for r in runs if r["beyond_burn_in"])
    print("\n=== summary: %d/%d recovered (nrmse < %.2f); "
          "%d/%d best trees beyond burn-in depth ==="
          % (n_rec, len(runs), RECOVERY_NRMSE, n_beyond, len(runs)))
    for r in runs:
        print("  %-9s seed %-3d %s  nrmse=%.4f  depth=%d  %s"
              % (r["target"], r["seed"],
                 "yes" if r["recovered"] else "NO ",
                 r["nrmse_test"], r["best_depth"], r["expression_fitted"]))

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    payload = {
        "config": {
            "suite": "depth3", "ladder": LADDER, "seeds": list(SEEDS),
            "size_per_grove": SIZE_PER_GROVE,
            "funcs": FUNCS, "ops": OPS, "noise_sd": NOISE_SD,
            "n_points": N_POINTS, "warmup_steps": WARMUP_STEPS,
            "rounds": ROUNDS, "steps_per_round": STEPS_PER_ROUND,
            "swaps_per_round": SWAPS_PER_ROUND,
            "max_tree_depth": MAX_TREE_DEPTH,
            "subtree_rebuild_max_height": REBUILD_MAX_HEIGHT,
            "recovery_nrmse_threshold": RECOVERY_NRMSE,
            "freeze_vocab": True, "sigma": NOISE_SD,
            "protocol": "v2 corrected + 2x budget; truths of height >= 3",
            "targets": [{"name": t[0], "truth": t[2], "domain": list(t[3])}
                        for t in TARGETS],
        },
        "runs": runs,
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(payload, fh, indent=2)
    print("results written to %s" % OUT_JSON)


if __name__ == "__main__":
    main()
