"""Phase 4.2a — hero application I: the Feynman univariate set.

Six closed-form R->R laws from the Feynman symbolic-regression database
(non-essential constants fixed so each is a single-variable equation),
run with the FULL corrected protocol (PLAN §4.1: sigma = known noise,
ONE shared spawn burn-in per (target, seed) seeding every rung with the
same top-K, tuned ladder) and a HERO budget: 800 rounds x 6 steps
(~3x the 4.1 budget), n = 100 points, spawn burn-in with 5 workers
(+ master = 6 MPI processes, the agreed cap).

Targets (graded, iconic):
  flux          I.12.2    a/x^2                inverse-square law
  scattering    III.17.37 a(1 + b cos x)       scattering rate
  velocity_add  I.16.6    (u+x)/(1+ux), u=.5   relativistic addition
  polarization  II.11.28  1 + x/(1 - x/3)      dielectric polarization
  planck        I.41.16   x^3/(e^x - 1)        blackbody radiancy —
                                               flagship; needs exp(x)
  gauss         I.6.20a   e^(-x^2/2)/sqrt(2pi) UNREACHABLE by safe-4
                                               composition -> Gaussian
                                               template A_gauss =
                                               c0*exp(-c1*x^2)
                                               (Phase 2.10 on real
                                               physics)

Unlike the 4.1 suites this is NOT a controlled inside/outside-burn-in
experiment: it is the pipeline pointed at real physical laws, mixed
difficulty by construction. Noise is per-target (SNR discipline,
RESULTS_PHASE4.md §2). NB: the ladder was tuned at n = 60 with
{sin,cos,x}; ladder_stats() is recorded per run as the transferability
check — report drift rather than silently re-tuning.

NEW NAMES THROUGHOUT (results/phase42_feynman.json, figs fig9+).

Run (module dir, MPI-capable env — spawn burn-in!):
    mpiexec -n 1 python exp_phase42_feynman.py [--only=<t>] [--smoke]
Writes: results/phase42_feynman.json
"""

import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

import gp_movesC
import real_functionC

import exp_common

LADDER = [1.0 / b for b in (1.0, 0.82, 0.64, 0.46, 0.28, 0.10)]
BURN_IN_WORKERS = 5          # + master = 6 MPI processes (agreed cap)
SEEDS = (17, 71)
SIZE_PER_GROVE = 4
FUNCS = ["sin(x)", "cos(x)", "exp(x)", "x"]
OPS = ["+", "-", "*", "/"]
N_POINTS = 100
WARMUP_STEPS = 60
ROUNDS = 800
STEPS_PER_ROUND = 6
SWAPS_PER_ROUND = 4
MAX_TREE_DEPTH = 4
REBUILD_MAX_HEIGHT = 2
RECOVERY_NRMSE = 0.05
OUT_JSON = os.path.join(_HERE, "results", "phase42_feynman.json")

_AGAUSS = real_functionC.AtomSpec("Agauss", "c0*exp(-c1*x**2)")

SMOKE = "--smoke" in sys.argv
RESUME = "--resume" in sys.argv    # reuse checkpointed runs, redo the rest
if SMOKE:
    ROUNDS = 40
    WARMUP_STEPS = 20
    BURN_IN_WORKERS = 2
    SEEDS = (17,)
    OUT_JSON = os.path.join(_HERE, "results", "phase42_feynman_smoke.json")

ONLY = None
for _a in sys.argv:
    if _a.startswith("--only="):
        ONLY = _a.split("=", 1)[1]
        OUT_JSON = OUT_JSON.replace(".json", "_%s.json" % ONLY)

# (name, feynman_id, truth-callable, truth string, domain, noise_sd, atoms)
TARGETS = (
    ("flux", "I.12.2",
     lambda x: 2.0 / (x * x),
     "2/x**2",                        (0.5, 3.0),  0.10, None),
    ("scattering", "III.17.37",
     lambda x: 1.2 * (1.0 + 0.7 * np.cos(x)),
     "1.2*(1 + 0.7*cos(x))",          (0.0, 6.28), 0.10, None),
    ("velocity_add", "I.16.6",
     lambda x: (0.5 + x) / (1.0 + 0.5 * x),
     "(0.5 + x)/(1 + 0.5*x)",         (0.0, 0.95), 0.05, None),
    ("polarization", "II.11.28",
     lambda x: 1.0 + x / (1.0 - x / 3.0),
     "1 + x/(1 - x/3)",               (0.1, 2.5),  0.10, None),
    ("planck", "I.41.16",
     lambda x: x ** 3 / (np.exp(x) - 1.0),
     "x**3/(exp(x) - 1)",             (0.5, 10.0), 0.05, None),
    ("gauss", "I.6.20a",
     lambda x: np.exp(-x * x / 2.0) / np.sqrt(2.0 * np.pi),
     "exp(-x**2/2)/sqrt(2*pi)",       (0.1, 3.0),  0.02, [_AGAUSS]),
)


def fitted_expression(desc, params, ndigits=4):
    try:
        import sympy as sp
        expr = sp.sympify(desc)
        subs = {sp.Symbol(k): sp.Float(round(float(v), ndigits), 6)
                for k, v in params.items()}
        return str(expr.subs(subs))
    except Exception:
        return "%s  with  %s" % (desc, {k: round(float(v), ndigits)
                                        for k, v in params.items()})


def run_target(name, fid, f_true, truth_str, dom, noise_sd, atoms, seed):
    rng = np.random.default_rng(seed)
    x = np.linspace(dom[0], dom[1], N_POINTS).astype(np.float64)
    y = (f_true(x) + rng.normal(0.0, noise_sd, N_POINTS)).astype(np.float64)

    roots, posts, burn_wall = exp_common.spawn_burn_in(
        x, y, FUNCS, OPS, noise_sd, seed, atoms, BURN_IN_WORKERS)
    print("    %s [%s] seed %d: burn-in %d candidates in %.1f s; top: %s"
          % (name, fid, seed, len(roots), burn_wall,
             roots[0].get_fun().get_description()), flush=True)
    forest = exp_common.seeded_forest(x, y, LADDER, SIZE_PER_GROVE, FUNCS,
                                      OPS, noise_sd, seed, roots,
                                      atoms=atoms)

    t0 = time.perf_counter()
    forest.run_all(y, WARMUP_STEPS)
    traces = exp_common.emc_loop(forest, y, ROUNDS, STEPS_PER_ROUND,
                                 SWAPS_PER_ROUND, trace_every=10,
                                 label="%s seed %d" % (name, seed), t0=t0,
                                 progress_every=25)
    wall = time.perf_counter() - t0

    best = forest.best_tree()
    g0 = forest.groves[0]
    best_post = float(g0.root_posteriors[g0.best_index()])

    yhat = np.asarray(best.evaluate(x), dtype=np.float64)
    mse_train = float(np.nanmean((y - yhat) ** 2))

    x_test = np.linspace(dom[0], dom[1], 250).astype(np.float64)
    f_test = f_true(x_test)
    yhat_test = np.asarray(best.evaluate(x_test), dtype=np.float64)
    resid = f_test - yhat_test
    nrmse = float(np.sqrt(np.nanmean(resid ** 2))
                  / max(np.sqrt(np.mean(f_test ** 2)), 1e-12))
    recovered = bool(nrmse < RECOVERY_NRMSE)

    params = dict(best.get_param_dicts()[1])
    out = {
        "target": name, "feynman_id": fid, "truth": truth_str, "seed": seed,
        "noise_sd": noise_sd, "domain": list(dom),
        "best_tree": best.tree_get_description(),
        "best_params": params,
        "expression_fitted": fitted_expression(best.tree_get_description(),
                                               params),
        "best_depth": int(best.get_root().depth()),
        "atomic_in_best": any(lf.is_atomic() for lf in best.get_leaves()),
        "best_posterior": best_post,
        "mse_train": mse_train, "nrmse_test": nrmse,
        "recovered": recovered,
        "wall_seconds": wall, "burn_in_seconds": burn_wall,
        "burn_in_top": roots[0].get_fun().get_description(),
        "ladder_stats": forest.ladder_stats(),
        "traces": traces,
    }
    print("  %-13s [%s] seed %-3d %s  nrmse=%.4f  %.0fs  best: %s"
          % (name, fid, seed, "RECOVERED" if recovered else "MISSED   ",
             nrmse, wall, out["expression_fitted"]), flush=True)
    return out


def main():
    gp_movesC.set_max_tree_depth(MAX_TREE_DEPTH)
    gp_movesC.set_subtree_rebuild_max_height(REBUILD_MAX_HEIGHT)

    print("Phase 4.2a HERO: Feynman univariate set — ladder %s%s"
          % (["%.2f" % t for t in LADDER], " [SMOKE]" if SMOKE else ""))
    print("conditions: funcs=%s ops=%s n=%d rounds=%dx%d seeds=%s "
          "workers=%d" % (FUNCS, OPS, N_POINTS, ROUNDS, STEPS_PER_ROUND,
                          list(SEEDS), BURN_IN_WORKERS), flush=True)

    targets = TARGETS
    if SMOKE:
        targets = tuple(t for t in TARGETS if t[0] in ("planck", "gauss"))
        print("smoke pass: exp primitive + Gaussian template, %d rounds"
              % ROUNDS, flush=True)
    if ONLY:
        targets = tuple(t for t in TARGETS if t[0] == ONLY)
        print("single-target rerun (--only=%s)" % ONLY, flush=True)

    jobs = [(name, fid, f_true, truth_str, dom, noise_sd, atoms, seed)
            for name, fid, f_true, truth_str, dom, noise_sd, atoms in targets
            for seed in SEEDS]
    runs = []
    suite_t0 = time.perf_counter()
    partial_path = OUT_JSON.replace(".json", "_partial.json")
    done = {}
    if RESUME and os.path.exists(partial_path):
        with open(partial_path) as fh:
            prev = json.load(fh)
        for r in prev.get("runs", []):
            done[(r["target"], r["seed"])] = r
        print("resume: reusing %d checkpointed runs from %s"
              % (len(done), os.path.basename(partial_path)), flush=True)
    for k, job in enumerate(jobs):
        name, fid, f_true, truth_str, dom, noise_sd, atoms, seed = job
        if (name, seed) in done:
            print("### RUN %d/%d — %s seed %d: reused from checkpoint ###"
                  % (k + 1, len(jobs), name, seed), flush=True)
            runs.append(done[(name, seed)])
            continue
        elapsed = (time.perf_counter() - suite_t0) / 60.0
        eta = (elapsed / k * (len(jobs) - k)) if k else float("nan")
        print("\n### RUN %d/%d — %s [%s] seed %d | elapsed %.1f min | "
              "ETA ~%.0f min ###"
              % (k + 1, len(jobs), name, fid, seed, elapsed, eta),
              flush=True)
        runs.append(run_target(name, fid, f_true, truth_str, dom,
                               noise_sd, atoms, seed))
        # crash-safe checkpoint: everything finished so far, every run
        with open(partial_path, "w") as fh:
            json.dump({"done": len(runs), "of": len(jobs), "runs": runs},
                      fh, indent=2)
        print("    [checkpoint] %d/%d runs saved to %s"
              % (len(runs), len(jobs), os.path.basename(partial_path)),
              flush=True)

    n_rec = sum(1 for r in runs if r["recovered"])
    print("\n=== summary: %d/%d recovered (nrmse < %.2f) ==="
          % (n_rec, len(runs), RECOVERY_NRMSE))
    for r in runs:
        print("  %-13s [%s] seed %-3d %s  nrmse=%.4f  %s"
              % (r["target"], r["feynman_id"], r["seed"],
                 "yes" if r["recovered"] else "NO ",
                 r["nrmse_test"], r["expression_fitted"]))

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    payload = {
        "config": {
            "suite": "phase42a_feynman", "ladder": LADDER,
            "seeds": list(SEEDS), "size_per_grove": SIZE_PER_GROVE,
            "funcs": FUNCS, "ops": OPS, "n_points": N_POINTS,
            "warmup_steps": WARMUP_STEPS, "rounds": ROUNDS,
            "steps_per_round": STEPS_PER_ROUND,
            "swaps_per_round": SWAPS_PER_ROUND,
            "burn_in_workers": BURN_IN_WORKERS,
            "max_tree_depth": MAX_TREE_DEPTH,
            "subtree_rebuild_max_height": REBUILD_MAX_HEIGHT,
            "recovery_nrmse_threshold": RECOVERY_NRMSE,
            "freeze_vocab": True, "smoke": SMOKE,
            "protocol": "v2 corrected + hero budget (800x6, n=100, "
                        "5+1 MPI); per-target noise",
            "targets": [{"name": t[0], "feynman_id": t[1], "truth": t[3],
                         "domain": list(t[4]), "noise_sd": t[5],
                         "atoms": [s.name for s in t[6]] if t[6] else None}
                        for t in TARGETS],
        },
        "runs": runs,
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(payload, fh, indent=2)
    print("results written to %s" % OUT_JSON)


if __name__ == "__main__":
    main()
