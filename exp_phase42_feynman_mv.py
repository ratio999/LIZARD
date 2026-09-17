"""Phase 4.2a-MV -- the Feynman MULTIVARIATE tier.

The univariate suite (exp_phase42_feynman.py, archived; untouched by
this script) reduced each law to R -> R because the kernel was
single-variable at the time. That limitation is gone: the flexible
domain is certified at d = 2 and d = 3 (test_phaseFD.py, FD-* and
FD3-*), so this tier runs the SAME laws in their multivariable form,
with the same ladder-EMC protocol and the same held-out numeric
recovery criterion, plus the optional AI-Feynman-comparable exactness
grading (--exactness; feynman_exactness.py -- equality modulo the
parameters).

Targets (per-target variables, vocabulary, and noise):
  flux_mv          I.12.2   x0*x1/x2**2          d=3  plain vocabulary
  velocity_add_mv  I.16.6   (x0+x1)/(1+x0*x1)    d=2  plain vocabulary
  polarization_mv  II.11.28 1 + p/(1-p/3), p=x0*x1  d=2  plain
  planck_mv        I.41.16  x0**3/(exp(x0/x1)-1) d=2  needs the Bose
                            factor as an opaque template: exp of a
                            QUOTIENT is not expressible at d > 1
                            (compose is d=1-only), the same
                            expressivity boundary found for
                            exp(-x0*x1) in the d=3 preparation
  gauss_mv         I.6.20   exp(-(x0/x1)**2/2)/(sqrt(2*pi)*x1)  d=2
                            Gaussian template, mirroring the
                            univariate suite's atom demonstration

Protocol notes (departures from 4.2a stated up front):
  * rows are drawn INDEPENDENTLY per variable, never linspace: at
    d > 1 collinear coordinate rows make a smaller model reproduce a
    larger law exactly and the recovery question ill-posed;
  * noise is a fixed fraction (5%) of the sampled truth's standard
    deviation, recorded per run (the univariate suite fixed absolute
    sigmas by the same SNR discipline);
  * MAX_TREE_DEPTH = 5 (4 in 4.2a): the polarization parse needs
    depth 4 plus a root slot;
  * the ladder was tuned at d = 1; ladder_stats is recorded per run as
    the transferability check, as in 4.2a -- report drift, do not
    silently re-tune;
  * for bit-reproducible reruns launch with PYTHONHASHSEED pinned
    (hash-randomised set order reaches the optimizer's parameter
    ordering across processes).

Run (module dir, MPI-capable env -- spawn burn-in):
    PYTHONHASHSEED=0 mpiexec -n 1 python exp_phase42_feynman_mv.py \
        [--only=<t>] [--smoke] [--exactness] [--burn-top-k K] [--burn-cap C]
Writes: results/phase42_feynman_mv.json (refuses overwrite)
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
BURN_IN_WORKERS = 5
SEEDS = (17, 71)
SIZE_PER_GROVE = 4
OPS = ["+", "-", "*", "/"]
N_POINTS = 250
NOISE_FRAC = 0.05
WARMUP_STEPS = 60
ROUNDS = 800
STEPS_PER_ROUND = 6
SWAPS_PER_ROUND = 4
MAX_TREE_DEPTH = 5
REBUILD_MAX_HEIGHT = 2
RECOVERY_NRMSE = 0.05
OUT_JSON = os.path.join(_HERE, "results", "phase42_feynman_mv.json")

_ABOSE = ("Abose", "c0/(exp(c1*x0/x1) - 1)")
_AGAUSS2 = ("Agauss2", "c0*exp(-c1*(x0/x1)**2)")

# (name, feynman_id, variables, truth-callable(cols), truth string,
#  per-variable domains, funcs, atom templates)
TARGETS = (
    ("flux_mv", "I.12.2", ("x0", "x1", "x2"),
     lambda c: c[0] * c[1] / (c[2] * c[2]),
     "x0*x1/x2**2",
     [(0.5, 2.0), (0.5, 2.0), (0.5, 3.0)],
     ["x0", "x1", "x2"], None),
    ("velocity_add_mv", "I.16.6", ("x0", "x1"),
     lambda c: (c[0] + c[1]) / (1.0 + c[0] * c[1]),
     "(x0 + x1)/(1 + x0*x1)",
     [(0.0, 0.95), (0.0, 0.95)],
     ["x0", "x1"], None),
    ("polarization_mv", "II.11.28", ("x0", "x1"),
     lambda c: 1.0 + c[0] * c[1] / (1.0 - c[0] * c[1] / 3.0),
     "1 + x0*x1/(1 - x0*x1/3)",
     [(0.1, 1.5), (0.1, 1.5)],
     ["x0", "x1"], None),
    ("planck_mv", "I.41.16", ("x0", "x1"),
     lambda c: c[0] ** 3 / (np.exp(c[0] / c[1]) - 1.0),
     "x0**3/(exp(x0/x1) - 1)",
     [(0.5, 8.0), (0.5, 2.0)],
     ["x0", "x1"], [_ABOSE]),
    ("gauss_mv", "I.6.20", ("x0", "x1"),
     lambda c: np.exp(-0.5 * (c[0] / c[1]) ** 2)
     / (np.sqrt(2.0 * np.pi) * c[1]),
     "exp(-(x0/x1)**2/2)/(sqrt(2*pi)*x1)",
     [(0.1, 3.0), (0.5, 2.0)],
     ["x0", "x1"], [_AGAUSS2]),
)

SMOKE = "--smoke" in sys.argv
EXACTNESS = "--exactness" in sys.argv
if SMOKE:
    ROUNDS = 40
    WARMUP_STEPS = 20
    BURN_IN_WORKERS = 2
    SEEDS = (17,)
    OUT_JSON = os.path.join(_HERE, "results",
                            "phase42_feynman_mv_smoke.json")

ONLY = None
BURN_TOP_K = 0
BURN_CAP = 0
for _a in sys.argv:
    if _a.startswith("--only="):
        ONLY = _a.split("=", 1)[1]
        OUT_JSON = OUT_JSON.replace(".json", "_%s.json" % ONLY)
    if _a.startswith("--burn-top-k"):
        BURN_TOP_K = int(_a.split()[-1] if " " in _a else
                         sys.argv[sys.argv.index(_a) + 1])
    if _a.startswith("--burn-cap"):
        BURN_CAP = int(_a.split()[-1] if " " in _a else
                       sys.argv[sys.argv.index(_a) + 1])


def make_domain(doms, n, rng):
    """(d, n) C-contiguous domain, rows drawn independently."""
    cols = [rng.uniform(lo, hi, n) for lo, hi in doms]
    return np.ascontiguousarray(np.vstack(cols).astype(np.float64))


def run_target(name, fid, variables, f_true, truth_str, doms, funcs,
               atom_defs, seed):
    real_functionC.set_variables(tuple(variables))
    atoms = ([real_functionC.AtomSpec(n_, t_) for n_, t_ in atom_defs]
             if atom_defs else None)
    rng = np.random.default_rng(seed)
    X = make_domain(doms, N_POINTS, rng)
    truth = f_true([X[i] for i in range(len(variables))])
    noise_sd = float(NOISE_FRAC * np.std(truth))
    y = (truth + rng.normal(0.0, noise_sd, N_POINTS)).astype(np.float64)

    roots, posts, burn_wall = exp_common.spawn_burn_in(
        X, y, funcs, OPS, noise_sd, seed, atoms, BURN_IN_WORKERS,
        burn_in_top_k=BURN_TOP_K, burn_in_cap=BURN_CAP)
    print("    %s [%s] seed %d: burn-in %d candidates in %.1f s; top: %s"
          % (name, fid, seed, len(roots), burn_wall,
             roots[0].get_fun().get_description()), flush=True)
    forest = exp_common.seeded_forest(X, y, LADDER, SIZE_PER_GROVE, funcs,
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
    yhat = np.asarray(best.evaluate(X), dtype=np.float64)
    mse_train = float(np.nanmean((y - yhat) ** 2))

    # held-out: a fresh independent draw, noise-free truth
    rng_t = np.random.default_rng(seed + 1000)
    X_test = make_domain(doms, 500, rng_t)
    f_test = f_true([X_test[i] for i in range(len(variables))])
    yhat_test = np.asarray(best.evaluate(X_test), dtype=np.float64)
    resid = f_test - yhat_test
    nrmse = float(np.sqrt(np.nanmean(resid ** 2))
                  / max(np.sqrt(np.mean(f_test ** 2)), 1e-12))
    recovered = bool(nrmse < RECOVERY_NRMSE)

    params = dict(best.get_param_dicts()[1])
    out = {
        "target": name, "feynman_id": fid, "truth": truth_str,
        "variables": list(variables), "seed": seed,
        "noise_sd": noise_sd, "noise_frac": NOISE_FRAC,
        "domain": [list(d) for d in doms],
        "funcs": list(funcs), "ops": OPS,
        "atoms": atom_defs,
        "best_tree": best.tree_get_description(),
        "best_params": params,
        "best_depth": int(best.get_root().depth()),
        "atomic_in_best": any(lf.is_atomic() for lf in best.get_leaves()),
        "best_posterior": best_post,
        "mse_train": mse_train, "nrmse_test": nrmse,
        "recovered": recovered,
        "wall_seconds": wall, "burn_in_seconds": burn_wall,
        "burn_in_top": roots[0].get_fun().get_description(),
        "burn_budget": {"top_k": BURN_TOP_K, "cap": BURN_CAP},
        "ladder_stats": forest.ladder_stats(),
        "traces": traces,
    }
    if EXACTNESS:
        from feynman_exactness import grade_record
        out["exactness"] = grade_record(out)
    print("  %-16s [%s] seed %-3d %s  nrmse=%.4f%s  %.0fs  best: %s"
          % (name, fid, seed, "RECOVERED" if recovered else "MISSED   ",
             nrmse,
             ("  [%s]" % out["exactness"]["grade"]) if EXACTNESS else "",
             wall, out["best_tree"][:80]), flush=True)
    return out


def main():
    gp_movesC.set_max_tree_depth(MAX_TREE_DEPTH)
    gp_movesC.set_subtree_rebuild_max_height(REBUILD_MAX_HEIGHT)
    if os.path.exists(OUT_JSON):
        raise SystemExit("REFUSING to overwrite %s" % OUT_JSON)
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)

    print("Phase 4.2a-MV: Feynman multivariate tier -- ladder %s%s"
          % (["%.2f" % t for t in LADDER], " [SMOKE]" if SMOKE else ""))
    print("conditions: ops=%s n=%d rounds=%dx%d seeds=%s workers=%d "
          "depth_cap=%d exactness=%s"
          % (OPS, N_POINTS, ROUNDS, STEPS_PER_ROUND, list(SEEDS),
             BURN_IN_WORKERS, MAX_TREE_DEPTH, EXACTNESS), flush=True)

    targets = TARGETS
    if SMOKE:
        targets = tuple(t for t in TARGETS
                        if t[0] in ("velocity_add_mv", "planck_mv"))
        print("smoke pass: one plain d=2 law + the Bose-template law, "
              "%d rounds" % ROUNDS, flush=True)
    if ONLY:
        targets = tuple(t for t in TARGETS if t[0] == ONLY)
        print("single-target rerun (--only=%s)" % ONLY, flush=True)

    runs = []
    t0 = time.time()
    try:
        for (name, fid, variables, f_true, truth_str, doms, funcs,
             atom_defs) in targets:
            for seed in SEEDS:
                runs.append(run_target(name, fid, variables, f_true,
                                       truth_str, doms, funcs, atom_defs,
                                       seed))
    finally:
        real_functionC.set_variables(("x",))
    rec = {"config": {"ladder": LADDER, "ops": OPS, "n_points": N_POINTS,
                      "noise_frac": NOISE_FRAC, "rounds": ROUNDS,
                      "steps_per_round": STEPS_PER_ROUND,
                      "swaps_per_round": SWAPS_PER_ROUND,
                      "warmup": WARMUP_STEPS, "seeds": list(SEEDS),
                      "max_tree_depth": MAX_TREE_DEPTH,
                      "recovery_nrmse": RECOVERY_NRMSE,
                      "burn_top_k": BURN_TOP_K, "burn_cap": BURN_CAP,
                      "smoke": SMOKE},
           "runs": runs}
    with open(OUT_JSON, "w") as fp:
        json.dump(rec, fp)
    n_rec = sum(r["recovered"] for r in runs)
    print("DONE: %d/%d recovered in %.0f s -> %s"
          % (n_rec, len(runs), time.time() - t0, OUT_JSON), flush=True)


if __name__ == "__main__":
    main()
