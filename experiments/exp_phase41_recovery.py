"""Phase 4.1 — sanity checks: recovering known functions (PLAN §4.1).

Six ground-truth targets of increasing structural difficulty, every one
expressible in the search vocabulary, sampled with the FULL production
configuration on the tuned PT ladder (exp_phase41_ladder.py, round 5):

    T = [1, 1.29, 1.82, 3.08, 5.50, 10.0]

Conditions match the ladder tuning exactly (frozen psi_const vocabulary,
safe-4 ops, depth caps 4/2, sigma = 1.0, Powell): a ladder tuned under
different conditions would be tuned for a different target family.
Target 6 registers the Asin = c0*sin(c1*x) opaque template (Phase 2.10):
with the safe-4 ops a frequency-scaled sinusoid is UNREACHABLE without
it, so this run exercises opaque primitives inside the full EMC loop.

Scoring: the winner must GENERALIZE — NRMSE of the best tree against the
noise-free truth on a dense held-out grid (recovery flag: < 0.05), plus
the training-side mse and the recovered expression for the paper table.
Each run also logs ladder_stats(), doubling as a transferability check
of the tuned schedule across targets.

Run (module dir, conda env active):   python exp_phase41_recovery.py
Writes: results/phase41_recovery.json
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
import real_functionC
from groveC import Forest

import exp_common

# ---------------------------------------------------------------------------
# Configuration — CORRECTED PROTOCOL v2 (2026-07-13): sigma = known noise,
# ONE shared spawn burn-in per (target, seed) seeding every rung with the
# same top-K, ladder from exp_phase41_ladder.py rounds 6-8. The v1 runs
# (sigma = 1.0, random init) are archived in results/phase41_recovery
# {,_long,_rational_v2}.json and document the sigma-miscalibration effect.
# ---------------------------------------------------------------------------
LADDER = [1.0 / b for b in (1.0, 0.82, 0.64, 0.46, 0.28, 0.10)]
BURN_IN_WORKERS = 2
SEEDS = (17, 71)
SIZE_PER_GROVE = 4
INIT_DEPTH = 2
FUNCS = ["sin(x)", "cos(x)", "x"]
OPS = ["+", "-", "*", "/"]
NOISE_SD = 0.1
N_POINTS = 60
WARMUP_STEPS = 40
ROUNDS = 200
STEPS_PER_ROUND = 4
SWAPS_PER_ROUND = 4
MAX_TREE_DEPTH = 4
REBUILD_MAX_HEIGHT = 2
RECOVERY_NRMSE = 0.05
OUT_JSON = os.path.join(_HERE, "results", "phase41_recovery_v2.json")

_ASIN = real_functionC.AtomSpec("Asin", "c0*sin(c1*x)")

# --long: budget-scaling pass — rerun the targets that MISSED at the
# standard budget (multimodal cases: inner-position parameters competing
# with broad approximant basins; the atom case with a cache-frozen theta)
# at 3x rounds, same seeds, to measure recovery vs compute.
LONG = "--long" in sys.argv
if LONG:
    ROUNDS = 600
    OUT_JSON = os.path.join(_HERE, "results", "phase41_recovery_long.json")

# --only=<target>: rerun a single (e.g. re-specified) target.
ONLY = None
for _a in sys.argv:
    if _a.startswith("--only="):
        ONLY = _a.split("=", 1)[1]
        OUT_JSON = OUT_JSON.replace(".json", "_%s_v2.json" % ONLY)

# (name, truth-callable, latex-ish truth string, domain, atoms)
TARGETS = (
    ("linear",          lambda x: 0.5 * x,             "0.5*x",
     (0.5, 3.5), None),
    ("sin_plus_linear", lambda x: np.sin(x) + 0.5 * x, "sin(x) + 0.5*x",
     (0.5, 3.5), None),
    ("x_sin",           lambda x: x * np.sin(x),       "x*sin(x)",
     (0.5, 3.5), None),
    # NB domain: on the original (0.5, 3.5) window this target spans only
    # 0.33-0.78 — signal sd ~0.13 vs noise 0.1 (SNR ~1.3), and the
    # posterior LEGITIMATELY favours near-constants (both long-budget runs
    # converged to a bare parameter leaf). Widened to (0.1, 10.0)
    # (span 0.09-0.91, SNR ~4) so recovery is well-posed. Benchmark-spec
    # fix, 2026-07-12; the low-SNR runs are kept in the round-1/long JSONs.
    ("rational",        lambda x: x / (1.0 + x),       "x/(1 + x)",
     (0.1, 10.0), None),
    ("poly2",           lambda x: x * x - x,           "x**2 - x",
     (0.5, 3.5), None),
    ("freq_sin_atom",   lambda x: 2.3 * np.sin(1.3 * x), "2.3*sin(1.3*x)",
     (0.1, 6.0), [_ASIN]),
)


def run_target(name, f_true, truth_str, dom, atoms, seed):
    rng = np.random.default_rng(seed)
    x = np.linspace(dom[0], dom[1], N_POINTS).astype(np.float64)
    y = (f_true(x) + rng.normal(0.0, NOISE_SD, N_POINTS)).astype(np.float64)

    # Corrected protocol: shared spawn burn-in once, seed every rung with
    # the same top-K, score with sigma = the known noise.
    roots, posts, burn_wall = exp_common.spawn_burn_in(
        x, y, FUNCS, OPS, NOISE_SD, seed, atoms, BURN_IN_WORKERS)
    print("    %s seed %d: burn-in %d candidates in %.1f s; top: %s"
          % (name, seed, len(roots), burn_wall,
             roots[0].get_fun().get_description()), flush=True)
    forest = exp_common.seeded_forest(x, y, LADDER, SIZE_PER_GROVE, FUNCS,
                                      OPS, NOISE_SD, seed, roots,
                                      atoms=atoms)

    t0 = time.perf_counter()
    forest.run_all(y, WARMUP_STEPS)
    traces = exp_common.emc_loop(forest, y, ROUNDS, STEPS_PER_ROUND,
                                 SWAPS_PER_ROUND, trace_every=5,
                                 label="%s seed %d" % (name, seed), t0=t0)
    wall = time.perf_counter() - t0

    best = forest.best_tree()
    g0 = forest.groves[0]
    best_post = float(g0.root_posteriors[g0.best_index()])

    # training-side fit
    yhat = np.asarray(best.evaluate(x), dtype=np.float64)
    mse_train = float(np.nanmean((y - yhat) ** 2))

    # held-out generalization against the NOISE-FREE truth
    x_test = np.linspace(dom[0], dom[1], 200).astype(np.float64)
    f_test = f_true(x_test)
    yhat_test = np.asarray(best.evaluate(x_test), dtype=np.float64)
    resid = f_test - yhat_test
    nrmse = float(np.sqrt(np.nanmean(resid ** 2))
                  / max(np.sqrt(np.mean(f_test ** 2)), 1e-12))
    recovered = bool(nrmse < RECOVERY_NRMSE)

    atomic_in_best = any(lf.is_atomic() for lf in best.get_leaves())
    out = {
        "target": name, "truth": truth_str, "seed": seed,
        "best_tree": best.tree_get_description(),
        "best_params": dict(best.get_param_dicts()[1]),
        "best_posterior": best_post,
        "mse_train": mse_train, "nrmse_test": nrmse,
        "recovered": recovered, "atomic_in_best": atomic_in_best,
        "wall_seconds": wall, "burn_in_seconds": burn_wall,
        "burn_in_top": roots[0].get_fun().get_description(),
        "ladder_stats": forest.ladder_stats(),
        "traces": traces,
    }
    print("  %-16s seed %-3d %s  nrmse=%.4f  mse=%.4f  %.0fs  best: %s"
          % (name, seed, "RECOVERED" if recovered else "MISSED   ",
             nrmse, mse_train, wall, out["best_tree"]), flush=True)
    return out


def main():
    gp_movesC.set_max_tree_depth(MAX_TREE_DEPTH)
    gp_movesC.set_subtree_rebuild_max_height(REBUILD_MAX_HEIGHT)

    print("Phase 4.1 recovery benchmarks — tuned ladder %s"
          % ["%.2f" % t for t in LADDER])
    print("conditions: funcs=%s ops=%s noise=%.2f n=%d rounds=%d seeds=%s"
          % (FUNCS, OPS, NOISE_SD, N_POINTS, ROUNDS, list(SEEDS)), flush=True)

    targets = TARGETS
    if LONG:
        hard = ("sin_plus_linear", "rational", "freq_sin_atom")
        targets = tuple(t for t in TARGETS if t[0] in hard)
        print("budget-scaling pass (--long): %d rounds, targets %s"
              % (ROUNDS, [t[0] for t in targets]), flush=True)
    if ONLY:
        targets = tuple(t for t in TARGETS if t[0] == ONLY)
        print("single-target rerun (--only=%s)" % ONLY, flush=True)

    runs = []
    for name, f_true, truth_str, dom, atoms in targets:
        for seed in SEEDS:
            runs.append(run_target(name, f_true, truth_str, dom, atoms, seed))

    n_rec = sum(1 for r in runs if r["recovered"])
    print("\n=== summary: %d/%d runs recovered (nrmse < %.2f) ==="
          % (n_rec, len(runs), RECOVERY_NRMSE))
    for r in runs:
        print("  %-16s seed %-3d %s  nrmse=%.4f  best: %s"
              % (r["target"], r["seed"],
                 "yes" if r["recovered"] else "NO ",
                 r["nrmse_test"], r["best_tree"]))

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    payload = {
        "config": {
            "ladder": LADDER, "seeds": list(SEEDS),
            "size_per_grove": SIZE_PER_GROVE, "init_depth": INIT_DEPTH,
            "funcs": FUNCS, "ops": OPS, "noise_sd": NOISE_SD,
            "n_points": N_POINTS, "warmup_steps": WARMUP_STEPS,
            "rounds": ROUNDS, "steps_per_round": STEPS_PER_ROUND,
            "swaps_per_round": SWAPS_PER_ROUND,
            "max_tree_depth": MAX_TREE_DEPTH,
            "subtree_rebuild_max_height": REBUILD_MAX_HEIGHT,
            "recovery_nrmse_threshold": RECOVERY_NRMSE,
            "freeze_vocab": True, "sigma": NOISE_SD, "long_pass": LONG,
            "protocol": "v2: shared spawn burn-in seeding + sigma=noise",
            "targets": [{"name": t[0], "truth": t[2], "domain": list(t[3]),
                         "atoms": [s.name for s in t[4]] if t[4] else None}
                        for t in TARGETS],
        },
        "runs": runs,
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(payload, fh, indent=2)
    print("results written to %s" % OUT_JSON)


if __name__ == "__main__":
    main()
