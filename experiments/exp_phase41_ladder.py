"""Phase 4.1 — PT ladder tuning (PLAN §4.1).

Protocol (as prescribed by the plan):
  * G = 8 rungs, geometric ladder T_k = ratio**k, k = 0..7;
  * sweep the ratio; monitor the SWAP acceptance rate per consecutive
    pair (Forest.ladder_stats(), Phase 4.1 monitors); target band
    0.2 <= rate <= 0.4 for every pair;
  * log ladder + rates (this script writes results/phase41_ladder.json)
    for the paper.

Configuration mirrors the sanity-check family the Phase 4.1 benchmarks
use: y = sin(x) + 0.5 x + eps on [0.5, 3.5] with the safe-4 op kernel,
run in the Stage 3c frozen-vocabulary paper mode. One fixed seed across
ratios so runs differ ONLY in the ladder.

Run (from the module directory, conda env active):

    python exp_phase41_ladder.py
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
from groveC import Forest

import exp_common

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 4141
G_RUNGS = 8
RATIOS = (1.2, 1.3, 1.4, 1.6)

# ROUND 2 (data-driven, from the geometric scan in results/phase41_ladder
# .json): no geometric ratio satisfies the band — swap acceptance is set by
# dbeta * dU, geometric ladders shrink dbeta like 1/T, and this bounded-BIC
# state space does not grow its energy fluctuations with T, so every hot
# pair pins at ~0.5-0.6 (adjacent hot rungs are near-prior twins). The
# cold pairs that DID land in band sit at dbeta ~ 0.23-0.37 — so round 2
# tests UNIFORM-IN-BETA (harmonic) ladders calibrated to that window.
# Set ROUND = 2 to run them (round 1 = the geometric scan, archived).
# ROUND 3: refinement — round 2's best (uniform-beta, d0.225) missed the
# band's lower edge by only 0.017 on two pairs (within one standard error
# at n~120 attempts), so shrink dbeta: straight linspace(beta = 1 -> 0.1)
# with 6 rungs (d0.18) and 7 rungs (d0.15). OUTCOME: overshoot — rates
# climb above 0.4, and the coldest pair saturates once both cold chains
# lock the dominant mode (a geometry-independent effect). Round 2's
# d0.225 ladder stands.
# ROUND 4: high-statistics validation of the chosen ladder at two seeds
# (double rounds, 6 swaps/round -> ~360 attempts/pair, SE ~0.026).
# OUTCOME: mid pairs solid (0.24-0.28, 0.19-0.21); the cold pair saturates
# high post-convergence (both cold chains lock the true model — free
# mixing, not a geometry defect); the TAIL pair (3.08 -> 10) sags to
# 0.11-0.17 in the long run (cold concentration widens the energy gap to
# the near-prior rung).
# ROUND 5: split the tail with one extra rung: T = [.., 3.08, 5.5, 10].
# OUTCOME: adopted under the sigma=1.0 scoring — but rounds 1-5 all ran
# with sigma=1.0 against noise 0.1 (the PLAN §4.1 mis-calibration
# finding), so their calibration is superseded.
# ---------------------------------------------------------------------
# ROUND 6 (2026-07-13, corrected protocol): sigma = NOISE_SD, and every
# forest is seeded from ONE shared spawn burn-in per seed (exp_common) —
# the tuning now measures the production protocol. Fresh geometric scan
# (the likelihood is ~100x sharper, so expect much finer dbeta near
# T = 1) + the round-5 ladder re-measured as a reference.
# ROUND 6 OUTCOME: the corrected sigma TILTS the profile — fine geometric
# ladders too easy everywhere (0.5-0.7), the old sigma1 ladder's COLD
# pairs sag under band (0.19/0.14: the 100x energy scale bites near T=1),
# hot region still flattens. ROUND 7: composites at HIGH statistics —
# finer cold spacing, keep the split tail.
# ROUND 7 OUTCOME: uniform-beta d0.18 (6 rungs) lands 5/5 pairs in band
# at high statistics (max deviation 0.000) and its best tree is the TRUE
# model a0*x + sin(x). ROUND 8: second-seed validation of that ladder.
ROUND = 8
_T6 = [1.0, 1.0 / 0.775, 1.0 / 0.55, 1.0 / 0.325, 5.5, 10.0]
_T18 = [1.0 / b for b in (1.0, 0.82, 0.64, 0.46, 0.28, 0.10)]
CANDIDATE_LADDERS = (
    ("final d0.18 6r seed999", _T18, 999),
)
SIZE_PER_GROVE = 4
INIT_DEPTH = 2
FUNCS = ["sin(x)", "cos(x)", "x"]
OPS = ["+", "-", "*", "/"]              # safe-4: no compose/pow blow-ups
NOISE_SD = 0.1
WARMUP_STEPS = 40                        # per grove, before counting swaps
ROUNDS = 120
STEPS_PER_ROUND = 4                      # per grove per round
SWAPS_PER_ROUND = 4                      # attempt_swaps calls per round
if ROUND in (4, 5) or ROUND >= 7:        # high-statistics rounds
    ROUNDS = 240
    SWAPS_PER_ROUND = 6
BAND = (0.2, 0.4)
# Corrected protocol (round >= 6): score with sigma = known noise, and
# seed every forest from one shared spawn burn-in per seed (exp_common).
SIGMA = NOISE_SD if ROUND >= 6 else 1.0
BURN_IN_SEEDED = ROUND >= 6
BURN_IN_WORKERS = 2
# Same bounded-support caps the Phase 2.8/2.9/2.12 fixtures (and hence the
# Phase 4.1 sanity benchmarks) run under. Without them the hot rungs drift
# toward the depth-8 default where per-step Powell cost explodes — and a
# ladder tuned on a different support than the benchmarks would be tuned
# for the wrong target.
MAX_TREE_DEPTH = 4
REBUILD_MAX_HEIGHT = 2
OUT_JSON = os.path.join(_HERE, "results", "phase41_ladder.json")


def make_data(seed=3, n=60):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.5, 3.5, n).astype(np.float64)
    y = (np.sin(x) + 0.5 * x + rng.normal(0.0, NOISE_SD, n)).astype(np.float64)
    return x, y


def run_ladder(label, temps, x, y, seed=SEED, top_roots=None):
    temps = np.asarray(temps, dtype=np.float64)
    if top_roots is not None:
        # Corrected protocol: every rung starts from the same top-K
        # burned-in candidates (posteriors computed inside).
        forest = exp_common.seeded_forest(x, y, temps, SIZE_PER_GROVE,
                                          FUNCS, OPS, SIGMA, seed,
                                          top_roots)
    else:
        forest = Forest(x, y, temps, SIZE_PER_GROVE, INIT_DEPTH, list(FUNCS),
                        sigma=SIGMA, Npars=1, operations=list(OPS),
                        use_gradients=False, seed=seed, freeze_vocab=True)
        forest.compute_posteriors_all(y)

    t0 = time.perf_counter()
    # Warm-up: let every rung find its region before rates are counted
    # (counters only start mattering after this — they are zero anyway,
    # but the warm-up swaps would bias the equilibrium rates, so none are
    # attempted here).
    forest.run_all(y, WARMUP_STEPS)

    for rnd in range(ROUNDS):
        forest.run_all(y, STEPS_PER_ROUND)
        for _ in range(SWAPS_PER_ROUND):
            forest.attempt_swaps(y)
        forest.attempt_crossovers(y)
        if (rnd + 1) % 20 == 0:
            print("    %s: round %d/%d (%.1f s)"
                  % (label, rnd + 1, ROUNDS, time.perf_counter() - t0),
                  flush=True)
    wall = time.perf_counter() - t0

    stats = forest.ladder_stats()
    best = forest.best_tree()
    stats["label"] = label
    stats["wall_seconds"] = wall
    stats["best_tree"] = best.tree_get_description() if best is not None else None
    stats["best_params"] = (dict(best.get_param_dicts()[1])
                            if best is not None else None)
    return stats


def report(stats):
    temps = stats["temperatures"]
    rates = stats["swap_rates"]
    atts = stats["swap_attempts"]
    lo, hi = BAND
    print("\n=== %s  (wall %.1f s)  best: %s" %
          (stats["label"], stats["wall_seconds"], stats["best_tree"]))
    print("  pair            T_g      T_g+1   attempts  rate   in-band")
    n_in = 0
    for g in range(len(rates)):
        ok = (rates[g] == rates[g]) and lo <= rates[g] <= hi
        n_in += ok
        print("  (%d,%d)     %8.3f  %8.3f   %6d    %5.3f    %s"
              % (g, g + 1, temps[g], temps[g + 1], atts[g], rates[g],
                 "yes" if ok else "NO"))
    print("  xover rates: %s"
          % ["%.2f" % r for r in stats["xover_rates"]])
    print("  in-band pairs: %d/%d" % (n_in, len(rates)))
    return n_in


def main():
    gp_movesC.set_max_tree_depth(MAX_TREE_DEPTH)
    gp_movesC.set_subtree_rebuild_max_height(REBUILD_MAX_HEIGHT)
    x, y = make_data()

    if ROUND == 1:
        candidates = [("geometric r=%.2f (%d rungs)" % (r, G_RUNGS),
                       [r ** k for k in range(G_RUNGS)], SEED)
                      for r in RATIOS]
        out_json = OUT_JSON
    else:
        candidates = [(e[0], list(e[1]), (e[2] if len(e) > 2 else SEED))
                      for e in CANDIDATE_LADDERS]
        out_json = OUT_JSON.replace(".json", "_round%d.json" % ROUND)

    print("Phase 4.1 PT ladder tuning (round %d) — band %.1f-%.1f, seed %d"
          % (ROUND, BAND[0], BAND[1], SEED))
    print("target: y = sin(x) + 0.5x + N(0, %.2f^2), n=%d, funcs=%s, ops=%s"
          % (NOISE_SD, len(x), FUNCS, OPS))

    # Corrected protocol: ONE shared burn-in per seed, reused across every
    # candidate ladder (burn-in scoring is temperature-independent).
    burn_roots = {}
    if BURN_IN_SEEDED:
        for _, _, cand_seed in candidates:
            if cand_seed not in burn_roots:
                roots, posts, wall = exp_common.spawn_burn_in(
                    x, y, FUNCS, OPS, SIGMA, cand_seed,
                    n_workers=BURN_IN_WORKERS)
                burn_roots[cand_seed] = roots
                print("burn-in seed %d: %d candidates in %.1f s; top: %s"
                      % (cand_seed, len(roots), wall,
                         roots[0].get_fun().get_description()), flush=True)

    all_stats = []
    scores = []
    for label, temps, cand_seed in candidates:
        stats = run_ladder(label, temps, x, y, cand_seed,
                           burn_roots.get(cand_seed))
        n_in = report(stats)
        all_stats.append(stats)
        # score a ladder by (#in-band pairs - #pairs, -max deviation):
        # all-in-band first, then smallest worst deviation
        devs = []
        for r in stats["swap_rates"]:
            if r != r:
                devs.append(1.0)
            elif r < BAND[0]:
                devs.append(BAND[0] - r)
            elif r > BAND[1]:
                devs.append(r - BAND[1])
            else:
                devs.append(0.0)
        scores.append((n_in - len(devs), -max(devs)))

    best_idx = max(range(len(candidates)), key=lambda k: scores[k])
    print("\n>>> recommended ladder: %s  (%d/%d pairs in band, "
          "max band deviation %.3f)"
          % (candidates[best_idx][0],
             scores[best_idx][0] + len(all_stats[best_idx]["swap_rates"]),
             len(all_stats[best_idx]["swap_rates"]), -scores[best_idx][1]))
    print("    T = %s" % ["%.3f" % t for t in candidates[best_idx][1]])

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    payload = {
        "config": {
            "round": ROUND, "seed": SEED,
            "candidates": [{"label": lab, "temperatures": t, "seed": sd}
                           for lab, t, sd in candidates],
            "size_per_grove": SIZE_PER_GROVE, "init_depth": INIT_DEPTH,
            "funcs": FUNCS, "ops": OPS, "noise_sd": NOISE_SD,
            "warmup_steps": WARMUP_STEPS, "rounds": ROUNDS,
            "steps_per_round": STEPS_PER_ROUND,
            "swaps_per_round": SWAPS_PER_ROUND, "band": list(BAND),
            "target": "sin(x) + 0.5*x + noise", "n_points": len(x),
            "freeze_vocab": True, "sigma": SIGMA,
            "burn_in_seeded": BURN_IN_SEEDED,
            "max_tree_depth": MAX_TREE_DEPTH,
            "subtree_rebuild_max_height": REBUILD_MAX_HEIGHT,
        },
        "runs": all_stats,
        "recommended_ladder": {
            "label": candidates[best_idx][0],
            "temperatures": candidates[best_idx][1],
        },
    }
    with open(out_json, "w") as fh:
        json.dump(payload, fh, indent=2)
    print("results written to %s" % out_json)


if __name__ == "__main__":
    main()
