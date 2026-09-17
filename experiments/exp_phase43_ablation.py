"""Phase 4.3 ablation I — FROZEN vs CONTINUOUSLY-ADAPTIVE Pólya prior.

The empirical companion to the paper's two-regime section (§9 / the
sec9_adaptation conjecture): reinforcement runs through burn-in-seeding
replay + warm-up, then Θ is FROZEN at its current value Θ̄
(`forest.polya.adapt_frozen = True` — the Phase 4.3 opt-in kernel
switch, default off = bit-identical legacy) for the entire EMC phase.

This script runs the FROZEN arm of the A/B on the fixed kernel. The
ADAPTIVE arm needs no new compute: the post-fix reproducibility reruns
(results/phase41_recovery_v2_repro.json, phase41_depth3_repro.json)
are same-kernel, same-seed, same-protocol baselines that differ from
this run in exactly one bit — the freeze flag.

Benchmarks (identical to the 4.1 suites, constants IMPORTED from the
suite modules so the arms cannot drift): depth-2 recovery v2 (6 targets
x 2 seeds, 200 rounds) + depth-3 discovery (3 x 2, 400 rounds).

Outputs (ALL NEW, in the dedicated ablation subfolder — additive
philosophy, nothing existing touched):
    results/ablation_results/phase43_frozen_d2.json
    results/ablation_results/phase43_frozen_d3.json
    results/ablation_results/phase43_frozen_vs_adaptive_report.md

Per-run sanity assertions: polya.t and the root-cell Dirichlet must be
CONSTANT from the freeze point onward (recorded in the JSON).

Run (module dir, MPI-capable env — spawn burn-in!):
    mpiexec -n 1 python exp_phase43_ablation.py [--smoke]
"""

import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

import gp_movesC

import exp_common

sys.argv_backup, sys.argv = sys.argv, ["exp_phase43_ablation"]
import exp_phase41_recovery as d2
import exp_phase41_depth3 as d3
sys.argv = sys.argv_backup

ABL = os.path.join(_HERE, "results", "ablation_results")
SMOKE = "--smoke" in sys.argv
# arm C: freeze BEFORE warm-up (symmetric untrained prior). Writes to
# separate *_frozenbirth_* files; the default arm-B behaviour and its
# outputs are untouched.
AT_BIRTH = "--at-birth" in sys.argv

REL_TOL = 1e-9


def run_frozen(name, f_true, truth_str, dom, atoms, seed, mod, rounds,
               freeze_at="warmup"):
    """One frozen-regime run, mirroring the suite's run_target exactly
    except for the freeze. freeze_at='warmup' (arm B: Theta-bar formed
    by warm-up reinforcement, then frozen) or 'birth' (arm C: frozen
    BEFORE any warm-up step — the symmetric untrained prior; Forest
    construction performs no reinforcement, so t_frozen should be 0
    and is recorded as proof)."""
    rng = np.random.default_rng(seed)
    x = np.linspace(dom[0], dom[1], mod.N_POINTS).astype(np.float64)
    y = (f_true(x) + rng.normal(0.0, mod.NOISE_SD, mod.N_POINTS)
         ).astype(np.float64)

    roots, posts, burn_wall = exp_common.spawn_burn_in(
        x, y, mod.FUNCS, mod.OPS, mod.NOISE_SD, seed, atoms,
        mod.BURN_IN_WORKERS)
    forest = exp_common.seeded_forest(x, y, mod.LADDER,
                                      mod.SIZE_PER_GROVE, mod.FUNCS,
                                      mod.OPS, mod.NOISE_SD, seed, roots,
                                      atoms=atoms)
    t0 = time.perf_counter()
    if freeze_at == "birth":
        # ---- arm C: freeze the SYMMETRIC prior before any learning ----
        forest.polya.adapt_frozen = True
    forest.run_all(y, mod.WARMUP_STEPS)

    # ---- arm B: freeze Theta at Theta-bar after warm-up ----
    forest.polya.adapt_frozen = True
    t_frozen = int(forest.polya.t)
    alphas_frozen = [float(a) for a in forest.polya.root.alphas]

    traces = exp_common.emc_loop(forest, y, rounds, mod.STEPS_PER_ROUND,
                                 mod.SWAPS_PER_ROUND, trace_every=5,
                                 label="frz %s s%d" % (name, seed),
                                 t0=t0, progress_every=50)
    wall = time.perf_counter() - t0

    # sanity: reinforcement really halted
    t_end = int(forest.polya.t)
    alphas_end = [float(a) for a in forest.polya.root.alphas]
    frozen_ok = (t_end == t_frozen and
                 all(abs(a - b) < 1e-15
                     for a, b in zip(alphas_frozen, alphas_end)))

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
    recovered = bool(nrmse < mod.RECOVERY_NRMSE)

    out = {
        "target": name, "truth": truth_str, "seed": seed,
        "regime": "frozen_at_%s" % freeze_at,
        "t_frozen": t_frozen, "t_end": t_end,
        "frozen_ok": frozen_ok,
        "best_tree": best.tree_get_description(),
        "best_params": dict(best.get_param_dicts()[1]),
        "best_posterior": best_post,
        "mse_train": mse_train, "nrmse_test": nrmse,
        "recovered": recovered, "wall_seconds": wall,
        "burn_in_seconds": burn_wall, "traces": traces,
    }
    print("  [frozen] %-16s seed %-3d %s  nrmse=%.4f  post=%.3f  "
          "urn-frozen=%s  %.0fs  best: %s"
          % (name, seed, "RECOVERED" if recovered else "MISSED   ",
             nrmse, best_post, frozen_ok, wall, out["best_tree"]),
          flush=True)
    return out


def compare(tag, baseline_path, frozen_runs, lines):
    if not os.path.exists(baseline_path):
        lines.append("## %s: NO ADAPTIVE BASELINE (%s missing)"
                     % (tag, os.path.basename(baseline_path)))
        return
    with open(baseline_path) as fh:
        base = json.load(fh)
    b = {(r["target"], r["seed"]): r for r in base["runs"]}
    lines.append("## %s" % tag)
    lines.append("| target | seed | adaptive rec/NRMSE/post | "
                 "frozen rec/NRMSE/post | Δpost (frozen−adaptive) |")
    lines.append("|---|---|---|---|---|")
    n_b = n_f = 0
    for r in frozen_runs:
        key = (r["target"], r["seed"])
        a = b.get(key)
        if a is None:
            continue
        n_b += 1 if a["recovered"] else 0
        n_f += 1 if r["recovered"] else 0
        lines.append("| %s | %d | %s / %.4f / %.2f | %s / %.4f / %.2f "
                     "| %+.2f |"
                     % (key[0], key[1],
                        "Y" if a["recovered"] else "n", a["nrmse_test"],
                        a["best_posterior"],
                        "Y" if r["recovered"] else "n", r["nrmse_test"],
                        r["best_posterior"],
                        r["best_posterior"] - a["best_posterior"]))
    lines.append("")
    lines.append("**%s: adaptive %d vs frozen %d recovered (of %d "
                 "paired runs)**" % (tag, n_b, n_f, len(frozen_runs)))
    lines.append("")


def main():
    os.makedirs(ABL, exist_ok=True)
    exp_common.set_caps()

    d2_targets = d2.TARGETS
    d3_targets = d3.TARGETS
    d2_rounds, d3_rounds = d2.ROUNDS, d3.ROUNDS
    seeds2, seeds3 = d2.SEEDS, d3.SEEDS
    if SMOKE:
        d2_targets = tuple(t for t in d2.TARGETS if t[0] == "linear")
        d3_targets = ()
        d2_rounds, seeds2 = 30, (17,)
        print("[smoke] linear/17 only, 30 rounds", flush=True)

    freeze_at = "birth" if AT_BIRTH else "warmup"
    stem = "phase43_frozenbirth" if AT_BIRTH else "phase43_frozen"
    frozen_d2, frozen_d3 = [], []
    print("=== frozen-at-%s arm, depth-2 suite (%d rounds) ==="
          % (freeze_at, d2_rounds), flush=True)
    for name, f_true, truth_str, dom, atoms in d2_targets:
        for seed in seeds2:
            frozen_d2.append(run_frozen(name, f_true, truth_str, dom,
                                        atoms, seed, d2, d2_rounds,
                                        freeze_at))
            with open(os.path.join(ABL, stem + "_d2.json"), "w") as fh:
                json.dump({"regime": "frozen_at_" + freeze_at,
                           "runs": frozen_d2}, fh, indent=2)
    print("=== frozen-at-%s arm, depth-3 suite (%d rounds) ==="
          % (freeze_at, d3_rounds), flush=True)
    for name, f_true, truth_str, dom in d3_targets:
        for seed in seeds3:
            frozen_d3.append(run_frozen(name, f_true, truth_str, dom,
                                        None, seed, d3, d3_rounds,
                                        freeze_at))
            with open(os.path.join(ABL, stem + "_d3.json"), "w") as fh:
                json.dump({"regime": "frozen_at_" + freeze_at,
                           "runs": frozen_d3}, fh, indent=2)

    lines = ["# Phase 4.3 ablation — %s vs adaptive Pólya prior"
             % ("frozen-at-BIRTH (symmetric untrained)" if AT_BIRTH
                else "frozen-at-warm-up"),
             "",
             "Frozen arm: this run (freeze point: %s). Adaptive arm: "
             "the post-fix repro reruns (same kernel, seeds, protocol)."
             % freeze_at, ""]
    if AT_BIRTH:
        t0s = sorted({r["t_frozen"] for r in frozen_d2 + frozen_d3})
        lines.append("t at freeze (should be 0 = genuinely untrained): "
                     "%s" % t0s)
        lines.append("")
    compare("depth-2 (recovery v2)",
            os.path.join(_HERE, "results",
                         "phase41_recovery_v2_repro.json"),
            frozen_d2, lines)
    compare("depth-3 (discovery)",
            os.path.join(_HERE, "results", "phase41_depth3_repro.json"),
            frozen_d3, lines)
    if AT_BIRTH:
        # also compare against arm B (frozen at warm-up)
        compare("depth-2 vs FROZEN-AT-WARM-UP arm",
                os.path.join(ABL, "phase43_frozen_d2.json"),
                frozen_d2, lines)
        compare("depth-3 vs FROZEN-AT-WARM-UP arm",
                os.path.join(ABL, "phase43_frozen_d3.json"),
                frozen_d3, lines)
    urn_ok = all(r["frozen_ok"] for r in frozen_d2 + frozen_d3)
    lines.append("Urn-freeze sanity (t and root Dirichlet constant "
                 "after freeze in every run): **%s**"
                 % ("PASS" if urn_ok else "FAIL"))
    report = os.path.join(ABL, ("phase43_frozenbirth_report.md"
                                if AT_BIRTH else
                                "phase43_frozen_vs_adaptive_report.md"))
    with open(report, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("report written to %s" % report, flush=True)


if __name__ == "__main__":
    main()
