"""Phase 4.3 — the remaining ablation arms, one campaign, three arms.

  notemper  single rung (LADDER = [1.0]): no parallel tempering. Swaps
            and trans-grove crossovers become no-ops (0 pairs). Same
            per-chain budget as the baseline — the question is "does
            the ladder matter at fixed per-chain compute".
  nobic     BIC complexity penalty OFF in the EMC scoring
            (function_treeC.set_bic_penalty(False) — the Phase 4.3
            opt-in kill switch; pure-likelihood selection). Burn-in
            ranking deliberately keeps BIC, so all arms share
            identical seeding; the ablation isolates parsimony's role
            DURING sampling. Expect overfitting pressure.
  affine    the user's affine-atom vocabulary pilot: the plain
            vocabulary PLUS affine-dressed oscillator templates
            A_sin^aff = c0 + c1*sin(c2*x), A_cos^aff = c0 + c1*cos(c2*x)
            (identifiability-pruned per the recorded rule: no phase d —
            sin AND cos both present span phases; never wrap x).
            Question: does one-move access to dressed primitives help
            or does the enlarged vocabulary dilute the prior?

Benchmarks: both 4.1 suites (depth-2 200 rounds, depth-3 400 rounds),
constants imported from the suite modules. Baseline for all
comparisons: the post-fix adaptive repro reruns. Everything additive;
outputs in results/ablation_results/:
    phase43_<arm>_d2.json / _d3.json / phase43_<arm>_report.md

Run (module dir, MPI-capable env — spawn burn-in!):
    mpiexec -n 1 python exp_phase43_arms.py [--smoke] [--arm=<name>]
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

import numpy as np

import function_treeC
import gp_movesC
import real_functionC

import exp_common

sys.argv_backup, sys.argv = sys.argv, ["exp_phase43_arms"]
import exp_phase41_recovery as d2
import exp_phase41_depth3 as d3
sys.argv = sys.argv_backup

ABL = os.path.join(_HERE, "results", "ablation_results")
SMOKE = "--smoke" in sys.argv
ONLY_ARM = None
for _a in sys.argv:
    if _a.startswith("--arm="):
        ONLY_ARM = _a.split("=", 1)[1]

_AFF_ATOMS = [real_functionC.AtomSpec("Asinaff", "c0 + c1*sin(c2*x)"),
              real_functionC.AtomSpec("Acosaff", "c0 + c1*cos(c2*x)")]

ARMS = ("notemper", "nobic", "affine")


def run_one(arm, name, f_true, truth_str, dom, atoms, seed, mod,
            rounds):
    rng = np.random.default_rng(seed)
    x = np.linspace(dom[0], dom[1], mod.N_POINTS).astype(np.float64)
    y = (f_true(x) + rng.normal(0.0, mod.NOISE_SD, mod.N_POINTS)
         ).astype(np.float64)

    ladder = [1.0] if arm == "notemper" else mod.LADDER
    use_atoms = atoms
    if arm == "affine":
        use_atoms = (list(atoms) if atoms else []) + _AFF_ATOMS

    roots, posts, burn_wall = exp_common.spawn_burn_in(
        x, y, mod.FUNCS, mod.OPS, mod.NOISE_SD, seed, use_atoms,
        mod.BURN_IN_WORKERS)
    forest = exp_common.seeded_forest(x, y, ladder, mod.SIZE_PER_GROVE,
                                      mod.FUNCS, mod.OPS, mod.NOISE_SD,
                                      seed, roots, atoms=use_atoms)
    t0 = time.perf_counter()
    if arm == "nobic":
        function_treeC.set_bic_penalty(False)
    try:
        forest.run_all(y, mod.WARMUP_STEPS)
        traces = exp_common.emc_loop(forest, y, rounds,
                                     mod.STEPS_PER_ROUND,
                                     mod.SWAPS_PER_ROUND, trace_every=5,
                                     label="%s %s s%d" % (arm, name,
                                                          seed),
                                     t0=t0, progress_every=100)
    finally:
        if arm == "nobic":
            function_treeC.set_bic_penalty(True)
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
    recovered = bool(nrmse < mod.RECOVERY_NRMSE)

    out = {
        "target": name, "truth": truth_str, "seed": seed, "arm": arm,
        "best_tree": best.tree_get_description(),
        "best_params": dict(best.get_param_dicts()[1]),
        "best_posterior": best_post,
        "best_k": len(dict(best.get_param_dicts()[1])),
        "best_depth": int(best.get_root().depth()),
        "mse_train": mse_train, "nrmse_test": nrmse,
        "recovered": recovered, "wall_seconds": wall,
        "burn_in_seconds": burn_wall, "traces": traces,
    }
    print("  [%s] %-16s seed %-3d %s  nrmse=%.4f  k=%d depth=%d  "
          "%.0fs  best: %s"
          % (arm, name, seed, "RECOVERED" if recovered else "MISSED   ",
             nrmse, out["best_k"], out["best_depth"], wall,
             out["best_tree"]), flush=True)
    return out


def compare(tag, baseline_path, arm_runs, lines):
    if not os.path.exists(baseline_path):
        lines.append("## %s: NO BASELINE" % tag)
        return
    with open(baseline_path) as fh:
        base = json.load(fh)
    b = {(r["target"], r["seed"]): r for r in base["runs"]}
    lines.append("## %s" % tag)
    lines.append("| target | seed | baseline rec/NRMSE | arm rec/NRMSE "
                 "| arm k | arm depth |")
    lines.append("|---|---|---|---|---|---|")
    n_b = n_a = 0
    for r in arm_runs:
        a = b.get((r["target"], r["seed"]))
        if a is None:
            continue
        n_b += 1 if a["recovered"] else 0
        n_a += 1 if r["recovered"] else 0
        lines.append("| %s | %d | %s / %.4f | %s / %.4f | %d | %d |"
                     % (r["target"], r["seed"],
                        "Y" if a["recovered"] else "n", a["nrmse_test"],
                        "Y" if r["recovered"] else "n", r["nrmse_test"],
                        r["best_k"], r["best_depth"]))
    lines.append("")
    lines.append("**%s: baseline %d vs arm %d recovered (of %d)**"
                 % (tag, n_b, n_a, len(arm_runs)))
    lines.append("")


def run_arm(arm):
    d2_targets, d3_targets = d2.TARGETS, d3.TARGETS
    d2_rounds, d3_rounds = d2.ROUNDS, d3.ROUNDS
    seeds2, seeds3 = d2.SEEDS, d3.SEEDS
    if SMOKE:
        d2_targets = tuple(t for t in d2.TARGETS if t[0] == "linear")
        d3_targets = ()
        d2_rounds, seeds2 = 30, (17,)

    # resume: reuse checkpointed runs (deterministic at fixed seeds, so
    # a rerun would reproduce them anyway — this just saves the time)
    done2, done3 = {}, {}
    for path, done in ((os.path.join(ABL, "phase43_%s_d2.json" % arm),
                        done2),
                       (os.path.join(ABL, "phase43_%s_d3.json" % arm),
                        done3)):
        if os.path.exists(path):
            with open(path) as fh:
                for r in json.load(fh).get("runs", []):
                    done[(r["target"], r["seed"])] = r
    if done2 or done3:
        print("[resume] arm %s: reusing %d checkpointed runs"
              % (arm, len(done2) + len(done3)), flush=True)

    runs2, runs3 = [], []
    print("=== arm %s, depth-2 (%d rounds) ===" % (arm, d2_rounds),
          flush=True)
    for name, f_true, truth_str, dom, atoms in d2_targets:
        for seed in seeds2:
            if (name, seed) in done2:
                runs2.append(done2[(name, seed)])
                continue
            runs2.append(run_one(arm, name, f_true, truth_str, dom,
                                 atoms, seed, d2, d2_rounds))
            with open(os.path.join(ABL, "phase43_%s_d2.json" % arm),
                      "w") as fh:
                json.dump({"arm": arm, "runs": runs2}, fh, indent=2)
    print("=== arm %s, depth-3 (%d rounds) ===" % (arm, d3_rounds),
          flush=True)
    for name, f_true, truth_str, dom in d3_targets:
        for seed in seeds3:
            if (name, seed) in done3:
                runs3.append(done3[(name, seed)])
                continue
            runs3.append(run_one(arm, name, f_true, truth_str, dom,
                                 None, seed, d3, d3_rounds))
            with open(os.path.join(ABL, "phase43_%s_d3.json" % arm),
                      "w") as fh:
                json.dump({"arm": arm, "runs": runs3}, fh, indent=2)

    lines = ["# Phase 4.3 ablation — arm: %s" % arm, ""]
    compare("depth-2 (recovery v2)",
            os.path.join(_HERE, "results",
                         "phase41_recovery_v2_repro.json"), runs2, lines)
    compare("depth-3 (discovery)",
            os.path.join(_HERE, "results",
                         "phase41_depth3_repro.json"), runs3, lines)
    report = os.path.join(ABL, "phase43_%s_report.md" % arm)
    with open(report, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("report written to %s" % report, flush=True)


def main():
    os.makedirs(ABL, exist_ok=True)
    exp_common.set_caps()
    arms = (ONLY_ARM,) if ONLY_ARM else ARMS
    for arm in arms:
        assert arm in ARMS, "unknown arm %r" % arm
        run_arm(arm)


if __name__ == "__main__":
    main()
