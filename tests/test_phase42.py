"""Phase 4.2 regression test — ONE Feynman run, truncated, fingerprinted.

Reruns the 4.2a `flux` (Feynman I.12.2) target, seed 17, with the exact
suite protocol (same data generation, same vocabulary, same shared spawn
burn-in, same 8->6-rung ladder and seeding) but a TRUNCATED EMC budget
(60 rounds instead of 800) and 2 burn-in workers, then compares a
fingerprint of the outcome against `test_phase42_reference.json`:

    burn-in top candidate (exact string) + candidate count,
    best tree (exact string) + fitted params (rel. 1e-6),
    best posterior / train mse / held-out NRMSE (rel. 1e-6).

This is NOT about the science result — it is an end-to-end
REPRODUCIBILITY check of the pipeline at fixed seeds: if any kernel
change silently perturbs the RNG stream, the prior, the moves, the
optimizer, or the wire format, the fingerprint breaks loudly here
instead of six hours into a hero run.

First execution RECORDS the reference (prints "[4.2reg] RECORDED").
Later executions COMPARE and print [4.2reg] PASS / FAIL per field.
If the 4.2a protocol in exp_phase42_feynman.py is changed ON PURPOSE,
delete test_phase42_reference.json and re-record.

Run (module dir, MPI-capable env — spawn burn-in!):
    mpiexec -n 1 python test_phase42.py
"""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_SRC = os.path.join(os.path.dirname(_HERE), "src")
_EXP = os.path.join(os.path.dirname(_HERE), "experiments")
for _p in (_EXP, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("OMP_NUM_THREADS", "1")

# neutralize argv so the suite module's flag parsing sees nothing
sys.argv = ["test_phase42"]

import gp_movesC

import exp_phase42_feynman as m

REF_PATH = os.path.join(_HERE, "test_phase42_reference.json")
REL_TOL = 1e-6


def _close(a, b, tol=REL_TOL):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def main():
    gp_movesC.set_max_tree_depth(m.MAX_TREE_DEPTH)
    gp_movesC.set_subtree_rebuild_max_height(m.REBUILD_MAX_HEIGHT)

    # truncated budget — protocol otherwise IDENTICAL to the suite
    m.ROUNDS = 60
    m.WARMUP_STEPS = 40
    m.BURN_IN_WORKERS = 2

    tgt = next(t for t in m.TARGETS if t[0] == "flux")
    name, fid, f_true, truth_str, dom, noise_sd, atoms = tgt
    print("[4.2reg] rerunning %s [%s] seed 17 at %d rounds, %d workers"
          % (name, fid, m.ROUNDS, m.BURN_IN_WORKERS), flush=True)
    run = m.run_target(name, fid, f_true, truth_str, dom, noise_sd,
                       atoms, 17)

    fp = {
        "target": run["target"], "feynman_id": run["feynman_id"],
        "seed": run["seed"], "rounds": m.ROUNDS,
        "warmup_steps": m.WARMUP_STEPS,
        "burn_in_workers": m.BURN_IN_WORKERS,
        "burn_in_top": run["burn_in_top"],
        "best_tree": run["best_tree"],
        "best_params": {k: float(v)
                        for k, v in sorted(run["best_params"].items())},
        "best_posterior": run["best_posterior"],
        "mse_train": run["mse_train"],
        "nrmse_test": run["nrmse_test"],
    }

    if not os.path.exists(REF_PATH):
        with open(REF_PATH, "w") as fh:
            json.dump(fp, fh, indent=2)
        print("[4.2reg] RECORDED reference -> %s"
              % os.path.basename(REF_PATH))
        print("[4.2reg] rerun this test to verify reproducibility.")
        return 0

    with open(REF_PATH) as fh:
        ref = json.load(fh)

    failures = []
    for key in ("target", "feynman_id", "seed", "rounds", "warmup_steps",
                "burn_in_workers", "burn_in_top", "best_tree"):
        if fp[key] != ref[key]:
            failures.append("%s: %r != reference %r"
                            % (key, fp[key], ref[key]))
    for key in ("best_posterior", "mse_train", "nrmse_test"):
        if not _close(fp[key], ref[key]):
            failures.append("%s: %.12g != reference %.12g"
                            % (key, fp[key], ref[key]))
    if sorted(fp["best_params"]) != sorted(ref["best_params"]):
        failures.append("best_params keys: %s != %s"
                        % (sorted(fp["best_params"]),
                           sorted(ref["best_params"])))
    else:
        for k in fp["best_params"]:
            if not _close(fp["best_params"][k], ref["best_params"][k]):
                failures.append("best_params[%s]: %.12g != %.12g"
                                % (k, fp["best_params"][k],
                                   ref["best_params"][k]))

    if failures:
        print("[4.2reg] FAIL — %d fingerprint mismatch(es):"
              % len(failures))
        for f in failures:
            print("    " + f)
        return 1
    print("[4.2reg] PASS  best: %s  post=%.6f  nrmse=%.6f  "
          "(fingerprint matches reference)"
          % (fp["best_tree"], fp["best_posterior"], fp["nrmse_test"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
