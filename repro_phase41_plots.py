"""REPRO plotting + comparison for the Phase 4.1 rerun suites.

Renders the run-dependent figures from the _repro JSONs into a SEPARATE
directory (same filenames, different folder — nothing archived is
touched) and prints/writes a field-by-field comparison against the
archived originals:

  inputs   results/phase41_recovery_v2_repro.json
           results/phase41_depth3_repro.json          (either optional)
  figures  results/figs_repro/   (fig1/3/4 for depth-2; fig6/7/8 for
           depth-3 — rendered by the ORIGINAL plot code, redirected)
  tables   results/repro/        (table_recovery / table_depth3 etc.)
  report   results/repro/phase41_repro_report.md
           per-run diff: best_tree (exact), recovered flag, NRMSE and
           posterior (tol 1e-9 relative) — the reproducibility verdict.

NOT part of the default test suite — run on demand after the repro
suites: python repro_phase41_plots.py
"""

import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(_HERE, "results")
FIGS_REPRO = os.path.join(RES, "figs_repro")
OUT_DIR = os.path.join(RES, "repro")

import exp_phase41_plots as p1        # fig1/3/4 renderers (v2 suite)
import exp_phase41_plots_v3 as p3     # fig6/7/8 + tables (depth-3)

REL_TOL = 1e-9


def _load(name):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        print("  [skip] %s not found" % name)
        return None
    with open(path) as fh:
        return json.load(fh)


def _close(a, b, tol=REL_TOL):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def compare(tag, orig, repro, lines):
    if orig is None or repro is None:
        lines.append("## %s: SKIPPED (missing original or repro file)"
                     % tag)
        return None
    lines.append("## %s" % tag)
    lines.append("| target | seed | best_tree | recovered | NRMSE | "
                 "posterior |")
    lines.append("|---|---|---|---|---|---|")
    o = {(r["target"], r["seed"]): r for r in orig["runs"]}
    n_diff = 0
    for r in repro["runs"]:
        key = (r["target"], r["seed"])
        ref = o.get(key)
        if ref is None:
            lines.append("| %s | %d | NO ORIGINAL | — | — | — |" % key)
            n_diff += 1
            continue
        same_tree = r["best_tree"] == ref["best_tree"]
        same_rec = r["recovered"] == ref["recovered"]
        same_nrmse = _close(r["nrmse_test"], ref["nrmse_test"])
        same_post = _close(r["best_posterior"], ref["best_posterior"])
        ok = same_tree and same_rec and same_nrmse and same_post
        n_diff += 0 if ok else 1
        lines.append("| %s | %d | %s | %s | %s | %s |" % (
            key[0], key[1],
            "=" if same_tree else "DIFF: %r vs %r" % (r["best_tree"],
                                                      ref["best_tree"]),
            "=" if same_rec else "DIFF",
            "=" if same_nrmse else "DIFF: %.6g vs %.6g"
            % (r["nrmse_test"], ref["nrmse_test"]),
            "=" if same_post else "DIFF: %.6g vs %.6g"
            % (r["best_posterior"], ref["best_posterior"])))
    verdict = ("REPRODUCED (all %d runs identical)" % len(repro["runs"])
               if n_diff == 0 else
               "NOT REPRODUCED: %d run(s) differ — REGRESSION SIGNAL"
               % n_diff)
    lines.append("")
    lines.append("**%s verdict: %s**" % (tag, verdict))
    lines.append("")
    print("  %s: %s" % (tag, verdict))
    return n_diff


def main():
    os.makedirs(FIGS_REPRO, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    v2_orig = _load("phase41_recovery_v2.json")
    v2_rep = _load("phase41_recovery_v2_repro.json")
    d3_orig = _load("phase41_depth3.json")
    d3_rep = _load("phase41_depth3_repro.json")

    # --- figures/tables via the ORIGINAL renderers, output redirected --
    p1.FIGS = FIGS_REPRO
    p1.RES = OUT_DIR
    p3.FIGS = FIGS_REPRO
    p3.RES = OUT_DIR
    print("rendering repro figures into %s" % FIGS_REPRO)
    if v2_rep is not None:
        p1.fig1_overlays(v2_rep)
        p1.fig3_traces(v2_rep)
        p1.fig4_polya(v2_rep)
    if d3_rep is not None:
        p3.fig6_overlays(d3_rep)
        p3.fig7_exploration(d3_rep)
        p3.fig8_depth(d3_rep)
        p3.table_depth3(d3_rep)

    # --- the reproducibility report ------------------------------------
    lines = ["# Phase 4.1 reproducibility report", ""]
    n1 = compare("depth-2 (recovery v2)", v2_orig, v2_rep, lines)
    n2 = compare("depth-3 (discovery)", d3_orig, d3_rep, lines)
    report = os.path.join(OUT_DIR, "phase41_repro_report.md")
    with open(report, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("report written to %s" % report)
    if (n1 or 0) + (n2 or 0) > 0:
        print("!!! REGRESSION SIGNAL — inspect the report")
        return 1
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
