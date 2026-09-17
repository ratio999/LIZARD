"""Phase 4.1 — NEW figures/exports for the depth-3 suite + expressions.

Additive companion to exp_phase41_plots.py: renders only NEW artifact
names (nothing from fig1-fig5 / table_recovery is touched):

  figs/fig6_depth3_overlays.png    best tree vs truth, 3 targets x 2 seeds
  figs/fig7_exploration.png        per-rung unique structures + accepted
                                   moves vs round (the non-max exploration
                                   picture fig3 could not show)
  figs/fig8_depth_penalization.png tree-depth statistics across the WHOLE
                                   sampling history, per temperature rung
  table_depth3.csv / .tex          depth-3 recovery matrix
  phase41_expressions.md / .csv    the regressed expression for EVERY
                                   testing case, all suites (v1, long,
                                   v2, depth-3), fitted params substituted

Pure numpy/sympy/matplotlib — runs anywhere. Colors: the dataviz
reference palette (same tokens as exp_phase41_plots.py).
"""

import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(_HERE, "results")
FIGS = os.path.join(RES, "figs")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
S = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
     "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
# sequential shades for depth 0..6+ (light -> dark)
DEPTH_SHADES = ["#e1e0d9", "#b9cfee", "#7fa8e0", "#2a78d6", "#123f78",
                "#0a2547", "#04101f"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 9,
    "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": BASE, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8,
    "axes.titlesize": 9.5, "axes.titlecolor": INK,
})


def _load(name):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        print("  [skip] %s not found" % name)
        return None
    with open(path) as fh:
        return json.load(fh)


TRUTHS3 = {
    "poly3": lambda x: x ** 3 - 2.0 * x,
    "inv_quad": lambda x: 3.0 * x / (1.0 + x * x),
    "mixed": lambda x: x * np.sin(x) + x / (1.0 + x),
}


def _eval_tree(desc, params, x):
    import sympy as sp
    expr = sp.sympify(desc)
    subs = {sp.Symbol(k): float(v) for k, v in (params or {}).items()}
    expr = expr.subs(subs)
    f = sp.lambdify(sp.Symbol("x"), expr, "numpy")
    out = np.asarray(f(x), dtype=np.float64)
    if out.shape != x.shape:
        out = np.full_like(x, float(out))
    return out


def _fitted(desc, params, ndigits=4):
    try:
        import sympy as sp
        expr = sp.sympify(desc)
        # fixed display precision — raw Python floats re-expand to 15
        # digits through arithmetic like x/a0 -> (1/a0)*x
        subs = {sp.Symbol(k): sp.Float(round(float(v), ndigits), 6)
                for k, v in (params or {}).items()}
        return str(expr.subs(subs))
    except Exception:
        return "%s  with  %s" % (desc, {k: round(float(v), ndigits)
                                        for k, v in (params or {}).items()})


def fig6_overlays(d3):
    cfg = d3["config"]
    doms = {t["name"]: t["domain"] for t in cfg["targets"]}
    order = [t["name"] for t in cfg["targets"]]
    seeds = cfg["seeds"]
    fig, axes = plt.subplots(len(seeds), len(order), figsize=(10.5, 6.2))
    axes = np.atleast_2d(axes)
    for si, seed in enumerate(seeds):
        for ti, name in enumerate(order):
            ax = axes[si, ti]
            run = next((r for r in d3["runs"]
                        if r["target"] == name and r["seed"] == seed), None)
            if run is None:
                ax.set_axis_off()
                continue
            dom = doms[name]
            rng = np.random.default_rng(seed)
            xd = np.linspace(dom[0], dom[1], cfg["n_points"])
            yd = TRUTHS3[name](xd) + rng.normal(0.0, cfg["noise_sd"],
                                                xd.size)
            xg = np.linspace(dom[0], dom[1], 300)
            ax.plot(xd, yd, "o", ms=2.6, mec="none", mfc=MUTED, alpha=0.55,
                    zorder=1)
            ax.plot(xg, TRUTHS3[name](xg), color=S[0], lw=2.0, zorder=2,
                    label="truth")
            try:
                yfit = _eval_tree(run["best_tree"], run["best_params"], xg)
                ax.plot(xg, yfit, color=S[1], lw=2.0, ls=(0, (5, 2.5)),
                        zorder=3, label="best tree")
            except Exception as e:
                print("  [warn] fit eval failed for %s/%d: %s"
                      % (name, seed, e))
            flag = "recovered" if run["recovered"] else "missed"
            ax.set_title("%s, seed %d — %s\nNRMSE %.3f, depth %d"
                         % (name, seed, flag, run["nrmse_test"],
                            run.get("best_depth", -1)), fontsize=8.5)
            ax.tick_params(labelsize=7.5)
    axes[0, 0].legend(loc="upper left")
    fig.suptitle("Depth-3 suite: targets outside the burn-in enumeration — "
                 "best sampled tree vs ground truth", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(FIGS, "fig6_depth3_overlays.png"), dpi=160)
    plt.close(fig)
    print("  fig6_depth3_overlays.png")


def fig7_exploration(d3):
    """Per-rung exploration: cumulative unique structures visited and
    cumulative accepted within-grove moves — the direct counters that
    fig3's running-max statistic hides."""
    cfg = d3["config"]
    temps = cfg["ladder"]
    seed = cfg["seeds"][0]
    runs = [r for r in d3["runs"] if r["seed"] == seed]
    fig, axes = plt.subplots(2, len(runs), figsize=(10.5, 6.0))
    axes = np.atleast_2d(axes)
    for ci, run in enumerate(runs):
        traces = run.get("traces") or []
        if not traces or "unique_structs" not in traces[0]:
            for ax in axes[:, ci]:
                ax.set_axis_off()
            continue
        rounds = np.arange(1, len(traces) + 1) * 5
        U = np.array([t["unique_structs"] for t in traces])   # (n, G)
        A = np.array([t["accepts"] for t in traces])          # (n, G)
        A = A - A[0]        # growth during the traced loop
        for g in range(U.shape[1]):
            cold = g == 0
            kw = dict(color=S[g % len(S)], lw=2.2 if cold else 1.4,
                      alpha=1.0 if cold else 0.75)
            axes[0, ci].plot(rounds, U[:, g], **kw)
            axes[1, ci].plot(rounds, A[:, g], **kw)
        axes[0, ci].set_title(run["target"])
        for ax in axes[:, ci]:
            ax.tick_params(labelsize=7.5)
        axes[1, ci].set_xlabel("round", fontsize=8)
    axes[0, 0].set_ylabel("unique structures visited\n(cumulative)",
                          fontsize=8)
    axes[1, 0].set_ylabel("accepted within-grove moves\n(cumulative)",
                          fontsize=8)
    handles = [plt.Line2D([], [], color=S[g % len(S)],
                          lw=2.2 if g == 0 else 1.4,
                          label="T=%.2f" % temps[g])
               for g in range(len(temps))]
    fig.legend(handles=handles, loc="lower center", ncol=len(temps),
               fontsize=8)
    fig.suptitle("Exploration per rung (depth-3 suite, seed %d): "
                 "every rung keeps moving — the flat cold lines in fig3 "
                 "are a max-statistic" % seed, fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(os.path.join(FIGS, "fig7_exploration.png"), dpi=160)
    plt.close(fig)
    print("  fig7_exploration.png")


def fig8_depth(d3):
    """How tree depth is penalized ACROSS THE WHOLE SAMPLING HISTORY:
    (a) population-mean depth per rung vs round, averaged over all runs;
    (b) the pooled depth distribution per rung (all snapshots, all runs,
        all slots) — the empirical complexity profile per temperature."""
    cfg = d3["config"]
    temps = cfg["ladder"]
    G = len(temps)
    runs = [r for r in d3["runs"] if (r.get("traces") or [])
            and "depths" in r["traces"][0]]
    if not runs:
        print("  [skip] fig8: no depth traces found")
        return
    n_snap = min(len(r["traces"]) for r in runs)
    rounds = np.arange(1, n_snap + 1) * 5

    # (a) mean depth per rung vs round, averaged over runs
    mean_depth = np.zeros((n_snap, G))
    for r in runs:
        for i in range(n_snap):
            for g in range(G):
                mean_depth[i, g] += np.mean(r["traces"][i]["depths"][g])
    mean_depth /= len(runs)

    # (b) pooled depth histogram per rung. Crossover grafts compound past
    # the per-move cap (depths up to 10 observed at hot rungs) — that
    # tail is part of the story, but 11 classes are unreadable, so
    # depths beyond the cap+1 fold into one top bucket.
    obs_max = max(d for r in runs for tr in r["traces"]
                  for gd in tr["depths"] for d in gd)
    cap = cfg.get("max_tree_depth", 4)
    max_d = min(obs_max, cap + 2)
    counts = np.zeros((G, max_d + 1))
    for r in runs:
        for tr in r["traces"]:
            for g in range(G):
                for d in tr["depths"][g]:
                    counts[g, min(d, max_d)] += 1
    props = counts / counts.sum(axis=1, keepdims=True)
    depth_labels = ["depth %d" % d for d in range(max_d)] + [
        "depth %d+" % max_d if obs_max > max_d else "depth %d" % max_d]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for g in range(G):
        cold = g == 0
        axes[0].plot(rounds, mean_depth[:, g], color=S[g % len(S)],
                     lw=2.2 if cold else 1.4, alpha=1.0 if cold else 0.75,
                     label="T=%.2f" % temps[g])
    axes[0].set_xlabel("round", fontsize=8)
    axes[0].set_ylabel("population-mean tree depth", fontsize=8)
    axes[0].set_title("mean depth per rung vs round\n"
                      "(averaged over %d runs)" % len(runs))
    axes[0].legend(fontsize=7, ncol=2)

    xs = np.arange(G)
    bottom = np.zeros(G)
    for d in range(max_d + 1):
        axes[1].bar(xs, props[:, d], width=0.62, bottom=bottom,
                    color=DEPTH_SHADES[min(d, len(DEPTH_SHADES) - 1)],
                    edgecolor="none", label=depth_labels[d], zorder=3)
        bottom += props[:, d]
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(["T=%.2f" % t for t in temps], fontsize=7.5)
    axes[1].set_ylabel("share of sampled trees", fontsize=8)
    axes[1].set_title("depth distribution over the whole history\n"
                      "(all rounds, runs, and slots pooled)")
    axes[1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    axes[1].grid(False)
    fig.suptitle("Tree-depth penalization during sampling "
                 "(depth-3 suite; prior + BIC vs tempered likelihood)",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(FIGS, "fig8_depth_penalization.png"), dpi=160)
    plt.close(fig)
    print("  fig8_depth_penalization.png")


def table_depth3(d3):
    rows = [(r["target"], r["seed"], "yes" if r["recovered"] else "no",
             r["nrmse_test"], r.get("best_depth", -1),
             "yes" if r.get("beyond_burn_in") else "no", r["best_tree"])
            for r in d3["runs"]]
    with open(os.path.join(RES, "table_depth3.csv"), "w") as fh:
        fh.write("target,seed,recovered,nrmse_test,best_depth,"
                 "beyond_burn_in,best_tree\n")
        for row in rows:
            fh.write("%s,%d,%s,%.4f,%d,%s,\"%s\"\n" % row)
    with open(os.path.join(RES, "table_depth3.tex"), "w") as fh:
        fh.write("\\begin{tabular}{lclrcll}\n\\hline\n"
                 "target & seed & rec. & NRMSE & depth & beyond & "
                 "best tree \\\\\n\\hline\n")
        for row in rows:
            fh.write("%s & %d & %s & %.4f & %d & %s & \\texttt{%s} \\\\\n"
                     % (row[0].replace("_", "\\_"), row[1], row[2], row[3],
                        row[4], row[5], row[6].replace("_", "\\_")))
        fh.write("\\hline\n\\end{tabular}\n")
    print("  table_depth3.csv / .tex  (%d rows)" % len(rows))


def expressions(suites):
    """The regressed expression for EVERY testing case, all suites."""
    md = os.path.join(RES, "phase41_expressions.md")
    csv = os.path.join(RES, "phase41_expressions.csv")
    n = 0
    with open(md, "w") as fm, open(csv, "w") as fc:
        fm.write("# Phase 4.1 — regressed expressions, all testing cases\n\n"
                 "Fitted parameter values substituted into the best "
                 "sampled tree (raw tree + parameter dict in the source "
                 "JSONs).\n")
        fc.write("suite,target,truth,seed,recovered,nrmse_test,"
                 "expression_fitted,best_tree\n")
        for tag, data in suites:
            if data is None:
                continue
            fm.write("\n## %s\n\n" % tag)
            fm.write("| target | truth | seed | rec. | NRMSE | "
                     "regressed expression |\n|---|---|---|---|---|---|\n")
            for r in data["runs"]:
                fitted = _fitted(r["best_tree"], r.get("best_params"))
                fm.write("| %s | `%s` | %d | %s | %.4f | `%s` |\n"
                         % (r["target"], r.get("truth", "?"), r["seed"],
                            "yes" if r["recovered"] else "NO",
                            r["nrmse_test"], fitted))
                fc.write("%s,%s,\"%s\",%d,%s,%.4f,\"%s\",\"%s\"\n"
                         % (tag, r["target"], r.get("truth", "?"),
                            r["seed"], "yes" if r["recovered"] else "no",
                            r["nrmse_test"], fitted, r["best_tree"]))
                n += 1
    print("  phase41_expressions.md / .csv  (%d cases)" % n)


def main():
    os.makedirs(FIGS, exist_ok=True)
    v1 = _load("phase41_recovery.json")
    vlong = _load("phase41_recovery_long.json")
    v2 = _load("phase41_recovery_v2.json")
    d3 = _load("phase41_depth3.json")

    print("rendering NEW figures into %s" % FIGS)
    if d3 is not None:
        fig6_overlays(d3)
        fig7_exploration(d3)
        fig8_depth(d3)
        table_depth3(d3)
    expressions([
        ("v1 (sigma = 1.0, random init)", v1),
        ("v1 --long (sigma = 1.0, 3x budget)", vlong),
        ("v2 (corrected protocol)", v2),
        ("depth-3 suite (corrected protocol, 400 rounds)", d3),
    ])
    print("done.")


if __name__ == "__main__":
    main()
