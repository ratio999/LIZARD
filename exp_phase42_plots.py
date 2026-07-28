"""Phase 4.2a — Feynman-suite figures and tables (fig9-fig11).

Reads results/phase42_feynman.json (10 runs: 5 laws x 2 seeds; gauss
dropped by design — see the JSON's config.note) and renders:

  fig9_feynman_overlays.png   data / truth / best tree, 2 seeds x 5 laws
  fig10_feynman_polya.png     root-cell predictive shares on planck —
                              does the urn learn exp(x) where blackbody
                              physics needs it?
  fig11_feynman_ladder.png    per-pair swap acceptance for all 10 runs
                              vs the 0.2-0.4 band (ladder tuned at n=60
                              {sin,cos,x}; these runs are n=100 + exp —
                              the transferability check)
  table_feynman.csv / .tex    run matrix
  phase42_feynman_expressions.md  regressed expressions

Pure numpy/sympy/matplotlib; dataviz reference palette. All names NEW.
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
BAND_FILL = "#f0efec"

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

TRUTHS = {
    "flux": lambda x: 2.0 / (x * x),
    "scattering": lambda x: 1.2 * (1.0 + 0.7 * np.cos(x)),
    "velocity_add": lambda x: (0.5 + x) / (1.0 + 0.5 * x),
    "polarization": lambda x: 1.0 + x / (1.0 - x / 3.0),
    "planck": lambda x: x ** 3 / (np.exp(x) - 1.0),
}


def _load():
    path = os.path.join(RES, "phase42_feynman.json")
    if not os.path.exists(path):
        print("  [skip] phase42_feynman.json not found")
        return None
    with open(path) as fh:
        return json.load(fh)


def _eval_tree(desc, params, x):
    import sympy as sp
    expr = sp.sympify(desc)
    subs = {sp.Symbol(k): float(v) for k, v in (params or {}).items()}
    expr = expr.subs(subs)
    f = sp.lambdify(sp.Symbol("x"), expr, "numpy")
    with np.errstate(all="ignore"):
        out = np.asarray(f(x), dtype=np.float64)
    if out.shape != x.shape:
        out = np.full_like(x, float(out))
    return out


def fig9_overlays(d):
    # paper-format layout: one law per ROW, seed 17 left / seed 71 right
    # (single-column-figure aspect ratio)
    cfg = d["config"]
    order = [t["name"] for t in cfg["targets"]]
    meta = {t["name"]: t for t in cfg["targets"]}
    seeds = cfg["seeds"]
    fig, axes = plt.subplots(len(order), len(seeds),
                             figsize=(7.0, 2.1 * len(order)))
    axes = np.atleast_2d(axes)
    for ti, name in enumerate(order):
        for si, seed in enumerate(seeds):
            ax = axes[ti, si]
            run = next((r for r in d["runs"]
                        if r["target"] == name and r["seed"] == seed),
                       None)
            if run is None:
                ax.set_axis_off()
                continue
            dom = meta[name]["domain"]
            rng = np.random.default_rng(seed)
            xd = np.linspace(dom[0], dom[1], cfg["n_points"])
            yd = TRUTHS[name](xd) + rng.normal(
                0.0, meta[name]["noise_sd"], xd.size)
            xg = np.linspace(dom[0], dom[1], 400)
            ax.plot(xd, yd, "o", ms=1.9, mec="none", mfc=MUTED,
                    alpha=0.45, zorder=1)
            ax.plot(xg, TRUTHS[name](xg), color=S[0], lw=1.8, zorder=2,
                    label="truth")
            try:
                yfit = _eval_tree(run["best_tree"], run["best_params"],
                                  xg)
                ax.plot(xg, yfit, color=S[1], lw=1.8, ls=(0, (5, 2.5)),
                        zorder=3, label="best tree")
            except Exception as e:
                print("  [warn] eval failed %s/%d: %s" % (name, seed, e))
            flag = "recovered" if run["recovered"] else "missed"
            ax.set_title("%s, NRMSE %.3f" % (flag, run["nrmse_test"]),
                         fontsize=7.5, pad=3)
            ax.tick_params(labelsize=6.5)
            if si == 0:
                ax.set_ylabel("%s\n[%s]" % (name, run["feynman_id"]),
                              fontsize=8, color=INK)
    for si, seed in enumerate(seeds):
        axes[0, si].set_title("seed %d\n%s" %
                              (seed, axes[0, si].get_title()),
                              fontsize=8, pad=4)
    axes[0, 0].legend(loc="best", fontsize=6.5)
    for ax in axes[-1]:
        ax.set_xlabel("x", fontsize=8)
    fig.suptitle("Feynman univariate laws:\nbest sampled tree vs "
                 "ground truth", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig(os.path.join(FIGS, "fig9_feynman_overlays.png"), dpi=160)
    plt.close(fig)
    print("  fig9_feynman_overlays.png")


def fig10_polya(d):
    labels = (["+", "-", "*", "/", "o", "**"] + d["config"]["funcs"]
              + ["<param>"])
    picks = [r for r in d["runs"] if r["target"] == "planck"
             and (r.get("traces") or []) and "root_alphas" in
             r["traces"][0]]
    if not picks:
        print("  [skip] fig10: no planck root_alphas traces")
        return
    fig, axes = plt.subplots(1, len(picks), figsize=(10.5, 3.8))
    axes = np.atleast_1d(axes)
    for ax, run in zip(axes, picks):
        A = np.array([t["root_alphas"] for t in run["traces"]])
        shares = A / A.sum(axis=1, keepdims=True)
        rounds = np.arange(1, A.shape[0] + 1) * 10
        # top-4 cells by final share PLUS the three function cells —
        # the root cell tracks ROOT choices (operators win there for
        # rational-exp forms); the function-cell comparison shows
        # whether exp(x) outgrew its sin/cos peers at the root level.
        order = [i for i in np.argsort(shares[-1])[::-1]]
        funcs_idx = [labels.index(f) for f in d["config"]["funcs"]
                     if f != "x"]
        top = list(dict.fromkeys(list(order[:4]) + funcs_idx))
        for slot, idx in enumerate(top):
            nm = labels[idx] if idx < len(labels) else "cell%d" % idx
            lw = 2.4 if nm == "exp(x)" else 1.5
            ax.plot(rounds, shares[:, idx], color=S[slot % len(S)],
                    lw=lw, label=nm)
        rest = [i for i in range(shares.shape[1]) if i not in top]
        if rest:
            ax.plot(rounds, shares[:, rest].sum(axis=1), color=MUTED,
                    lw=1.3, ls=":", label="other")
        ax.set_title("planck [I.41.16], seed %d" % run["seed"])
        ax.set_xlabel("round", fontsize=8)
        ax.legend(fontsize=6.5, ncol=2)
        ax.tick_params(labelsize=7.5)
        ax.set_yscale("log")
    axes[0].set_ylabel("root-cell predictive share (log)", fontsize=8)
    fig.suptitle("Pólya urn on blackbody data: root-cell shares — "
                 "operator grammar reinforced; exp(x) vs its sin/cos "
                 "peers", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(FIGS, "fig10_feynman_polya.png"), dpi=160)
    plt.close(fig)
    print("  fig10_feynman_polya.png")


def fig11_ladder(d):
    runs = [r for r in d["runs"] if r.get("ladder_stats")]
    if not runs:
        print("  [skip] fig11: no ladder_stats")
        return
    n = len(runs)
    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    ax.axhspan(0.2, 0.4, color=BAND_FILL, zorder=0)
    ax.axhline(0.2, color=BASE, lw=0.8, zorder=1)
    ax.axhline(0.4, color=BASE, lw=0.8, zorder=1)
    npairs = len(runs[0]["ladder_stats"]["swap_rates"])
    w = 0.8 / n
    for i, r in enumerate(runs):
        rates = r["ladder_stats"]["swap_rates"]
        xs = np.arange(npairs) + (i - n / 2 + 0.5) * w
        ax.bar(xs, rates, width=w * 0.9,
               color=S[i // 2 % len(S)], alpha=1.0 if i % 2 == 0 else 0.55,
               edgecolor="none",
               label="%s s%d" % (r["target"], r["seed"]), zorder=3)
    temps = runs[0]["ladder_stats"]["temperatures"]
    ax.set_xticks(np.arange(npairs))
    ax.set_xticklabels(["%.2f\n%.2f" % (temps[g], temps[g + 1])
                        for g in range(npairs)], fontsize=7)
    ax.set_ylabel("swap acceptance rate", fontsize=8)
    ax.set_xlabel("consecutive pair (T$_g$ / T$_{g+1}$)", fontsize=8)
    ax.legend(fontsize=6.5, ncol=5)
    ax.set_title("Ladder transferability: tuned at n=60 {sin,cos,x}, "
                 "run at n=100 + exp — per-pair swap acceptance, "
                 "all 10 runs")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig11_feynman_ladder.png"), dpi=160)
    plt.close(fig)
    print("  fig11_feynman_ladder.png")


def tables(d):
    rows = [(r["target"], r["feynman_id"], r["seed"],
             "yes" if r["recovered"] else "no", r["nrmse_test"],
             r.get("expression_fitted", r["best_tree"]))
            for r in d["runs"]]
    with open(os.path.join(RES, "table_feynman.csv"), "w") as fh:
        fh.write("target,feynman_id,seed,recovered,nrmse_test,"
                 "expression_fitted\n")
        for row in rows:
            fh.write("%s,%s,%d,%s,%.4f,\"%s\"\n" % row)
    with open(os.path.join(RES, "table_feynman.tex"), "w") as fh:
        fh.write("\\begin{tabular}{llclrl}\n\\hline\n"
                 "law & Feynman & seed & rec. & NRMSE & "
                 "regressed expression \\\\\n\\hline\n")
        for row in rows:
            fh.write("%s & %s & %d & %s & %.4f & \\texttt{%s} \\\\\n"
                     % (row[0].replace("_", "\\_"), row[1], row[2],
                        row[3], row[4], row[5].replace("_", "\\_")))
        fh.write("\\hline\n\\end{tabular}\n")
    md = os.path.join(RES, "phase42_feynman_expressions.md")
    with open(md, "w") as fh:
        fh.write("# Feynman suite — regressed expressions (gauss "
                 "dropped by design)\n\n"
                 "| law | Feynman ID | seed | rec. | NRMSE | "
                 "regressed expression |\n|---|---|---|---|---|---|\n")
        for row in rows:
            fh.write("| %s | %s | %d | %s | %.4f | `%s` |\n" % row)
    print("  table_feynman.csv / .tex + phase42_feynman_expressions.md "
          "(%d rows)" % len(rows))


def main():
    os.makedirs(FIGS, exist_ok=True)
    d = _load()
    if d is None:
        return
    print("rendering Feynman figures into %s" % FIGS)
    fig9_overlays(d)
    fig10_polya(d)
    fig11_ladder(d)
    tables(d)
    print("done.")


if __name__ == "__main__":
    main()
