"""Phase 4.1 — figures and tables from the archived experiment JSONs.

Reads results/phase41_*.json and renders paper-grade PNGs into
results/figs/ plus CSV/LaTeX table exports. Pure numpy/sympy/matplotlib —
no compiled modules needed, so it runs anywhere.

Figures
  fig1_recovery_overlays.png   best-fit vs truth vs data, per target (v2)
  fig2_ladder_rates.png        old-vs-retuned ladder under corrected sigma
  fig3_convergence_traces.png  best log-posterior per rung vs round (v2)
  fig4_polya_evolution.png     root-cell predictive shares vs round (v2)
  fig5_sigma_calibration.png   selection pressure in nats, sigma=1 vs 0.1
Tables
  table_recovery.csv / .tex    v1 vs v2 recovery matrix

Colors: the dataviz reference palette (validated instance) — categorical
slots in fixed order, chrome/ink tokens for text and axes.
"""

import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(_HERE, "results")
FIGS = os.path.join(RES, "figs")

# --- dataviz reference palette (light mode) --------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
S = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
     "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]   # fixed slot order
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


def _load(name):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        print("  [skip] %s not found" % name)
        return None
    with open(path) as fh:
        return json.load(fh)


# Truth callables (must match exp_phase41_recovery.py)
TRUTHS = {
    "linear": lambda x: 0.5 * x,
    "sin_plus_linear": lambda x: np.sin(x) + 0.5 * x,
    "x_sin": lambda x: x * np.sin(x),
    "rational": lambda x: x / (1.0 + x),
    "poly2": lambda x: x * x - x,
    "freq_sin_atom": lambda x: 2.3 * np.sin(1.3 * x),
}


def _eval_tree(desc, params, x):
    """Evaluate a stored best-tree description at its stored parameters."""
    import sympy as sp
    expr = sp.sympify(desc)
    subs = {sp.Symbol(k): float(v) for k, v in (params or {}).items()}
    expr = expr.subs(subs)
    f = sp.lambdify(sp.Symbol("x"), expr, "numpy")
    out = np.asarray(f(x), dtype=np.float64)
    if out.shape != x.shape:                 # constant expression
        out = np.full_like(x, float(out))
    return out


def fig1_overlays(v2):
    runs = [r for r in v2["runs"] if r["seed"] == v2["config"]["seeds"][0]]
    doms = {t["name"]: t["domain"] for t in v2["config"]["targets"]}
    n = len(runs)
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2))
    for ax, run in zip(axes.ravel(), runs):
        name = run["target"]
        dom = doms[name]
        rng = np.random.default_rng(run["seed"])
        xd = np.linspace(dom[0], dom[1], v2["config"]["n_points"])
        yd = TRUTHS[name](xd) + rng.normal(0.0, v2["config"]["noise_sd"],
                                           xd.size)
        xg = np.linspace(dom[0], dom[1], 300)
        ax.plot(xd, yd, "o", ms=2.6, mec="none", mfc=MUTED, alpha=0.55,
                zorder=1)
        ax.plot(xg, TRUTHS[name](xg), color=S[0], lw=2.0, zorder=2,
                label="truth")
        try:
            yfit = _eval_tree(run["best_tree"], run["best_params"], xg)
            ax.plot(xg, yfit, color=S[1], lw=2.0, ls=(0, (5, 2.5)),
                    zorder=3, label="best tree")
        except Exception as e:
            print("  [warn] fit eval failed for %s: %s" % (name, e))
        flag = "recovered" if run["recovered"] else "missed"
        ax.set_title("%s — %s, NRMSE %.3f" % (name, flag,
                                              run["nrmse_test"]))
        ax.tick_params(labelsize=7.5)
    axes[0, 0].legend(loc="upper left")
    fig.suptitle("Phase 4.1 recovery (corrected protocol, seed %d): "
                 "best sampled tree vs ground truth"
                 % v2["config"]["seeds"][0], fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(FIGS, "fig1_recovery_overlays.png"), dpi=160)
    plt.close(fig)
    print("  fig1_recovery_overlays.png")


def _rate_panel(ax, stats, title, color, label=None, x_offset=0.0, w=0.34):
    temps = stats["temperatures"]
    rates = stats["swap_rates"]
    xs = np.arange(len(rates)) + x_offset
    ax.bar(xs, rates, width=w, color=color, edgecolor="none", label=label,
           zorder=3)
    ax.axhspan(0.2, 0.4, color=BAND_FILL, zorder=0)
    ax.axhline(0.2, color=BASE, lw=0.8, zorder=1)
    ax.axhline(0.4, color=BASE, lw=0.8, zorder=1)
    ax.set_xticks(np.arange(len(rates)))
    ax.set_xticklabels(["%.2f\n%.2f" % (temps[g], temps[g + 1])
                        for g in range(len(rates))], fontsize=7)
    ax.set_ylim(0, 0.75)
    ax.set_title(title)


def fig2_ladders(r6, r7, r8):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    if r6 is not None:
        old = [r for r in r6["runs"] if r["label"].startswith("old")]
        if old:
            _rate_panel(axes[0], old[0],
                        "old $\\sigma$=1-tuned ladder under corrected "
                        "$\\sigma$ — cold pairs sag", S[5])
    if r7 is not None:
        win = [r for r in r7["runs"] if "d0.18" in r["label"]]
        if win:
            _rate_panel(axes[1], win[0], "re-tuned ladder "
                        "(uniform-$\\beta$ $\\Delta\\beta$=0.18)",
                        S[0], label="seed 4141", x_offset=-0.18)
    if r8 is not None and r8["runs"]:
        _rate_panel(axes[1], r8["runs"][0], axes[1].get_title(), S[1],
                    label="seed 999", x_offset=0.22)
    axes[1].legend(loc="upper right")
    for ax in axes:
        ax.set_ylabel("swap acceptance rate", fontsize=8)
        ax.set_xlabel("consecutive pair (T$_g$ / T$_{g+1}$)", fontsize=8)
    fig.suptitle("PT ladder swap acceptance per pair — target band 0.2–0.4",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(FIGS, "fig2_ladder_rates.png"), dpi=160)
    plt.close(fig)
    print("  fig2_ladder_rates.png")


def fig3_traces(v2):
    runs = [r for r in v2["runs"] if r["seed"] == v2["config"]["seeds"][0]]
    temps = v2["config"]["ladder"]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.0))
    for ax, run in zip(axes.ravel(), runs):
        traces = run.get("traces") or []
        if not traces:
            continue
        rounds = np.arange(1, len(traces) + 1) * 5
        M = np.array([t["best_post"] for t in traces])   # (n, G)
        for g in range(M.shape[1]):
            cold = g == 0
            ax.plot(rounds, M[:, g],
                    color=S[g % len(S)], lw=2.2 if cold else 1.4,
                    alpha=1.0 if cold else 0.75)
        ax.set_title(run["target"])
        ax.tick_params(labelsize=7.5)
    for ax in axes[1]:
        ax.set_xlabel("round", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("best log-posterior", fontsize=8)
    handles = [plt.Line2D([], [], color=S[g % len(S)],
                          lw=2.2 if g == 0 else 1.4,
                          label="T=%.2f" % temps[g])
               for g in range(len(temps))]
    fig.legend(handles=handles, loc="lower center", ncol=len(temps),
               fontsize=8)
    fig.suptitle("Convergence: best log-posterior per rung "
                 "(corrected protocol, seed %d)" % v2["config"]["seeds"][0],
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(os.path.join(FIGS, "fig3_convergence_traces.png"), dpi=160)
    plt.close(fig)
    print("  fig3_convergence_traces.png")


def fig4_polya(v2):
    # root-cell predictive shares over rounds, one representative run each
    # for a plain-vocab and the atom target
    ops = ["+", "-", "*", "/", "o", "**"]
    funcs = v2["config"]["funcs"]
    picks = [r for r in v2["runs"] if r["seed"] == v2["config"]["seeds"][0]
             and r["target"] in ("sin_plus_linear", "freq_sin_atom")]
    fig, axes = plt.subplots(1, len(picks), figsize=(10.5, 3.8))
    if len(picks) == 1:
        axes = [axes]
    for ax, run in zip(axes, picks):
        traces = run.get("traces") or []
        if not traces or "root_alphas" not in traces[0]:
            continue
        labels = list(ops) + list(funcs) + ["<param>"]
        if run["target"] == "freq_sin_atom":
            labels = labels + ["Asin"]
        A = np.array([t["root_alphas"] for t in traces])   # (n, cells)
        shares = A / A.sum(axis=1, keepdims=True)
        rounds = np.arange(1, A.shape[0] + 1) * 5
        # top cells by final share; fold the rest into "other"
        order = np.argsort(shares[-1])[::-1]
        top = list(order[:6])
        slot = 0
        for idx in top:
            nm = labels[idx] if idx < len(labels) else "cell%d" % idx
            ax.plot(rounds, shares[:, idx], color=S[slot % len(S)], lw=1.8,
                    label=nm)
            slot += 1
        rest = [i for i in range(shares.shape[1]) if i not in top]
        if rest:
            ax.plot(rounds, shares[:, rest].sum(axis=1), color=MUTED,
                    lw=1.4, ls=":", label="other")
        ax.set_title(run["target"])
        ax.set_xlabel("round", fontsize=8)
        ax.legend(fontsize=7, ncol=2)
        ax.tick_params(labelsize=7.5)
    axes[0].set_ylabel("root-cell predictive share", fontsize=8)
    fig.suptitle("Pólya urn: what the prior learned "
                 "(root Dirichlet shares over the run)",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(FIGS, "fig4_polya_evolution.png"), dpi=160)
    plt.close(fig)
    print("  fig4_polya_evolution.png")


def fig5_sigma():
    # selection pressure (nats of log-likelihood separating the true model
    # from the best constant) vs the 2-nat/parameter BIC cost, on the
    # rational target: Delta_mse ~ 0.04, n = 60.
    n, dmse = 60, 0.04
    sig = [1.0, 0.1]
    nats = [0.5 * n * dmse / (s * s) for s in sig]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    xs = np.arange(2)
    ax.bar(xs, nats, width=0.5, color=[S[5], S[0]], edgecolor="none",
           zorder=3)
    ax.axhline(0.5 * np.log(n), color=INK2, lw=1.2, ls=(0, (4, 3)),
               zorder=4)
    ax.text(-0.38, 0.5 * np.log(n) * 1.3, "BIC cost of one\nparameter "
            "(~2 nats)", fontsize=8, color=INK2, ha="left")
    for x, v in zip(xs, nats):
        ax.text(x, v * 1.15, "%.1f nats" % v, ha="center", fontsize=9,
                color=INK)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(["$\\sigma$ = 1.0\n(mis-calibrated)",
                        "$\\sigma$ = 0.1\n(= true noise)"], fontsize=9)
    ax.set_ylabel("log-likelihood gain of the true model", fontsize=8)
    ax.set_title("Why fixed $\\sigma$=1 scoring loses low-amplitude "
                 "structure\n(rational target: truth vs best constant)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig5_sigma_calibration.png"), dpi=160)
    plt.close(fig)
    print("  fig5_sigma_calibration.png")


def tables(v1, v2):
    rows = []
    for tag, data in (("v1 sigma=1.0", v1), ("v2 corrected", v2)):
        if data is None:
            continue
        for r in data["runs"]:
            rows.append((tag, r["target"], r["seed"],
                         "yes" if r["recovered"] else "no",
                         r["nrmse_test"], r["best_tree"]))
    csv_path = os.path.join(RES, "table_recovery.csv")
    with open(csv_path, "w") as fh:
        fh.write("protocol,target,seed,recovered,nrmse_test,best_tree\n")
        for row in rows:
            fh.write("%s,%s,%d,%s,%.4f,\"%s\"\n" % row)
    tex_path = os.path.join(RES, "table_recovery.tex")
    with open(tex_path, "w") as fh:
        fh.write("\\begin{tabular}{llclrl}\n\\hline\n"
                 "protocol & target & seed & rec. & NRMSE & best tree \\\\\n"
                 "\\hline\n")
        for row in rows:
            fh.write("%s & %s & %d & %s & %.4f & \\texttt{%s} \\\\\n"
                     % (row[0], row[1].replace("_", "\\_"), row[2], row[3],
                        row[4], row[5].replace("_", "\\_")))
        fh.write("\\hline\n\\end{tabular}\n")
    print("  table_recovery.csv / .tex  (%d rows)" % len(rows))


def main():
    os.makedirs(FIGS, exist_ok=True)
    v1 = _load("phase41_recovery.json")
    v2 = _load("phase41_recovery_v2.json")
    r6 = _load("phase41_ladder_round6.json")
    r7 = _load("phase41_ladder_round7.json")
    r8 = _load("phase41_ladder_round8.json")

    print("rendering figures into %s" % FIGS)
    if v2 is not None:
        fig1_overlays(v2)
        fig3_traces(v2)
        fig4_polya(v2)
    fig2_ladders(r6, r7, r8)
    fig5_sigma()
    tables(v1, v2)
    print("done.")


if __name__ == "__main__":
    main()
