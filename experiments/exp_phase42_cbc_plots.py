"""Phase 4.2b — CBC diagnostics: figures fig12-fig18 + tables.

Reads results/phase42_cbc.json (falls back to the smoke JSON if the
full suite has not run yet) and renders the full diagnostic panel:

  fig12_cbc_overlays.png     data / truth / best tree per run (semilogx)
  fig13_cbc_traces.png       best log-posterior per rung vs round
  fig14_cbc_exploration.png  per-rung unique structures + accepted moves
  fig15_cbc_depth.png        depth vs round per rung + pooled histogram
  fig16_cbc_polya.png        root-cell predictive shares vs round
  fig17_cbc_ladder.png       per-pair swap acceptance vs the 0.2-0.4 band
  fig18_cbc_exponent.png     fitted power terms vs the PN truths; the
                             correction coefficient vs e0 (slope-2 check)
  table_cbc.csv / .tex       run matrix
  phase42_cbc_expressions.md regressed expressions, params substituted

Pure numpy/sympy/matplotlib — runs anywhere. Colors: the dataviz
reference palette (same tokens as the other plot scripts). All names
NEW — nothing existing is overwritten.
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
DEPTH_SHADES = ["#e1e0d9", "#b9cfee", "#7fa8e0", "#2a78d6", "#123f78",
                "#0a2547"]

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

P_CIRC = -5.0 / 3.0
P_ECC = -34.0 / 9.0


def _load():
    for name in ("phase42_cbc.json", "phase42_cbc_smoke.json"):
        path = os.path.join(RES, name)
        if os.path.exists(path):
            with open(path) as fh:
                d = json.load(fh)
            print("  source: %s (%d runs)" % (name, len(d["runs"])))
            return d
    print("  [skip] no phase42_cbc JSON found")
    return None


def _truth_fn(run):
    """Rebuild the truth callable from the stored physics config."""
    e0 = run["e0"]
    B = _PHYS["B_phase_rad"]
    C = _PHYS["ecc_coef"] * e0 * e0 * B
    if run["stage"] == "A":
        return lambda x: C * x ** P_ECC
    return lambda x: B * x ** P_CIRC + C * x ** P_ECC


def _eval_tree(desc, params, x):
    import sympy as sp
    expr = sp.sympify(desc)
    subs = {sp.Symbol(k): float(v) for k, v in (params or {}).items()}
    expr = expr.subs(subs)
    f = sp.lambdify(sp.Symbol("x"), expr, "numpy")
    with np.errstate(all="ignore"):
        out = np.asarray(f(x), dtype=np.complex128)
    out = np.where(np.abs(out.imag) < 1e-12, out.real, np.nan)
    if out.shape != x.shape:
        out = np.full_like(x, float(out))
    return out.astype(np.float64)


def _run_label(r):
    return "%s s%d" % (r["target"], r["seed"])


def fig12_overlays(d):
    runs = d["runs"]
    n = len(runs)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(10.5, 3.4 * nrow))
    axes = np.atleast_2d(axes)
    cfg = d["config"]
    for ax, run in zip(axes.ravel(), runs):
        rng = np.random.default_rng(run["seed"])
        x = np.geomspace(*cfg["x_band"], num=cfg["n_points"])
        f = _truth_fn(run)
        y = f(x) + rng.normal(0.0, run["sigma"], x.size)
        xg = np.geomspace(*cfg["x_band"], num=400)
        ax.semilogx(x, y, "o", ms=2.4, mec="none", mfc=MUTED, alpha=0.5,
                    zorder=1)
        ax.semilogx(xg, f(xg), color=S[0], lw=2.0, zorder=2, label="truth")
        try:
            yfit = _eval_tree(run["best_tree"], run["best_params"], xg)
            ax.semilogx(xg, yfit, color=S[1], lw=2.0, ls=(0, (5, 2.5)),
                        zorder=3, label="best tree")
        except Exception as e:
            print("  [warn] eval failed for %s: %s" % (_run_label(run), e))
        flag = "recovered" if run["recovered"] else "missed"
        ax.set_title("%s — %s\nNRMSE %.4f, valid %.0f%%, p̂=%s"
                     % (_run_label(run), flag, run["nrmse_test"],
                        100 * run.get("valid_frac_test", 1.0),
                        "%.3f" % run["ecc_exponent_closest"]
                        if run.get("ecc_exponent_closest") is not None
                        else "—"), fontsize=8.5)
        ax.set_xlabel("x = f/f$_0$", fontsize=8)
        ax.tick_params(labelsize=7.5)
    for ax in axes.ravel()[n:]:
        ax.set_axis_off()
    axes.ravel()[0].legend(loc="best")
    axes.ravel()[0].set_ylabel("phase [rad]", fontsize=8)
    fig.suptitle("CBC eccentric dephasing (frequency domain): "
                 "best sampled tree vs truth", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(FIGS, "fig12_cbc_overlays.png"), dpi=160)
    plt.close(fig)
    print("  fig12_cbc_overlays.png")


def _per_rung_panels(d, key, fname, title, ylabel, delta_from_first):
    runs = [r for r in d["runs"] if (r.get("traces") or [])
            and key in r["traces"][0]]
    if not runs:
        print("  [skip] %s: no '%s' traces" % (fname, key))
        return
    temps = d["config"]["ladder"]
    n = len(runs)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(10.5, 3.2 * nrow))
    axes = np.atleast_2d(axes)
    for ax, run in zip(axes.ravel(), runs):
        traces = run["traces"]
        rounds = np.arange(1, len(traces) + 1) * 10
        M = np.array([t[key] for t in traces], dtype=np.float64)
        if delta_from_first:
            M = M - M[0]
        for g in range(M.shape[1]):
            cold = g == 0
            ax.plot(rounds, M[:, g], color=S[g % len(S)],
                    lw=2.2 if cold else 1.3,
                    alpha=1.0 if cold else 0.7)
        ax.set_title(_run_label(run), fontsize=8.5)
        ax.set_xlabel("round", fontsize=8)
        ax.tick_params(labelsize=7.5)
    for ax in axes.ravel()[n:]:
        ax.set_axis_off()
    axes.ravel()[0].set_ylabel(ylabel, fontsize=8)
    handles = [plt.Line2D([], [], color=S[g % len(S)],
                          lw=2.2 if g == 0 else 1.3,
                          label="T=%.2f" % temps[g])
               for g in range(len(temps))]
    fig.legend(handles=handles, loc="lower center", ncol=min(8, len(temps)),
               fontsize=7.5)
    fig.suptitle(title, fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(os.path.join(FIGS, fname), dpi=160)
    plt.close(fig)
    print("  %s" % fname)


def fig13_traces(d):
    _per_rung_panels(d, "best_post", "fig13_cbc_traces.png",
                     "Convergence: best log-posterior per rung (8 rungs)",
                     "best log-posterior", delta_from_first=False)


def fig14_exploration(d):
    runs = [r for r in d["runs"] if (r.get("traces") or [])
            and "unique_structs" in r["traces"][0]]
    if not runs:
        print("  [skip] fig14: no exploration traces")
        return
    temps = d["config"]["ladder"]
    picks = runs[:3]
    fig, axes = plt.subplots(2, len(picks), figsize=(10.5, 6.0))
    axes = np.atleast_2d(axes)
    for ci, run in enumerate(picks):
        traces = run["traces"]
        rounds = np.arange(1, len(traces) + 1) * 10
        U = np.array([t["unique_structs"] for t in traces])
        A = np.array([t["accepts"] for t in traces])
        A = A - A[0]
        for g in range(U.shape[1]):
            cold = g == 0
            kw = dict(color=S[g % len(S)], lw=2.2 if cold else 1.3,
                      alpha=1.0 if cold else 0.7)
            axes[0, ci].plot(rounds, U[:, g], **kw)
            axes[1, ci].plot(rounds, A[:, g], **kw)
        axes[0, ci].set_title(_run_label(run), fontsize=8.5)
        axes[1, ci].set_xlabel("round", fontsize=8)
        for ax in axes[:, ci]:
            ax.tick_params(labelsize=7.5)
    axes[0, 0].set_ylabel("unique structures\n(cumulative)", fontsize=8)
    axes[1, 0].set_ylabel("accepted moves\n(cumulative)", fontsize=8)
    handles = [plt.Line2D([], [], color=S[g % len(S)],
                          lw=2.2 if g == 0 else 1.3,
                          label="T=%.2f" % temps[g])
               for g in range(len(temps))]
    fig.legend(handles=handles, loc="lower center", ncol=min(8, len(temps)),
               fontsize=7.5)
    fig.suptitle("Exploration per rung (CBC suite)", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(os.path.join(FIGS, "fig14_cbc_exploration.png"), dpi=160)
    plt.close(fig)
    print("  fig14_cbc_exploration.png")


def fig15_depth(d):
    runs = [r for r in d["runs"] if (r.get("traces") or [])
            and "depths" in r["traces"][0]]
    if not runs:
        print("  [skip] fig15: no depth traces")
        return
    temps = d["config"]["ladder"]
    G = len(temps)
    n_snap = min(len(r["traces"]) for r in runs)
    rounds = np.arange(1, n_snap + 1) * 10
    mean_depth = np.zeros((n_snap, G))
    for r in runs:
        for i in range(n_snap):
            for g in range(G):
                mean_depth[i, g] += np.mean(r["traces"][i]["depths"][g])
    mean_depth /= len(runs)
    max_d = max(dd for r in runs for tr in r["traces"]
                for gd in tr["depths"] for dd in gd)
    counts = np.zeros((G, max_d + 1))
    for r in runs:
        for tr in r["traces"]:
            for g in range(G):
                for dd in tr["depths"][g]:
                    counts[g, min(dd, max_d)] += 1
    props = counts / counts.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for g in range(G):
        cold = g == 0
        axes[0].plot(rounds, mean_depth[:, g], color=S[g % len(S)],
                     lw=2.2 if cold else 1.3, alpha=1.0 if cold else 0.7,
                     label="T=%.2f" % temps[g])
    axes[0].set_xlabel("round", fontsize=8)
    axes[0].set_ylabel("population-mean tree depth", fontsize=8)
    axes[0].set_title("mean depth per rung vs round (all runs)")
    axes[0].legend(fontsize=6.5, ncol=2)
    xs = np.arange(G)
    bottom = np.zeros(G)
    for dd in range(max_d + 1):
        axes[1].bar(xs, props[:, dd], width=0.62, bottom=bottom,
                    color=DEPTH_SHADES[dd % len(DEPTH_SHADES)],
                    edgecolor="none", label="depth %d" % dd, zorder=3)
        bottom += props[:, dd]
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(["%.2f" % t for t in temps], fontsize=7)
    axes[1].set_xlabel("rung temperature", fontsize=8)
    axes[1].set_ylabel("share of sampled trees", fontsize=8)
    axes[1].set_title("depth distribution over the whole history")
    axes[1].legend(fontsize=6.5, ncol=2, loc="lower right")
    axes[1].grid(False)
    fig.suptitle("Tree-depth penalization during sampling (CBC suite)",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(FIGS, "fig15_cbc_depth.png"), dpi=160)
    plt.close(fig)
    print("  fig15_cbc_depth.png")


def fig16_polya(d):
    # root-cell order: full op universe, then primitives, then <param>,
    # then registered template atoms (frozen-vocabulary cell order)
    labels = (["+", "-", "*", "/", "o", "**"] + d["config"]["funcs"]
              + ["<param>"] + list(d["config"].get("atoms") or []))
    picks = [r for r in d["runs"]
             if (r.get("traces") or []) and "root_alphas" in r["traces"][0]]
    picks = picks[:2]
    if not picks:
        print("  [skip] fig16: no root_alphas traces")
        return
    fig, axes = plt.subplots(1, len(picks), figsize=(10.5, 3.8))
    axes = np.atleast_1d(axes)
    for ax, run in zip(axes, picks):
        A = np.array([t["root_alphas"] for t in run["traces"]])
        shares = A / A.sum(axis=1, keepdims=True)
        rounds = np.arange(1, A.shape[0] + 1) * 10
        order = np.argsort(shares[-1])[::-1]
        top = list(order[:6])
        for slot, idx in enumerate(top):
            nm = labels[idx] if idx < len(labels) else "cell%d" % idx
            ax.plot(rounds, shares[:, idx], color=S[slot % len(S)], lw=1.8,
                    label=nm)
        rest = [i for i in range(shares.shape[1]) if i not in top]
        if rest:
            ax.plot(rounds, shares[:, rest].sum(axis=1), color=MUTED,
                    lw=1.4, ls=":", label="other")
        ax.set_title(_run_label(run))
        ax.set_xlabel("round", fontsize=8)
        ax.legend(fontsize=7, ncol=2)
        ax.tick_params(labelsize=7.5)
    axes[0].set_ylabel("root-cell predictive share", fontsize=8)
    fig.suptitle("Pólya urn on the CBC vocabulary: what the prior learned",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(FIGS, "fig16_cbc_polya.png"), dpi=160)
    plt.close(fig)
    print("  fig16_cbc_polya.png")


def fig17_ladder(d):
    runs = [r for r in d["runs"] if r.get("ladder_stats")]
    if not runs:
        print("  [skip] fig17: no ladder_stats")
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
        ax.bar(xs, rates, width=w * 0.9, color=S[i % len(S)],
               edgecolor="none", label=_run_label(r), zorder=3)
    temps = runs[0]["ladder_stats"]["temperatures"]
    ax.set_xticks(np.arange(npairs))
    ax.set_xticklabels(["%.2f\n%.2f" % (temps[g], temps[g + 1])
                        for g in range(npairs)], fontsize=7)
    ax.set_ylabel("swap acceptance rate", fontsize=8)
    ax.set_xlabel("consecutive pair (T$_g$ / T$_{g+1}$)", fontsize=8)
    ax.legend(fontsize=7, ncol=3)
    ax.set_title("8-rung ladder transferability: per-pair swap acceptance "
                 "(band 0.2–0.4; finer Δβ ⇒ high side expected)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig17_cbc_ladder.png"), dpi=160)
    plt.close(fig)
    print("  fig17_cbc_ladder.png")


def fig18_exponent(d):
    runs = d["runs"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    # (a) fitted power terms per run vs the PN truths
    ax = axes[0]
    for i, r in enumerate(runs):
        terms = [t for t in r.get("power_terms", [])
                 if isinstance(t, dict) and t.get("power") is not None
                 and t.get("coeff") not in (None, 0.0)]
        for t in terms:
            ax.plot(i, t["power"], "o", ms=6,
                    mfc=S[1] if r["recovered"] else S[5], mec="none",
                    alpha=0.85, zorder=3)
    ax.axhline(P_CIRC, color=S[0], lw=1.2, ls=(0, (4, 3)), zorder=2)
    ax.axhline(P_ECC, color=S[4], lw=1.2, ls=(0, (4, 3)), zorder=2)
    ax.text(len(runs) - 0.4, P_CIRC + 0.06, "−5/3 (circular)", fontsize=7.5,
            color=S[0], ha="right")
    ax.text(len(runs) - 0.4, P_ECC + 0.06, "−34/9 (eccentric)", fontsize=7.5,
            color=S[4], ha="right")
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels([_run_label(r) for r in runs], fontsize=7,
                       rotation=20)
    ax.set_ylabel("fitted exponent p in c·x$^p$", fontsize=8)
    ax.set_title("power-law decomposition of the winners\n"
                 "(green = recovered run, red = missed)")
    # (b) correction coefficient vs e0 — expect slope 2 in log-log
    ax = axes[1]
    pts = []
    for r in runs:
        if r["stage"] not in ("B", "C"):
            continue
        cand = [t for t in r.get("power_terms", [])
                if isinstance(t, dict) and t.get("power") is not None
                and abs(t["power"] - P_ECC) < 0.5]
        if cand:
            c = min(cand, key=lambda t: abs(t["power"] - P_ECC))
            pts.append((r["e0"], abs(c["coeff"]), r["seed"]))
    if pts:
        e0s = np.array([p[0] for p in pts])
        cs = np.array([p[1] for p in pts])
        ax.loglog(e0s, cs, "o", ms=7, mfc=S[1], mec="none", zorder=3)
        phys = d["config"]["physics"]
        B = phys["B_phase_rad"] / phys.get("unit_rad", 1.0)
        coef = abs(phys["ecc_coef"]) * B
        e_ref = np.array([0.04, 0.25])
        ax.loglog(e_ref, coef * e_ref ** 2, color=INK2, lw=1.2,
                  ls=(0, (4, 3)), zorder=2, label="truth ∝ e$_0^2$")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no correction terms found",
                transform=ax.transAxes, ha="center", color=MUTED)
    ax.set_xlabel("e$_0$", fontsize=8)
    ax.set_ylabel("|fitted correction coefficient| [rad]", fontsize=8)
    ax.set_title("eccentric correction amplitude vs e$_0$\n"
                 "(the physics-validation check)")
    fig.suptitle("Did the sampler measure the PN signature of "
                 "eccentricity?", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(FIGS, "fig18_cbc_exponent.png"), dpi=160)
    plt.close(fig)
    print("  fig18_cbc_exponent.png")


def tables(d):
    rows = [(r["target"], r["stage"], r["e0"], r["seed"],
             "yes" if r["recovered"] else "no", r["nrmse_test"],
             r.get("valid_frac_test", 1.0),
             r.get("ecc_exponent_closest"), r["best_tree"])
            for r in d["runs"]]
    with open(os.path.join(RES, "table_cbc.csv"), "w") as fh:
        fh.write("target,stage,e0,seed,recovered,nrmse_test,valid_frac,"
                 "ecc_exponent_closest,best_tree\n")
        for row in rows:
            fh.write("%s,%s,%.2f,%d,%s,%.4f,%.3f,%s,\"%s\"\n"
                     % (row[0], row[1], row[2], row[3], row[4], row[5],
                        row[6],
                        "%.4f" % row[7] if row[7] is not None else "",
                        row[8]))
    with open(os.path.join(RES, "table_cbc.tex"), "w") as fh:
        fh.write("\\begin{tabular}{llclcrrl}\n\\hline\n"
                 "target & st. & $e_0$ & seed & rec. & NRMSE & "
                 "$\\hat p_{ecc}$ & best tree \\\\\n\\hline\n")
        for row in rows:
            fh.write("%s & %s & %.2f & %d & %s & %.4f & %s & "
                     "\\texttt{%s} \\\\\n"
                     % (row[0].replace("_", "\\_"), row[1], row[2], row[3],
                        row[4], row[5],
                        "%.3f" % row[7] if row[7] is not None else "—",
                        row[8].replace("_", "\\_")))
        fh.write("\\hline\n\\end{tabular}\n")
    md = os.path.join(RES, "phase42_cbc_expressions.md")
    with open(md, "w") as fh:
        fh.write("# CBC frequency-domain suite — regressed expressions\n\n"
                 "| target | stage | e0 | seed | rec. | NRMSE | "
                 "regressed expression |\n|---|---|---|---|---|---|---|\n")
        for r in d["runs"]:
            fh.write("| %s | %s | %.2f | %d | %s | %.4f | `%s` |\n"
                     % (r["target"], r["stage"], r["e0"], r["seed"],
                        "yes" if r["recovered"] else "NO",
                        r["nrmse_test"], r.get("expression_fitted", "?")))
    print("  table_cbc.csv / .tex + phase42_cbc_expressions.md "
          "(%d rows)" % len(rows))


_PHYS = None


def main():
    global _PHYS
    os.makedirs(FIGS, exist_ok=True)
    d = _load()
    if d is None:
        return
    _PHYS = d["config"]["physics"]
    print("rendering CBC diagnostics into %s" % FIGS)
    fig12_overlays(d)
    fig13_traces(d)
    fig14_exploration(d)
    fig15_depth(d)
    fig16_polya(d)
    fig17_ladder(d)
    fig18_exponent(d)
    tables(d)
    print("done.")


if __name__ == "__main__":
    main()
