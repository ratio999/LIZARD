"""Phase 4.3 ablation figures (fig25-fig27) — from archived JSONs only.

  fig25_ablation_summary.png    (a) recoveries per arm (of 18), split
                                by suite; (b) target x arm recovery
                                heatmap — WHERE each component matters
  fig26_ablation_complexity.png winner complexity (k, depth) per arm —
                                the "BIC is what makes the output
                                symbolic" visual
  fig27_ablation_theta.png      the three-regime Theta dynamics on one
                                run: urn counter and root-predictive
                                entropy vs round for adaptive vs
                                frozen-at-warm-up vs frozen-at-birth

Additive: new script, new figure names. Baselines: the post-fix repro
reruns. Run any time: python exp_phase43_plots.py
"""

import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(_HERE, "results")
ABL = os.path.join(RES, "ablation_results")
FIGS = os.path.join(RES, "figs")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
S = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
     "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
RED = "#e34948"
GREEN = "#1baf7a"

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

ARMS = [
    ("adaptive", None,
     [os.path.join(RES, "phase41_recovery_v2_repro.json"),
      os.path.join(RES, "phase41_depth3_repro.json")]),
    ("frozen@Θ̄", "frozen",
     [os.path.join(ABL, "phase43_frozen_d2.json"),
      os.path.join(ABL, "phase43_frozen_d3.json")]),
    ("frozen@birth", "frozenbirth",
     [os.path.join(ABL, "phase43_frozenbirth_d2.json"),
      os.path.join(ABL, "phase43_frozenbirth_d3.json")]),
    ("no-temper", "notemper",
     [os.path.join(ABL, "phase43_notemper_d2.json"),
      os.path.join(ABL, "phase43_notemper_d3.json")]),
    ("no-BIC", "nobic",
     [os.path.join(ABL, "phase43_nobic_d2.json"),
      os.path.join(ABL, "phase43_nobic_d3.json")]),
    ("affine", "affine",
     [os.path.join(ABL, "phase43_affine_d2.json"),
      os.path.join(ABL, "phase43_affine_d3.json")]),
]

D2_TARGETS = ["linear", "sin_plus_linear", "x_sin", "rational",
              "poly2", "freq_sin_atom"]
D3_TARGETS = ["poly3", "inv_quad", "mixed"]
SEEDS = [17, 71]


def load_arm(paths):
    """-> (d2_runs, d3_runs) lists (either may be empty)."""
    out = []
    for p in paths:
        if os.path.exists(p):
            with open(p) as fh:
                out.append(json.load(fh)["runs"])
        else:
            out.append([])
    return out


def k_of(run):
    if "best_k" in run:
        return int(run["best_k"])
    return len(run.get("best_params") or {})


def depth_of(run):
    if "best_depth" in run:
        return int(run["best_depth"])
    return None


def main():
    os.makedirs(FIGS, exist_ok=True)
    data = {name: load_arm(paths) for name, _t, paths in ARMS}

    # ---------------- fig25: summary + heatmap ----------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6),
                             gridspec_kw={"width_ratios": [1, 1.35]})
    names = [n for n, _, _ in ARMS]
    d2c = [sum(1 for r in data[n][0] if r["recovered"]) for n in names]
    d3c = [sum(1 for r in data[n][1] if r["recovered"]) for n in names]
    xs = np.arange(len(names))
    axes[0].bar(xs, d2c, width=0.6, color=S[0], label="depth-2 (of 12)",
                zorder=3)
    axes[0].bar(xs, d3c, width=0.6, bottom=d2c, color=S[4],
                label="depth-3 (of 6)", zorder=3)
    ref = d2c[0] + d3c[0]
    for i, (a, b) in enumerate(zip(d2c, d3c)):
        tot = a + b
        lab = "%d (ref)" % tot if i == 0 else "%d (%+d)" % (tot,
                                                            tot - ref)
        axes[0].text(i, tot + 0.3, lab, ha="center", fontsize=8.5,
                     color=INK)
    axes[0].axhline(18, color=BASE, lw=0.8, ls=(0, (4, 3)))
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(names, fontsize=8, rotation=15)
    axes[0].set_ylabel("recovered runs (of 18)", fontsize=8)
    axes[0].set_ylim(0, 20.4)
    axes[0].legend(loc="lower left", fontsize=7.5, frameon=True,
                   facecolor=SURFACE, edgecolor=BASE, framealpha=0.95)
    axes[0].set_title("recoveries per arm (delta vs adaptive baseline)")

    # heatmap: rows = target/seed (18), cols = arms
    rows = [(t, s) for t in D2_TARGETS + D3_TARGETS for s in SEEDS]
    M = np.full((len(rows), len(names)), np.nan)
    for j, n in enumerate(names):
        runs = data[n][0] + data[n][1]
        idx = {(r["target"], r["seed"]): r for r in runs}
        for i, key in enumerate(rows):
            r = idx.get(key)
            if r is not None:
                M[i, j] = 1.0 if r["recovered"] else 0.0
    ax = axes[1]
    from matplotlib.colors import ListedColormap
    ax.imshow(M, aspect="auto", cmap=ListedColormap([RED, GREEN]),
              vmin=0, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8, rotation=15)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(["%s s%d" % k for k in rows], fontsize=6.5)
    ax.grid(False)
    for i in range(len(rows)):
        for j in range(len(names)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, "✓" if M[i, j] else "✗",
                        ha="center", va="center", fontsize=6,
                        color=SURFACE)
    ax.set_title("where each component matters (green = recovered)")
    fig.suptitle("Ablations: recovery by arm and by target "
                 "(baseline = adaptive, certified kernel)",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(FIGS, "fig25_ablation_summary.png"),
                dpi=160)
    plt.close(fig)
    print("  fig25_ablation_summary.png")

    # ---------------- fig26: complexity ----------------
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    rng = np.random.default_rng(0)
    for j, n in enumerate(names):
        runs = data[n][0] + data[n][1]
        ks = [k_of(r) for r in runs]
        ds = [depth_of(r) for r in runs if depth_of(r) is not None]
        xj = j + rng.uniform(-0.16, 0.16, len(ks))
        axes[0].plot(xj, ks, "o", ms=4, mfc=S[j % len(S)], mec="none",
                     alpha=0.75, zorder=3)
        axes[0].plot([j - 0.25, j + 0.25], [np.median(ks)] * 2,
                     color=INK, lw=1.6, zorder=4)
        if ds:
            xj = j + rng.uniform(-0.16, 0.16, len(ds))
            axes[1].plot(xj, ds, "o", ms=4, mfc=S[j % len(S)],
                         mec="none", alpha=0.75, zorder=3)
            axes[1].plot([j - 0.25, j + 0.25], [np.median(ds)] * 2,
                         color=INK, lw=1.6, zorder=4)
    for ax, lab in ((axes[0], "winner k (continuous parameters)"),
                    (axes[1], "winner tree depth")):
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=8, rotation=15)
        ax.set_ylabel(lab, fontsize=8)
    axes[0].set_title("parameter count of the winning tree, per arm\n"
                      "(bars = medians; no-BIC reaches k = 19, 20)")
    axes[1].set_title("depth of the winning tree, per arm\n"
                      "(adaptive baseline depth not stored — arms only)")
    fig.suptitle("What the parsimony term buys: winner complexity "
                 "under ablation", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(FIGS, "fig26_ablation_complexity.png"),
                dpi=160)
    plt.close(fig)
    print("  fig26_ablation_complexity.png")

    # ---------------- fig27: three-regime Theta dynamics ----------
    pick = ("mixed", 17)     # a 400-round depth-3 run present in all
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for name, col in (("adaptive", S[0]), ("frozen@Θ̄", S[1]),
                      ("frozen@birth", S[4])):
        runs = data[name][1]
        r = next((x for x in runs
                  if (x["target"], x["seed"]) == pick), None)
        if r is None or not r.get("traces"):
            continue
        tr = r["traces"]
        rounds = np.arange(1, len(tr) + 1) * 5
        t = np.array([s["polya_t"] for s in tr], dtype=float)
        A = np.array([s["root_alphas"] for s in tr], dtype=float)
        q = A / A.sum(axis=1, keepdims=True)
        ent = -np.sum(np.where(q > 0, q * np.log(q), 0.0), axis=1)
        axes[0].plot(rounds, t, color=col, lw=1.8, label=name)
        axes[1].plot(rounds, ent, color=col, lw=1.8, label=name)
    axes[0].set_ylabel("Pólya urn counter t", fontsize=8)
    axes[0].set_title("reinforcement: growing / halted at "
                      "Θ̄ / never started")
    axes[1].set_ylabel("root-predictive entropy H(q$_t$) [nats]",
                       fontsize=8)
    axes[1].set_title("prior concentration: adapting / frozen-informed "
                      "/ frozen-symmetric")
    for ax in axes:
        ax.set_xlabel("round", fontsize=8)
        ax.legend(fontsize=8)
    fig.suptitle("The three regimes in one run (%s, seed %d): what "
                 "each freeze point means for Θ" % pick,
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(FIGS, "fig27_ablation_theta.png"), dpi=160)
    plt.close(fig)
    print("  fig27_ablation_theta.png")


if __name__ == "__main__":
    main()
