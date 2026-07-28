"""CBC post-hoc diagnostics II — urn dynamics & mixing (fig23, fig24).

Reads ONLY the archived results/phase42_cbc.json (traces every 10
rounds) and renders two new figures:

  fig23_cbc_urn.png    the adaptive prior's own dynamics:
    (a) urn counter t vs round (reinforcement velocity, per run)
    (b) Pólya trie size vs round (construction-path diversity)
    (c) root-predictive Shannon entropy vs round (concentration)
    (d) ||q_t - q_{t-1}||_1 vs round, log scale — the EMPIRICAL
        COMPANION TO ASSUMPTION qconv (§9): does the root predictive
        stabilize as gamma_t decays?

  fig24_cbc_mixing.png sampler mixing health:
    (a) accepted moves per round, per rung (Stage B seed 17)
    (b) new unique structures per round, per rung (same run)
    (c) cold-hot best-posterior gap vs round (all runs)
    (d) cold-rung acceptance rate vs round (all runs)

Additive: new script, new figure names; nothing existing touched.
Run any time: python exp_phase42_cbc_diag2.py
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

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 9,
    "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": BASE, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 7.5,
    "axes.titlesize": 9.5, "axes.titlecolor": INK,
})

TRACE_EVERY = 10


def _label(r):
    return "%s s%d" % (r["target"].replace("phi_ecc_", "B/C "), r["seed"])


def main():
    os.makedirs(FIGS, exist_ok=True)
    with open(os.path.join(RES, "phase42_cbc.json")) as fh:
        d = json.load(fh)
    runs = [r for r in d["runs"] if r.get("traces")]
    temps = d["config"]["ladder"]
    G = len(temps)

    # ---------------- fig23: urn dynamics ----------------
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0))
    for i, r in enumerate(runs):
        tr = r["traces"]
        rounds = np.arange(1, len(tr) + 1) * TRACE_EVERY
        col = S[i % len(S)]
        t = np.array([s["polya_t"] for s in tr], dtype=float)
        nodes = np.array([s["polya_nodes"] for s in tr], dtype=float)
        A = np.array([s["root_alphas"] for s in tr], dtype=float)
        q = A / A.sum(axis=1, keepdims=True)
        ent = -np.sum(np.where(q > 0, q * np.log(q), 0.0), axis=1)
        dq = np.abs(np.diff(q, axis=0)).sum(axis=1)

        axes[0, 0].plot(rounds, t, color=col, lw=1.4, label=_label(r))
        axes[0, 1].plot(rounds, nodes, color=col, lw=1.4)
        axes[1, 0].plot(rounds, ent, color=col, lw=1.4)
        axes[1, 1].semilogy(rounds[1:], np.maximum(dq, 1e-12),
                            color=col, lw=1.2)

    axes[0, 0].set_title("urn counter t vs round — reinforcement "
                         "velocity (one increment per accepted move)")
    axes[0, 0].set_ylabel("Pólya t", fontsize=8)
    axes[0, 0].legend(ncol=2, fontsize=6.5)
    axes[0, 1].set_title("Pólya trie size vs round — construction-path "
                         "diversity")
    axes[0, 1].set_ylabel("trie nodes", fontsize=8)
    axes[1, 0].set_title("root-predictive Shannon entropy vs round — "
                         "prior concentration")
    axes[1, 0].set_ylabel("H(q$_t$) [nats]", fontsize=8)
    axes[1, 0].set_xlabel("round", fontsize=8)
    axes[1, 1].set_title("||q$_t$ − q$_{t-1}$||$_1$ vs round (log) — "
                         "empirical companion to Assumption qconv")
    axes[1, 1].set_ylabel("root-predictive movement", fontsize=8)
    axes[1, 1].set_xlabel("round", fontsize=8)
    fig.suptitle("The adaptive prior's own dynamics during the CBC "
                 "runs (continuous-adaptation regime)",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(FIGS, "fig23_cbc_urn.png"), dpi=160)
    plt.close(fig)
    print("  fig23_cbc_urn.png")

    # ---------------- fig24: mixing health ----------------
    hero = next(r for r in runs if r["target"] == "phi_ecc_e10"
                and r["seed"] == 17)
    tr = hero["traces"]
    rounds = np.arange(1, len(tr) + 1) * TRACE_EVERY
    ACC = np.array([s["accepts"] for s in tr], dtype=float)
    UNI = np.array([s["unique_structs"] for s in tr], dtype=float)
    acc_rate = np.diff(ACC, axis=0) / TRACE_EVERY
    nov_rate = np.diff(UNI, axis=0) / TRACE_EVERY

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0))
    for g in range(G):
        cold = g == 0
        kw = dict(color=S[g % len(S)], lw=1.8 if cold else 1.1,
                  alpha=1.0 if cold else 0.7,
                  label="T=%.2f" % temps[g])
        axes[0, 0].plot(rounds[1:], acc_rate[:, g], **kw)
        axes[0, 1].plot(rounds[1:], nov_rate[:, g], **kw)
    axes[0, 0].set_title("accepted moves per round, per rung "
                         "(Stage B, seed 17)")
    axes[0, 0].set_ylabel("accepts / round", fontsize=8)
    axes[0, 0].legend(ncol=2)
    axes[0, 1].set_title("NEW unique structures per round, per rung "
                         "(same run) — novelty rate")
    axes[0, 1].set_ylabel("new structures / round", fontsize=8)

    for i, r in enumerate(runs):
        tr_i = r["traces"]
        rd = np.arange(1, len(tr_i) + 1) * TRACE_EVERY
        M = np.array([s["best_post"] for s in tr_i], dtype=float)
        gap = M[:, 0] - M[:, -1]
        # log axis: the start-up transient (hot rungs briefly holding
        # penalty-scored trees, gaps ~1e7-1e8 nats) and the settled
        # regime (O(1-100) nats) differ by 6+ orders of magnitude
        axes[1, 0].semilogy(rd, np.maximum(gap, 1e-3),
                            color=S[i % len(S)], lw=1.3,
                            label=_label(r))
        ACCi = np.array([s["accepts"] for s in tr_i], dtype=float)
        axes[1, 1].plot(rd[1:], np.diff(ACCi[:, 0]) / TRACE_EVERY,
                        color=S[i % len(S)], lw=1.3)
    axes[1, 0].set_title("cold-minus-hot best-posterior gap vs round "
                         "(all runs) — ladder utilization")
    axes[1, 0].set_ylabel("best$_{cold}$ − best$_{hot}$ [nats]",
                          fontsize=8)
    axes[1, 0].set_xlabel("round", fontsize=8)
    axes[1, 0].legend(ncol=2, fontsize=6.5)
    axes[1, 1].set_title("cold-rung acceptance rate vs round (all runs)")
    axes[1, 1].set_ylabel("cold accepts / round", fontsize=8)
    axes[1, 1].set_xlabel("round", fontsize=8)
    fig.suptitle("Sampler mixing health during the CBC runs",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(FIGS, "fig24_cbc_mixing.png"), dpi=160)
    plt.close(fig)
    print("  fig24_cbc_mixing.png")


if __name__ == "__main__":
    main()
