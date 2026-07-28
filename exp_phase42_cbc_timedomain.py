"""Time-domain comparison: the ECCENTRIC CBC signal we generated vs the
CIRCULAR (eccentric-less) model the template represents.

Renders figs/fig20_cbc_timedomain.png:
  (a) h(t) overlay in the first 12 s from band entry (waveforms aligned
      at t=0): the eccentric signal visibly slips ~5 rad (~0.8 cycle)
      by the end of the window — the perturbation onset;
  (b) the same overlay 300 s into the observation: ~127 rad (~20
      cycles) of accumulated dephasing — the waveforms are completely
      decohered even though their envelopes are indistinguishable;
  (c) the accumulated dephasing delta-phi(t) over the whole inspiral
      (leading-order secular term, mapped through f(t)): ~1983 rad
      ~ 316 cycles by merger, most of it accumulated at low frequency
      — the 3G science case in the time domain.

Leading-order illustrative mapping: phi_ecc(t) = phi_circ(t) +
[dPhi(f(t)) - dPhi(f0)] with dPhi the SPA secular dephasing; Newtonian
chirp relations f(tau), phi(tau), A ~ f^(2/3). Constants imported from
exp_phase42_cbc so figure and experiment cannot drift apart.

Run any time (pure numpy/matplotlib): python exp_phase42_cbc_timedomain.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import exp_phase42_cbc as cbc

FIGS = os.path.join(_HERE, "results", "figs")

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
    "legend.frameon": False, "legend.fontsize": 8,
    "axes.titlesize": 9.5, "axes.titlecolor": INK,
})

MC = cbc.MC_S
E0 = 0.10
F0 = cbc.F0_HZ


def tau_of_f(f_hz):
    return 5.0 * MC * (8.0 * np.pi * MC * f_hz) ** (-8.0 / 3.0)


def f_of_tau(tau_s):
    return (1.0 / (8.0 * np.pi * MC)) * (5.0 * MC / tau_s) ** (3.0 / 8.0)


def phi_circ_t(tau_s):
    return -2.0 * (tau_s / (5.0 * MC)) ** (5.0 / 8.0)


def dephase(tau_s):
    """Accumulated eccentric dephasing since band entry (rad, >= 0)."""
    x = f_of_tau(tau_s) / F0
    return cbc.dphi_ecc(x, E0) - cbc.dphi_ecc(1.0, E0)


TAU0 = tau_of_f(F0)               # time to merger at band entry


def waveforms(t_start, t_len, n=3000):
    t = np.linspace(t_start, t_start + t_len, n)
    tau = TAU0 - t
    amp = (np.pi * MC * f_of_tau(tau)) ** (2.0 / 3.0)
    amp = amp / amp.max()
    # both waveforms share phase 0 at BAND ENTRY (not per window)
    ph_c = phi_circ_t(tau) - phi_circ_t(TAU0)
    h_c = amp * np.cos(ph_c)
    h_e = amp * np.cos(ph_c + dephase(tau))
    return t, h_c, h_e


def main():
    os.makedirs(FIGS, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.2))

    # (a) band entry
    t, h_c, h_e = waveforms(0.0, 12.0)
    ax = axes[0]
    ax.plot(t, h_c, color=S[0], lw=1.1, label="circular (template)")
    ax.plot(t, h_e, color=S[5], lw=1.1, ls=(0, (4, 2)),
            label="eccentric (e$_0$=0.1)")
    ax.set_title("band entry (f ≈ 5 Hz), aligned at t = 0 — the "
                 "eccentric signal slips ≈ %.1f rad in 12 s"
                 % abs(dephase(TAU0 - 12.0)))
    ax.legend(ncol=2, loc="upper right", frameon=True,
              facecolor=SURFACE, edgecolor=BASE, framealpha=0.95)
    ax.set_ylabel("h(t) [norm.]", fontsize=8)

    # (b) 300 s later
    t, h_c, h_e = waveforms(300.0, 12.0)
    ax = axes[1]
    ax.plot(t, h_c, color=S[0], lw=1.1)
    ax.plot(t, h_e, color=S[5], lw=1.1, ls=(0, (4, 2)))
    ax.set_title("300 s into the observation — ≈ %.0f rad (≈ %.0f "
                 "cycles) accumulated: same envelope, fully decohered"
                 % (abs(dephase(TAU0 - 300.0)),
                    abs(dephase(TAU0 - 300.0)) / (2 * np.pi)))
    ax.set_ylabel("h(t) [norm.]", fontsize=8)
    ax.set_xlabel("time since band entry [s]", fontsize=8)

    # (c) accumulated dephasing over the whole inspiral
    ax = axes[2]
    tt = np.geomspace(1.0, TAU0 - tau_of_f(200.0 * F0), 500)
    dp = np.abs(dephase(TAU0 - tt))
    ax.loglog(tt, dp, color=S[4], lw=1.8)
    total = abs(cbc.dphi_ecc(200.0, E0) - cbc.dphi_ecc(1.0, E0))
    ax.axhline(total, color=BASE, lw=1.0, ls=(0, (4, 3)))
    ax.text(tt[2], total * 1.25,
            "total ≈ %.0f rad ≈ %.0f cycles by 1 kHz" %
            (total, total / (2 * np.pi)), fontsize=8, color=INK2)
    ax.axhline(2 * np.pi, color=BASE, lw=0.8, ls=":")
    ax.text(tt[2], 2 * np.pi * 1.3, "one full cycle", fontsize=7,
            color=MUTED)
    ax.set_xlabel("time since band entry [s]  (merger at ≈ %.0f s)"
                  % TAU0, fontsize=8)
    ax.set_ylabel("accumulated dephasing [rad]", fontsize=8)
    ax.set_title("eccentric vs circular: accumulated phase difference "
                 "over the inspiral (leading-order secular term)")

    fig.suptitle("Time domain: the generated eccentric CBC signal vs "
                 "the circular template it perturbs "
                 "($\\mathcal{M}$=1.2 M$_\\odot$, e$_0$=0.1 at 5 Hz)",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(FIGS, "fig20_cbc_timedomain.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("wrote %s" % out)
    print("dephasing: 12 s -> %.2f rad; 300 s -> %.1f rad; total "
          "(to 1 kHz) -> %.0f rad (%.0f cycles)"
          % (abs(dephase(TAU0 - 12.0)), abs(dephase(TAU0 - 300.0)),
             total, total / (2 * np.pi)))


if __name__ == "__main__":
    main()
