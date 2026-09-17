"""The VISIBLE chirp: the last instants of the inspiral.

Renders figs/fig21_cbc_chirp.png (new name — overwrites nothing):
  (a) h(t) over the final ~2.2 s (100 Hz -> 1 kHz, ~130 cycles):
      amplitude grows as f^(2/3) while the cycles compress — the
      classic chirp morphology;
  (b) zoom on the final 0.15 s (~300 Hz -> 1 kHz, ~18 cycles): the
      sweep visible cycle by cycle;
  (c) the instantaneous frequency f(t) on the same axis as (a) —
      the "chirp" as a curve.

Single (circular) waveform: at these frequencies the eccentric
dephasing is < 0.03 rad — the binary has circularized, which is
exactly why eccentricity must be measured at the LOW-frequency end
(figs 19/20). Newtonian chirp, cut at the 1 kHz band edge.

Run any time (pure numpy/matplotlib): python exp_phase42_cbc_chirp.py
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
from exp_phase42_cbc_timedomain import (tau_of_f, f_of_tau, phi_circ_t,
                                        dephase, TAU0)

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

F_END = 1000.0                      # band edge
TAU_END = tau_of_f(F_END)


def h_window(tau_hi, tau_lo, n):
    """Waveform between times-to-merger tau_hi -> tau_lo (tau_hi > tau_lo).
    Time axis measured backwards from the 1 kHz cut ('time before cut')."""
    tau = np.linspace(tau_hi, tau_lo, n)
    f = f_of_tau(tau)
    amp = (np.pi * cbc.MC_S * f) ** (2.0 / 3.0)
    h = amp * np.cos(phi_circ_t(tau) - phi_circ_t(TAU0))
    t_axis = tau_hi - tau           # seconds since window start
    return t_axis, h / np.abs(h).max(), f


def main():
    os.makedirs(FIGS, exist_ok=True)
    tau_a = tau_of_f(100.0)         # ~2.2 s before the 1 kHz cut
    tau_b = tau_of_f(300.0)         # ~0.15 s before the cut

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.0))

    # (a) final ~2.2 s
    t, h, f = h_window(tau_a, TAU_END, 40000)
    axes[0].plot(t, h, color=S[0], lw=0.55)
    axes[0].set_title("the audible part: final %.2f s of the inspiral "
                      "(100 Hz → 1 kHz, ≈ %d cycles) — amplitude ∝ "
                      "f$^{2/3}$, cycles compressing"
                      % (tau_a - TAU_END,
                         int((phi_circ_t(TAU_END) - phi_circ_t(tau_a))
                             / (2 * np.pi))))
    axes[0].set_ylabel("h(t) [norm.]", fontsize=8)
    axes[0].set_xlim(t[0], t[-1])

    # (b) zoom: final 0.15 s
    t, h, f = h_window(tau_b, TAU_END, 12000)
    axes[1].plot(t, h, color=S[5], lw=0.9)
    axes[1].set_title("zoom: final %.3f s (≈ 300 Hz → 1 kHz, ≈ %d "
                      "cycles) — the chirp, cycle by cycle"
                      % (tau_b - TAU_END,
                         int((phi_circ_t(TAU_END) - phi_circ_t(tau_b))
                             / (2 * np.pi))))
    axes[1].set_ylabel("h(t) [norm.]", fontsize=8)
    axes[1].set_xlabel("time within window [s]", fontsize=8)
    axes[1].set_xlim(t[0], t[-1])

    # (c) instantaneous frequency over the same span as (a)
    t, h, f = h_window(tau_a, TAU_END, 2000)
    axes[2].plot(t, f, color=S[1], lw=1.8)
    axes[2].set_ylabel("GW frequency [Hz]", fontsize=8)
    axes[2].set_xlabel("time since window start [s]  (1 kHz cut at "
                       "the right edge; merger %.1f ms later)"
                       % (TAU_END * 1e3), fontsize=8)
    axes[2].set_title("f(t): the chirp as a curve — the sweep that "
                      "panels (a)/(b) trace")
    axes[2].set_xlim(t[0], t[-1])

    dp = abs(dephase(tau_a) - dephase(TAU_END))
    fig.suptitle("The late inspiral: where the chirp becomes visible\n"
                 "($\\mathcal{M}$=1.2 M$_\\odot$; eccentric dephasing "
                 "accrued in this window: only %.3f rad — circularized)"
                 % dp, fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(FIGS, "fig21_cbc_chirp.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("wrote %s" % out)
    print("window: %.3f s (100 Hz cut-in), zoom %.3f s; dephasing "
          "accrued in this window: %.4f rad"
          % (tau_a - TAU_END, tau_b - TAU_END, dp))


if __name__ == "__main__":
    main()
