"""Eccentric vs circular in the LAST instants of the chirp — and their
difference. Renders figs/fig22_cbc_lastdiff.png (new name):

  (a) both waveforms in the final ~0.114 s, BAND-ENTRY ALIGNED (the
      matched-filter situation: template locked at 5 Hz): the eccentric
      signal arrives carrying the full accumulated dephasing — mod 2pi
      a large, visible offset;
  (b) the residual h_ecc - h_circ over the final ~2.2 s: order of the
      signal itself — what a circular-template filter fails to capture;
  (c) the same final window but LOCALLY RE-ALIGNED at the window start:
      the two curves are indistinguishable (the dephasing accrued
      within the window is ~0.024 rad) — the binary has circularized;
      the offset in (a) is inherited from low frequency, not generated
      here.

Run any time (pure numpy/matplotlib): python exp_phase42_cbc_lastdiff.py
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

F_END = 1000.0
TAU_END = tau_of_f(F_END)


def pair(tau_hi, tau_lo, n, local_align=False):
    tau = np.linspace(tau_hi, tau_lo, n)
    amp = (np.pi * cbc.MC_S * f_of_tau(tau)) ** (2.0 / 3.0)
    amp = amp / amp.max()
    ph = phi_circ_t(tau) - phi_circ_t(TAU0)
    d = dephase(tau)
    if local_align:
        d = d - d[0]
    return tau_hi - tau, amp * np.cos(ph), amp * np.cos(ph + d)


def main():
    os.makedirs(FIGS, exist_ok=True)
    tau_a = tau_of_f(100.0)          # ~2.2 s window
    tau_b = tau_of_f(300.0)          # ~0.114 s window

    total = abs(dephase(TAU_END))
    frac_cycle = (total / (2 * np.pi)) % 1.0
    offset_rad = frac_cycle * 2 * np.pi

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.2))

    # (a) band-entry aligned, final 0.114 s
    t, h_c, h_e = pair(tau_b, TAU_END, 12000, local_align=False)
    ax = axes[0]
    ax.plot(t, h_c, color=S[0], lw=1.0, label="circular (template)")
    ax.plot(t, h_e, color=S[5], lw=1.0, ls=(0, (4, 2)),
            label="eccentric (e$_0$=0.1)")
    ax.legend(ncol=2, loc="upper left", frameon=True, facecolor=SURFACE,
              edgecolor=BASE, framealpha=0.95)
    ax.set_title("final %.3f s, BAND-ENTRY ALIGNED: the eccentric "
                 "signal arrives shifted by the accumulated %.0f rad "
                 "— visible offset ≈ %.2f rad (%.2f cycle) mod 2π"
                 % (tau_b - TAU_END, total, offset_rad, frac_cycle))
    ax.set_ylabel("h(t) [norm.]", fontsize=8)
    ax.set_xlim(t[0], t[-1])

    # (b) the residual over the final 2.2 s
    t, h_c, h_e = pair(tau_a, TAU_END, 40000, local_align=False)
    ax = axes[1]
    ax.plot(t, h_e - h_c, color=S[4], lw=0.55)
    ax.set_title("the residual h$_{ecc}$(t) − h$_{circ}$(t) over the "
                 "final %.2f s — order of the signal itself: what a "
                 "circular-template matched filter leaves behind"
                 % (tau_a - TAU_END))
    ax.set_ylabel("residual [norm.]", fontsize=8)
    ax.set_xlim(t[0], t[-1])

    # (c) locally re-aligned, same final window
    t, h_c, h_e = pair(tau_b, TAU_END, 12000, local_align=True)
    ax = axes[2]
    ax.plot(t, h_c, color=S[0], lw=1.0)
    ax.plot(t, h_e, color=S[5], lw=1.0, ls=(0, (4, 2)))
    ax.set_title("same window, LOCALLY RE-ALIGNED at its start: "
                 "indistinguishable (only %.3f rad accrued here) — "
                 "the offset in (a) is inherited from low frequency"
                 % abs(dephase(tau_b) - dephase(TAU_END)))
    ax.set_ylabel("h(t) [norm.]", fontsize=8)
    ax.set_xlabel("time within window [s]  (1 kHz cut at right edge)",
                  fontsize=8)
    ax.set_xlim(t[0], t[-1])

    fig.suptitle("Where the eccentricity shows up in the last instants: "
                 "as an inherited phase shift, not a local distortion",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    out = os.path.join(FIGS, "fig22_cbc_lastdiff.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("wrote %s" % out)
    print("total dephasing %.1f rad = %.2f cycles; offset mod 2pi = "
          "%.2f rad" % (total, total / (2 * np.pi), offset_rad))


if __name__ == "__main__":
    main()
