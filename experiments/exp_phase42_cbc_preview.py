"""Phase 4.2b — CBC data preview (run BEFORE the suite).

Renders figs/fig19_cbc_data.png: exactly the data the suite will
regress (same constants, same seed-17 noise draw), plus the physical
context — the time-frequency chirp track and the time-domain waveform
h(t) whose PHASING is the regression target:

  (a) the frequency-domain phase Phi(f): circular vs eccentric + the
      noisy data points (Stage-B target)
  (b) the eccentric dephasing |dPhi(f)| (Stage-A target) against the
      x**(-34/9) truth and the sigma = 1 rad noise floor
  (c) the Newtonian time-frequency track f(t): a ~1.8 h signal from
      5 Hz (3G) vs the last ~3 min above 20 Hz (current detectors)
  (d) h(t) snippets, early (f ~ 5 Hz) and final 0.35 s (200 Hz -> 1 kHz)

Pure numpy/matplotlib for the physics; imports exp_phase42_cbc for the
constants so preview and suite can never drift apart.
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

MC = cbc.MC_S                      # chirp mass in seconds
E0 = 0.10


def tau_of_f(f_hz):
    """Newtonian time to merger at GW frequency f."""
    return 5.0 * MC * (8.0 * np.pi * MC * f_hz) ** (-8.0 / 3.0)


def f_of_tau(tau_s):
    return (1.0 / (8.0 * np.pi * MC)) * (5.0 * MC / tau_s) ** (3.0 / 8.0)


def h_of_tau(tau_s):
    """Normalized Newtonian chirp h(t) ~ f^{2/3} cos(Phi(t))."""
    phase = -2.0 * (tau_s / (5.0 * MC)) ** (5.0 / 8.0)
    amp = (np.pi * MC * f_of_tau(tau_s)) ** (2.0 / 3.0)
    return amp / amp.max() * np.cos(phase)


def main():
    os.makedirs(FIGS, exist_ok=True)
    x = np.geomspace(cbc.X_LO, cbc.X_HI, cbc.N_POINTS)
    f_hz = x * cbc.F0_HZ
    rng = np.random.default_rng(17)
    phi_c = cbc.phi_circ(x)
    dphi = cbc.dphi_ecc(x, E0)
    y = phi_c + dphi + rng.normal(0.0, cbc.SIGMA_RAD, x.size)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))

    # (a) the Stage-B target
    ax = axes[0, 0]
    ax.loglog(f_hz, y, "o", ms=2.6, mec="none", mfc=MUTED, alpha=0.6,
              zorder=2, label="data (noisy, $\\sigma$=1 rad)")
    ax.loglog(f_hz, phi_c, color=S[0], lw=1.8, zorder=3,
              label="circular $B\\,x^{-5/3}$")
    ax.loglog(f_hz, phi_c + dphi, color=S[5], lw=1.4,
              ls=(0, (4, 2.5)), zorder=4, label="eccentric ($e_0$=0.1)")
    ax.set_xlabel("f [Hz]", fontsize=8)
    ax.set_ylabel("phase $\\Phi(f)$ [rad]", fontsize=8)
    ax.set_title("Stage-B target: SPA phasing, 5 Hz – 1 kHz\n"
                 "(%.0f rad accumulated at 5 Hz)" % phi_c[0])
    ax.legend(fontsize=7)

    # (b) the Stage-A target
    ax = axes[0, 1]
    resid = np.abs(y - phi_c)
    ax.loglog(f_hz, resid, "o", ms=2.6, mec="none", mfc=MUTED,
              alpha=0.6, zorder=2, label="|data − circular|")
    ax.loglog(f_hz, np.abs(dphi), color=S[4], lw=1.8, zorder=3,
              label="|$\\Delta\\Phi$| $\\propto x^{-34/9}$")
    ax.axhline(cbc.SIGMA_RAD, color=BASE, lw=1.0, ls=(0, (4, 3)),
               zorder=1)
    ax.text(f_hz[-1], cbc.SIGMA_RAD * 1.4, "noise floor",
            fontsize=7, color=MUTED, ha="right")
    ax.axvspan(cbc.BAND_A[0] * cbc.F0_HZ, cbc.BAND_A[1] * cbc.F0_HZ,
               color="#f0efec", zorder=0)
    ax.text(cbc.BAND_A[1] * cbc.F0_HZ * 0.9, 3e-4,
            "Stage-A band\n(signal > noise)", fontsize=7, color=INK2,
            ha="right")
    ax.set_xlabel("f [Hz]", fontsize=8)
    ax.set_ylabel("eccentric dephasing [rad]", fontsize=8)
    ax.set_title("Stage-A target: $\\Delta\\Phi(f)$, e$_0$=0.1 at 5 Hz\n"
                 "(%.0f rad at band entry, circularizes as f grows)"
                 % abs(dphi[0]))
    ax.legend(fontsize=7)

    # (c) time-frequency track
    ax = axes[1, 0]
    ff = np.geomspace(cbc.F0_HZ, cbc.X_HI * cbc.F0_HZ, 400)
    ax.loglog(tau_of_f(ff), ff, color=S[0], lw=1.8, zorder=3)
    for fmark, note in ((5.0, "3G band entry:\n~%.1f h to merger"
                         % (tau_of_f(5.0) / 3600.0)),
                        (20.0, "current detectors:\nlast ~%.0f s"
                         % tau_of_f(20.0)),
                        (1000.0, "1 kHz:\n%.0f ms" %
                         (tau_of_f(1000.0) * 1e3))):
        ax.plot(tau_of_f(fmark), fmark, "o", ms=5, mfc=S[5], mec="none",
                zorder=4)
        ax.annotate(note, (tau_of_f(fmark), fmark),
                    textcoords="offset points", xytext=(8, 6),
                    fontsize=7, color=INK2)
    ax.invert_xaxis()
    ax.set_xlabel("time to merger [s]  (time runs →)", fontsize=8)
    ax.set_ylabel("GW frequency [Hz]", fontsize=8)
    ax.set_title("the same signal in time–frequency\n"
                 "(Newtonian chirp, $\\mathcal{M}$=1.2 M$_\\odot$)")

    # (d) time-domain waveform snippets
    ax = axes[1, 1]
    # early: 3 s at band entry (f ~ 5 Hz)
    tau0 = tau_of_f(cbc.F0_HZ)
    t_e = np.linspace(0.0, 3.0, 3000)
    h_e = h_of_tau(tau0 - t_e)
    ax.plot(t_e, h_e * 0.42 + 0.55, color=S[0], lw=0.9)
    # late: final 0.35 s (f: 200 Hz -> 1 kHz)
    tau_l0 = tau_of_f(200.0)
    t_l = np.linspace(0.0, tau_l0 - tau_of_f(1000.0), 3000)
    h_l = h_of_tau(tau_l0 - t_l)
    ax.plot(t_l / (tau_l0) * 3.0, h_l * 0.42 - 0.55, color=S[5], lw=0.7)
    ax.text(0.02, 0.98, "band entry (f≈5 Hz), 3 s", fontsize=7.5,
            color=S[0], transform=ax.transAxes, va="top")
    ax.text(0.02, 0.08, "final %.2f s (200 Hz → 1 kHz, rescaled)"
            % tau_l0, fontsize=7.5, color=S[5], transform=ax.transAxes)
    ax.set_yticks([])
    ax.set_xlabel("time within snippet [s]", fontsize=8)
    ax.set_title("time-domain representation h(t) ∝ f$^{2/3}$cos$\\Phi$\n"
                 "(the regression works on the PHASING of this chirp)")
    ax.grid(False)

    fig.suptitle("CBC hero data preview — what the sampler will see "
                 "($\\mathcal{M}$=1.2 M$_\\odot$, e$_0$=0.1 at 5 Hz, "
                 "$\\sigma$=1 rad)", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(FIGS, "fig19_cbc_data.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("wrote %s" % out)
    print("B = %.6g rad, dPhi(x=1, e0=0.1) = %.1f rad, band %g-%g Hz, "
          "n = %d" % (cbc.B_PHASE, dphi[0], cbc.F0_HZ,
                      cbc.F0_HZ * cbc.X_HI, cbc.N_POINTS))


if __name__ == "__main__":
    main()
