"""Phase 4.2b — hero application II: CBC eccentricity, FREQUENCY DOMAIN.

NO flexible atoms, NO kernel changes: the practitioner's knowledge
"phasing terms are decaying power laws" is encoded ONCE as the opaque
template atom A_pow = c0*x**c1 (Phase 2.10 machinery), over safe-4 ops
with the standard corrected protocol (shared spawn burn-in + same-top-K
seeding + tuned ladder). Frequency-domain formulation per PLAN §4.2b:
the physics lives in the smooth SPA phasing and the leading
eccentricity contribution is an ADDITIVE secular term.

Physics (self-contained numpy; no LALSuite) — 3G-detector realism:
  x = f/f0, band [1, 200] = 5 Hz .. 1 kHz (ET/CE-like: a ~1.8 h
  BNS-type inspiral, ~1.2e5 rad of phase, eccentricity measurable at
  the low-frequency end before circularization)
  Data preview BEFORE running: python exp_phase42_cbc_preview.py
  (renders figs/fig19_cbc_data.png incl. the time-domain chirp)
  Newtonian TaylorF2 phase (t_c = phi_c = 0):
      Phi_circ(x) = B * x**(-5/3),   B = (3/128) * (pi*Mc*f0)**(-5/3)
      (Mc in seconds: Mc = 1.2 Msun * G Msun/c^3)
  Leading-order secular eccentric dephasing (TaylorF2Ecc-type, e.g.
  Favata 2014 / Moore et al. 2016; e0 = eccentricity at f0):
      dPhi(x; e0) = -(2355/1462) * e0**2 * B * x**(-34/9)
  So the full phase is a SUM OF TWO POWER LAWS with exponents -5/3 and
  -34/9 ~ -3.778 — the second exponent IS the PN signature of
  eccentricity, and the headline question is whether the sampler
  measures it from data.

Stages (PLAN §4.2b; claim framing per the 2026-07-15 forensics — see
the OPS/_APOW block below):
  A  dphi_e10     regress the POSITIVE dephasing magnitude -dPhi(x) on
                  the eccentricity-informative band (5-40 Hz); the
                  deliverable is the MEASURED EXPONENT vs -34/9
  B  phi_ecc_e10  regress Phi_ecc = Phi_circ + dPhi whole on the full
                  3G band. A_pow + A_pow is burn-in-enumerable, so this
                  is CALIBRATED SELECTION + JOINT REFINEMENT of a
                  sub-dominant term (1.6% of the signal at band entry),
                  not structural discovery — the headline is whether
                  the fitted sub-dominant exponent lands on -34/9.
  C  phi_ecc_e05 / phi_ecc_e20   e0 scan; check the fitted correction
                  coefficient scales as e0**2 (physics validation).

STATISTICAL-POWER SPEC (user, 2026-07-15): 8 temperatures (uniform-beta
extension of the tuned ladder), primitives {x, log, sqrt, exp, sin,
cos} — logs as the physically-motivated distractor (real PN phasing
has log terms at higher order) — x safe-4 ops + the A_pow template;
8 MPI processes (7 burn-in workers + master); hero chains (Stage A 800,
B/C 1600 rounds x 6 steps). sigma = 1 rad fixed, KILORADIAN units.
Grid: 150 log-spaced points. Diagnostics: exp_phase42_cbc_plots.py
renders fig12-fig18 from the archived JSON.

Run (module dir, MPI-capable env — spawn burn-in!):
    mpiexec -n 1 python exp_phase42_cbc.py [--only=<t>] [--smoke]
Writes: results/phase42_cbc.json (+ _partial checkpoint per run)
"""

import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_SRC = os.path.join(os.path.dirname(_HERE), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

import gp_movesC
import real_functionC

import exp_common

# ---------------------------------------------------------------------------
# Physics constants
# ---------------------------------------------------------------------------
MSUN_S = 4.925490947e-6          # G*Msun/c^3 in seconds
MCHIRP_MSUN = 1.2                # chirp mass, BNS-like (GW170817-ish)
# 3G-detector configuration (user, 2026-07-15): an ET/CE-like band —
# in-band from 5 Hz to 1 kHz (x = f/f0 in [1, 200], 2.3 decades).
# For this chirp mass that is a ~1.8-hour signal accumulating ~1.2e5
# rad of phase, with ~2e3 rad of eccentric dephasing at e0=0.1 —
# realistic magnitudes, and the low-frequency end is precisely where
# eccentricity survives circularization (the 3G science case).
F0_HZ = 5.0                      # reference frequency (band start)
MC_S = MCHIRP_MSUN * MSUN_S
B_PHASE = (3.0 / 128.0) * (np.pi * MC_S * F0_HZ) ** (-5.0 / 3.0)
ECC_COEF = -(2355.0 / 1462.0)    # leading-order secular coefficient
P_CIRC = -5.0 / 3.0
P_ECC = -34.0 / 9.0

def phi_circ(x):
    return B_PHASE * x ** P_CIRC

def dphi_ecc(x, e0):
    return ECC_COEF * (e0 ** 2) * B_PHASE * x ** P_ECC

# REGRESSION UNITS: KILORADIANS (2026-07-15). The raw problem spans 8+
# decades (phase 1.2e5 rad, sigma 1 rad, warm starts ~1) and strands
# Powell's coordinate slices: coefficient -> ~0 flattens the exponent
# slice -> guarded bracket failure -> the TRUE power-law candidates get
# discarded as failed fits (observed in the 3G smoke: burn-in topped by
# a0**exp(x) junk). In kiloradians the fitted coefficients are O(1)-
# O(100); the likelihood is UNCHANGED (mse/sigma^2 is scale-invariant)
# — a pure conditioning fix. All stored coefficients are in krad;
# config.physics.unit_rad records the conversion.
UNIT = 1000.0
BK = B_PHASE / UNIT              # circular coefficient in krad
SIGMA_RAD = 1.0                  # detector-ish phase noise, radians

# ---------------------------------------------------------------------------
# Protocol — STATISTICAL-POWER SPEC (user, 2026-07-15): 8 temperatures,
# wide vocabulary (logs included), hero-length chains, 8 MPI processes.
#   * 8 rungs: the tuned 6-rung uniform-beta ladder (d-beta = 0.18)
#     extended to 8 rungs over the same [0.10, 1.0] beta range
#     (d-beta = 0.9/7 ~ 0.129). Finer spacing => swap rates expected at
#     or ABOVE the 0.2-0.4 band (harmless side); per-run ladder_stats()
#     records reality — report drift, don't silently re-tune.
#   * MPI: the 8 processes are used where MPI parallelism exists in the
#     corrected protocol — the shared spawn burn-in (7 workers + master).
#     The EMC loop itself is the serial in-process multi-rung sampler
#     (all 8 temperatures are sampled every round; statistical power is
#     identical, only wall-clock is serial).
#   * Vocabulary: 6 primitives x safe-4 ops + the A_pow template (see
#     the OPS/_APOW block). Composition 'o' stays excluded, and free
#     '**' was REMOVED after the burn-in forensics — the template
#     provides the power-law family robustly.
# ---------------------------------------------------------------------------
# FEYNMAN-PARITY RESPEC (user, 2026-07-15, after the NaN-leak fix):
# the 8-rung statistical-power ladder is superseded by the TUNED 6-rung
# schedule (the family every 4.1/4.2a result was validated on — swap
# band 0.2-0.4 holds by construction), and budgets drop to one-Feynman-
# run scale: Stage A 400 rounds, B/C 800 rounds, 4 swaps/round.
# Rationale: the healthy smoke recovered Stage A in 60 rounds; suite
# wall ~3-4 h instead of ~22 h. --long remains for any stalling target.
_BETAS = (1.0, 0.82, 0.64, 0.46, 0.28, 0.10)
LADDER = [1.0 / b for b in _BETAS]
BURN_IN_WORKERS = 7              # + master = 8 MPI processes (user spec)
SIZE_PER_GROVE = 4
FUNCS = ["x", "log(x)", "sqrt(x)", "exp(x)", "sin(x)", "cos(x)"]
# POWER-LAW TEMPLATE, NOT free '**' (user decision 2026-07-15, after the
# burn-in forensics): free-exponent '**' composites inherit junk warm
# starts from stage-1 sub-fits and single-start Powell buries the true
# power laws (probes: clean-form fits converge from theta0=(1,1) on
# positive data from EVERY start; '**' composites do not). The GW
# practitioner's knowledge "phasing terms are decaying power laws" is
# encoded EXACTLY ONCE, as an opaque template atom A_pow = c0*x**c1
# (Phase 2.10 machinery, validated by freq_sin_atom 2/2 in 4.1), and
# ops return to safe-4 — the family the ladder was tuned for. Claim
# framing: Stage B truth (A_pow + A_pow) is enumerable by the burn-in,
# so the headline is CALIBRATED MEASUREMENT of the PN exponent -34/9
# and the e0^2 scaling, not structural discovery (RESULTS C2-style).
OPS = ["+", "-", "*", "/"]
_APOW = real_functionC.AtomSpec("Apow", "c0*x**c1")
ATOMS = [_APOW]
N_POINTS = 150
X_LO, X_HI = 1.0, 200.0          # f = 5 Hz .. 1 kHz (inspiral-only)
SIGMA_PHASE = SIGMA_RAD / UNIT   # = 1 rad, expressed in kiloradians
WARMUP_STEPS = 100
STEPS_PER_ROUND = 6
SWAPS_PER_ROUND = 4              # Feynman parity (5 pairs, tuned band)
MAX_TREE_DEPTH = 4
REBUILD_MAX_HEIGHT = 2
RECOVERY_NRMSE = 0.05
OUT_JSON = os.path.join(_HERE, "results", "phase42_cbc.json")

# (name, stage, e0, truth-callable [KILORADIANS], truth string, rounds,
#  seeds, x-band)
#
# PER-STAGE BANDS (2026-07-15, after the wide-band smoke): on [1, 200]
# the dephasing is BELOW the noise floor for x >~ 8 (f >~ 40 Hz), so in
# Stage A 60% of the log-spaced points are pure noise and Powell's
# coordinate descent loses all traction (coefficient -> 0 on the noise
# tail, exponent slice flat — unit-independent, verified). Stage A
# therefore runs on the ECCENTRICITY-INFORMATIVE band x in [1, 8]
# (5-40 Hz, where the preview shows signal > noise: "analyze the
# dephasing where it is measurable"). Stages B/C keep the full 3G band
# [1, 200]: there the CIRCULAR term dominates the noise everywhere
# (>17 sigma even at 1 kHz), the burn-in power-law fit has traction,
# and hunting the small correction is the EMC's job, not Powell's.
# Stage A regresses the dephasing MAGNITUDE -dPhi > 0 (2026-07-15):
# Powell probes show exponent-first coordinate descent from the
# standard positive warm start converges to the exact power law on
# POSITIVE data from every start, and jams on negative data (the
# coefficient's sign basin). Presentation-only: the physical dephasing
# is the negative of the fitted expression.
def _t_dphi(e0):
    return lambda x: -dphi_ecc(x, e0) / UNIT

def _t_phi(e0):
    return lambda x: (phi_circ(x) + dphi_ecc(x, e0)) / UNIT

BAND_A = (1.0, 8.0)              # 5-40 Hz: the eccentricity band
BAND_FULL = (X_LO, X_HI)         # 5 Hz - 1 kHz: the full 3G band

TARGETS = (
    ("dphi_e10",    "A", 0.10, _t_dphi(0.10),
     "%.6g*x**(-34/9)" % (-ECC_COEF * 0.01 * BK),     400, (17, 71),
     BAND_A),
    ("phi_ecc_e10", "B", 0.10, _t_phi(0.10),
     "%.6g*x**(-5/3) %+.6g*x**(-34/9)"
     % (BK, ECC_COEF * 0.01 * BK),                    800, (17, 71),
     BAND_FULL),
    ("phi_ecc_e05", "C", 0.05, _t_phi(0.05),
     "%.6g*x**(-5/3) %+.6g*x**(-34/9)"
     % (BK, ECC_COEF * 0.0025 * BK),                  800, (17,),
     BAND_FULL),
    ("phi_ecc_e20", "C", 0.20, _t_phi(0.20),
     "%.6g*x**(-5/3) %+.6g*x**(-34/9)"
     % (BK, ECC_COEF * 0.04 * BK),                    800, (17,),
     BAND_FULL),
)

SMOKE = "--smoke" in sys.argv
if SMOKE:
    OUT_JSON = os.path.join(_HERE, "results", "phase42_cbc_smoke.json")

ONLY = None
for _a in sys.argv:
    if _a.startswith("--only="):
        ONLY = _a.split("=", 1)[1]
        OUT_JSON = OUT_JSON.replace(".json", "_%s.json" % ONLY)


def fitted_expression(desc, params, ndigits=4):
    try:
        import sympy as sp
        expr = sp.sympify(desc)
        subs = {sp.Symbol(k): sp.Float(round(float(v), ndigits), 6)
                for k, v in params.items()}
        return str(expr.subs(subs))
    except Exception:
        return "%s  with  %s" % (desc, {k: round(float(v), ndigits)
                                        for k, v in params.items()})


def power_terms(desc, params):
    """Decompose the fitted expression into c*x**p terms (p=None for
    non-power terms). This is the analysis hook: Stage A's deliverable
    is p-hat vs -34/9; Stage B/C's is the correction term's (c, p)."""
    try:
        import sympy as sp
        xs = sp.Symbol("x")
        expr = sp.sympify(desc).subs({sp.Symbol(k): float(v)
                                      for k, v in params.items()})
        out = []
        for t in sp.expand(expr).as_ordered_terms():
            c, rest = t.as_independent(xs)
            if rest == 1:
                out.append({"coeff": float(c), "power": 0.0})
            elif rest == xs:
                out.append({"coeff": float(c), "power": 1.0})
            elif (rest.func == sp.Pow and rest.args[0] == xs
                  and rest.args[1].is_number):
                out.append({"coeff": float(c),
                            "power": float(rest.args[1])})
            else:
                out.append({"coeff": float(c) if c.is_number else None,
                            "power": None,
                            "term": str(rest)})
        return out
    except Exception as e:
        return [{"error": str(e)}]


def run_target(name, stage, e0, f_true, truth_str, rounds, seed, band):
    rng = np.random.default_rng(seed)
    x = np.geomspace(band[0], band[1], N_POINTS).astype(np.float64)
    y = (f_true(x) + rng.normal(0.0, SIGMA_PHASE, N_POINTS)
         ).astype(np.float64)

    roots, posts, burn_wall = exp_common.spawn_burn_in(
        x, y, FUNCS, OPS, SIGMA_PHASE, seed, ATOMS, BURN_IN_WORKERS)
    print("    %s [stage %s, e0=%.2f] seed %d: burn-in %d candidates in "
          "%.1f s; top: %s"
          % (name, stage, e0, seed, len(roots), burn_wall,
             roots[0].get_fun().get_description()), flush=True)
    forest = exp_common.seeded_forest(x, y, LADDER, SIZE_PER_GROVE, FUNCS,
                                      OPS, SIGMA_PHASE, seed, roots,
                                      atoms=ATOMS)

    t0 = time.perf_counter()
    forest.run_all(y, WARMUP_STEPS)
    traces = exp_common.emc_loop(forest, y, rounds, STEPS_PER_ROUND,
                                 SWAPS_PER_ROUND, trace_every=10,
                                 label="%s seed %d" % (name, seed), t0=t0,
                                 progress_every=25)
    wall = time.perf_counter() - t0

    best = forest.best_tree()
    g0 = forest.groves[0]
    best_post = float(g0.root_posteriors[g0.best_index()])

    yhat = np.asarray(best.evaluate(x), dtype=np.float64)
    mse_train = float(np.nanmean((y - yhat) ** 2))

    x_test = np.geomspace(band[0], band[1], 250).astype(np.float64)
    f_test = f_true(x_test)
    yhat_test = np.asarray(best.evaluate(x_test), dtype=np.float64)
    resid = f_test - yhat_test
    # NaN-hardened metric: with '**' in the vocabulary, pathological
    # trees (negative base, real exponent) evaluate NaN on most of the
    # grid; a plain nanmean would judge them on their few valid points
    # (loophole observed in the smoke run). Invalid points count as
    # failures, and recovery additionally requires >= 99% valid.
    valid = np.isfinite(resid)
    valid_frac = float(np.mean(valid))
    if valid.any():
        nrmse = float(np.sqrt(np.mean(resid[valid] ** 2) / valid_frac)
                      / max(np.sqrt(np.mean(f_test ** 2)), 1e-12))
    else:
        nrmse = float("inf")
    recovered = bool(nrmse < RECOVERY_NRMSE and valid_frac >= 0.99)

    params = dict(best.get_param_dicts()[1])
    terms = power_terms(best.tree_get_description(), params)
    # closest fitted exponent to the eccentric PN power
    ecc_p = None
    cands = [t["power"] for t in terms
             if isinstance(t, dict) and t.get("power") is not None]
    if cands:
        ecc_p = min(cands, key=lambda p: abs(p - P_ECC))
    out = {
        "target": name, "stage": stage, "e0": e0, "truth": truth_str,
        "seed": seed, "sigma": SIGMA_PHASE, "rounds": rounds,
        "best_tree": best.tree_get_description(),
        "best_params": params,
        "expression_fitted": fitted_expression(best.tree_get_description(),
                                               params),
        "power_terms": terms,
        "ecc_exponent_closest": ecc_p,
        "ecc_exponent_truth": P_ECC,
        "best_depth": int(best.get_root().depth()),
        "best_posterior": best_post,
        "mse_train": mse_train, "nrmse_test": nrmse,
        "valid_frac_test": valid_frac,
        "recovered": recovered,
        "wall_seconds": wall, "burn_in_seconds": burn_wall,
        "burn_in_top": roots[0].get_fun().get_description(),
        "ladder_stats": forest.ladder_stats(),
        "traces": traces,
    }
    print("  %-12s seed %-3d %s  nrmse=%.4f  valid=%.0f%%  "
          "p_ecc-hat=%s (truth %.4f)  %.0fs\n      best: %s"
          % (name, seed, "RECOVERED" if recovered else "MISSED   ", nrmse,
             100.0 * valid_frac,
             ("%.4f" % ecc_p) if ecc_p is not None else "none", P_ECC,
             wall, out["expression_fitted"]), flush=True)
    return out


def main():
    global BURN_IN_WORKERS
    gp_movesC.set_max_tree_depth(MAX_TREE_DEPTH)
    gp_movesC.set_subtree_rebuild_max_height(REBUILD_MAX_HEIGHT)

    print("Phase 4.2b HERO: CBC eccentric dephasing, frequency domain%s"
          % (" [SMOKE]" if SMOKE else ""))
    print("physics: Mc=%.2f Msun, f0=%.0f Hz, B=%.6g rad, x in [%.0f,%.0f]"
          % (MCHIRP_MSUN, F0_HZ, B_PHASE, X_LO, X_HI))
    print("truth exponents: circular %.4f, eccentric %.4f (= -34/9)"
          % (P_CIRC, P_ECC))
    print("conditions: funcs=%s ops=%s n=%d sigma=%.1f rad "
          "(regression units: KILORADIANS) workers=%d"
          % (FUNCS, OPS, N_POINTS, SIGMA_RAD, BURN_IN_WORKERS),
          flush=True)

    targets = TARGETS
    if SMOKE:
        targets = tuple(t for t in TARGETS if t[0] == "dphi_e10")
        targets = tuple((n, s, e, f, tr, 60, (17,), band)
                        for n, s, e, f, tr, _r, _sd, band in targets)
        print("smoke pass: Stage A only, 60 rounds, seed 17", flush=True)
    if ONLY:
        targets = tuple(t for t in TARGETS if t[0] == ONLY)
        print("single-target rerun (--only=%s)" % ONLY, flush=True)

    if SMOKE:
        BURN_IN_WORKERS = 4   # +master = 5 procs; smoke burn-in ~2x faster

    jobs = [(name, stage, e0, f, tr, rounds, seed, band)
            for name, stage, e0, f, tr, rounds, seeds, band in targets
            for seed in seeds]
    runs = []
    suite_t0 = time.perf_counter()
    partial_path = OUT_JSON.replace(".json", "_partial.json")
    for k, job in enumerate(jobs):
        name, stage, e0, f, tr, rounds, seed, band = job
        elapsed = (time.perf_counter() - suite_t0) / 60.0
        eta = (elapsed / k * (len(jobs) - k)) if k else float("nan")
        print("\n### RUN %d/%d — %s [stage %s] seed %d | elapsed %.1f min "
              "| ETA ~%.0f min ###"
              % (k + 1, len(jobs), name, stage, seed, elapsed, eta),
              flush=True)
        runs.append(run_target(name, stage, e0, f, tr, rounds, seed,
                               band))
        with open(partial_path, "w") as fh:
            json.dump({"done": len(runs), "of": len(jobs), "runs": runs},
                      fh, indent=2)
        print("    [checkpoint] %d/%d runs saved to %s"
              % (len(runs), len(jobs), os.path.basename(partial_path)),
              flush=True)

    n_rec = sum(1 for r in runs if r["recovered"])
    print("\n=== summary: %d/%d recovered (nrmse < %.2f) ==="
          % (n_rec, len(runs), RECOVERY_NRMSE))
    for r in runs:
        print("  %-12s [%s] seed %-3d %s  nrmse=%.4f  p_ecc-hat=%s  %s"
              % (r["target"], r["stage"], r["seed"],
                 "yes" if r["recovered"] else "NO ",
                 r["nrmse_test"],
                 ("%.4f" % r["ecc_exponent_closest"])
                 if r["ecc_exponent_closest"] is not None else "none",
                 r["expression_fitted"]))
    # Stage C physics check: correction coefficient vs e0^2
    cs = [(r["e0"], t["coeff"]) for r in runs if r["stage"] in ("B", "C")
          for t in r["power_terms"]
          if isinstance(t, dict) and t.get("power") is not None
          and abs(t["power"] - P_ECC) < 0.5]
    if len(cs) >= 2:
        print("  e0-scaling check (correction coeff, expect ratio = "
              "(e0/e0')^2): %s" % ["e0=%.2f: %.4g" % c for c in cs])

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    payload = {
        "config": {
            "suite": "phase42b_cbc_freq", "ladder": LADDER,
            "size_per_grove": SIZE_PER_GROVE, "funcs": FUNCS, "ops": OPS,
            "n_points": N_POINTS, "x_band": [X_LO, X_HI],
            "sigma_rad": SIGMA_PHASE, "warmup_steps": WARMUP_STEPS,
            "steps_per_round": STEPS_PER_ROUND,
            "swaps_per_round": SWAPS_PER_ROUND,
            "burn_in_workers": BURN_IN_WORKERS,
            "max_tree_depth": MAX_TREE_DEPTH,
            "subtree_rebuild_max_height": REBUILD_MAX_HEIGHT,
            "recovery_nrmse_threshold": RECOVERY_NRMSE,
            "freeze_vocab": True, "smoke": SMOKE,
            "physics": {"mchirp_msun": MCHIRP_MSUN, "f0_hz": F0_HZ,
                        "B_phase_rad": B_PHASE, "ecc_coef": ECC_COEF,
                        "unit_rad": UNIT, "sigma_rad": SIGMA_RAD,
                        "p_circ": P_CIRC, "p_ecc": P_ECC,
                        "note": "x=f/f0; TaylorF2 Newtonian phase, "
                                "t_c=phi_c=0; leading-order secular "
                                "eccentric dephasing (TaylorF2Ecc-type)"},
            "protocol": "v2 corrected; A_pow template + safe-4; "
                        "FEYNMAN-PARITY budgets on the tuned 6-rung "
                        "ladder (2026-07-15, post NaN-leak fix — see "
                        "RESULTS_PHASE4.md §7). Headline = calibrated "
                        "measurement of the eccentric PN exponent and "
                        "its e0^2 scaling",
            "atoms": ["Apow"],
            "targets": [{"name": t[0], "stage": t[1], "e0": t[2],
                         "truth": t[4], "rounds": t[5],
                         "seeds": list(t[6]), "x_band": list(t[7])}
                        for t in TARGETS],
        },
        "runs": runs,
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(payload, fh, indent=2)
    print("results written to %s" % OUT_JSON)


if __name__ == "__main__":
    main()
