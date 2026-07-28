# Results — Phase 4 experimental program

Running log of Phase 4 outcomes, written to be lifted into the paper's
experiments chapter. Companion to PLAN.md §4 (protocol decisions and
their rationale live there; this file records what was run and what came
out). Data: `results/*.json`; figures: `results/figs/`; scripts:
`exp_common.py`, `exp_phase41_ladder.py`, `exp_phase41_recovery.py`,
`exp_phase41_plots.py` (+ the depth-3 suite scripts, see §4).
Nothing in this file is overwritten by later runs: new experiments get
new file/figure names.

---

## 1. PT ladder tuning (§4.1, first deliverable)

**Protocol.** Per-consecutive-pair swap acceptance monitored via the
Forest pair counters (`ladder_stats()`); target band 0.2–0.4; candidates
run on the sanity target `y = sin(x) + 0.5x + N(0, 0.1²)`, n = 60,
safe-4 ops, frozen ψ_const vocabulary, depth caps 4/2.

**Finding L1 — geometric ladders cannot meet the band** on this target
family. Scanned r ∈ {1.2, 1.3, 1.4, 1.6} (σ = 1 scoring, rounds 1–5)
and r ∈ {1.05, 1.1, 1.2, 1.4} (corrected σ, round 6): hot pairs pin at
0.5–0.7 for every ratio, because a geometric ladder shrinks Δβ ∝ 1/T
while the bounded-BIC discrete target does not grow its energy
fluctuations with T — adjacent hot rungs become near-prior twins.
Uniform-in-β (harmonic) ladders are the correct family; the band
calibrates Δβ directly.

**Finding L2 — the tuning is σ-dependent** (see §2). Under σ = 1 the
adopted schedule was T = [1, 1.29, 1.82, 3.08, 5.50, 10] (Δβ = 0.225 +
split tail; 5/5 pairs in band, seed 4141). Re-measured under the
corrected σ, its cold pairs sag to 0.14–0.19: the ×100 sharper
likelihood demands finer cold spacing while the hot end still flattens
(fig2, left panel).

**Adopted schedule (corrected protocol, rounds 6–8):**

    T = [1, 1.22, 1.56, 2.17, 3.57, 10]      (uniform-β, Δβ = 0.18, 6 rungs)

High-statistics validation (240 rounds × 6 swap attempts, ~280–300
attempts/pair, SE ≈ 0.026): seed 4141 → [0.31, 0.28, 0.26, 0.26, 0.24],
5/5 in band, max deviation 0.000; seed 999 → [0.30, 0.34, 0.35, 0.43,
0.33], 4/5, single excursion +0.029 on the harmless high side.

**Finding L3 — long-run vs transient rates differ.** The coldest pair
saturates high after both cold chains lock the dominant mode (free
mixing, a target-concentration effect no spacing can or should remove);
the hot tail sags as cold rungs concentrate. Rates for production-length
runs are the ones to tune on.

Data: `results/phase41_ladder.json` (geometric scan, σ=1),
`_round2/3/4/5` (σ=1 refinement → superseded schedule),
`_round6/7/8` (corrected protocol → adopted schedule).
Figure: `figs/fig2_ladder_rates.png`.

---

## 2. The σ-calibration finding (decides the whole protocol)

Benchmarks initially scored with the test-suite convention σ = 1.0
against injected noise σ = 0.1. The likelihood term −0.5·N·mse/σ² is
then ~100× too flat: on the rational target the true model beats the
best constant by ≈ 1.2 nats of likelihood while one BIC parameter costs
≈ 2 nats — the constant wins **legitimately** under the mis-specified
scoring. Verified by widening the target's domain to SNR ≈ 4 and
observing constants still win (`results/phase41_recovery_rational_v2
.json`). High-variance targets mask the effect; low-amplitude structure
is posterior-suppressed.

**Consequence (adopted, 2026-07-13):** benchmark scoring uses σ fixed at
the known noise level, end to end (scoring, burn-in, parameter refresh).
This is the PLAN 5.1.6 fixed-vs-profiled choice surfacing in practice;
profiled σ remains the choice for unknown-noise applications (hero app).
Figure: `figs/fig5_sigma_calibration.png` — the paper's
likelihood-treatment section should carry this example.

---

## 3. Recovery benchmarks (§4.1, main deliverable)

Six targets, all expressible in the vocabulary; scored by NRMSE of the
best tree against the **noise-free truth on a dense held-out grid**
(recovered := NRMSE < 0.05). Two independent seeds per target.

**v1 (σ = 1, random init + warmup): 7/12**, plus a 3× budget pass on the
misses (atom target → 2/2; sin_plus_linear → 1/2; rational → constants,
which exposed §2). Files: `results/phase41_recovery.json`, `_long.json`.

**v2 (corrected protocol: σ = noise, ONE shared spawn burn-in per
(target, seed) seeding every rung with the same top-K, adopted ladder):
11/12.**

| target | truth | seed 17 | seed 71 |
|---|---|---|---|
| linear | 0.5·x | ✔ `a0*x + x` (0.023) | ✔ `a0*x - x` (0.007) |
| sin_plus_linear | sin(x)+0.5x | ✔ `x + sin(x) - x/a1` (0.017) | ✔ `a0*x + sin(x)` (0.0015) |
| x_sin | x·sin(x) | ✔ exact (0.0000) | ✔ exact (0.0000) |
| rational | x/(1+x) | ✔ `x/(a0 + x)` (0.032) | ✔ `x/(a0 + x)` (0.003) |
| poly2 | x²−x | ✔ exact (0.0000) | ✘ `x**2` (0.57) |
| freq_sin_atom | 2.3·sin(1.3x) | ✔ `b0*sin(b1*x)` (0.009) | ✔ `b3*sin(b4*x)` (0.009) |

(parenthetical = held-out NRMSE; fitted parameter values in the JSON and
in `results/phase41_expressions.md`.)

Highlights: `rational` recovers the exact structure on both seeds (was
0/2 — the burn-in enumerates it and the corrected σ makes it beat the
constant by ~120 nats); the Ψ₂ opaque-primitive target recovers 2/2 with
the atom surviving the full EMC machinery. The single miss (poly2, seed
71, stalled at `x²`) is a one-run stochastic event — exact on the other
seed and recovered in v1; open item: inspect that run's burn-in ranking.

File: `results/phase41_recovery_v2.json` (incl. per-round traces:
best posterior per rung, Pólya urn counter, trie size, root-cell
Dirichlet). Figures: `figs/fig1_recovery_overlays.png`,
`figs/fig3_convergence_traces.png`, `figs/fig4_polya_evolution.png`.
Table: `results/table_recovery.csv/.tex`.

---

## 4. Interpretation, caveats, and the depth-3 follow-up

**C1 — fig3's flat cold lines are a max-statistic, not frozen chains.**
The plotted quantity is the running best posterior per rung; a cold
chain rarely accepts downhill so the max is sticky. Exploration counters
from the same runs: the Pólya urn counter t (one increment per accepted
structural move, population-wide) grows by ~700–1,700 per 200-round run,
and the Pólya trie grows from ~100–250 to ~1,400–4,700 nodes — thousands
of previously-unvisited construction paths accepted. Cold-pair swaps
were accepted ~85 times per run.

**C2 — for 5 of 6 v1/v2 targets the true structure is depth ≤ 2, i.e.
inside the burn-in's exhaustive enumeration.** For those, the EMC
phase's contribution to the headline metric is confirmation plus θ
refinement; 11/12 is a statement about the pipeline, not a stress test
of the MCMC kernel. The exception, sin_plus_linear seed 17, was won by
a depth-3 tree the burn-in cannot produce (`x + sin(x) - x/a1`) —
genuine sampler discovery, and the v1 suite (no burn-in) shows the
sampler finds depth-2 modes unaided, just less reliably.

**Follow-up (completed 2026-07-14): the depth-3 suite** — targets whose
true structure is strictly outside the burn-in enumeration, so recovery
measures the sampler's reach beyond it. Outcomes in §5.

---

## 5. The depth-3 suite: discovery beyond the burn-in (2026-07-14)

**Design.** Three targets whose true structure has tree height ≥ 3 —
outside the exhaustive height-≤2 burn-in, so the EMC kernel must GROW
the winning structure from seeds that cannot contain it. Chosen to
grade the difficulty of escaping the initialization: `poly3` x³−2x
(the nearby height-2 mode x³ IS a seed and must be escaped), `inv_quad`
3x/(1+x²) (fully outside; peaked/non-monotone, no height-2 form fits;
SNR 3.3 by design per §2's lesson), `mixed` x·sin(x)+x/(1+x) (the
dominant part x·sin(x) IS a seed; a fitted constant instead of the
rational correction leaves NRMSE ≈ 0.09 — the correction is mandatory).
Protocol identical to v2 except a doubled budget (400 rounds); same
seeds (17, 71); same NRMSE < 0.05 held-out criterion.

**Result: 5/6 recovered, and 6/6 winning trees have height ≥ 3 — every
single winner lies beyond the burn-in's reach.**

| target | truth | seed | NRMSE | depth | regressed expression |
|---|---|---|---|---|---|
| poly3 | x³−2x | 17 | ✔ 0.0018 | 4 | `x²(x − 2.0114/x)` = x³−2.011x — **exact** |
| poly3 | x³−2x | 71 | ✔ 0.0008 | 4 | `0.9994x³ − 2x` — **exact** |
| inv_quad | 3x/(1+x²) | 17 | ✘ 0.0692 | 4 | trig-rational approximant (near miss) |
| inv_quad | 3x/(1+x²) | 71 | ✔ 0.0046 | 3 | `0.5906/(0.1984x + 0.1957/x)` ≡ 2.98x/(x²+0.987) — **exact** |
| mixed | x·sin(x)+x/(1+x) | 17 | ✔ 0.0166 | 4 | `0.928x·sin(x) + 0.714 − 0.424·sin(x)cos(x)/x` — functional |
| mixed | x·sin(x)+x/(1+x) | 71 | ✔ 0.0071 | 5 | `x·sin(x) + 0.0862x(sin(x)+3.517) + 0.217cos(x)` − … — functional |

Structural scorecard: 3/6 exact (both poly3; inv_quad/71 in an
algebraically equivalent parametrization — divide through by 0.1984),
2/6 functional approximants passing held-out, 1 near miss (0.069 vs the
0.05 threshold). Together with §3 this completes the claim pair:
**§3 = posterior selection is calibrated; §5 = the kernel discovers
structure beyond its initialization.**

**New diagnostics** (trace fields added 2026-07-13: per-rung cumulative
accepted moves, unique structures visited, tree-depth lists):

- **fig7 (exploration).** Cumulative unique structures and accepted
  moves grow roughly linearly at every rung all run long — no stalling;
  the COLD rung shows the LOWEST churn (~60–90 unique structures vs
  ~150–190 at hot rungs per 400 rounds), which is the correct division
  of labour, and settles fig3's max-statistic question with data.
- **fig8 (depth penalization across the whole history — the paper's
  complexity-control figure).** Population-mean depth rises from the
  seeded ≈2 and stratifies MONOTONICALLY by temperature: the cold rung
  saturates at ≈3.5 while T=10 climbs to ≈4.9. Pooled over the history,
  the cold rung holds ~91% of its mass at depths 3–4; hot rungs shift
  mass into 5 and 6+. The tempered posterior visibly relaxes the
  complexity penalty as T grows — prior + BIC pin the cold chain to the
  most parsimonious adequate structure.
- **Depth-cap escalation (kernel observation, worth a paper footnote).**
  The height cap (4) binds per MOVE, but trans-grove crossover grafts
  compound: sampled depths up to 10 were observed at hot rungs, and two
  winning trees (inv_quad/71 first run at depth 5; mixed/71 at depth 5)
  exceeded the nominal cap. Not a soundness issue (the prior still
  penalizes; posteriors are exact), but the support is larger than the
  per-move cap suggests — either document as "soft cap" or add a
  crossover-height guard in a future revision.

Data: `results/phase41_depth3.json` (full traces with the new fields).
Figures: `figs/fig6_depth3_overlays.png`, `figs/fig7_exploration.png`,
`figs/fig8_depth_penalization.png`. Tables: `results/table_depth3.csv/
.tex`; expressions for ALL 36 runs (v1, long, v2, depth-3):
`results/phase41_expressions.md/.csv`. Script: `exp_phase41_depth3.py`
(+ `exp_phase41_plots_v3.py` for rendering).

---

## 6. Hero application I — the Feynman univariate set (4.2a, 2026-07-15)

**Design.** Five closed-form ℝ→ℝ laws from the Feynman database
(constants fixed so each is single-variable), full corrected protocol
at a hero budget: 800 rounds × 6 steps (~3× the 4.1 budget), n = 100,
vocabulary {sin, cos, exp, x} × safe-4, spawn burn-in on 5+1 MPI
processes, per-target noise chosen by the §2 SNR discipline. Unlike
§§3/5 this is not a controlled inside/outside-burn-in experiment — it
is the pipeline pointed at real physical laws of mixed difficulty.

**Result: 9/10 recovered** (`results/phase42_feynman.json`):

| law | truth | seed 17 | seed 71 |
|---|---|---|---|
| flux I.12.2 | 2/x² | ✔ `1.987/x²` (0.006) | ✔ **exact** `2/x²` (0.0000) |
| scattering III.17.37 | 1.2(1+0.7cos x) | ✔ exact structure (0.013) | ✔ exact structure (0.009) |
| velocity_add I.16.6 | (½+x)/(1+½x) | ✔ functional (0.006) | ✔ `(x+eˣ)/(eˣ+1.04)` (0.006) |
| polarization II.11.28 | 1+x/(1−x/3) | ✘ 0.174* | ✔ **exact** `1+2.98x/(3.00−x)` (0.001) |
| planck I.41.16 | x³/(eˣ−1) | ✔ `0.85x²(x+0.72)e⁻ˣ` (0.023) | ✔ `0.74x²(x+1.93)/(x+eˣ)` (0.015) |

\* the sole formal miss is structurally CORRECT: `−2.05 − 9.06/(x−3.001)`
is the right rational family with the pole at 3.001 (truth: −2 − 9/(x−3)
by partial fractions); fig9 shows it visually indistinguishable from
truth — the NRMSE lives entirely in the steep pole approach, where a
0.1% pole offset costs ~1 unit. Metric caveat for the paper: NRMSE
punishes pole sensitivity, not structure.

Notes for the write-up: (i) exact-vs-functional matters — flux and
polarization/71 are exact; velocity_add's `(x+eˣ)/(eˣ+1.04)` is an
elegant impostor (the sampler used the new exp primitive as an
approximation device); planck recovers the correct functional anatomy
(both winners are x²·polynomial against an eˣ-family denominator —
seed 71's `x+eˣ` mirrors the true `eˣ−1`). (ii) **gauss (I.6.20a with
the A_gauss template) was DROPPED by design decision**: with the
template in the burn-in vocabulary the target is tautological — the
chain locks the primitive it was handed. Rule adopted: informative
template tests must be template-PLUS-correction (this is exactly the
CBC Stage-B design, §4.2b). (iii) The gauss run also exposed a
robustness bug (queued): scipy Powell's `RuntimeError("Too many
iterations.")` in a θ-fit is uncaught → process death; fix = guard →
force-reject. (iv) fig10 finding: at the ROOT level all three function
cells decay identically (operators absorb root mass) — the urn's
function preference lives deeper in the trie than the root-cell trace
shows; a per-depth trie-share diagnostic is a candidate future trace
field. (v) fig11: the ladder tuned at n=60/{sin,cos,x} transfers to
n=100/+exp — middle pairs in band; coldest pair saturates high on easy
targets (L3 again); hot tail sags mildly.

Data: `results/phase42_feynman.json` (assembled from per-run
checkpoints after a crash in the dropped gauss run — see config.note).
Figures: `figs/fig9_feynman_overlays.png`, `figs/fig10_feynman_polya
.png`, `figs/fig11_feynman_ladder.png`. Tables: `results/table_feynman
.csv/.tex`; expressions: `results/phase42_feynman_expressions.md`.
Scripts: `exp_phase42_feynman.py` (`--resume` reuses checkpoints),
`exp_phase42_plots.py`.

**Next (4.2b, built and armed):** the CBC eccentric-dephasing hero in
the frequency domain — `exp_phase42_cbc.py` at the statistical-power
spec (8 rungs, {x, log, sqrt, exp, sin, cos} × {+,−,*,/,**}, Stage A
800 / B–C 1600 rounds × 6, 7+1 MPI), diagnostics fig12–fig18 via
`exp_phase42_cbc_plots.py`. Launch: `bash run_phase42_cbc.sh` (~6–9 h).

---

## 7. The NaN-scoring incident (2026-07-15) — how the sampler found
## our bug, and what it teaches

**What happened.** The first full CBC run (§4.2b spec: 8 rungs,
800/1600 rounds) produced three completed runs, all compromised: the
winning trees evaluated to NaN on up to 99.6% of the grid (fitted
exponents like x^(8.6×10¹⁴)) yet carried internal posteriors of −10 to
−12.5 — beating the exact truth's legitimate −87 by ~75 nats. The
held-out metric (NaN-hardened earlier) correctly flagged every run as
MISSED; the leak was upstream, in the scoring that steers the chains.

**Root cause.** The kernel MSE was `np.nanmean(residual²)`: NaN
predictions were silently dropped, so a mostly-NaN tree was scored on
its few finite points — and the θ-optimizer *actively drove*
parameters into NaN territory because that shrank the mean. No gate
caught it: the post-fit validity check evaluates at reference
parameters (θ = 1), never at the fitted values, and the BIC cache then
memoized the flattered scores. inf predictions were never exploitable
(they poison the mean); NaN specifically was. The exploit needed a
vocabulary with unbounded parametric exponents (the CBC power-law
template) and enough sampling time to find it — 60-round smokes
passed; 800-round chains reward-hacked it reliably. **Methodological
moral: an MCMC sampler is an adversarial search over your scoring
function; any leniency in the likelihood WILL eventually be found and
maximized.** Short validation runs cannot certify a scoring function.

**The fix (kernel, minimal).** Zero-likelihood semantics for any
non-finite prediction: the MSE is now a plain sum with a scalar
finiteness check and a large finite penalty (10¹⁰⁰) — finite so the
Powell line search stays bracket-stable. The parameter-refresh
likelihood's identical hole (`nansum`: an all-NaN residual gave
ssr = 0 and *passed*) was closed the same way, and penalty-scale
scores are excluded from the BIC cache so one bad optimizer landing
cannot permanently condemn a topology. On all-finite data the new code
is bit-identical to the old (nanmean degenerates to the same pairwise
sum) and strictly cheaper (no NaN-mask pass).

**Validation.** Finite-path exactness probe: the fitted power law's
posterior reproduces the pre-fix value −87.025718642 at 2×10⁻¹⁴
relative. All four test suites pass with bit-identical historical
fingerprints; wall times within the environmental band. The
`test_phase42` pipeline fingerprint passes against its PRE-fix
reference — i.e., on a NaN-free trajectory the fixed kernel is
literally bit-identical, which is the strongest possible compatibility
statement.

**End-to-end assay (CBC re-smoke, the monsters' native habitat):**
RECOVERED at NRMSE 0.0003 with valid = 100%, p̂ = −3.7787 (truth
−3.7778), winner `1.98295/x^3.7787`, cold chain at the legitimate
−87.026 from round 25 — no monster anywhere in a vocabulary that
previously bred them within 60 rounds.

**Full 4.1 repro-toolkit comparison against the archives**
(`results/repro/phase41_repro_report.md`): 15 of 18 runs reproduce
BIT-IDENTICALLY (all four fields: tree string, recovered, NRMSE,
posterior at 1e-9). The three flagged diffs, inspected:
  * freq_sin_atom/17 — identical model, identical NRMSE and posterior;
    only the parameter NAMES differ (a-band vs b-band): a single
    trajectory desync converging to the same endpoint.
  * inv_quad/17 — the archived NEAR-MISS (0.0692) is now RECOVERED at
    0.0271 with a proper rational form: **the depth-3 suite improves
    to 6/6 under the fixed kernel.**
  * inv_quad/71 — same model family, equal NRMSE (0.00463 vs 0.00462),
    but a MORE PARSIMONIOUS parametrization (posterior −26.1 vs −36.3).
Interpretation: pre-fix trajectories occasionally routed through
inf-scored candidates whose reject path consumed no RNG draw; the fix
changes that bookkeeping, so a few runs desync — and everywhere they
do, the science is preserved or improved. No recovery was lost
anywhere. The archived 4.1/Feynman conclusions stand unchanged (the
depth-3 headline is now conservative: 5/6 archived, 6/6 on the fixed
kernel).

**Forensic corpus** (preserved for the paper/thesis honesty section):
`results/phase42_cbc_partial_INCIDENT_20260715.json` (three complete
runs with traces) and `..._run_INCIDENT_20260715.log`. The incident
also yielded two prior findings now folded into protocol: the burn-in
ranking's warm-start inheritance trap for free-exponent composites
(resolved by the A_pow template + safe-4 design), and the per-run
`valid%` now printed alongside NRMSE in all CBC logs.

---

## 8. Hero application II — CBC eccentric dephasing, FINAL (2026-07-16)

**Run** (Feynman-parity spec on the fixed kernel): 6 runs — Stage A
(dephasing magnitude, 5–40 Hz band, 400 rounds) × 2 seeds; Stage B
(full phase, 5 Hz–1 kHz, 800 rounds) × 2 seeds; Stage C (e₀ scan,
800 rounds) × 2 values — tuned 6-rung ladder, A_pow template + safe-4,
σ = 1 rad, kiloradian units. **Result: 6/6 recovered, all with 100%
finite predictions.** `results/phase42_cbc.json`.

| run | fitted expression (krad) | notes |
|---|---|---|
| A/17 | `1.983/x^3.7787` | truth 1.9832·x^(−3.7778) |
| A/71 | `1.9834/x^3.7786` | exponent to 0.02% |
| B/17 | `−2.0047·x^(−3.7555) + 123.141·x^(−1.6668)` | full anatomy |
| B/71 | `−1.9966·x^(−3.7645) + 123.133·x^(−1.6667)` | full anatomy |
| C e₀=0.05 | `0.8679·log(x)/x^2.7621 + 122.625/x^1.6655` | impostor (below) |
| C e₀=0.20 | `−7.9545·x^(−3.7721) + 123.141·x^(−1.6668)` | tightest of all |

**The headline (Stage B, both seeds):** regressing 123,000 rad of
phase containing a 1.6% eccentric correction, the sampler returned the
exact two-power-law anatomy — circular coefficient right to 0.02%,
circular exponent to 0.006%, eccentric coefficient to ~1%, eccentric
exponent −3.755/−3.764 vs the PN truth −34/9 = −3.7778 (0.3–0.6%).

**The e₀² physics check:** fitted correction coefficients −2.005 /
−1.997 (e₀ = 0.1) and −7.954 (e₀ = 0.2): ratio 3.98 vs the theoretical
(0.2/0.1)² = 4 — **the amplitude walks the e₀² line to 0.6%**
(fig18, right panel; truth values −1.9834 and −7.934).

**The honest caveat (C, e₀ = 0.05):** at the weakest eccentricity the
winner passes held-out at NRMSE 1e-5 but with an IMPOSTOR correction —
`log(x)/x^2.76` in place of `x^(−3.778)`, with the circular
coefficient absorbing 0.4% of the difference. Within σ = 1 rad the two
forms are functionally degenerate at this amplitude: the exponent
measurement is solid at e₀ ≥ 0.1 and becomes FORM-degenerate at 0.05
(the velocity_add-class lesson recurring exactly where identifiability
theory predicts it — the term is 4× weaker). This is a finding, not a
failure: it maps the e₀ floor of symbolic identifiability at this
noise level, and it is why the e₀²-scaling check draws on the three
runs with clean power-law corrections.

Costs: 808 s–8,013 s per run, suite ≈ 7.5 h wall. Figures
fig12–fig18 (+ the data/story figures fig19–fig22); tables
`table_cbc.csv/.tex`; expressions `phase42_cbc_expressions.md`.
Scripts: `exp_phase42_cbc.py`, `exp_phase42_cbc_plots.py`, previews
`exp_phase42_cbc_{preview,timedomain,chirp,lastdiff}.py`. Combined
with §7, the CBC chapter is: a compromised first run whose forensics
produced two kernel-level fixes, then a clean suite measuring a
post-Newtonian exponent and its amplitude scaling from noisy phase
data. **§4.2b IS CLOSED.**

---

## 9. Ablation I — frozen vs continuously-adaptive Pólya prior
## (2026-07-16): the two regimes are empirically indistinguishable

**Design.** The paper's two-regime question (§9 of the theory: frozen
Θ̄ = rigorous baseline vs continuous adaptation = flagship needing the
self-interacting-chain analysis) tested head-to-head. A new opt-in
kernel switch (`polya.adapt_frozen`, default off, one C-bool branch on
accepted moves only, state-carrying; flag-off bit-identity certified
by suites + the pipeline fingerprint) freezes reinforcement after
warm-up. Frozen arm: both 4.1 suites rerun with the freeze (18 runs).
Adaptive arm: the post-fix repro reruns — same kernel, seeds, protocol;
the arms differ by literally one bit. Urn-freeze sanity (t and root
Dirichlet constant after the freeze): PASS in all 18 runs.

**Result** (`results/ablation_results/phase43_frozen_vs_adaptive_
report.md`): **depth-2: 11 vs 11 recovered; depth-3: 6 vs 6 — parity
on all 18 paired runs, including the same single miss (poly2/71).**
On depth-2, 10 of 12 pairs are outcome-identical to the reported
digits; on depth-3 the harder trajectories diverge (Δposterior −4.7 to
+5.9 nats, NRMSE differences within a factor ~2 in both directions)
with no systematic winner.

**Interpretation.** Combined with fig23(d) — the root-predictive
movement ‖Δq‖₁ decaying to ~6×10⁻⁴ per snapshot under γ_t = 1/(1+t) —
the picture is coherent: after a warm-up that includes the burn-in
seeding replay, Θ̄ is already well-formed, and the continuous
adaptation that follows is too small to change benchmark-scale
outcomes. Two paper-ready consequences: (i) the "frozen regime as
recommended default" now has DIRECT empirical support — nothing is
lost by using the regime whose guarantees are theorems; (ii) the
adaptive flagship is not refuted (it loses nothing either), but its
practical benefit is not visible at these budgets with seeded warm
starts — where it should matter is vocabulary learning over longer
horizons / unseeded searches, an honest open question for the outlook.
This is the empirical companion the §9 review prompt asked the
experiments section to provide.

**Arm C — frozen at BIRTH (2026-07-17): the symmetric untrained prior
DEGRADES.** Same 18 runs with Θ frozen before any warm-up step
(`t_frozen = 0` recorded as proof; urn sanity PASS ×18):

| arm | Θ's history | depth-2 | depth-3 | total |
|---|---|---|---|---|
| adaptive | learns all run | 11/12 | 6/6 | **17/18** |
| frozen at Θ̄ (post-warm-up) | learns through warm-up only | 11/12 | 6/6 | **17/18** |
| frozen at birth (symmetric) | never learns | 10/12 | 4/6 | **14/18** |

The untrained prior loses three recoveries: rational/71 collapses
(NRMSE 0.0034 → 0.393 — an exact-form target lost outright),
inv_quad/17 and mixed/17 degrade 2–3× past the threshold (0.027 →
0.067, 0.017 → 0.055). Equally telling below the threshold:
freq_sin_atom/17 survives but 5× worse (0.0094 → 0.0465) — without
warm-up reinforcement the template cell is never up-weighted and the
atom target barely scrapes by. (poly2/71 misses in ALL arms —
confirming it as a burn-in/seed artifact, not a regime effect.)

**The completed conclusion — sharper than either two-arm reading:**
the reinforcement machinery PAYS OFF, and its entire benefit is
front-loaded into the warm-up. Learning Θ̄ matters (+3 recoveries over
symmetric); learning BEYOND Θ̄ adds nothing measurable at these
budgets. The practical recipe with both rigor and performance is
therefore: reinforce through warm-up, freeze at Θ̄, sample in the
regime whose guarantees are theorems. For §9 of the theory this is the
ideal empirical companion: the adaptive prior is neither decorative
(arm C shows real value) nor indispensable in its continuous form
(arm B shows the frozen regime suffices once Θ̄ is formed).

Artifacts: `results/ablation_results/` (both frozen-arm JSONs +
reports, run logs). Script: `exp_phase43_ablation.py` (`--at-birth`
for arm C); kernel switch in `polya_treeC` (5 additive edits).

---

## 10. The ablation program complete (2026-07-19) — every component
## justified by measurement. PHASE 4 CLOSES HERE.

Five arms, all on the certified fixed kernel, all against the same
adaptive baseline (the post-fix repro reruns; 17/18), all seeds and
protocols pinned, all outputs in `results/ablation_results/`:

| component | arm | recovered | verdict |
|---|---|---|---|
| prior LEARNING | C: frozen-at-birth (symmetric Θ) | 14/18 | the urn's warm-up learning is worth 3 recoveries — the machinery pays, front-loaded |
| continued adaptation | B: frozen-at-Θ̄ (post-warm-up) | 17/18 (parity) | "learn briefly, then freeze": full rigor, zero cost |
| tempering | notemper (single chain) | 16/18 | earns its keep EXACTLY on the multimodal target: inv_quad 0/2 without the ladder (0.101/0.091); depth-2 unaffected |
| parsimony (BIC k-penalty) | nobic | 12/18 | the collapse is qualitative: `linear` itself missed both seeds; winners with k = 10, 19, 20 at depth 8; even numerical "recoveries" are baroque (freq_sin_atom "recovered" at k = 19 vs the baseline's k = 2). **The BIC term is what makes the output symbolic.** |
| dressed vocabulary | affine pilot (a+b·f(cx+d) templates) | **18/18** | best recovery rate of any arm — but every depth-3 winner is TEMPLATE-DRESSED (cosine-cubics for poly3, dressed-trig rationals for inv_quad/mixed), and runs cost 5–25×. Where the truth IS a dressed primitive (freq_sin_atom), the winner is a single depth-0 template leaf — the intended use-case, perfectly realized. **Dressed vocabularies trade symbolic fidelity and compute for robustness.** |

**Cross-cutting observation:** poly2/71 — the baseline's lone,
knife-edge stochastic miss — was recovered by EVERY perturbed arm
(notemper exact at 0.0021, nobic at 0.0036, affine exact), each by a
different accidental trajectory shift. Strong confirmation it was
never systematic; equally strong illustration that single-run
outcomes at this multiplicity carry trajectory luck.

**The claim ladder this completes** (the paper/thesis architecture):
§3 selection is calibrated → §5 the kernel discovers beyond its
initialization → §6 real physical laws (Feynman 9/10) → §8 templated
symbolic measurement of a PN scaling law (CBC 6/6, e₀² to 0.6%) →
§§9–10 every load-bearing component individually justified: prior
learning (arm C), the freeze recipe (arm B), tempering (notemper),
parsimony (nobic), vocabulary policy (affine) — plus the σ convention
(§2) and kernel soundness (§7 + the certification framework).

**PHASE 4 — THE EXPERIMENTAL PROGRAM — IS COMPLETE.**
Deferred beyond it: the no-Pólya arm (user-deferred), profiled-σ /
real-data extension, flexible atoms (time-domain CBC), multi-start
Powell, and the open question of adaptation's value in unseeded /
longer-horizon searches.

---

## Protocol reference (corrected, v2 onward)

`exp_common.py`: σ = known noise everywhere; ONE spawn burn-in per
(target, seed) (`_burn_in_stage2_parallel_spawn`), every rung seeded
with the same top-K candidates as deepcopies; `compute_posteriors_all`
before stepping; EMC loop = run_all + swap attempts + trans-grove
crossover per round; depth caps 4/2; frozen ψ_const vocabulary
(`freeze_vocab=True`); adopted ladder T = [1, 1.22, 1.56, 2.17, 3.57,
10]. NB: `Forest(burn_in=True)` runs one redundant burn-in per rung —
never used; seeding goes through `exp_common.seeded_forest`.
