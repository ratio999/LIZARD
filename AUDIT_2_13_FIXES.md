# Phase 2.13 Soundness Audit — Implementation Tracker

**Started:** 2026-07-09  
**Status:** ✅ COMPLETE AND VERIFIED (2026-07-09) — all code fixes
(a1, b1, c, d, e, f, g, h, i, m), the a1 occupancy test, and the R2
connectivity deliverables implemented; full suite recompiled and run
green by the user (test_phase23, test_cython4, test_phase28 incl. 2.8c,
test_phase29 incl. 2.9d2, test_phase2mpi).  
**Gate:** Phase 4 experiments UNBLOCKED. Remaining audit items (j)–(l) are
docs-only (fold into the paper per PAPER_PLAN.md).

---

## Fix (e): `_propose` retry loop — max_attempts=1

**Issue:** `max_attempts=2` creates an unaccounted proposal mixture; redrawing on validity failure biases the kernel.  
**Location:** `groveC.pyx:488` (_propose method)  
**Severity:** Medium (affects every step that hits a validity failure)  
**Change:** Set `max_attempts=1`; invalid → hard reject (None path already exists).  

| Status | Date | Notes |
|--------|------|-------|
| ✅ IMPLEMENTED — pending recompile | 2026-07-09 | `groveC.pyx` `_propose` default `max_attempts=2 → 1` + docstring stating why it MUST stay 1. pxd unchanged (`=?`). Theory: THEORY_2_13.md §3.2. |

---

## Fix (d): NaN root_posteriors on node_list init

**Issue:** `_init_trees` scores only the fresh-build branch; `node_list` path leaves posteriors as NaN. Lazy cache-fill in `_step` is commented out (deprecated), so `old_post = NaN` → silent hard rejection forever.  
**Location:** `groveC.pyx:268–288` (_init_trees), `groveC.pyx:704–712` (_step lazy-fill block)  
**Severity:** High (silently breaks any Grove built with node_list)  
**Change:** Restore lazy fill via `log_marginal_bic` (mirror `_score_post`), and/or score the node_list branch too in `_init_trees`.  

| Status | Date | Notes |
|--------|------|-------|
| ✅ IMPLEMENTED — pending recompile | 2026-07-09 | `groveC.pyx` `_step`: dead commented block replaced by live `if np.isnan(...): root_posteriors[i] = self._score_post(trees[i], y, True)` — shares validity gate + BIC cache with candidate scoring, so old/new comparable. Pair-crossover helper lazy-fills BOTH slots the same way. |

---

## Fix (c): Pólya auto-register ordering bias

**Issue:** `old_prior` computed before `extract_path(cand)` may auto-register new symbols and grow Dirichlet. Old and new priors evaluated against different `n_total` → incumbent favored.  
**Location:** `groveC.pyx:775–779` (_step prior block)  
**Severity:** Medium (biases tree acceptance by ~(#new-symbols)·log((n+1)/n) per step)  
**Change:** Extract both paths first (registration happens), then evaluate both priors against the final Θ.  
**Related:** Same ordering issue in `legacy_step`, Forest swap/crossover (harmless there only if both trees pre-registered).  

| Status | Date | Notes |
|--------|------|-------|
| ✅ IMPLEMENTED — pending recompile | 2026-07-09 | Extract-then-score applied at SIX sites in `groveC.pyx`: `_step`, `legacy_step` (+ `old_path` decl), `_attempt_swap_pair`, `_attempt_crossover_pair` (all four paths extracted before any scoring; old paths captured BEFORE the in-place crossover mutates the trees), the two `run_mpi_spawn` master blocks (same before-mutation capture; MPI note: frozen 32-name vocab makes master-side registration — which would desync replicas — practically unreachable), and the new `_step_pair_crossover`. Theory: THEORY_2_13.md §3.1. |

---

## Fix (b): Within-grove crossover reversibility

**Issue:** `_propose` deep-copies donor, swaps subtrees, on accept installs only slot i. Donor copy (holding S_v) discarded → reverse density is zero, yet `lqr` uses pair-swap node counts. Detailed balance broken.  
**Location:** `groveC.pyx:522–533` (within-grove crossover in _propose)  
**Severity:** High (breaks MH correctness for the move)  
**Options:**  
  - **(b1, recommended)** Genuine pair move: install both children (slots i and j, same temperature), joint acceptance `ΔU_i/T + ΔU_j/T + (lqr - lqf)`.  
  - **(b2)** Drop within-grove crossover entirely (Forest trans-grove crossover already supplies recombination).  

| Status | Date | Notes |
|--------|------|-------|
| ✅ IMPLEMENTED as (b1) — pending recompile | 2026-07-09 | `groveC.pyx`: (1) `_propose` returns a 6-tuple `(cand, lqf, lqr, move_name, cand2, donor_idx)` — donor child kept, validity required of BOTH; (2) `_step` unpacks and dispatches `cand2 is not None` to the new `cdef bint _step_pair_crossover(y, i, j, candA, candB, lqf, lqr)` — lazy fills both slots, scores both via `_score_post`, extract-then-score priors (c-discipline), joint `log_alpha = dtarget/T + q_corr` with the same greedy-T convention, installs BOTH trees + posts on accept, reinforces BOTH paths (deferred under polya_defer); (3) `groveC.pxd`: helper declared. size==1 fallback to point_mutation unchanged. Theory: THEORY_2_13.md §2. |

---

## Fix (a): θ-bookkeeping inconsistency (the deep one)

**Issue:** Target is θ-free (profiled BIC), but proposal corrections include g(θ) densities. That bookkeeping is exact *only if θ is retained in the chain state*. It isn't: scorer immediately overwrites θ → θ̂, parameter_refresh diffuses θ under a different law. Result: "Monte Carlo within Metropolis," not valid pseudo-marginal → topology marginal is biased.  
**Location:**  
  - Birth/death g terms: `gp_movesC.pyx:879–881` (point_mutation case c/d)  
  - Build densities: `gp_movesC.pyx:197–217` (_eval_build_density)  
  - ETR: `gp_movesC.pyx:1890–1897` (_etr_mutation)  

**Severity:** Critical (biases the entire topology posterior)  
**Recommended fix (a1):** Topology-marginal view: θ̂ is deterministic per topology (cache enforces first-visit determinism per y). Drop *all* g(θ) terms from proposal corrections — keep only structural factors (Bernoulli p_par, uniform choices, etc.). Birth draw θ~g stays as warm-start noise (integrates to 1). Parameter_refresh then becomes harmless (perturbs warm starts only).  
**Cost:** Rewrite _point_mutation BIRTH/DEATH cases, _eval_build_density, _etr_mutation; update test_phase23 3.3(b)/(c) expected densities.  

| Status | Date | Notes |
|--------|------|-------|
| ✅ IMPLEMENTED (a1) — pending recompile + test run on Linux | 2026-07-09 | Decision: (a1) chosen. See change list below. |

**Changes applied (2026-07-09):**

`gp_movesC.pyx` — five edits, all deletions of log-g density terms (the θ~g
draws themselves are KEPT everywhere as optimizer warm starts):
1. Module comment block at `_sample_g`/`_log_g` (~line 40): documents g's
   new warm-start-only role; points to THEORY_2_13.md §1.
2. `_point_mutation` DEATH branch: removed `theta_dying` capture and
   `log_q_rev += _log_g(theta_dying, g_tau)`. Now: `lqf = -log M, lqr = 0`.
3. `_point_mutation` BIRTH branch: removed
   `log_q_fwd += _log_g(theta_new, g_tau)`. Now: `lqf = 0, lqr = -log M`.
   Docstring cases (c)/(d) updated to the structural contract.
4. `_eval_build_density`: removed the `log_g_sum` accumulation over
   parameter leaves; returns structural factors only
   (K_S·log p_par + (2^d−K_S)(log(1−p_par) − log M) − (2^d−1)·log O).
   Signature UNCHANGED (g_tau kept but unused) so gp_movesC.pxd and all six
   call sites (add_node, split, leaf_hoist, internal_hoist, legacy
   hoist_mutation ×2 branches) are untouched — they inherit the fix through
   the shared helper.
5. `_etr_mutation`: removed the `old_param_leaves`/`log_g_old` accumulation
   and dropped `+ log_g_new` / `+ log_g_old` from lqf/lqr. Docstring
   density formulas updated. `_draw_leaf_et`/`_build_random_et` signatures
   unchanged (still return their log-g sums; caller ignores them).

`gp_movesC.pxd` — NO changes (signatures preserved by design).

`test_phase23.py` — three tests re-targeted to the structural contract:
1. `test_3_3b_birth_density`: rewritten. Branch detection now reads the
   TREE (which leaf flipped kind) instead of inferring from densities;
   asserts BIRTH = (0, −log M), DEATH = (−log M, 0), and that the born
   warm-start value is finite. The old θ-recovery round-trip via the
   Gaussian quadratic form is gone (no g in densities anymore).
2. `test_3_3c_point_mutation_symmetric`: regime signatures updated to
   BIRTH = (0, −log M), DEATH = (−log M, 0); docstring updated.
3. `test_3_3c_etr_density`: `expected_lqf` no longer includes log_g_new;
   born values asserted finite; dead `g_tau` local removed.

NOT changed (already g-free or unaffected): `_subtree_mutation`,
`_subtree_crossover`, `parameter_refresh`/AMState, groveC._step (acceptance
formula unchanged — it just consumes the corrected lqf/lqr),
add_node/split/leaf_hoist/internal_hoist selection tests (they never
asserted g terms).

**Verification still owed (Linux box):** recompile gp_movesC (+ anything
that cimports it), run test_phase23.py (3.3b/3.3c updated), test_cython4.py,
test_phase28.py, test_phase29.py, test_phase2mpi.py. NEW proposed test:
`test_a1_toy_occupancy` (THEORY_2_13.md §1.7) — two-topology birth/death
chain must recover 1:1 occupancy regardless of θ̂ and τ.

### Mathematical appendix (precise statement, 2026-07-09)

State stored by the code is x = (T, θ); acceptance uses the θ-free target
π(T) ∝ exp(−BIC(T)/2 + log π_P(T|Θ)). Collect structural factors of a
birth/death pair into A = [π(T')/π(T)]·[s_rev·(1/M)/s_fwd]. The code accepts

    birth:  α = min{1, A / g(θ*)},        θ* ~ g = N(0, τ²)   (g enters via lqf)
    death:  α = min{1, g(θ_stored) / A}                        (g enters via lqr)

These are the EXACT RJMCMC acceptances for the improper product target
π(T) ⊗ Lebesgue(dθ), because detailed balance holds POINTWISE in θ*:
z·min(1, w/z) = min(z, w) with z = g(θ*), w = A — PROVIDED the θ* drawn at
birth is the same θ the death later evaluates g at (retention).

Retention is violated twice:
  1. The scorer replaces θ* by θ̂(T') (optimize / cache re-install) BEFORE
     accept. Realized topology-marginal probabilities become
        a_birth = E_{θ*~g}[min{1, A/g(θ*)}]   (an average)
        a_death = min{1, g(θ̂)/A}             (a point evaluation)
     and invariance would need E[min{g(θ*), A}] = min{g(θ̂), A} — false in
     general ⇒ net flow ⇒ π not stationary. This is Monte-Carlo-within-
     Metropolis, NOT pseudo-marginal (no unbiased retained estimator).
  2. parameter_refresh preserves ∝ L(y|T,θ)^{1/T} in θ; the optimizer
     projection preserves δ_{θ̂}; the structural kernel (retained reading)
     preserves π⊗Leb. Three kernels, three invariant laws ⇒ the composite
     chain has no common stationary distribution.

Exact toy bias: M=1, π(T0)=π(T1), A=1, τ=1 ⇒ a_birth = 1 (φ(θ*)<1 always),
a_death = φ(θ̂) ⇒ stationary odds T1:T0 = 1/φ(θ̂) = √(2π)·exp(θ̂²/2) instead
of 1. θ̂=2 → 18.5:1; θ̂=4 → ≈7500:1. General bias factor per parameter:
exp(θ̂²/2τ²) — huge for parameters whose fitted values ≫ τ = std(y)
(frequencies, offsets). Direction: chain hoards parameters far from 0 on the
τ scale (death prob ∝ g(θ̂) collapses).

Fix (a1) in this language: declare state = T only; θ̂(·) a deterministic map
(frozen per topology per y by _bic_cache); the birth draw θ* does not affect
which T' results, so it integrates out of q: q(T'|T) = q_struct(T'|T).
Acceptance = min{1, π(T')q_struct(T|T') / π(T)q_struct(T'|T)} — no g terms.
Per move: point_mutation birth q_corr = −log M, death = +log M;
_eval_build_density keeps K_S·log p_par + (2^d−K_S)(log(1−p_par) − log M)
− (2^d−1)·log O, drops Σ log g; ETR keeps −logΩ − log2 − log s(o), drops both
log-g sums. PR becomes warm-start jitter (reads/writes nothing the acceptance
uses) — harmless. Note: in EVERY coherent construction (a1 or a2) the naked
g factors vanish from the final ratio (in a2 they cancel against the g-prior);
their survival in the current code is itself the red flag.

Residual caveat under (a1): θ̂ must be path-independent — the cache enforces
this ex post (target defined by frozen first-visit values); a badly converged
first visit under-scores that topology permanently (quality, not balance).

---

## Fix (g): Single-leaf root pairing edge case

**Issue:** `subtree_mutation` on a leaf root silently delegates to `add_node` but is selected with `p_subtree ≠ p_leaf_hoist` → Green move-choice factor doesn't cancel for that transition.  
**Location:** `gp_movesC.pyx:312–315` (subtree_mutation early return)  
**Severity:** Low (edge case; most trees are not single leaves)  
**Change:** Return `(0, -inf)` (no-op) from subtree_mutation on single-leaf trees, or dispatch under add_node's own name/weight.  

| Status | Date | Notes |
|--------|------|-------|
| ✅ IMPLEMENTED — pending recompile | 2026-07-09 | `gp_movesC.pyx` `_subtree_mutation`: single-leaf root now returns `(0.0, -inf)` (forced reject, tree untouched) instead of silently delegating to add_node under p_subtree's selection weight. Irreducibility unaffected (add_node/split/ETR grow single leaves under their own weights). No test relied on the delegation (density fixtures are never single-leaf); grove runs just count a rejected proposal. Theory: THEORY_2_13.md §3.4. |

---

## Fix (h): NaN hiding in nanmean fitness

**Issue:** `objective`/`fitness` use `np.nanmean((y-f)²)`: partial-domain NaN is scored on the finite subset with full-N BIC penalty. Optimizer can exploit this.  
**Location:** `real_functionC.pyx:298` (fitness), `groveC.pyx:737–741` (cache logic)  
**Severity:** Low-medium (the optimizer CAN actively settle in NaN-exploiting optima)  
**Change:** Re-validate at θ̂, or let NaN propagate (mean instead of nanmean).  

| Status | Date | Notes |
|--------|------|-------|
| ✅ IMPLEMENTED (re-validate at θ̂) — pending recompile | 2026-07-09 | `function_treeC.pyx` `Tree._log_marginal_bic`: after optimization, `if not self.is_valid(self.domain): return -inf` — on BOTH branches (fast `_update_parameters_and_fitness` path and slow `_update_parameters` path). Chosen over mean-instead-of-nanmean: surgical, uses the SAME validity criterion (finiteness + max_abs 1e12) as the pre-scoring gate, so no asymmetry. If validation passes, the residuals at θ̂ are all finite ⇒ the optimizer's nanmean == mean, closing the partial-domain loophole exactly. K=0 / optimize=False paths unchanged (values not moved; caller's pre-gate covers them). Theory: THEORY_2_13.md §3.5. |

---

## Fix (i): profiled-σ BIC penalty k → k+1

**Issue:** With `sigma=None`, σ̂² = SSR/N is *estimated*, so σ counts as a parameter; Phase 2.7(a) promised the k+1 penalty but both scorers used k. Constant +0.5·log N shift across all models → MH ratios unaffected; reported evidences and fixed-vs-profiled comparisons were off.  
**Location:** `function_treeC.pyx` `_shortcut_bic` (σ=None branch), `real_functionC.pyx` `Node._log_marginal_bic`  
**Severity:** Low (cosmetic for sampling; matters for reported evidence)  

| Status | Date | Notes |
|--------|------|-------|
| ✅ IMPLEMENTED — pending recompile | 2026-07-09 | Both σ=None scorers now penalize `0.5*(k+1)*log N` (fixed-σ branches unchanged: σ known ⇒ k params). Side effect: burn-in's `root._log_marginal_bic(y)` (params=None → k=0) shifts by a constant −0.5·log N — rank-preserving, harmless. Tests updated in `test_phase23.py`: `test_3_3d_bic_formula_consistency` expected value → k+1; the 3.3(d) header comment and `test_3_3d_bic_sigma_modes_differ` docstring formulas updated. Ordering/finiteness tests (3.3d penalty/finite) unaffected (constant shift preserves both). |

---

## Fix (f): `operations` plumbing (build_tree vs groove)

**Issue:** `build_tree` samples ops from hardcoded 6-op list ([function_treeC.pyx:1001](function_treeC.pyx#L1001)); `PolyaTree.__init__` silently overrides `operations` arg; densities use caller's `len(operations)`. Consistent *only* for default 6 ops. Test_phase29 uses 4-op fixture → densities numerically wrong.  
**Location:** `function_treeC.pyx:1001`, `polya_treeC.pyx:77`, `gp_movesC.pyx` density calculations  
**Severity:** Medium (affects non-default operation sets)  
**Change:** Thread `operations` through `build_tree`/_build_subtree_with_params, or enforce `operations == canonical 6` at Grove.__init__.  

| Status | Date | Notes |
|--------|------|-------|
| ✅ IMPLEMENTED (threading) — pending recompile | 2026-07-09 | PROMOTED from second tier (direct suspect for the 2.8c failure — its fixture runs the 4-op safe list). Changes: (1) `function_treeC.pyx` `build_tree` gains trailing `list operations=None`; op drawn as SYMBOL from it, branch dispatch on symbols, `raise ValueError` on unknown symbol; default list ordered to match the old internal list index-for-index → default-ops RNG streams bit-identical (determinism tests unaffected); recursion threads it down; (2) `function_treeC.pxd` declaration updated; (3) `Tree.__init__` gains `operations=None`, forwarded to build_tree (support closure of initial trees); (4) `groveC._init_trees` fresh-build branch passes `operations=self.operations`; (5) `gp_movesC.pyx` `_build_subtree_with_params` forwards `operations` (fixes add_node/split fwd + hoists rev densities via the shared `_eval_build_density`, whose `-log O` now matches the true sampling law); (6) `subtree_mutation` gains `operations` (rebuild + root-leaf add_node delegation; support closure — safe-ops kernels no longer leak '**'/'o'); `gp_movesC.pxd` + `groveC._propose` call site updated. CAVEAT (documented): user-supplied `node_list` trees must respect the kernel vocabulary (unchecked). PolyaTree still hardcodes the 6 ops for its own vocabulary — prior-side only, target-consistent, untouched. Theory: THEORY_2_13.md §3.3. |

---

## Test-failure log

### 2026-07-09 — test_phase28 2.8a CRASH after (f) recompile — HOTFIXED
`NameError: name 'build_tree' is not defined` at gp_movesC.pyx:364
(and latently :238). **Cause:** Cython gotcha — a cpdef function called
ACROSS a cimport boundary cannot take keyword arguments; the kwarg call
`function_treeC.build_tree(..., operations=operations)` bypasses the C fast
path and falls back to a Python lookup of 'build_tree' in gp_movesC's own
module dict, which fails. In-module kwarg recursion inside function_treeC
and cdef-class ctors (Tree(..., operations=...)) are unaffected.
**Hotfix:** both gp_movesC call sites now call build_tree FULLY POSITIONALLY
(middle args spell out the signature defaults: None, 0, None, None, None,
1.0, 0.0, 0.01, 1.0, 1.0, operations) with an explanatory comment. No
function_treeC changes; only gp_movesC needs recompiling.
**Rule for future edits:** never use kwargs on cross-module cpdef calls.



### 2026-07-09 — test_phase28 2.8c FAIL (first run after (a1) recompile)
`TV(off,on)=0.263 > TV(off,off)=0.103 + 0.15`. 2.8a/2.8b PASS.

**Diagnosis:** 2.8c is a weight-perturbation test — with exact acceptance
math the stationary law is independent of move weights, so any failure
indicates a move whose bias *scales with its weight*. Live weight-dependent
biases in this build, in order of suspicion:
  1. **(f) — ACTIVE IN THIS FIXTURE**: test uses `_OPS_SAFE` (4 ops) but
     build_tree samples subtree ops from the hardcoded 6; build densities
     off by n_int(S)·log(6/4) per built subtree. Not absorbable into a
     modified target (ETR/crossover provide correctly-densitied alternate
     size routes → Kolmogorov loop criterion fails) → invariant law is
     mixture-dependent → adaptation shifts it. (f) upgraded from second
     tier to REQUIRED for 2.8c/2.9c fixtures.
  2. **(b)** within-grove crossover irreversibility (crossover in kernel,
     grove size 3).
  3. **(e)** _propose retry mixture.
  4. Test power: exceedance is marginal (0.263 vs 0.253); 400 rounds with
     gamma_t = 1/(100+t) leaves a large inhomogeneous transient in the
     collected window that the off/off null cannot calibrate.
Excluded: Pólya drift (fixture freezes t=1e8, target fixed both arms);
(a1) itself (contract change is identical in both arms — though new
acceptance rates change where the logits drift, exposing 1–3 differently).
Note: likely the FIRST 2.8c run with ETR active (test's _MOVE_LABELS still
lists 7 moves — predates 2.11).

**Action:** implement (e)+(d)+(c)+(b1)+(f) in one batch → recompile → rerun.
If still marginal, raise rounds 400→1200 (power fix, not correctness).
**(b) decision REVISED:** (b1) (install both children, joint acceptance)
now recommended over (b2) — delisting the move fights the adapt_floor
containment (every physical move is floored ≥ eps) and would restructure
the 8-move dispatch; (b1) is localized to _propose/_step.

---

## New finding (m): Tree.__deepcopy__ drops g_tau (2026-07-09, found while writing the a1 test)

**Issue:** `function_treeC.pyx` `Tree.__deepcopy__` rebuilds via
`Tree(depth, functions, x, root, Npars, value, T, alpha, sigma)` — `g_tau`
is NOT forwarded, so every deepcopy re-defaults to `g_tau = 1.0`. Since
`Grove._propose` deepcopies the current tree before every move, all
proposal-side g_tau reads happen on the copy:
  - POST-(a1): impact is minor — g_tau no longer enters any density; the
    only effect is that birth WARM-START draws use N(0, 1) instead of
    N(0, std(y)²). Wrong scale for the optimizer's starting point; no
    correctness impact on the chain.
  - PRE-(a1) (historical note): the g-densities themselves were evaluated
    with tau = 1.0 regardless of the configured value — yet another
    inconsistency in the old bookkeeping, now moot.
**Fix:** forward `self.g_tau` as the 10th positional arg to the `Tree(...)`
constructor in both `Tree.__deepcopy__` and `Tree.__copy__`.  
**Test note:** `test_a1_toy_occupancy`'s `set_g_tau` re-install is now
belt-and-suspenders (idempotent), retained as a regression guard.

| Status | Date | Notes |
|--------|------|-------|
| ✅ IMPLEMENTED — pending recompile | 2026-07-09 | `function_treeC.pyx` `__deepcopy__` + `__copy__` now pass `self.g_tau` (slot after sigma). Post-a1 impact is warm-start scale only (no density); the value was historically re-defaulting to 1.0 on every `_propose` deepcopy. Constructor calls are Python-level (cdef-class ctor), so positional args are safe. |

---

## New test: test_a1_toy_occupancy (2026-07-09) ✅ WRITTEN — pending run

`test_phase23.py::test_a1_toy_occupancy` — the sharpest regression pin for
fix (a1) (THEORY_2_13.md §1.4/§1.7). Two-topology chain (function leaf ↔
param leaf) under point_mutation with M=1, UNIFORM target (dtarget=0), and
a theta_hat projection after every accepted birth (simulating the
optimizer). Post-fix: every kind-flip accepted ⇒ occupancy 0.5 ± 0.05
(>6σ band) for BOTH theta_hat ∈ {2, 4}. Pre-fix code would read ≈0.95 and
≈0.9999 (parameter hoarding via death prob = g(theta_hat)) — loud failure.
Registered in the __main__ runner.

---

## Phase 2.9 R2 deliverables (2026-07-09) ✅ WRITTEN — pending run

**(a) Proof sketch** — THEORY_2_13.md §4: connectivity lemma for the
mutation kernel (ETR alone connects all shapes through the single leaf —
2-ET→0-ET reductions down, 0-ET→2-ET constructions up — and
point_mutation connects leaf kinds / op labels; the perfect-binary moves
conserve parities and are NOT needed for connectivity, only for mixing),
plus the ergodicity corollary (positive target on finite depth-capped
space ⇒ irreducible + aperiodic ⇒ uniformly ergodic; adaptation layered
via R&R 2007). Feeds paper §7.7 / Appendix C.

**(b) Strong empirical test** — `test_phase29.py::test_2_9d2_full_support_coverage`:
micro-space M=1 ("x"), O=1 ("+"), depth cap 2 ⇒ exactly 38 enumerable
parse-structures. Driver = raw proposal kernel with always-accept-on-support
(valid + finite lqr) — faithful for reachability since MH accepts every
on-support proposal with positive probability; avoids the flakiness of
driving a concentrating BIC target. Asserts SET EQUALITY per seed:
coverage (⊇, the R2 claim) AND closure (⊆, checked every step with the
offending move named — doubles as a regression test for the 2.13(f)
vocabulary threading and the depth caps). Signatures built from the raw
Node structure, NOT tree_get_description() (sympy canonicalizes x+x → 2*x
and would conflate distinct parse trees). 3 seeds, early-stop on coverage,
max 12k steps (>6× the typical coupon-collector time; runtime dominated by
tiny-tree deepcopies).

**Cleanup:** removed a stray per-step `print(st)` in the pre-existing
`test_2_9d_within_chain_irreducibility` (thousands of lines of debug spam
per run); its header now labels 2.9d "dynamic form" vs 2.9d2 "strong form".

---

## Phase 2.10 Stage 1 — DP completion of the Pólya prior (2026-07-09) ✅ WRITTEN — pending compile+run

Implements DESIGN_2_10_PRIMITIVES.md §9 item 1 / THEORY_2_13.md §5.
**Opt-in and legacy-safe:** `remainder_mass=0` (the default everywhere) is
bit-identical legacy behaviour; no existing caller changes.

`polya_treeC.pyx` / `.pxd` changes:
1. `PolyaNode.alpha_rem` — the remainder cell as a SEPARATE scalar, not an
   array slot; excluded from `alpha_total`, so `log_prob`/`sample_choice`/
   `update` are UNCHANGED (restriction by neutrality, THEORY §5.4).
2. `PolyaTree.__init__(..., remainder_mass=0.0)` — >0 activates dp_mode;
   fixes total base mass c = (n_ops+n_prims)·alpha_0 + remainder_mass;
   new fields dp_mode / rem_base / base_alphas / base_reg_total.
3. `_ensure_child` — fresh lazily-created nodes inherit CURRENT
   base_alphas + rem_base in dp_mode (keeps every node a projection of the
   same DP).
4. `log_prior` unvisited branch — dp_mode uses the base-measure predictive
   log(base_alphas[cell]) − log(base_reg_total) (equals −log n_total when
   masses are symmetric); legacy branch untouched. Also PROMOTED
   `log_prior` cdef → cpdef (additive; lets tests probe hand-built paths).
5. `add_primitive(desc, mass=None)` — dp_mode CARVES mass (default
   alpha_0) from the remainder at every node (`_grow_all_nodes(new_mass)`:
   append slot = mass, alpha_total += mass, alpha_rem −= mass ⇒
   alpha_total + alpha_rem conserved per node); raises on remainder
   exhaustion and on custom mass in legacy mode. Cross-module callers must
   pass `mass` POSITIONALLY (cpdef+cimport kwargs trap).
6. `get_state`/`set_state` + `_node_to_state`/`_node_from_state` — DP
   fields serialized; read back with legacy-safe defaults (pre-2.10 MPI
   snapshots restore as legacy trees).

**New test file `test_phase210.py`** — the §5 lemmas executable:
(a) legacy regression; (b) conservation per node after reinforcement
(Lemma 5.2); (c) registration-order commutation on probe symbol sequences
(Theorem 5.3, dyadic masses for bit-exactness); (d) unvisited base-measure
predictive with a custom-mass primitive; (e) fresh-node inheritance;
(f) both error paths; (g) DP state round-trip + stripped pre-2.10 snapshot
restore.

**To run:** recompile polya_treeC (pxd changed ⇒ also rebuild its
cimporters: groveC, mpi_worker), then `python test_phase210.py` and the
full existing suite (legacy invariance is the claim — everything must stay
green untouched).

---

## Retest checklist

Once fixes land, recompile and run (in order):
- [ ] (e) / (d) only: `test_cython4.py` (smoke test, should pass faster)
- [ ] (e) / (d) only: `test_phase23.py` (all 3.3 sub-tests)
- [ ] (c) / (b) added: `test_phase29.py` (2.9 suite: shared Θ, trans-grove moves, target invariance)
- [ ] (a) added + 3.3(b)/(c) expectations updated: `test_phase23.py` again
- [ ] (f) fixed: `test_phase29.py` with 4-op fixture (verify densities)
- [ ] (b1 or b2 decided): check Forest.attempt_crossovers does NOT call Grove within-grove crossover, or that the within-grove variant (if kept) lands correctly in the pair-move framework

---

## Notes & links

- Full findings: [PLAN.md §2.13](PLAN.md#213-soundness-audit-2026-07-09-full-code-bayesian-review--new-hard-gate-before-phase-4)
- Critical path updated: Phase 2.13(a)–(e) → then Phase 4
- θ-refactor decision point: (a1) topology-marginal (recommended) vs (a2) full product-space RJMCMC
- Build_tree ops issue affects test_phase29 fixture; other tests use default 6 ops (pass despite the bug)
