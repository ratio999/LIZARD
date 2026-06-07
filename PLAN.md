# Implementation Plan — Bayesian Symbolic Regression (Pólya-GP MCMC)

**Goal**: Take research-prototype GP-MCMC SR tool to publishable artifact (PhD thesis chapter + ≥1 technical paper).
**Headline novelty**: Adaptive Pólya-tree structural prior over GP construction paths inside parallel-tempered MCMC.

## Critical-path summary
Phase 0 → Phase 1.3 + Phase 2.1 + Phase 2.2 (proposal-correction work) → **Phase 2.7 (SOLID POSTERIOR — hard gate)** → Phase 2.3 (vocabulary) → Phase 3.3 (tests) → Phase 4.1 (sanity) → Phase 4.3 (ablations) → Phase 4.2 (hero) → Phase 5 (writing).
Phase 1.1, 1.2, 1.5, 1.6 can run in parallel with the Phase 2 code work.

---

## Phase 0 — Scope
Pick venue (methods journal vs ML conf vs astro), headline claim (recommend astro-application with method extractable as separate paper), and one hero benchmark (Mc recovery / CCSNe / Feynman).

---

## Phase 1 — Theory (parallelizable with Phase 2)

### 1.1 Formal model as adaptive PCFG with Dirichlet productions
### 1.2 Diminishing adaptation γ_t=1/(1+t); ergodicity via Roberts-Rosenthal containment.
### 1.3 Proper MH acceptance ratio (this is the single biggest formal hole)
- Derive log α = log L(T') − log L(T) + log π(T'|Θ) − log π(T|Θ) + **log q(T|T') − log q(T'|T)**.
- Per move (point_mutation, hoist, add_node, subtree_mutation, subtree_crossover):
  - Identify proposal kernel q(·|·) explicitly as a product of (a) selection probability of the modification site and (b) generative probability of the replacement subtree.
  - Derive log q(T'|T) and log q(T|T') in closed form OR show the move is symmetric.
  - For trans-dimensional moves (add_node, subtree_mutation that changes size, hoist), follow Green 1995 RJMCMC: include the |Jacobian| term. Here = 1 because θ is profiled out (not transformed), so no continuous-parameter Jacobian appears in the move correction. State and justify carefully; cross-ref Phase 1.6 / Phase 2.7(b).
- Decide for each move: (i) keep with explicit ratio, (ii) restrict to a symmetric variant, or (iii) drop.

### 1.4 Lazy-trie / unvisited-region lemma.
### 1.5 Vocabulary growth (add_primitive) invariance.

### 1.6 Marginal likelihood (committed: BIC-as-Laplace; see Phase 2.7)
- Likelihood: y | θ, T ~ N(f_T(x; θ), σ² I).
- Treatment of θ: profile/MAP via scipy.optimize (not marginalization). The
  resulting plug-in is upgraded to the BIC penalty in Phase 2.7.
- Identification: under flat prior on θ, regularity at θ̂, iid data,
    log p(y|T) = log L(y|T, θ̂) - 0.5 * k log n + O(1) = -BIC/2 + O(1).
  This is the Schwarz / BIC-as-Laplace result. Cross-ref Phase 2.7(b).
- Jacobian on θ in RJMCMC moves: =1 because θ is integrated out via
  profiling, not transformed within the MH proposal. Cross-ref Phase 1.3.
- σ treatment: configurable per run (see Phase 2.7(a)) — either fixed
  (user supplies noise scale, e.g. CCSNe instrument) or profiled
  (σ̂² = SSR/n, then k → k+1 in BIC). Both supported; user decides per data.
- Deferred upgrades (future work, not v1): full Laplace with -0.5 log|H|;
  pseudo-marginal MCMC (Andrieu & Roberts 2009) if optimizer noise is problematic.

### 1.7 Identifiability: prior on parsed trees, not equivalence classes.

---

## Phase 2 — Code: critical correctness (blocks all benchmarking)

### 2.1 Make moves report proposal log-densities (gp_movesC)
- Refactor each public move to return (new_tree, log_q_forward, log_q_reverse) instead of just mutating in place.
- Catalog of moves to update:
  - point_mutation — see 2.1.PM below
  - hoist_mutation — derive both, OR remove if too painful
  - add_node — derive both (forward = pick leaf · choose op · sample subtree; reverse = pick the new internal node · collapse to one child · choose which child)
  - subtree_mutation — derive both (selection × generation densities)
  - subtree_crossover — derive both for the swap with renaming
- Delete dead/broken: root_replacement, prune_node (broken), size_fair_mutation (commented), old add_node.

#### 2.1.SM (subtree_mutation v1.5, parameter-recycling — DONE ✅)
- Uniform selection over non-param-leaf nodes; |N(T)| in correction.
- Fresh perfect-binary subtree of height d_max via build_tree(Npars=0).
- Old K parametric leaves recycled (names+values) into K uniformly-chosen
  leaf positions of the new subtree, name-sorted assignment.
- Net log_q_rev - log_q_fwd = log|N(T)| - log|N(T')| on support (S perfect-binary
  of height d_max), -inf off support.
- K is preserved by this move (recycling discipline, Phase 2.1-only).

#### 2.1.PM (point_mutation v2 — RJMCMC kind-crossing — NEXT ⏳)
- Commits to v2 from the start (kind-crossing allowed: function ↔ parameter).
- Four sub-cases:
  (a) internal → internal: resample op ω' ∈ Ω uniformly. Symmetric (returns 0,0).
  (b) fun leaf → fun leaf: resample f' ∈ F uniformly. Symmetric (returns 0,0).
  (c) fun leaf → param (BIRTH): select leaf (1/|all-leaves|), choose "param" with prob p=1/2,
      draw θ_new ~ g (Normal(0, τ²) or Unif(-c,c)); new param name = a_{K+1}.
      log_q_fwd += log(p) + log(g(θ_new)).
      log_q_rev += log(1-p) + log(1/M)  [reverse is a death].
  (d) param leaf → fun (DEATH): select param leaf (1/|param-leaves|), choose "function",
      pick f' ∈ F uniformly; remove θ_j from vocabulary.
      log_q_fwd += log(1-p) + log(1/M).
      log_q_rev += log(p) + log(g(θ_j))  [reverse is a birth at the dying value].
- Selection: uniform over all leaves for (c)/(d); uniform over all non-param-leaves
  for (a)/(b). Mixture weight p=1/2 between function and param relabeling.
- g(θ) = Normal(0, τ²) with τ = std(y) (or configurable). NOT a Dirac (Dirac is degenerate).
- After birth: run full update_parameters (joint re-optimization). Warm-start from θ_{old} + 0.
- After death: run update_parameters with K-1 params. Surviving param names are unchanged.

#### 2.1.PR (Parameter-refresh move — DONE ✅)
- Separate from point_mutation. Decouples continuous-θ adaptation from structural changes.
- AMState: adaptive MH state per tree topology (covariance adapts between visits).
- Symmetric proposal ⇒ log_q_rev - log_q_fwd = 0.
- Called as a dedicated move within _step.

#### 2.1.K (K-CHANGING MOVES — must remain free)
The following moves CHANGE K and must NOT be locked to a fixed K:
  * hoist_mutation: K can decrease (sibling subtree discarded, params in it lost).
  * add_node: K can increase (new subtrees may contain parameter leaves).
  * subtree_crossover: K can change in BOTH trees (subtrees swapped between trees).
  * point_mutation v2 (case (c)/(d)): explicit birth/death of one parameter.

When K changes, MH ratio must include:
  1. Vocabulary registration: update Pólya/vocabulary for new/removed symbols.
  2. Birth density g(θ_new) in the acceptance ratio (Green 1995 RJMCMC).
  3. Death: evaluate g at the dying parameter value (time-reversal symmetry).

These items are DEFERRED from v1 of hoist/add_node/crossover (Phase 2.1 v1 uses
simplified bookkeeping). Phase 2.3 will centralize vocabulary; Phase 1.3 documents
the Jacobian = 1 argument.

### 2.2 groveC._step consumes log_q correction; tempering only on (post+prior).
- Wire (log_q_fwd, log_q_rev) returned by each move into the acceptance computation.
- log α = β * [log π(T') - log π(T)] + [log q_rev - log q_fwd].
- Proposal correction is NOT tempered.

### 2.3 Single source of truth for vocabulary; auto-register fresh symbols.
- Pólya structural prior ✅ done (integrated).
- Vocabulary registration for K-changing moves: pending.

### 2.4 Fix or remove fake gradient in real_functionC.
### 2.5 Determinism & reproducibility
'''- Plumb a `seed` argument through Grove/Forest/moves down to numpy.random and python.random.
- Document scipy optimizer non-determinism. Note: per-tree warm-start of θ
  is handled by Phase 2.7(e) (implicit via `self.name_values` inheritance
  + Phase 2.7(f) caching).'''
  
2.5(a) — Grove.__init__ accepts seed=None; store _py_rng = random.Random(seed) and _np_rng = np.random.default_rng(seed); add the two fields to groveC.pxd. No call-site changes yet.
2.5(b) — parallel with 2.5(c). Inside groveC.pyx replace global RNG calls with self._py_rng.* / self._np_rng.*. Targets: _choose_move (groveC.pyx:233), _step PR coin and u-draw (groveC.pyx:431, groveC.pyx:552), _step_parameter_refresh, _propose donor pick, legacy_step, parallel-tempering swap.
2.5(c) — parallel with 2.5(b). Add set_rng(py_rng, np_rng) module setter + _PY_RNG, _NP_RNG globals to gp_movesC.pyx (mirrors set_g_tau). Grove calls it once at construction.
2.5(d) — depends on 2.5(c). Replace random.* / np.random.* inside gp_movesC.pyx with the module RNGs. Targets: _select_uniform_node, _sample_g, _sample_truncated_geometric, _build_subtree_with_params, _point_mutation (kind-cross coin + picks), _subtree_mutation (slot sample), _subtree_crossover (selection), _hoist_mutation, _add_node, _split, AMState._propose (np.random.normal / standard_normal), any reachable legacy_*.
2.5(e) — Forest derives per-Grove seeds via np.random.SeedSequence(seed).spawn(n_groves).
2.5(f) — Audit polya_treeC.pyx for any RNG use (likely no-op since extract_path / update_path are deterministic). Route via same setter if found.
2.5(g) — Docstring on Grove: "Seed controls the structural-MCMC trajectory. Continuous-θ optimization is reproducible only up to scipy/BLAS noise; for bit-exact runs set OMP_NUM_THREADS=1 and pin BLAS."
2.5(h) — depends on (a)–(g); test file creation deferred per freeze. Smoke test: two Groves, same seed, same n_steps → identical tree_get_description() per slot AND identical move_stats(). This is the canonical correctness check for the seed plumbing.

### 2.6 Pólya tree v2 (recommended after Phase 1.1 decision)
- If kind-conditional Dirichlets are chosen: refactor PolyaNode to have separate (α_op, α_prim) blocks; descent and predictive use only the relevant block according to the GP grammar context (which kind is required at this position).
- This puts the prior strictly on the support of T; cleaner math, slightly more bookkeeping.

### 2.7 SOLID POSTERIOR (HARD GATE before any benchmarking) ← CURRENT FOCUS
After all gp_moves are refactored (2.1), and BEFORE any experiments (Phase 4),
the posterior pipeline in groveC._step must be made rigorous and consistent.

Requirements (must all be met):
  (a) **BIC as the model evidence proxy.** ✅ DONE
      log_evidence(T) := -0.5 * BIC(T)
                     := log L(y | T, θ̂_MLE) - 0.5 * k(T) * log n
      where:
        - θ̂ comes from `update_parameters` (existing scipy.optimize call);
        - k(T) = number of effective parameters (start with #params; flag
          identifiability caveats in writeup);
        - n = len(y);
        - log L = -0.5 * n * log(2π σ²) - 0.5 * SSR/σ²
        - σ treatment is **user-configurable per run** (data-dependent):
          * if `sigma` provided (float > 0): fixed-σ BIC, k unchanged.
          * if `sigma=None`: profile σ̂² = SSR/n; then
              -2 log L = n log(2π SSR/n) + n,
              and k → k+1 in the BIC penalty (σ counts as a parameter).
          API exposes both via a single `sigma=None` default.
        - Implementation: `_log_marginal_bic` in `real_functionC.pyx`, `object sigma` arg
          threads through Node → Tree → Grove.
        - RuntimeWarning fix: evaluate wrapped in warnings.catch_warnings + np.errstate(all="ignore").
        - NaN/inf guards: INVALID = -1e300 sentinel returned for bad models.

  (b) **Laplace identification.** ✅ (documented in 1.6)
      Under flat prior, regularity at θ̂, iid data:
        log p(y|T) ≈ -BIC(T)/2 + O(1).

  (c) **Total log-target.**
        log π(T) = -BIC(T)/2 + log π_Pólya(T|Θ)
      Used in MH acceptance (tempered by β = 1/T).

  (d) **MH acceptance rule (final form).** With β = 1/T (tempering):
        log α = β * [log π(T') - log π(T)]
              + [log q_rev - log q_fwd]
      Tempering applies to log π only; proposal correction is NOT tempered.
      When log_q_rev = -inf, skip the log-target evaluation (BIC + Pólya)
      for the candidate and reject immediately.

  (e) **Parameter optimization warm-start (implicit only, v1).**
      `update_parameters` already inits scipy.optimize from
      `self.name_values`, which is automatically inherited from the parent
      tree for any parameter name that survives the move (subtree_mut v1.5
      recycling, point_mut v1 — names unchanged). No explicit cross-tree
      copy needed in v1. Multi-start / explicit cross-tree warm-start
      deferred to future work.

  (f) **BIC topology cache (PENDING ⏳)**
      Cache {topology_key → (log_evidence, θ̂)} keyed on `tree_get_description()`.
      Invalidate only on new y. Avoids redundant optimize calls for re-visited topologies.
      TODO markers at groveC.pyx lines ~361, 393, 398, 459, 533.

  (g) **Numerical guards.** ✅ (implemented in _log_marginal_bic):
      - Ignore NaN/inf from evaluate (bad lambdify expressions).
      - Skip trees with n_eff <= k+1.
      - Return INVALID = -1e300 sentinel instead of -inf.

  Implementation order (6 steps):
    1. _log_marginal_bic in real_functionC.pyx ✅ DONE.
    2. Thread sigma through Node → Tree → Grove ✅ DONE.
    3. Wire _log_marginal_bic into groveC._step ← PENDING (replace get_superficial_posterior).
    4. Add Pólya prior term (log π_Pólya) to log-target ← PENDING.
    5. Implement BIC cache (item f) ← PENDING.
    6. Update test_cython4 grove_smoke_test ✅ DONE (sigma=sigma_data, Npars=2, use_mpi=0).

  Future-work flags (not v1): full Laplace with -0.5 log|H|; pseudo-marginal MCMC
  (Andrieu & Roberts 2009); multi-start optimizer.

### 2.8 Move-rate calibration (after 2.7)
- Feed move_stats() acceptance rates back into _choose_move probability weights.
- Adaptive move weights per temperature level.

---

## Phase 3 — Code: hygiene, structure, tests
### 3.1 Remove dead code.
### 3.2 API documentation.
### 3.3 Test suite (blocks Phase 4).
### 3.4 MPI: decide keep vs drop for paper version.

---

## Phase 4 — Experimental program
Cannot start before Phase 2.1, 2.2, 2.3, **and 2.7** are done (otherwise results are biased).
### 4.1 Sanity checks (recovering known functions)
### 4.2 Hero application (CCSNe / Mc recovery / Feynman — choose one)
### 4.3 Ablations (no Pólya, no tempering, no BIC)

---

## Phase 5 — Writing
### 5.1 Chapter T: theory
  1. Informal motivation
  2. State space T (parse trees, not equivalence classes)
  3. Pólya tree prior construction path
  4. RJMCMC acceptance ratios per move
  5. Parallel tempering schedule
  6. Likelihood treatment: BIC-as-Laplace identification (Schwarz, regularity at θ̂, flat prior on θ, iid data) — see Phase 1.6, Phase 2.7. Identifiability/symmetry handling (Phase 1.7). Configurable σ (fixed vs profiled) and the resulting k+1 bookkeeping. Limitations: redundant params, boundary opt, multimodal likelihood.
  7. Diminishing adaptation + ergodicity
### 5.2 Chapter I: implementation
### 5.3 Chapter E: experiments

---

## Decisions still open (will need vscode_askQuestions later)
1. Target venue / paper count (Phase 0).
2. Hero application choice (Phase 4.2).
3. Kind-conditional vs unified Pólya nodes (Phase 1.1 / 2.6).
4. Keep MPI in the paper version or drop it (Phase 3.4).

## Decisions closed
- **σ in BIC** (Phase 2.7(a)): both modes supported via `sigma=None` (profiled) vs `sigma>0` (fixed). User picks per dataset.
- **Profile vs Laplace marginal** (Phase 1.6): committed to BIC = -2 (log L - k/2 log n) as the v1 evidence proxy; full Laplace deferred.
- **Warm-start scope** (Phase 2.7(e)): implicit only (inheritance via `self.name_values`).
- **K-changing freedom**: hoist, add_node, crossover, point_mut v2 are all allowed to change K. Fixed-K was only Phase 2.1.SM's recycling convention.
- **point_mutation**: goes to v2 (kind-crossing) from the start (no v1-first step).
- **Birth density g**: Normal(0, τ²) with τ = std(y). NOT Dirac/fixed-init (degenerate).

## Current status summary (as of this session)
- ✅ 2.1.SM subtree_mutation
- ✅ 2.1.PR parameter_refresh / AMState
- ✅ 2.3 Pólya structural prior integration
- ✅ 2.7(a) σ-aware BIC (_log_marginal_bic in real_functionC.pyx)
- ✅ 2.7(b) Laplace identification (documented)
- ✅ 2.7(g) Numerical guards (RuntimeWarning suppression, NaN-safe BIC)
- ✅ test_cython4 grove_smoke_test fixes
- ⏳ 2.1.PM point_mutation v2 (kind-crossing RJMCMC) — NEXT
- ⏳ 2.7(f) BIC topology cache
- ⏳ 2.2 wire (log_q_fwd, log_q_rev) into groveC._step
- ⏳ 2.7(c/d) wire total log-target + MH rule into groveC._step
- ⏳ 2.1.K hoist, add_node, crossover (K-changing)
- ⏳ 2.8 Move-rate calibration
