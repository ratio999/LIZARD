# Implementation Plan — Bayesian Symbolic Regression (Pólya-GP MCMC)

**Goal**: Take research-prototype GP-MCMC SR tool to publishable artifact (PhD thesis chapter + ≥1 technical paper).
**Headline novelty**: Adaptive Pólya-tree structural prior over GP construction paths inside parallel-tempered MCMC.

## Critical-path summary
Phase 0 → Phase 1.3 + Phase 2.1 + Phase 2.2 (proposal-correction work) → **Phase 2.7 (SOLID POSTERIOR — hard gate)** → **Phase 2.13 (SOUNDNESS AUDIT fixes (a)–(e) — new hard gate, 2026-07-09)** → Phase 2.3 (vocabulary) → Phase 3.3 (tests) → Phase 4.1 (sanity) → Phase 4.3 (ablations) → Phase 4.2 (hero) → Phase 5 (writing).
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

#### 2.1.PM (point_mutation v2 — RJMCMC kind-crossing — DONE ✅)
- Commits to v2 from the start (kind-crossing allowed: function ↔ parameter).
- Implemented in `gp_movesC._point_mutation`: BIRTH adds `log g(theta_new)`
  to log_q_fwd and `-log M` to log_q_rev; DEATH is the mirror. Mixture
  weight `_KIND_CROSS_PROB = 0.5`. Validated by test_phase23 3.3(b)/(c).
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

### 2.2 groveC._step consumes log_q correction; tempering only on (post+prior). ✅ DONE
- Wire (log_q_fwd, log_q_rev) returned by each move into the acceptance computation. ✅
- log α = β * [log π(T') - log π(T)] + [log q_rev - log q_fwd]. ✅
- Proposal correction is NOT tempered. ✅
- Implemented in `groveC._step`: `q_corr = lqr - lqf`,
  `log_alpha = dtarget / self.T + q_corr`; non-finite lqr -> hard reject.
  Validated by test_phase23 3.3(e)/(f).

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

  (f) **BIC topology cache ✅ DONE**
      Cache {topology_key → (log_evidence, θ̂)} keyed on `tree_get_description()`.
      Invalidate only on new y. Avoids redundant optimize calls for re-visited topologies.
      Implemented in `groveC._step` via `self._bic_cache` / `self._cache_y`;
      on a hit the cached θ̂ is re-installed on the candidate so accepts warm-start.

  (g) **Numerical guards.** ✅ (implemented in _log_marginal_bic):
      - Ignore NaN/inf from evaluate (bad lambdify expressions).
      - Skip trees with n_eff <= k+1.
      - Return INVALID = -1e300 sentinel instead of -inf.

  (h) **Canonical deduplication in _bic_cache** (BMS-inspired — RE-SCOPED, optional)
      BMS permanently blacklists any tree whose SymPy canonical form matches a
      previously visited one. The original plan text here called for keying
      the cache on `str(sp.expand(expr))` / `str(sp.cancel(expr))` — REJECTED
      after review: running `expand`/`cancel`/`simplify` on a compose/pow-heavy
      expression is EXACTLY the operation that causes the catastrophic sympy
      blow-up documented above under "build_tree sympy blow-up" (symbolic size
      ~2**(2**(h-1)), as_real_imag spinning). Doing this unconditionally inside
      `_step`'s hot path would reintroduce that hang risk for any accepted tree
      that happens to compose/pow deeply, not just pathological build_tree
      output. Do NOT expand/cancel/simplify unconditionally.
      Lower-risk observation: RealFunctionC's arithmetic ops (`__add__` etc.)
      call plain sympy operators (`self.expr + other.expr`, never
      `evaluate=False`), so sympy ALREADY canonically reorders commutative
      Add/Mul arguments and folds trivial identities (x+0, x*1, ...) at
      construction time. `tree_get_description()` = `str(root_expr)` is
      therefore already dedup-safe for the common case (x+a and a+x hash
      identically today) — the original problem statement overstated the risk.
      REVISED scope (if ever picked up): (i) OPT-IN, node-count-bounded
      canonicalization only (e.g. run `sp.simplify` only when node count is
      below a small cap, say <=10) so cost stays bounded regardless of what
      compose/pow could produce; (ii) pure CACHE-KEY refinement only (extra
      cache hits = free speed) — never a correctness-affecting ban-list. A
      ban-list that forces `log_q_rev = -inf` for an "already seen" canonical
      form changes the chain's effective support and would need its own
      detailed-balance proof, which is not worth the complexity for a
      speed-only optimization.
      STATUS: re-scoped as OPTIONAL / low-priority speed polish, NOT a
      correctness gate. Removed from the pre-Phase-4 critical path; revisit
      only if profiling shows material cache misses from algebraic duplicates.

  Implementation order (6 steps):
    1. _log_marginal_bic in real_functionC.pyx ✅ DONE.
    2. Thread sigma through Node → Tree → Grove ✅ DONE.
    3. Wire _log_marginal_bic into groveC._step ✅ DONE (replaces get_superficial_posterior).
    4. Add Pólya prior term (log π_Pólya) to log-target ✅ DONE (extract_path/log_prior/update_path in _step).
    5. Implement BIC cache (item f) ✅ DONE (_bic_cache keyed on tree_get_description(), reset on new y).
    6. Update test_cython4 grove_smoke_test ✅ DONE (sigma=sigma_data, Npars=2, use_mpi=0).

  Future-work flags (not v1): full Laplace with -0.5 log|H|; pseudo-marginal MCMC
  (Andrieu & Roberts 2009); multi-start optimizer.

### 2.8 Move-rate calibration (after 2.7) ✅ DONE
- Feed move_stats() acceptance rates back into _choose_move probability weights.
- Adaptive move weights per temperature level.
- Feature fully implemented in groveC (`adapt_moves=False` default;
  diminishing-adaptation + logit-box/floor containment; birth/death pair-equality
  preserved so `_step` math is unchanged; plumbed through Grove AND Forest).
    * Added `_compute_move_probs` (shared source of truth for _choose_move) +
      `cpdef get_move_probs()` introspection getter (groveC.pyx + .pxd).
    * `test_phase28.py` (plain script) with three checks:
        (a) kernel invariants at init AND after adaptation -- pair equality
            (p_add==p_leaf_hoist, p_split==p_internal_hoist) + containment floor;
        (b) determinism: same seed -> identical adapted logits + move_stats +
            trees;
        (c) target invariance: adapt-on vs adapt-off cold-chain size marginal
            within a null-calibrated noise floor (Roberts & Rosenthal 2007).

### 2.11 ETR-style small-step move (BMS-inspired) ✅ DONE
Elementary Tree Replacement: select an ET (a subtree with <=1 operation) uniformly
and replace it with a freshly drawn ET of possibly different arity. Cheaper and
more targeted than subtree_mutation and stays on-support by design (an ET has
<=1 op, so it can NEVER trigger the build_tree pow/compose sympy blow-up -> no
subtree-rebuild height cap needed).
- GRAMMAR NOTE (this codebase != BMS): functions (sin, cos, x, ...) are LEAVES,
  not unary operators; internal nodes are ONLY binary ops. So ETs have exactly
  two arities: 0-ET = a leaf (function OR parameter); 2-ET = op(leaf, leaf).
  There are no 1-ETs. Structural counts: N0 = M+1 (M functions + 1 param option),
  N2 = O*N0*N0.
- Implemented `etr_mutation` / `_etr_mutation` (+ helpers `_collect_ets`,
  `_draw_leaf_et`, `_build_random_et`) in gp_movesC.pyx (+ .pxd). Forward: pick
  ET uniformly (Omega_i); pick new arity ~ U{0,2}; draw the new ET uniformly
  within that arity (s_f = N0 or N2); param leaves born ~ g = N(0, tau^2).
  Reverse selects at the same position among Omega_f ETs of T', draws the OLD
  arity/structure, re-births old params at their values:
    log_q_fwd = -log Omega_i - log 2 - log s(o_f) + sum_new log g(theta_new)
    log_q_rev = -log Omega_f - log 2 - log s(o_i) + sum_old log g(theta_old)
  (the -log 2 arity choice is common and cancels). Self-inverse move -> no
  birth/death pairing correction. Depth cap (only 0-ET->2-ET can deepen):
  force-reject with log_q_rev = -inf, same convention as add_node/split.
- Wired into groveC as the 8th physical move: `_choose_move` dispatch,
  `move_proposals`/`move_accepts`, fixed weight `p_etr = 0.10*f + 0.05`, and the
  Phase 2.8 adaptive map (now 6 groups -> 8 physical; group 5 = etr; floor 1/8).
- Density unit tests in test_phase23.py:
    * `test_3_3c_etr_density` -- EXACT reconstruction of lqf AND lqr from a
      single-leaf start (Omega_i=1, whole tree replaced), both o_f in {0,2};
    * `test_3_3c_etr_offsupport` -- depth-cap force-reject gives lqr == -inf.
- ⏳ PENDING: recompile (gp_movesC + groveC) then run `python3 test_phase23.py`
  (and the full suite). ETR is ACTIVE in the default kernel once compiled.

### 2.12 MPI parallelization of the SAMPLING loop (spawn model) ✅ IMPLEMENTED (pending compile+run)
Parallelizes the EMC sampling loop only — all four burn-in implementations and
`mpi_worker.worker_main` are UNTOUCHED. Mirrors the burn-in spawn pattern:
`MPI.COMM_SELF.Spawn` (no mpirun needed), config bcast, gather replies, and the
same tree wire format (sympy expr, name_values, post) rebuilt via `_expr_to_tree`.
- **Topology**: `Forest.run_mpi_spawn(y, n_rounds, n_steps_per_round, seed=,
  max_tree_depth=, subtree_rebuild_max_height=)` spawns ONE worker per
  temperature rung (`mpi_worker.sampler_main`). Each worker builds and
  persistently OWNS one Grove (same ctor args via new `Forest._ctor` stash +
  the master's CURRENT trees shipped at startup). The mutation phase (BIC
  scoring + scipy optimize — the dominant cost) runs concurrently; each round
  the master coordinates ONE swap + ONE crossover attempt.
- **Shared-Θ rigor (R1) by deterministic replay**: new `Grove.polya_defer`
  flag (default False = zero serial change): accepted construction paths are
  COLLECTED (`drain_deferred_paths`) instead of applied. Master gathers all
  paths each round (ordered by rank), replays into its Θ
  (`PolyaTree.replay_paths`), and broadcasts the same ordered list; every
  worker replays identically ⇒ all Θ replicas bit-identical. Within-round
  priors use the last-synced Θ = DELAYED adaptation (γ_t→0 + containment
  still hold; Roberts & Rosenthal 2007). Vocabulary is frozen up-front
  (a0..a31/b0..b31 pre-registered) so ('kind', idx) tuples mean the same on
  every replica. New `PolyaTree.get_state()/set_state()` ship the (possibly
  evolved) master Θ to workers at startup.
- **Exchange/crossover on the master** with the synced Θ and master RNG, using
  exactly the serial acceptance forms. Control flow is bcast-driven
  (('run',n) / ('sync',paths) / ('get_tree',rank,slot) /
  ('set_tree',rank,slot,wire,post) / ('stop',)) ⇒ deadlock-free by
  construction; workers answer collectives even on internal errors (lesson
  from the burn-in worker). Rebuild-fidelity failure on any wire tree turns
  the exchange into a REJECT (self-loop; target-preserving).
- On exit the master's `self.groves`/`self.polya` hold the final state, so
  `best_tree()` etc. work unchanged. Returns stats dict (swap/xover accepts,
  paths replayed).
- **Test**: `test_phase2mpi.py` — (a) end-to-end integration (valid trees,
  finite posts, best_tree), (b) replay fingerprint `t_after == t_before +
  paths_replayed`, (c) exploration progress, (d) serial-path regression guard
  (polya_defer defaults off; serial loop unchanged).
- ⏳ PENDING: recompile (polya_treeC + groveC + mpi_worker) then
  `python3 test_phase2mpi.py`.
- Known limits (documented, acceptable v1): worker adapt_moves state is not
  synced back at 'stop'; swap/crossover remain serial per round (cheap);
  bit-parity with the serial trajectory is NOT expected (different but equally
  valid random schedule).

### 2.13 SOUNDNESS AUDIT (2026-07-09, full-code Bayesian review) — NEW HARD GATE before Phase 4
Independent audit of the posterior/acceptance math across groveC._step,
gp_movesC densities, real_functionC/function_treeC BIC, polya_treeC prior,
Forest EMC operators, and mpi_worker. Verdict: the overall architecture
(BIC-as-Laplace target, Pólya adaptive prior with diminishing adaptation,
tempered EMC with untempered proposal correction, birth/death pair-equality,
bounded-support depth caps via log_q_rev = -inf, MPI deterministic replay)
is sound in design. The following items break or bias exactness and are
ordered by urgency. (a)–(e) are BLOCKING for Phase 4; (f)–(l) are
second-order / documentation.

  (a) **θ-bookkeeping inconsistency (g-density terms vs profiled-BIC target).**
      The target is θ-free (log-evidence = -BIC/2 at θ̂, Pólya prior), yet the
      structural proposal corrections include continuous densities over θ:
      point_mutation BIRTH adds log g(θ_new) to lqf / DEATH adds log g(θ_dying)
      to lqr; _eval_build_density and ETR add Σ log g(θ) terms. That bookkeeping
      is exact ONLY if θ is retained as part of the chain state (product-space
      RJMCMC with an implicit improper-uniform θ-measure). It is NOT retained:
      the scorer (`_log_marginal_bic(optimize=True)` / the BIC cache) overwrites
      θ with θ̂ immediately, and parameter_refresh diffuses θ under a *different*
      target (raw tempered likelihood). The g(θ) factors therefore inject noise
      into the acceptance ratio that does not cancel in expectation
      (Monte-Carlo-within-Metropolis, not pseudo-marginal) → the topology
      marginal is biased.
      FIX (recommended, minimal): treat the chain as living on TOPOLOGY space
      only, with θ̂ a deterministic per-topology statistic (the _bic_cache
      already enforces first-visit determinism per y). Then all proposal
      corrections must be PURELY STRUCTURAL: delete the log g terms from
      _point_mutation (birth/death), _eval_build_density (keep the Bernoulli
      p_par and uniform-choice factors), and _etr_mutation; the birth draw
      θ_new ~ g remains as optimizer warm-start randomness that integrates
      to 1 in q. parameter_refresh then no longer affects the topology
      chain's correctness (it only perturbs warm starts) and may be kept.
      Alternative (NOT recommended, bigger change): abandon BIC and score
      L(y|T,θ) at the CURRENT θ with explicit priors g(θ) in the target —
      full product-space RJMCMC; then the g bookkeeping is right but
      optimize/refresh must be removed or reworked into valid kernels.
      Cross-refs: Phase 1.3, 1.6, 2.7(b). Update test_phase23 3.3(b)/(c)
      expected densities accordingly.

  (b) **Within-grove subtree_crossover violates reversibility.**
      _propose deep-copies a donor, swaps subtrees, and on accept installs
      ONLY slot i; the donor copy (which received S_v) is discarded. From the
      new state the reverse proposal density is generally ZERO — S_v no longer
      exists anywhere — yet the code uses the pair-swap counts
      (-log n_int(a')-log n_int(b')) as lqr. Detailed balance is broken.
      FIX: either (i) make it a genuine pair move: install BOTH children in
      slots i and j of the SAME grove with joint acceptance
      ΔU_i/T + ΔU_j/T + (lqr - lqf) [both slots share T, so this is the
      within-grove analog of the Forest crossover, which is CORRECT]; or
      (ii) drop within-grove crossover from the kernel (Forest trans-grove
      crossover already supplies recombination).

  (c) **Pólya auto-register ordering bias in _step.**
      old_prior is evaluated BEFORE new_path extraction. When the candidate
      introduces an unregistered symbol, extract_path(cand) auto-registers it
      and grows every Dirichlet (n_total += 1), so new_prior is evaluated on a
      LARGER vocabulary than old_prior within the same acceptance ratio —
      systematically favoring the incumbent by ~(#unvisited steps)·log((n+1)/n).
      FIX: extract BOTH paths first (letting registration happen), then
      evaluate both log_priors against the same Θ. Same ordering issue exists
      in legacy_step and in the Forest swap/crossover prior evaluations
      (harmless there only when both trees are already covered).

  (d) **NaN root_posteriors when Grove is built with node_list (or any path
      that skips scoring).** The lazy cache-fill in _step is commented out
      (deprecated block), so old_post = NaN → log_alpha = NaN → every
      comparison is False → the slot can never accept anything, silently.
      FIX: restore the lazy fill via log_marginal_bic (mirror _score_post),
      and/or score in _init_trees for the node_list branch too.

  (e) **_propose retry loop (max_attempts=2) is an unaccounted proposal
      mixture.** On validity failure a SECOND move is drawn and only its
      (lqf, lqr) enters the ratio; the effective kernel q_eff includes the
      retry mixture, which differs between forward and reverse states.
      FIX: max_attempts=1 — an invalid candidate is a hard reject (the
      None path already does this). Cheap and exact.

  (f) **`operations` plumbing is consistent ONLY for the default 6-op set.**
      build_tree draws internal ops from the HARDCODED
      ['add','sub','mul','div','pow','compose'] (function_treeC.pyx:1001),
      ignoring the caller's list; PolyaTree.__init__ likewise overrides the
      `operations` argument with module _OPERATIONS (polya_treeC.pyx:77).
      But _eval_build_density and the move densities use O = len(operations)
      as passed by the Grove. With any non-default operations list (e.g. the
      +,-,*,/ used by test_phase29 fixtures) the build densities are WRONG
      (claim 1/4 per op, sample 1/6) and grown subtrees can contain ops the
      Grove kernel cannot produce elsewhere. FIX: honor `operations` in
      build_tree/_build_subtree_with_params (thread it through), or assert
      at Grove.__init__ that operations == the canonical 6 until then.

  (g) **Single-leaf-root edge case breaks pairing.** subtree_mutation on a
      leaf root silently delegates to add_node but is selected with
      p_subtree ≠ p_leaf_hoist, so the Green move-choice factor no longer
      cancels for that transition. FIX: return (0, -inf) (no-op) from
      subtree_mutation on single-leaf trees, or dispatch add_node under its
      own name/probability.

  (h) **nanmean hides partial-domain invalidity in the BIC.** objective/
      fitness use np.nanmean((y-f)^2): a model NaN on part of the domain is
      scored on the finite subset while the BIC penalty uses full N —
      inflating evidence for partially-defined expressions, and the optimizer
      can actively exploit it. is_valid gates only at pre-optimization θ.
      FIX: re-validate at θ̂ (or replace nanmean with mean so any NaN →
      non-finite → reject) before caching/accepting.

  (i) **Profiled-σ BIC omits the promised k+1.** _shortcut_bic /
      _log_marginal_bic use k = #params in BOTH σ modes; Phase 2.7(a) says
      σ=None should penalize k+1. Constant shift (every model pays the same
      +σ), so MH ratios are unaffected — but reported evidences and any
      cross-mode comparison are off by 0.5·log N. Fix or document.

  (j) **Burn-in scores have no complexity penalty.** _process_candidate and
      mpi_worker._evaluate_tree call root._log_marginal_bic(y) with
      params=None → k=0 → pure-fit ranking. Acceptable (burn-in is
      initialization only, per the user's design: trees built regardless of
      score), but document that "post" wire values from burn-in are NOT the
      Phase 2.7 log-evidence and must be re-scored before use in acceptance
      ratios (run_mpi_spawn currently ships them as-is into swap ratios on
      round 0 if compute_posteriors was not called first).

  (k) **Proposal-path multiplicity ignored (document in Phase 1.3).** q is
      computed as the density of ONE generative path; symmetric trees admit
      multiple (leaf, subtree) choices mapping T→T'. Standard practice, but
      the writeup must state the convention (single-path densities with
      matched forward/reverse path pairing).

  (l) **Pólya adaptation limit (writeup care, Phase 1.2).** Σ γ_t = Σ 1/(1+t)
      diverges, so total urn mass grows without bound and Θ need not converge
      to a deterministic limit; diminishing adaptation + containment (finite
      support under the depth cap) still give ergodicity in the R&R 2007
      sense, but the "self-consistent posterior" limit language needs a
      stochastic-approximation argument or should be weakened. Propriety of
      the prior is NOT an issue given the depth-cap-restricted support
      (finite space).

  Verified-correct (no action): untempered q_corr with tempered target;
  birth/death pair probability equality preserved under 2.8 adaptation;
  Forest swap (1/Ta-1/Tb)(Ub-Ua) with U = post+prior on shared Θ; trans-grove
  crossover acceptance form; prior re-evaluation of the incumbent every step
  under evolving Θ; depth caps as bounded support (restriction, not bias);
  BIC cache θ̂ re-install (helps determinism, required by fix (a));
  MPI deterministic replay / delayed adaptation; AMState Robbins-Monro
  update-on-every-step; subtree_mutation size-ratio correction with
  perfect-binary support gate; add_node↔leaf_hoist and split↔internal_hoist
  density pairing (selection, op, trunc-geom depth, side, build factors all
  mirrored).

  Implementation order for 2.13: (e) → (d) → (c) → (b) → (a) → (g) → (h) →
  (f), then recompile + rerun test_phase23 (adjusting 3.3(b)/(c) expectations
  for (a)), test_phase28, test_phase29, test_phase2mpi. (i)–(l) are
  docs/flags. Phase 4 remains gated until (a)–(e) land.

  STATUS 2026-07-09 (second batch): ALL FIVE BLOCKING FIXES IMPLEMENTED,
  pending recompile + full test rerun on the Linux box.
    (a) ✅ (a1) — g-density terms removed from _point_mutation birth/death,
        _eval_build_density, _etr_mutation; θ~g draws kept as warm starts;
        test_phase23 3.3(b)/(c) re-targeted. THEORY_2_13.md §1.
    (b) ✅ (b1) — within-grove crossover is now a genuine PAIR move:
        _propose returns the donor child, new Grove._step_pair_crossover
        installs both with joint acceptance (ΔU_i + ΔU_j)/T + q_corr.
        THEORY_2_13.md §2.
    (c) ✅ — extract-then-score Pólya ordering at six sites (_step,
        legacy_step, Forest swap/xover, MPI-master blocks, pair helper).
        THEORY_2_13.md §3.1.
    (d) ✅ — lazy NaN root_posteriors fill in _step via _score_post.
    (e) ✅ — _propose max_attempts = 1 (invalid candidate = hard reject).
        THEORY_2_13.md §3.2.
    (f) ✅ (PROMOTED from second tier — prime suspect for the 2.8c
        target-invariance failure, whose fixture uses a 4-op vocabulary) —
        operations threaded through build_tree / Tree.__init__ /
        _init_trees / _build_subtree_with_params / subtree_mutation;
        default preserves the historical RNG stream bit-exactly; initial
        trees now support-closed. THEORY_2_13.md §3.3.
  Technical change log: AUDIT_2_13_FIXES.md. Once the suite passes
  (incl. test_phase28 2.8c), the Phase 4 gate REOPENS.

  STATUS 2026-07-09 (third batch): (g)(h)(i) ✅ IMPLEMENTED, pending
  recompile (gp_movesC + function_treeC + real_functionC) + suite rerun.
    (g) ✅ single-leaf subtree_mutation → forced reject (0, -inf), no more
        mislabelled add_node delegation. THEORY_2_13.md §3.4.
    (h) ✅ re-validation at θ̂ in Tree._log_marginal_bic (both branches)
        closes the nanmean partial-domain loophole exactly. THEORY §3.5.
    (i) ✅ profiled-σ BIC penalty k → k+1 in both σ=None scorers +
        test_phase23 3.3(d) formula test/docs updated. THEORY §3.6.
  Remaining in 2.13: (j)–(l) are docs-only (fold into the paper per
  PAPER_PLAN.md §5.5/§9.5/§12.4). Next CODE work after the suite is green:
  the a1 toy-occupancy regression test (THEORY §1.7, needs go-ahead),
  then Phase 2.10 (custom opaque primitives — design first, PAPER_PLAN
  §10.3), then Phase 3 dead-code removal.

---

## Phase 2.9 — Evolutionary Monte Carlo: population-coherent moves (rigorous, Bayesian)

**Agreed roadmap order (this session):**
  0. Verify base builds + tests pass (hoist-split + 2.8 adaptive are UNCOMPILED). ← gate
  1. Cross-temperature crossover.                                          ← IMMEDIATE
  2. Custom atomic primitives (Phase 2.10).
  3. Full parallel/EMC hardening + MPI of the *sampling* loop (today MPI = burn-in only).

**Target.** Population product of tempered posteriors  Π = Π_k π_{T_k},
π_T(x) ∝ exp(U(x)/T),  U(x) = log-evidence_BIC(x) + log π_Pólya(x).

**Canonical EMC operator set (Liang & Wong 2000) — NO fourth operator is required:**
  - Mutation  = within-grove MCMC step (run_all → _step).              ✅ exists
  - Exchange  = parallel-tempering swap (Forest.attempt_swaps).        ⚠️ exists BUT Forest
                has no `_py_rng` (neither in __init__ nor .pxd) → `_attempt_swap_pair`
                would AttributeError. RNG fix folds into the crossover work.
  - Crossover = cross-temperature subtree crossover.                  ⏳ IMMEDIATE deliverable
      log α = ΔU_a/T_a + ΔU_b/T_b + (lqr − lqf); preserves every marginal; reduces to the
      PT swap when the "crossover" is a whole-tree swap. Reuses gp_movesC.subtree_crossover
      (returns lqf, lqr; is its own inverse on the chosen pair). Design frozen in session
      memory /memories/session/plan.md.

**Rigor requirements BEYOND the three operators (properties, not new moves):**

  (R1) DECIDED — proper joint target that INCLUDES the prior, never frozen.
       Forest is rewritten from scratch. It builds ONE SHARED PolyaTree and passes the
       SAME instance to every grove (Grove.__init__ already accepts `polya=`). The prior
       is NOT frozen: it keeps evolving via the existing `update_path` DIMINISHING
       adaptation gamma_t = 1/(1+t) (Roberts & Rosenthal 2007 + Robbins-Monro), reinforced
       by EVERY grove's accepted WITHIN-grove moves on a global counter. Because Theta is
       shared, the population is a canonical tempered ladder of ONE self-consistent
       posterior; the cold chain (beta=1) samples it. Formal note for the writeup: the
       limit is the self-consistent (Polya-urn-reinforced) posterior, not the fixed-alpha0
       posterior — a modeling statement, not a bug. Rejected alternatives: freeze-after-
       burn-in (works but less elegant) and per-grove Theta_k (rigorous but throttles swap
       acceptance via a prior-mismatch penalty and needs the non-cancelling acceptance
       form + a joint proof).
       - Acceptance form: general per-grove-prior MH; with shared Theta it REDUCES to
         (beta_a - beta_b)(U_b - U_a) with U(x) = post(x) + log pi_Polya(x | Theta) (prior
         now INCLUDED; the current post-only swap is therefore WRONG and is replaced).
       - Reinforcement: within-grove moves reinforce (already in _step); the trans-grove
         swap and trans-grove crossover are prior-preserving relabelings that DO NOT
         reinforce. (Within-grove crossover is mutation-family and DOES reinforce; _step
         already updates only slot i, donor untouched.)
       - Confirm the Polya prior is normalizable over tree space (proper => proper posterior).

  (R2) ✅ DONE 2026-07-09 (pending run) — lemma sketch: THEORY_2_13.md §4
       (ETR carries connectivity through the single leaf; point_mutation the
       labels; perfect-binary moves conserve parities, only add mixing).
       Strong test: test_phase29 test_2_9d2_full_support_coverage (38-state
       micro-space, set-equality coverage+closure, 3 seeds). Dynamic
       companion: existing test_2_9d. Original spec follows:
       add a WITHIN-CHAIN irreducibility lemma + a small connectivity test.
       The mutation move-set (point/subtree mutation, add_node/hoist, split, within-grove
       crossover) must connect the whole tree space: from any tree, every other tree is
       reachable in finitely many positive-probability moves. Crossover/exchange add
       cross-chain connectivity but do NOT substitute for within-chain irreducibility.
       Deliverable: (a) short proof sketch (birth/death pairs + subtree replacement span
       the space up to any depth bound; cross-ref Phase 1.4 lazy-trie lemma); (b) a unit
       test that, on a tiny vocabulary + shallow depth cap, empirically visits every tree
       in the finite support within N steps from several seeds.

  (R3) DECIDED — random-scan pair selection: ONE random consecutive temperature pair per
       step, for BOTH swap and trans-grove crossover (reversible per step; matches the
       reference PT). Replaces the current all-adjacent left-to-right sweep.

**Non-required but standard PT/EMC hygiene (still open / later):**
  - **principled geometric temperature ladder** (BMS-inspired): BMS uses 40 rungs at
    T_k = 1.05^k (k=0..39, top T ≈ 7). Swap acceptance degrades rapidly with temperature
    gap; too few rungs → swaps/crossovers rarely fire. For production runs use ≥8–12
    rungs at a geometric ratio ~1.3–1.5. For hard problems (e.g. CCSNe hero) BMS also
    uses 4 independent restarts; implement as parallel Forest seeds + take best cold-
    chain result. Concrete ladder choice is a 4.1 sanity-run tuning decision: monitor
    swap acceptance rate (target ~0.2–0.4 per consecutive pair); if < 0.1, add rungs.
  - burn-in-ONLY adaptive tempering / annealing to escape traps (rigor: never during the
    inference phase);
  - extend theta-hat warm-start caching across the Forest (speed only; per-grove
    _bic_cache exists).

---

## Phase 2.10 — Custom atomic (protected) primitives — parametric terminals
User-registrable indivisible basis functions (e.g. A·sin(ω·x)) treated as ONE opaque leaf:
  - carries its own internal parameters (≥1 param that cannot be individually deleted);
  - atomic to every structural move (cannot be split/hoisted-into; internal params tuned
    only via parameter_refresh; whole leaf deleted/replaced as a unit);
  - rigorous, generalized analog of a protected fixed sub-expression: a user primitive set
    with proper prior mass assigned to each primitive symbol.
Touches: function_treeC (Node opaque flag + is_leaf/is_param semantics), gp_movesC (moves
skip opaque internals when picking targets), real_functionC (primitive definition), Pólya
prior (assign each primitive a symbol/mass). Paradigm-independent; highest value/lowest risk.
OPEN: exact prior-mass treatment for opaque primitives.

---

## Phase 3 — Code: hygiene, structure, tests
### 3.1 Remove dead code.
### 3.2 API documentation.
### 3.3 Test suite (blocks Phase 4). ✅ DONE (a)–(g)
All new tests in tests/test_phase23.py (runnable as a plain script, no pytest).
Sub-tasks (a) determinism, (b) birth density, (c) per-move densities,
(d) BIC scorer, (e) MH rule, (f) Grove integration, (g) Pólya prior — all
implemented and passing. Existing coverage in test_cython4.py retained.

Sub-tasks:
  (a) Determinism smoke test (closes 2.5(h)).
      Two Groves, same seed, same n_steps, same y:
      assert tree_get_description() per slot matches, move_stats() matches
      exactly, root_posteriors matches within 1e-9.

  (b) Birth density g unit test.
      - get_g_tau() reflects set_g_tau().
      - For a seeded BIRTH: lqf == log N(theta_new; 0, tau^2) analytically.
      - BIRTH/DEATH time-reversal symmetry:
        force BIRTH at known theta, force DEATH at same theta;
        lqf_birth + lqr_birth == lqf_death + lqr_death.

  (c) Per-move proposal log-density checks.
      For each of subtree_mutation, hoist, add_node, split,
      point_mutation symmetric cases:
      - On-support: lqr != -inf for small well-formed trees.
      - Off-support: lqr == -inf when precondition not met.
      - subtree_mutation ratio: lqr - lqf == log(size_T) - log(size_Tp).
      - add_node/split: reconstruct expected density from (K, M, O, d)
        and compare.

  (d) BIC scorer unit test.
      On y = a*x + b (analytic optimum):
      - _log_marginal_bic is finite.
      - More parameters → lower log_evidence (BIC penalises complexity).
      - sigma=None vs sigma=fixed give different penalties, both finite.

  (e) MH acceptance rule unit test.
      Known T, T', (lqf, lqr): verify
      - dtarget >> 0 and q_corr >= 0 → always accepted.
      - lqr == -inf → always rejected (hard reject path).
      - T=0 (greedy) → accept iff dtarget > 0.

  (f) Grove integration test (extends grove_smoke_test).
      Seeded Grove, 200 steps, y = sin(x):
      - Acceptance rate in [0.05, 0.95].
      - move_stats() non-zero proposals for every move type.
      - best_tree() has finite root_posteriors.
      - Same seed → same final state (inherits (a)).

  (g) Pólya prior unit test.
      Known ops+primitives, hand-constructed tree:
      - log_prior matches analytic Dirichlet-Multinomial predictive.
      - update_path changes log_prior on next call.
      - auto_register=True: unknown leaf doesn't raise, log_prior stays finite.
### 3.4 MPI: decide keep vs drop for paper version.

---

## Phase 4 — Experimental program
Cannot start before Phase 2.1, 2.2, 2.3, **and 2.7** are done (otherwise results are biased).
### 4.1 Sanity checks (recovering known functions)
- **PT ladder tuning**: before running sanity benchmarks, fix the temperature schedule.
  Start with ≥8 rungs at geometric ratio ~1.4 (e.g. T_k = 1.4^k, k=0..7, top T≈14.7).
  Monitor swap acceptance rate per consecutive pair; target 0.2–0.4. Adjust ratio or
  add rungs until target is met. Log ladder + swap rates in results for the paper.
  (See Phase 2.9 hygiene note.)
### 4.2 Hero application (CCSNe / Mc recovery / Feynman — choose one)
### 4.3 Ablations (no Pólya, no tempering, no BIC)

---

## Phase 5 — Writing
**→ The full, consequential section-by-section writing roadmap now lives in
`PAPER_PLAN.md` (created 2026-07-09). It supersedes the thin skeleton below,
which is retained only as a quick index. PAPER_PLAN.md flags the CENTRAL
FORMAL DECISION (fixed vs adaptive Pólya prior framing) that must be resolved
before drafting §4/§9, and folds the Phase 2.10 opaque-primitive prior-mass
question into the paper's formalism (its §10.3).**

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
5. (Phase 2.10) Prior-mass treatment for user atomic/opaque primitives.
6. (Phase 2.9 hygiene) Temperature-ladder policy + burn-in-only annealing (later).

## Decisions closed
- **σ in BIC** (Phase 2.7(a)): both modes supported via `sigma=None` (profiled) vs `sigma>0` (fixed). User picks per dataset.
- **Profile vs Laplace marginal** (Phase 1.6): committed to BIC = -2 (log L - k/2 log n) as the v1 evidence proxy; full Laplace deferred.
- **Warm-start scope** (Phase 2.7(e)): implicit only (inheritance via `self.name_values`).
- **K-changing freedom**: hoist, add_node, crossover, point_mut v2 are all allowed to change K. Fixed-K was only Phase 2.1.SM's recycling convention.
- **point_mutation**: goes to v2 (kind-crossing) from the start (no v1-first step).
- **Birth density g**: Normal(0, τ²) with τ = std(y). NOT Dirac/fixed-init (degenerate).
- **EMC Pólya layout** (Phase 2.9 R1): SHARED single Pólya across all groves, NEVER frozen,
  reinforced by ALL groves' accepted within-grove moves (diminishing γ_t=1/(1+t), global
  counter). Rejected: freeze-after-burn-in and per-grove Θ_k. Swap/crossover use U=post+prior
  and reduce to (β_a−β_b)(U_b−U_a); trans-grove moves do NOT reinforce.
- **EMC pair selection** (Phase 2.9 R3): one random consecutive pair per step (swap + xover).
- **EMC irreducibility** (Phase 2.9 R2): within-chain irreducibility lemma + connectivity
  test added to the plan (Phase 3.3 extension).

## Current status summary (as of this session)
- ✅ 2.1.SM subtree_mutation
- ✅ 2.1.PM point_mutation v2 (kind-crossing RJMCMC, BIRTH/DEATH)
- ✅ 2.1.K hoist_mutation, add_node, split, subtree_crossover (all return (lqf, lqr))
- ✅ 2.1.PR parameter_refresh / AMState
- ✅ 2.2 (log_q_fwd, log_q_rev) wired into groveC._step; tempering on target only
- ✅ 2.3 Pólya structural prior integration (auto-register)
- ✅ 2.5 determinism / seed plumbing (Grove + gp_movesC + function_treeC set_rng)
- ✅ 2.7(a) σ-aware BIC (_log_marginal_bic in real_functionC.pyx)
- ✅ 2.7(b) Laplace identification (documented)
- ✅ 2.7(c/d) total log-target + MH rule wired into groveC._step
- ✅ 2.7(f) BIC topology cache (_bic_cache)
- ✅ 2.7(g) Numerical guards (RuntimeWarning suppression, NaN-safe BIC)
- ✅ 3.3 test suite (a)–(g)
- ✅ test_cython4 grove_smoke_test fixes
- ✅ perf: lambdify caching in real_functionC.from_string; vectorised conversion
- ✅ 2.8 Move-rate calibration — feature done + test_phase28.py written
- ✅ 2.11 ETR move (etr_mutation) implemented + wired (8th move) + density tests
- ⏳ 3.1 MPI keep/drop decision
- ⏳ 3.4 dead-code removal (legacy_* moves, root_replacement, replace_with_subtree)
- ⏳ **2.13 soundness-audit fixes (a)–(e)** — NEW hard gate (2026-07-09): θ/g-density
  bookkeeping, within-grove crossover reversibility, Pólya auto-register ordering,
  NaN root_posteriors on node_list init, _propose retry mixture
- ⏳ Phase 4 experiments (4.1 sanity → 4.3 ablations → 4.2 hero) — gate REOPENS after 2.13(a)–(e)
