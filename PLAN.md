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
STAGE 1 (DP completion in polya_treeC) ✅ IMPLEMENTED 2026-07-09, pending
compile+run: remainder cell + carve-on-register + neutrality restriction +
state round-trip; theorem-tests in test_phase210.py; legacy mode
(remainder_mass=0 default) bit-identical.
STAGE 2 (atomic-leaf object, DESIGN §9 items 2-3) ✅ IMPLEMENTED 2026-07-09,
pending compile+run: AtomSpec template + instantiate_atom/rename_leaf_params
(real_functionC); Node.theta + is_atomic/leaf_k/leaf_param_names/vocab_symbol
+ name_values absorbs θ_ψ (so _shortcut_bic's k = len(name_values) IS the
DESIGN §5 effective dimension) + optimizer/apply write-backs (function_treeC);
extract_path/assert_covers key on vocab_symbol — ONE Pólya cell per template
(polya_treeC); every leaf-choice cardinality M+1 → M+1+A via a trailing
`atoms` arg on point_mutation/ETR/add_node/split/hoists (None = A=0 =
bit-identical legacy), force-rejects where a reverse move can't rebuild an
atom (subtree_mutation always; death moves when A=0), crossover renames
opaque internals, parameter_refresh treats θ_ψ as ordinary coordinates
(gp_movesC). Stage-2 tests (h)-(n) in test_phase210.py.
STAGE 3a (single-process grove wiring) ✅ IMPLEMENTED + COMPILED + TESTED
2026-07-10: Forest/Grove gain `atoms=` (list of AtomSpec); Forest
pre-registers each template symbol in the shared Pólya at construction
(one cell, registration-invariant; pass a pre-registered `polya=` to
control per-atom G_Ψ masses in DP mode); Grove._propose threads `atoms`
into ALL atom-aware moves (point, both hoists, add_node, split, ETR —
the ONLY place atoms enter the kernel); AMState is built from
gp_movesC.pr_coord_names(tree) (single source of truth with
parameter_refresh — PR now refreshes θ_ψ); BIC cache hits install opaque
θ̂ via _install_cached_theta (was plain-params-only: installed tree and
cached score could disagree); optimize gate in _step switched
param_values→name_values (atomic-only trees now optimized); atom-birth
warm start re-centred at the template reference θ₀=1 (zero-centred draws
regularly dropped Powell into a degenerate frequency basin that the
description-keyed cache then froze); run_mpi_spawn raises
NotImplementedError when atoms are set (wire format is Stage 3b).
End-to-end test 2.10(o): single-rung Forest with Asin template + safe-4
ops (frequency unreachable by composition) recovers 2.3·sin(1.3x) to
mse ~5e-28. Full suite green on the final build (210 a–o, 23, 29).
STAGE 3c (FULL vocabulary freeze) ✅ IMPLEMENTED + COMPILED + TESTED
2026-07-10 — settles Phase 2.3 / DESIGN §9 item 4's deferred half. Key
realization: NO finite parameter-name pool can be frozen — new_par
ratchets indices upward without bound (births after deaths raise the max
index; only crossover compaction resets it), so the pre-registered-pool
idea is unsound in principle, not just under-sized. The correct object is
DESIGN §1's own unification: the parameter leaf is ONE atom of Ψ₁
(ψ_const); names are bound labels chosen deterministically by new_par and
carry no prior mass. Implementation: PolyaTree(..., unify_params=True)
appends the reserved cell PARAM_CELL_TOKEN ("<param>") and maps EVERY
plain parameter leaf to it in extract/assert (opaque leaves keep their
template cells; guards reject the token as a primitive/registration);
get_state/set_state carry the flag (MPI replicas). Forest(...,
freeze_vocab=True) then builds a CLOSED vocabulary {ops} ∪ {funcs} ∪
{ψ_const} ∪ {atom templates} with auto_register=False — unknown symbols
raise the guarded KeyError instead of growing the prior; run_mpi_spawn
skips the legacy 32-name pool when unified (it would only shift the
symmetric prior). Payoffs: the prior is a fixed measure from t=0 AND
renaming-invariant (α-equivalent trees get equal mass — test 2.10(r)
asserts lp(a0-tree) == lp(b7-tree) exactly, before and after
reinforcement, which now lands on the shared ψ_const urn cell); MPI path
indices are replica-stable by construction, not best-effort. Defaults
UNCHANGED (freeze_vocab=False, unify_params=False → bit-identical legacy,
whole suite green); freeze is opt-in like DP mode — the paper runs use
Forest(freeze_vocab=True) (+ a caller polya with remainder_mass>0 for DP).
Tests: 2.10(r) (α-equivalence, closed-vocab KeyError, token guards, atom
coexistence, state round-trip, 300-step frozen Forest run with vocab
bit-stable); test_phase2mpi (e) now runs in the frozen paper mode
(freeze+atoms+skeleton wire, vocabulary bit-stable across all replicas).
Still open (minor): burn-in enumeration stays atom-free by design (moves
introduce atoms); revisit if burn-in should seed atom-bearing depth-1
trees.
STAGE 3b (MPI wire format) ✅ IMPLEMENTED + COMPILED + TESTED 2026-07-10:
the wire stays a (payload, name_values, post) 3-tuple; payload is the raw
sympy expr for atom-free trees (HISTORICAL format, bit-identical) or
('skel3b', skeleton, description) for atom-bearing ones, where the
skeleton (function_treeC.tree_to_skel / node_from_skel) is a picklable
recursive tuple — ('L', desc) | ('A', atom_symbol, names) | ('O', op,
left, right) — that preserves the EXACT tree shape and reconstructs
opaque leaves as units (a raw expr cannot: a0*sin(a1*x) is a sympy Mul
and _expr_to_tree would decompose it). groveC._tree_to_wire/_tree_from_wire
promoted to cpdef with an atom_specs registry arg; mpi_worker's duplicate
pair now DELEGATES to them (one wire implementation); AtomSpec pickles
via __reduce__ and rides the worker startup config (Grove kernels +
rebuilds on every rank); run_mpi_spawn guard lifted. Rebuild failures
(unknown template, fidelity mismatch) keep the REJECT-the-exchange
convention. Tests: 2.10(p) wire round-trip unit (both payloads, pickle,
exact-shape, reject-on-missing-registry); 2.10(q) in test_phase2mpi —
live 2-rung run_mpi_spawn with a planted Asin tree: startup wire ships a
skeleton, atom PROPAGATES across the population (5/6 final slots atomic,
2 accepted swaps), best = b0*sin(b1*x) at mse 0.0. Legacy 2.12 tests
unchanged-green on the dual wire. NB singleton MPI_Comm_spawn is not
configured in the WSL env — run the MPI suite as
`mpiexec -n 1 python test_phase2mpi.py`.
BURN-IN SEEDING ✅ IMPLEMENTED + COMPILED + TESTED 2026-07-11 (the last
2.10 item — the phase is now CLOSED). Opaque templates enter the burn-in
enumeration as plain-string tokens '@atom:<symbol>' (all itertools
sweeps, work-item tuples and pickling untouched); one shared resolver
_burn_in_primitive_node builds the leaves (parameter 'a0' → value-1.0
leaf as before; tokens → opaque leaves at the θ₀ reference; else
from_string). Name discipline: stage-1 candidates stay all-'a' (right
leaf's band offset past a left atom's k) because _combine_depth1 renames
right operands wholesale to 'b'; mpi_worker's combiner renames nothing,
so its primitive operand uses the 'c' band and it now renames the right
side to 'b' ONLY when it carries atomics (atom-free path bit-identical;
the plain-a0 coupling quirk of that combiner is preserved as-is). The
spawn burn-in's three wire crossings (depth-1 down, worker results up,
master rebuild) use the Stage 3b dual-payload wire for atom-bearing
trees; the worker's optimize gate generalized tree.params→name_values
(Stage 3a pattern). _burn_in_stage2_parallel (rank-parallel, non-spawn)
rebuilds candidates from DESCRIPTION STRINGS and cannot carry atoms —
guarded with NotImplementedError (use serial or spawn). Legacy
burn_in_combinations (not in the Grove.__init__ flow) stays atom-free.
Tests: 2.10(s) serial (best a1*sin(a2*x), mse ~2e-28, atom-pair
candidates verified name-disjoint); test_phase2mpi (f) spawn (same
recovery, 323 atomic candidates through the wire, guard verified).
KNOWN LIMITATION (pre-existing, documented not patched): the spawn
master's progress-wait loop exits only after ≥1 worker progress message,
and workers send one per 50 processed items — a spawn burn-in whose
per-worker chunk is < 50 stage-2 combinations hangs the master forever
(never occurs with realistic vocabularies; the (f) test sizes its
workload accordingly). Fix if ever needed: drop the sum(progress)>0
requirement — the subsequent blocking gather is the real barrier, the
loop is only cosmetic for the progress bar.
Full parameter-pool vocabulary freeze: settled by Stage 3c (ψ_const).
PRIOR-MASS QUESTION RESOLVED (2026-07-09): see DESIGN_2_10_PRIMITIVES.md —
DP(c, G₀) formalization; registration = partition refinement carving mass
from a remainder cell (replaces the append-α₀ scheme, which is NOT
registration-invariant); parameter leaves unify as k=1 constant primitives;
k_ψ enters the BIC dimension count only. Implementation order in that doc §9.
Corollaries: settles Phase 2.3 (freeze vocabulary at construction;
auto_register demoted to guarded fallback) and corrects Phase 1.5 (append-
growth invariance is FALSE as implemented; true under the splitting rule).

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
- **PT ladder tuning** ✅ DONE 2026-07-12 (exp_phase41_ladder.py; logs in
  results/phase41_ladder*.json; per-pair monitors added to Forest —
  _attempt_swap_pair/_attempt_crossover_pair counters + ladder_stats(),
  pure observability, suite regression green).
  TUNED SCHEDULE — CORRECTED PROTOCOL (rounds 6–8, 2026-07-13, σ = noise
  + shared-burn-in seeding): **T = [1, 1.22, 1.56, 2.17, 3.57, 10.0]**
  (uniform-in-β, Δβ = 0.18, 6 rungs). High-stats validation (~280–300
  attempts/pair, SE≈0.026): seed 4141 → [0.31, 0.28, 0.26, 0.26, 0.24],
  5/5 in band, max deviation 0.000 (and its best tree is the TRUE model
  a0*x + sin(x)); seed 999 → [0.30, 0.34, 0.35, 0.43, 0.33], 4/5, single
  excursion +0.029 on the harmless high side. The corrected σ TILTS the
  profile: cold pairs need finer Δβ (the old σ=1 ladder's cold pairs sag
  to 0.14–0.19) while the hot region still flattens (fine geometric
  ladders 0.5–0.7 everywhere) — see results/phase41_ladder_round{6,7,8}
  .json.
  [SUPERSEDED σ=1.0 artifact, rounds 1–5: T = [1, 1.29, 1.82, 3.08,
  5.50, 10.0] (Δβ=0.225 + split tail); kept for the paper's
  σ-sensitivity discussion.] Original σ=1 validation:
  seed 4141 → per-pair swap rates [0.31, 0.27, 0.28, 0.34, 0.24], 5/5 in
  the 0.2–0.4 band, max deviation 0.000; seed 999 → [0.44, 0.20, 0.28,
  0.25, 0.26] (cold-pair excess = post-convergence saturation, see below).
  FINDINGS (paper-relevant, method section):
  (i) NO geometric ratio meets the band — scanned r ∈ {1.2, 1.3, 1.4,
  1.6} at 8 rungs: hot pairs pin at 0.5–0.6 because Δβ ∝ 1/T for
  geometric ladders while this bounded-BIC discrete target does not grow
  its energy fluctuations with T (adjacent hot rungs become near-prior
  twins — redundant, not harmful). (ii) Uniform-in-β ladders are the
  right family; the in-band window calibrates to Δβ ≈ 0.2–0.25.
  (iii) Long-run (post-convergence) rates differ from transient ones:
  the coldest pair saturates high once both cold chains lock the
  dominant mode (free mixing — a target-concentration effect, not
  ladder geometry), and the hot tail sags as cold rungs concentrate —
  hence the split tail (3.08 → 5.5 → 10).
  Conditions (must match the benchmarks): freeze_vocab=True, safe-4 ops,
  max_tree_depth=4, subtree_rebuild_max_height=2, sigma=1.0, target
  y = sin(x)+0.5x+N(0,0.1²) n=60. Re-tune if the benchmark family
  changes materially (the script's ROUND mechanism reproduces the
  whole tuning ladder-by-ladder).
- **Recovery benchmarks** ◐ FIRST PASS DONE 2026-07-12
  (exp_phase41_recovery.py; results/phase41_recovery.json). Six known
  targets × 2 seeds on the tuned ladder, production config (frozen vocab,
  safe-4 ops, caps 4/2, σ_noise = 0.1, n = 60), scored by NRMSE vs the
  noise-free truth on a held-out grid (recovered := < 0.05).
  RESULT at budget 200 rounds × 4 steps (840 steps/grove + EMC): 7/12.
    linear          2/2  (a0*x ; x/a0)
    x_sin           2/2  (x*sin(x) EXACT, nrmse 0.0000)
    poly2           2/2  (a0*x + x² ; x*(a0+x)/a1)
    sin_plus_linear 0/2  — competing basin of rational/additive
                          approximants (nrmse 0.13–0.15)
    rational        0/2  — same failure class
    freq_sin_atom   1/2  — seed 17 recovers b6*sin(b7*x) (nrmse 0.009,
                          Ψ₂ opaque primitive recovered inside the full
                          EMC loop); seed 71 found the RIGHT structure
                          (-a0*sin(a1*x)) but with θ̂ frozen at a bad
                          Powell optimum by the description-keyed BIC
                          cache (known cache property).
  READING: perfect recovery when parameters sit in top-level
  multiplicative slots; inner-position parameters (denominator constant,
  sub-unit coefficient) lose to broad approximant basins at this budget —
  per-run recovery ≈ 25% for those (consistent with the ~3/13 rate seen
  across the ladder-tuning runs on the same target). This is a search-
  budget effect, not a model defect: the true modes score ~50+ nats
  better when found. Budget-scaling pass (--long, 3× rounds, same seeds;
  results/phase41_recovery_long.json): freq_sin_atom → 2/2 (the θ̂-cache
  miss vanishes with more rebirths); sin_plus_linear → 1/2 (true
  structure b0*x + sin(x) found); rational → still 0/2, converging to
  BARE CONSTANTS — which exposed the real defect:
  **σ MIS-CALIBRATION FINDING (2026-07-12, decides the next step):** the
  runs score with FIXED σ = 1.0 while the injected noise is σ = 0.1, so
  the likelihood term −0.5·N·mse/σ² is ~100× too flat: constant-vs-truth
  on the rational target differs by ~1.2 nats of likelihood against a
  ~2 nats/parameter BIC penalty — the constant wins LEGITIMATELY under
  the mis-specified scoring (verified: widening the domain to SNR≈4,
  results/phase41_recovery_rational_v2.json, still yields constants).
  High-variance targets mask the effect; low-amplitude structure is
  posterior-suppressed. CONSEQUENCE: benchmark scoring must use σ
  matched to the known noise (Forest(sigma=NOISE_SD) — coherent
  everywhere incl. PR) or profiled σ (trees sigma=None via set_sigmas;
  NB Grove’s ctor coerces None→1.0, and PR would still score at
  grove.sigma — the fixed-true-σ route avoids that quirk). The PT ladder
  must then be RE-TUNED under the corrected scoring (energy scale ×100 →
  swap rates change; the uniform-β methodology and the tuning script
  carry over unchanged — rerun ROUND 1 → refine). Decision on the σ
  convention (fixed-true vs profiled) is the PLAN 5.1.6 fixed-vs-profiled
  choice surfacing in practice — flagged to the author. Recovery results
  above remain valid AS a documented property of σ=1 scoring; the paper's
  likelihood-treatment section should carry this example.
- **RECOVERY v2 — CORRECTED PROTOCOL ✅ 2026-07-13
  (results/phase41_recovery_v2.json): 11/12 recovered** (was 7/12).
  Protocol: σ = noise, ONE shared spawn burn-in per (target, seed)
  seeding every rung with the same top-K (exp_common.py), re-tuned
  ladder T = [1, 1.22, 1.56, 2.17, 3.57, 10]. Highlights: rational →
  EXACT x/(a0+x) on both seeds (was 0/2 — burn-in enumerates it, σ fix
  makes it beat the constant by ~120 nats); sin_plus_linear → both seeds
  (seed 71 exact a0*x + sin(x), NRMSE 0.0015); x_sin & poly2(17) exact;
  atom target 2/2. Single miss: poly2 seed 71 stalled at x² (stochastic
  one-run event — same target exact on the other seed and recovered in
  v1; open item: inspect that burn-in's ranking).
  FIGURES + TABLES (exp_phase41_plots.py → results/figs/): fig1 recovery
  overlays; fig2 old-vs-retuned ladder rates with band; fig3 per-rung
  convergence traces (burn-in start + tempering visible); fig4 Pólya
  root-cell predictive evolution — the urn visibly LEARNS the Asin cell
  on the atom target (the paper's adaptive-prior figure); fig5
  σ-calibration nats comparison; table_recovery.csv/.tex (v1 vs v2).
- **DEPTH-3 DISCOVERY SUITE ✅ 2026-07-14 (exp_phase41_depth3.py →
  results/phase41_depth3.json): 5/6 recovered, 6/6 winning trees of
  height ≥ 3 — beyond the burn-in enumeration entirely.** Targets
  x³−2x (EXACT both seeds), 3x/(1+x²) (seed 71 EXACT in equivalent
  parametrization; seed 17 near-miss 0.069), x·sin(x)+x/(1+x) (both
  functional). Protocol = v2 + 400 rounds. New ADDITIVE trace fields in
  exp_common.snapshot (accepts, unique_structs, depths, nodes) feed
  fig7 (per-rung exploration counters — settles the fig3 max-statistic
  question) and fig8 (depth penalization over the whole history: mean
  sampled depth stratifies monotonically by T; cold rung ~91% mass at
  depths 3–4). Rendering: exp_phase41_plots_v3.py → figs 6/7/8 +
  table_depth3 + phase41_expressions.md/.csv (all 36 runs, fitted
  params substituted). KERNEL NOTE: depth cap 4 binds per-move only;
  crossover grafts compound (observed depth ≤ 10 hot, two winners at 5)
  — "soft cap", document in paper; crossover-height guard = possible
  future revision. Full write-up: RESULTS_PHASE4.md §§1–5 (+ handoff
  prompt PROMPT_RESULTS_SECTION.md for drafting the paper section).
  **§4.1 IS NOW CLOSED.**
  REPRODUCIBILITY TOOLKIT (2026-07-15, ON-DEMAND — deliberately NOT in
  the default test suite): `bash repro_phase41.sh` reruns both 4.1
  suites verbatim (workers pinned to 2 — burn-in ranking is
  worker-count sensitive) into `*_repro.json`, renders figures into
  `results/figs_repro/`, tables into `results/repro/`, and writes the
  field-by-field diff vs the archived originals to
  `results/repro/phase41_repro_report.md` (expect "REPRODUCED"; any
  diff = regression signal). Wrappers import the archived experiment
  scripts unmodified — nothing existing can be overwritten. ~1–1.5 h;
  never run concurrently with another experiment.
  NEXT: 4.2 hero application choice; 4.3 ablations; optional affine-atom
  vocabulary-policy A/B pilot (user-proposed; AtomSpec already supports
  it — identifiability-prune the wrapper per primitive); open items:
  poly2/seed-71 burn-in ranking inspection, crossover-height guard
  decision.
### 4.2 Hero applications (DECIDED 2026-07-14: TWO heroes)
**4.2a — Feynman univariate set (IN PROGRESS 2026-07-14).** Closed-form
ℝ→ℝ equations from the Feynman database (constants fixed so each is a
single-variable law), full corrected protocol with LONG chains:
`exp_phase42_feynman.py` → `results/phase42_feynman.json`, figs `fig9+`.
Six targets, iconic and graded:
  flux I.12.2 a/x² · scattering III.17.37 a(1+b·cosx) · velocity_add
  I.16.6 (u+x)/(1+ux) · polarization II.11.28 1+x/(1−x/3) · planck
  I.41.16 x³/(eˣ−1) [flagship — needs the exp primitive] · gauss I.6.20a
  e^(−x²/2)/√2π [UNREACHABLE by composition under safe-4 → registered as
  a Gaussian template A_gauss = c0·exp(−c1·x²), the Phase-2.10 custom
  primitive demonstrated on a real physics form].
Vocabulary {sin, cos, exp, x} + safe-4 (+A_gauss on the gauss target
only); budget 800 rounds × 6 steps (~3× the 4.1 budget), n = 100 points,
burn-in via spawn with 5 workers (+master = 6 MPI processes, the agreed
cap). NB the ladder was tuned at n = 60 / {sin,cos,x}: per-run
ladder_stats() is recorded as the transferability check — report drift,
re-tune only if rates leave the band materially.
**COMPLETE 2026-07-15: 9/10 recovered** (results/phase42_feynman.json,
assembled from per-run checkpoints). flux EXACT both seeds (2/x²);
scattering exact structure both seeds; velocity_add 2/2 functional
(seed 71: the elegant (x+eˣ)/(eˣ+1.04) impostor); polarization seed 71
EXACT (1+2.98x/(3.00−x)), seed 17 the sole formal miss at NRMSE 0.174
— but structurally CORRECT (−2.05−9.06/(x−3.001), right family, pole
at 3.001; the metric punishes pole sensitivity — write-up should show
the partial-fraction equivalence). planck 2/2 functional
(x²-polynomial × e⁻ˣ family; seed 71's x²(x+1.93)/(x+eˣ) mirrors the
true denominator). **gauss (A_gauss demo) DROPPED by user decision:
with the template in the burn-in, recovery is tautological — the chain
locks the primitive it was given; 4.1's freq_sin_atom already covers
the machinery demo. Informative template tests must be
template-PLUS-correction (= CBC Stage B design).** Its run also
exposed a ROBUSTNESS BUG: scipy Powell raises
RuntimeError("Too many iterations.") inside warmup θ-fits and nothing
catches it → process death. **FIXED 2026-07-15 (user go-ahead): all
three scipy minimize sites in real_functionC (optimize /
optimize_sigma / _optimize_with_grad fallback) guarded with
except (RuntimeError, ValueError, FloatingPointError, OverflowError)
→ OptimizeResult(success=False, x=warm-start θ, fun=+inf) → the
NaN-fill path rejects the candidate and sampling continues. Root
cause: scipy bracket() raises during the Powell line-search when a
1-D slice is non-finite (exp overflow) — NOT preventable via maxiter
options. The except branch fires only where the old code died ⇒
bit-identical on green runs; VERIFIED: full rebuild + all 4 suites
PASS with literally identical fingerprints (2.10o mse 5.28e-28,
2.10s/2.10f 1.72e-28). Optional layer 2 (finite-penalty objective
clamp to reduce incidence) deliberately deferred — it changes
optimizer trajectories.** Figures fig9–11 + table + expressions:
DONE (fig9 re-rendered in portrait/paper format 2026-07-15).
**4.2b — CBC gravitational-wave signal (PLANNED, ADDRESS IN A LATER
PROMPT — user decision 2026-07-14).** Compact-binary-coalescence
waveforms: templates exist for BOTH time-domain and frequency-domain
signals, with and without physical features such as ECCENTRICITY. Plan:
build an expression TREE from such a template — explicitly NOT via the
Phase-2.10 atomic-leaf machinery (user: "don't call it atomic"); the
template's known functional form is encoded as an ordinary tree over
the vocabulary, with its (estimated) parameters as the tree's
name_values — then seed/plant that tree and run SR on data from an
ECCENTRIC waveform to test whether the sampler GROWS the eccentricity
contribution as a correction the circular template lacks. Open design
choices for that prompt: time vs frequency domain first; which template
family (PN/phenomenological); whether the circular part is frozen
(fixed subtree) or free; residual-vs-full-signal formulation; profiled
σ (PLAN 5.1.6 — real-data route; NB Grove ctor coerces sigma=None→1.0).
**STAGED PLAN (2026-07-15).** Work in the FREQUENCY DOMAIN on the
SPA/TaylorF2 phasing, NOT time-domain strain: (a) strain is an
oscillatory chirp — SR on the carrier is hopeless and uninformative;
(b) the physics lives in smooth real functions Φ(f), A(f) — exactly the
ℝ→ℝ pipeline's food; (c) the leading eccentricity contribution is an
ADDITIVE secular e₀² phase term with a known fractional power of f, so
the sought correction is a top-level `+` graft, perfectly matched to
the grammar; (d) no composition operator needed (a time-domain template
A(t)·cos(φ(t)) is not even expressible under safe-4). Data generated
self-contained in numpy (TaylorF2 = closed-form PN polynomial in
v=(πMf)^(1/3); + leading TaylorF2Ecc e₀² term, coefficients from the
literature at implementation time; no LALSuite dependency).
  Stage A — residual calibration: regress ΔΦ(f)=Φ_ecc−Φ_circ directly;
  truth is a fractional power law → headline question "does the sampler
  measure the PN exponent of eccentricity?". No seeding needed.
  Stage B — template-seeded hero: regress Φ_ecc(f) whole with Φ_circ
  encoded as an ordinary TRANSPARENT tree (NOT atomic, per user),
  coefficients = the estimated (circular-fit) parameters, frozen-vs-free
  as a design flag; the sampler must graft the correction. Compare to A.
  Stage C — physics validation: e₀ grid, check fitted correction
  amplitude ∝ e₀² (the referee-memorable figure); amplitude channel
  A(f) optional.
  Prerequisites to verify before coding: `**` op semantics with a
  parameter exponent (natural power-law primitive; outside safe-4 —
  identifiability/irreducibility status needs a look); profiled σ or
  scaled fixed σ; log-spaced f grid; regress ΔΦ (O(rad), conditioned)
  rather than raw Φ (10³ rad).
  **BUILT 2026-07-15 (exp_phase42_cbc.py + run_phase42_cbc.sh +
  exp_phase42_cbc_plots.py → fig12–fig18 + table_cbc + expressions).**
  Smoke green ('**' works end-to-end); found+fixed the nanmean NRMSE
  loophole ('**' trees NaN on most of the grid were judged on valid
  points only — now valid_frac≥0.99 required, NaNs count as failure).
  STATISTICAL-POWER SPEC (user): 8 temperatures (uniform-β extension
  of the tuned ladder to Δβ=0.9/7≈0.129 — swap rates expected at/above
  band, logged per run), vocabulary {x,log,sqrt,exp,sin,cos}×{+,−,*,/,
  **} ('o' stays excluded), Stage A 800 / B–C 1600 rounds × 6 steps,
  8 MPI processes = 7 spawn burn-in workers + master (EMC loop remains
  the serial in-process multi-rung sampler). ~6–9 h; START AFTER the
  Feynman suite finishes (CPU contention).
  **BURN-IN FORENSICS → TEMPLATE REDESIGN (2026-07-15, user chose
  Option A).** Three smokes + four probes traced Stage A's failure to
  its true root: (i) NOT ambiguity — the fitted power law beats the
  best zero-parameter impostor by 47,410 nats at σ=0.001 (5-nat LOSS
  at σ=1: the §4.1-σ finding quantified); (ii) NOT enumeration — 
  a0/(x·x**a1) ≡ a0·x^(−1−a1) was candidate rank 239; (iii) NOT the
  optimizer per se — the kernel's own Powell fit of the clean form
  converges to (1.983, −3.7787), mse=σ², from θ₀=(1,1); (iv) the
  killer: '**'-composite stage-2 candidates INHERIT warm starts from
  their stage-1 sub-fits (x**a1 alone → a1≈−2.08 → combined form
  starts in a growing-exponent basin) and single-start coordinate
  Powell buries them with junk θ̂ (post 19.9 vs the ~810 a good fit
  earns). Along the way: Stage A regresses the POSITIVE dephasing
  magnitude (sign flip de-traps coefficient-first descent — probes);
  kiloradian units kept (harmless; likelihood scale-invariant);
  burn-in ranking confirmed to use the PROFILED-σ BIC branch.
  RESOLUTION (no kernel changes): drop '**' from ops (back to safe-4,
  the tuned-ladder family) and encode "phasing terms are decaying
  power laws" ONCE as the opaque template atom A_pow = c0·x^c1
  (Phase 2.10 machinery, freq_sin_atom-validated; θ₀=1 references
  suffice — clean-form fits converge from (1,1) on positive data).
  CLAIM REFRAMING: A_pow + A_pow is burn-in-enumerable ⇒ the CBC
  headline is CALIBRATED MEASUREMENT of the eccentric PN exponent
  (−34/9) and its e₀² scaling, not structural discovery (C2-style
  honesty). Option B (multi-start Powell, keeps '**' discovery
  framing) DEFERRED as deliberate 4.3-adjacent kernel work.
  **3G-REALISM UPDATE (user, 2026-07-15): band widened to an ET/CE-like
  configuration** — f₀ = 5 Hz, x ∈ [1, 200] (5 Hz → 1 kHz, 2.3 decades,
  n = 150 log-spaced): a ~1.8 h BNS-type signal (ℳ = 1.2 M☉),
  Φ ≈ 1.23e5 rad at band entry, eccentric dephasing ≈ 2×10³ rad at
  e₀ = 0.1 — and ΔΦ crosses the σ = 1 rad noise floor near ~40 Hz, so
  eccentricity is measurable ONLY at the low-frequency end (the 3G
  science case, visible in the preview). Data preview (incl. the
  time-domain chirp): `exp_phase42_cbc_preview.py` →
  `figs/fig19_cbc_data.png`.
**FLEXIBLE ATOMS / TEMPLATE SCAFFOLDS (design sketch, DEFERRED — do not
build preemptively).** User question 2026-07-15: atoms whose internal
nodes are immutable but whose designated leaves (slots) can expand into
subtrees; composition Ψ(g(x)) is the arity-1 special case (current
atoms hard-fill the argument slot with x). Formal view: a flexible atom
is a USER-DEFINED m-ARY OPERATOR whose expansion emits a deterministic
frozen skeleton — prior accounting needs nothing new: one Pólya cell
for the Ψ-choice (carve-from-remainder, as now); frozen interior =
probability 1 (grammar production, not part of the sample space); each
slot = an ordinary child subtrie of the Ψ trie node (slot-specific
reinforcement automatic); BIC k via name_values unchanged; per-template
identifiability pruning required (skeleton θ redundant against slot
content — same class as the affine-wrapper rule). The REAL cost is the
kernel, not the prior: move-site masking (frozen nodes never targets;
site-count factors in acceptance ratios count mutable sites only),
slot-birth/death + atom-death with force-reject reverses, irreducibility
re-verification (THEORY §4 style), skel wire + burn-in token extension —
a Phase-2.10-sized project (label it 2.11 if/when opened). GATE: run
4.2b Stage B on pure seeding first; implement scaffolds only if the
seeded template is observed being destroyed before corrections attach.
**HARD IMPLEMENTATION CONSTRAINT (user, 2026-07-15): flexible atoms
must be PURELY ADDITIVE code additions — zero effect on existing code
paths, OFF BY DEFAULT, opt-in by explicit flag/argument (same contract
as atoms=None ⇒ bit-identical legacy in Phase 2.10). A feature one may
activate, never a change one must accommodate.** Context: the
frequency-domain CBC hero (below) needs NO flexible atoms; the
TIME-DOMAIN CBC treatment (A(t)·cos(φ(t)) — composition into the
carrier) is the use case that will need them, deferred accordingly.
**NaN SCORING LEAK — FOUND AND FIXED (2026-07-15, user-approved plan;
full plan file: ~/.claude/plans/delegated-foraging-rabin.md).** The
first full CBC run (8 rungs, 800/1600 rounds) produced 0/3 compromised
runs: trees NaN on ~99.6% of the grid carried posteriors (−10..−12.5)
BEATING the exact truth (−87, probe-verified). ROOT CAUSE (traced by
agents + manual reads): real_functionC.fitness used
np.nanmean(resid²) — NaN predictions silently DROPPED, so Powell
actively drives parameters into NaN territory (x^8.6e14) because it
shrinks the mean over surviving points; is_valid can't catch it (no
params arg → evaluates at reference θ=1, NOT θ̂ — the 2.13(h) comment
overstated it, now corrected); the description-keyed BIC cache
memoized the flattered scores. inf was never a hole (poisons the
mean); NaN specifically was. Short smokes passed because 60 rounds is
too little for MCMC to FIND the exploit — long chains always do.
FIX (kernel, minimal, hot-path FASTER): fitness/fitness_sigma/
posterior_fitness → plain np.sum + scalar C isfinite + finite
_MSE_PENALTY=1e100 (zero-likelihood semantics for any non-finite
prediction; finite penalty keeps Powell bracket-stable);
_log_likelihood_at nansum→sum (all-NaN gave ssr=0 and PASSED — closed);
groveC: penalty-scale posts (< −1e90) are NOT cached (pre-fix NaN
posts weren't either — preserves retry-with-fresh-warm-start, prevents
a single bad Powell landing from permanently condemning the TRUE
topology). On all-finite vectors np.sum is bit-identical to what
nanmean computed and strictly cheaper (no isnan mask/count).
VALIDATION: probe (a) finite-path exactness PASS at 2e-14 rel
(−87.025718642 reproduced); ALL 4 suites PASS with bit-identical
historical fingerprints (5.28e-28 / 1.72e-28); walls within the
environmental band; test_phase42 fingerprint PASSES against the
PRE-FIX reference (flux/17's 60-round trajectory is NaN/inf-free ⇒
kernels literally bit-identical there; no re-record needed). Penalty-
branch assay (CBC re-smoke): RECOVERED 0.0003, valid=100%,
p̂=−3.7787, cold −87.026 — no monsters in their native habitat.
4.1 repro vs archives: 15/18 runs BIT-IDENTICAL; 3 diffs inspected =
1 name-band-only (same model/score), inv_quad/17 near-miss→RECOVERED
(depth-3 now 6/6 on fixed kernel), inv_quad/71 same family but more
parsimonious (post −26.1 vs −36.3). Nothing lost, one gained.
**FIX FULLY VALIDATED 2026-07-15.** Incident
corpus preserved: results/phase42_cbc_partial_INCIDENT_20260715.json
+ _run_INCIDENT_20260715.log. Residual risks documented in the plan
file (Jacobian nansum if gradients re-enabled; Feynman held-out nrmse
lacks valid_frac correction — reporting-side; worker burn-in
params=None/profiled-σ anomaly — pre-existing).
### 4.3 Ablations (no Pólya, no tempering, no BIC)
**ABLATION I — FROZEN vs ADAPTIVE Θ: DONE 2026-07-16 (RESULTS §9).**
Kernel: `polya_treeC.adapt_frozen` opt-in switch (5 additive edits,
default off = bit-identical — certified by suites + fingerprint;
travels through get_state with legacy-safe default). Experiment:
`exp_phase43_ablation.py` frozen arm (18 runs, both 4.1 suites,
Θ frozen after warm-up) vs the post-fix repro reruns as the adaptive
arm (same kernel/seeds/protocol, one-bit difference). **RESULT:
PARITY — 11v11 (depth-2) and 6v6 (depth-3), same single miss; no
systematic Δposterior direction on the diverging depth-3 pairs.**
With fig23(d) (root-predictive movement decaying to ~6e-4) the
reading: post-warm-up adaptation is too small to change benchmark
outcomes ⇒ direct empirical support for "frozen = recommended
default"; adaptive not refuted, its benefit deferred to unseeded/
longer-horizon settings (outlook). Outputs in results/ablation_results/
(dedicated subfolder, additive). ALSO NEW: fig23_cbc_urn.png (urn
dynamics + the qconv empirical companion) and fig24_cbc_mixing.png
(mixing health) via exp_phase42_cbc_diag2.py from archived CBC traces.
**ARM C — FROZEN-AT-BIRTH: DONE 2026-07-17 (user-run; RESULTS §9).
The symmetric untrained prior DEGRADES: 14/18 vs 17/18 for both
adaptive and frozen-at-Θ̄** (rational/71 lost outright 0.0034→0.393;
inv_quad/17 + mixed/17 past threshold; freq_sin_atom/17 5× worse but
surviving — the un-reinforced template cell mechanism; poly2/71 misses
in ALL arms ⇒ burn-in artifact). COMPLETED CONCLUSION: reinforcement
pays off and is FRONT-LOADED — learning Θ̄ matters (+3 recoveries),
learning beyond Θ̄ adds nothing at these budgets. Recipe: reinforce
through warm-up, freeze at Θ̄, sample with theorem-backed guarantees.
REMAINING 4.3 ARMS: **ALL DONE 2026-07-19 (RESULTS §10) — PHASE 4
COMPLETE.** notemper 16/18 (inv_quad 0/2: tempering matters exactly
on the multimodal target); nobic 12/18 with winners at k up to 20 /
depth 8 (parsimony = what makes the output symbolic); affine 18/18 —
best rate of any arm, fixed poly2/71, freq_sin_atom won by a single
depth-0 template leaf, BUT every depth-3 winner template-dressed
(cosine-cubics etc.) at 5–25× compute — dressed vocabularies trade
symbolic fidelity + compute for robustness. Cross-cutting: EVERY
perturbed arm fixed poly2/71 (knife-edge miss confirmed).
Component-justification table: RESULTS §10. Deferred past Phase 4:
no-Pólya arm (user), profiled-σ real-data route, flexible atoms
(2.11), multi-start Powell, adaptation-in-unseeded-searches question.

### 4.4 Structural-constraint prior tilt (opt-in) — DONE 2026-07-21
**Origin (user question):** can constraints like "this has to diverge
like x²" be added to the LOSS, possibly via loss_function.pyx, without
breaking the formalism? **Formal answer that fixed the design:** a
candidate-dependent, data-independent C(τ) is a PRIOR statement.
Sound wiring: multiplicative tilt π(τ) ∝ π_polya(τ)·exp(−λC(τ)) —
Bayes/evidence/BIC intact, MH ratios gain one term, bounded penalties
preserve irreducibility, whole-posterior tempering treats it like any
prior factor. Loss-side wiring (all a (y_true,y_pred) API can say)
multiplies C by n/(2σ²) and tempers it as likelihood → constraint
strength scales with n and noise, evidence semantics collapse (Gibbs-
posterior trap). loss_function.pyx / custom_loss_functionC stay DEAD —
not the vehicle. **Kernel (additive, off by default):**
`structural_prior=None` kwarg on Grove AND Forest (threaded like
atoms) accepting a callable tree→log-prior increment; pxd fields
structural_prior/_sp_cache/structural_prior_cache; cdef `_sp` helper
(None → 0.0 single branch; description-keyed cache, opt-out kwarg for
θ-sensitive callables; non-finite returns floored at −1e90 = the BIC-
cache penalty threshold); tilt added at the SIX acceptance sums where
the Pólya prior enters (_step dtarget, pair crossover, swap U's,
trans-grove U's, both spawn-master mirrors — mirrors wired correctly
incl. pre-crossover old-tilt capture, though unreachable in v1) and
NOWHERE else: never root_posteriors, never mse/BIC cache, never
burn-in ranking, never the Powell objective. `run_mpi_spawn` + tilt →
NotImplementedError (callables don't pickle; declarative specs = v2).
**Experiment layer:** NEW exp_constraints.py — formal contract in the
module docstring; `asymptotic_power(p, domain, lam, c_max)` builder
(symbolic-first: sympy limit of log|f|/log x at +∞ with generic
real-nonzero parameter symbols; numeric fallback: log-log slope on
the outermost pair of probes at hi×(4,16,64) beyond the data domain;
evaluation failure = max penalty; bounded −λ·min(C,c_max));
`tree_name_values` accessor (get_param_dicts()[1]). **Validation
2026-07-21: test_phase44.py 14/14 PASS** — T1 zero-returning callable
reproduces structural_prior=None BIT-IDENTICAL (descriptions + posts;
proves the six insertions inject exactly 0.0 and consume no RNG); T3
steering on y=x² (noiseless, [1,2.5], polynomial ops): tilt p=3
λ=200 → final population powers [3.00, 2.998, 3.017, 3.0, 3.34,
2.86]; agreeing tilt p=2 → [2.0…2.11] and CLEANER than no-tilt;
NEGATIVE CONTROL (no tilt) powers scatter [3.36, 9.22, 5.07, 4.25,
1.21, 0.75] — the data alone do NOT pin asymptotics (near-zero
dressed limbs shift the exponent at ~2 nats BIC each): this control
IS the feature's motivation, and doubles as the planned wrong-
constraint demo. T4 cache: 296 evals / 296 unique descriptions / 296
cached, values exactly the callable's returns. T5 guards (TypeError,
Forest forwarding, spawn NotImplementedError). **Certification
(feature off): all four suites + test_phase2mpi PASS with bit-
identical fingerprints (2.10o 5.28e-28; 2.10s and 2.10f spawn
1.72e-28); test_phase42 pipeline fingerprint PASS (post −64.534224,
nrmse 0.009240); walls in the environmental band.** NB best_tree/
root_posteriors stay BIC-only by invariant: the tilt shapes WHICH
structures the chain retains (and therefore what can win), not the
reported evidence. Paper/thesis note queued:
PROMPT_PAPER_CONSTRAINTS_NOTE.md (≤1 page, methods note + the
custom-loss-rejection paragraph). Deferred: non-plug-in θ-dependent
constraints; x→0 / two-sided asymptotics (builder currently +∞ only).

**v2 — DECLARATIVE-SPEC ROUTE (parallel shipping) + WIRE-FIDELITY
FINDING, 2026-07-22.** Implemented the picklable
`structural_prior_spec=('module','builder',kwargs)` route so the tilt
can cross the MPI-spawn boundary (raw closures can't): resolver
`groveC._resolve_structural_prior_spec` (lazy name-driven import — no
kernel↔experiment coupling), new `structural_prior_spec` field
(pxd + Grove/Forest ctor, mutually exclusive with the raw callable),
stashed in `_ctor`, and `mpi_worker` rebuilds it LOCALLY (`.get`
defaults keep old configs valid). Spawn guard relaxed: raw callable
still → NotImplementedError, a spec passes. Builder `exp_constraints.
spec()` + `theta_policy='current'|'reference'`. **Post-v2
certification GREEN and bit-identical** (four suites + test_phase2mpi
+ test_phase42 fingerprints unchanged). test_phase44 T1–T6 PASS incl.
all v2 resolution/guard checks (spec↔callable equivalence, mutual
exclusion, malformed-spec rejection, theta_policy).
**WIRE-FIDELITY FINDING (pre-existing kernel defect, NOT a v2 bug):**
the spec provably ships to and fires in the workers (probe P1), but
the spawn SAMPLING wire `groveC._tree_from_wire` gates on
`tree_get_description() == str(sympy_expr)`; sympy folds the tilt's
target `x*x*x → x**3`, which the **-free vocabulary can't rebuild, so
the master DISCARDS exactly the steered trees and keeps stale initial
ones (probes: 4/6 and 6/6 slots stale; master `_sp_cache` starved).
Latent until now because `exp_common.emc_loop` is SERIAL — the
4.1/4.3/Feynman/CBC suites only spawned burn-in, never sampling, and
test_phase2mpi [2.12] never asserted the master population REFRESHES.
No published result affected (all serial sampling). **Per user
(2026-07-22): parallel steering demo PAUSED; bug handed to the
codebase-developer lane in PROMPT_DEV_WIRE_FIDELITY.md** (preferred
fix: ship the tree skeleton not the sympy expr for atom-free trees,
mirroring the 'skel3b' atom path; re-extend [2.12] to assert refresh).
T7 now asserts only the shipping contract (spec accepted + stats dict)
and prints the wire footprint as a diagnostic; the steering assertion
returns after the engine fix. exp_phase44_spec.py demo (serial arms +
paused spawn arms) and fig28 deferred to that point. Serial tilt path
(T3b clean steering) unaffected and certified. Also OPEN (my lane):
clean idle-machine timed cert to answer the user's "did test time
double" — earlier walls were background-contention-polluted.

**DEV PLAN — spawn-sampling wire fidelity fix (2026-07-24, kernel
lane).** Fleshes out the handoff in PROMPT_DEV_WIRE_FIDELITY.md; extends
Phase 2.12. GOAL: make the master's post-`run_mpi_spawn` population
faithfully reflect worker exploration over folding-prone (`**`-free)
vocabularies, unblocking (a) run_mpi_spawn SAMPLING generally and (b)
the Phase 4.4 tilt parallel path. Root cause recap: `_tree_to_wire`
ships atom-free trees as a RAW SYMPY EXPR; sympy folds on the way onto
the wire (`x*x*x → x**3`, `x+x → 2*x`, reorder); `_tree_from_wire`
rebuilds via `_expr_to_tree` then gates on
`tree_get_description() == str(expr)`; when the vocab lacks `**` the
fold is unrepresentable, the gate returns None, and the master keeps
the stale initial slot.

CHOSEN DIRECTION — option 1 (skeleton payload for ALL trees), the
in-design fix. `function_treeC.tree_to_skel`/`node_from_skel` ALREADY
handle atom-free trees: tag `'L'` rebuilds a plain leaf from its own
`get_description()` via `from_string`, tag `'O'` rebuilds an operator
node preserving the exact op, and the `'A'` atom branch is simply never
taken when `atom_specs=None` (verified function_treeC.pyx:1136–1178).
So a skeleton round-trip is SHAPE-EXACT and never invokes sympy
canonicalization — `x*x*x` stays three `*` nodes, description stays
in-vocab, and the fidelity gate (now comparing the rebuilt desc to the
SHIPPED desc `skel[2]`, not a re-canonicalized `str(expr)`) passes.

WORK ITEMS (kernel, additive):
1. `groveC._tree_to_wire`: drop the atom-free raw-expr branch — always
   emit `('skel3b', tree_to_skel(root), tree_get_description())`. One
   payload form for all trees; the `has_atomic` split disappears.
   (Alternatively keep both branches and just force the skeleton path —
   but a single format is cleaner and `_tree_from_wire` already
   dispatches on the `'skel3b'` tag.)
2. `groveC._tree_from_wire`: NO functional change needed — the skel
   branch (2400–2421) already rebuilds via `node_from_skel(..., None)`,
   installs `name_values` via `_install_cached_theta`, and gates on
   `expr[2]`. Keep the legacy raw-expr rebuild branch (2422+) in place,
   harmless (never produced now), for wire-format tolerance.
3. Confirm the plain-param install path: `_install_cached_theta` must
   set `a0,a1,…` on a skeleton-rebuilt atom-free tree identically to the
   old raw-expr param loop (2428+). Spot-check a param-bearing leaf
   (`from_string("a0")` → param node → value set) round-trips value +
   description.

CERTIFICATION STRATEGY (the crux — this fix is NOT off-by-default; it
changes the spawn-sampling path on purpose):
- SERIAL suites (4.1/4.3/Feynman/CBC, 2.10o) never touch the wire →
  MUST stay bit-identical (fingerprint 2.10o mse 5.28e-28). Hard gate.
- ATOM-BEARING spawn tests (2.10s/2.10f, 1.72e-28) already use the
  skel3b payload → unaffected → MUST stay bit-identical. Hard gate.
- ATOM-FREE spawn sampling is the ONLY path that changes. It moves for
  TWO legitimate reasons: (i) previously-DISCARDED folded trees are now
  retained (the fix's whole point); (ii) previously-ACCEPTED trees that
  `_expr_to_tree` silently RE-NORMALIZED (reorder/restructure) now keep
  their exact worker-side shape. So expect test_phase2mpi [2.12] and any
  atom-free spawn fingerprint to MOVE. Procedure: run the fixed build
  against the full fingerprint suite FIRST; verify only the atom-free
  spawn numbers moved; for each mover, confirm the delta is "population
  now refreshes / shape now exact" (not a regression) then RE-BASELINE
  that fingerprint with a dated PLAN note recording old→new and the
  justification. Serial + atom-bearing fingerprints unchanged is the
  pass condition.
- PERF: skeleton build/rebuild is CHEAPER than sympy parse+`str()`
  (no sympy on the hot spawn path), so if anything spawn sampling gets
  faster; the change is confined to the spawn wire (off by default in
  every serial run) so the "now it's fast" guarantee is untouched.

TEST EXTENSION [2.12] (my lane, after the kernel change lands):
- New unit: the three planted strings from the repro (`(x*(x*x))`,
  `a0*(x*(x*x))`, `((x*x)+a0*(x*(x*x)))`) now round-trip through
  `_tree_to_wire`→`_tree_from_wire` to a NON-rejected tree whose
  description stays in the `**`-free vocab (was `<REJECTED>`).
- New END-TO-END assertion: after `run_mpi_spawn` over a folding-prone
  vocab (x²/x³ target, ops `["+","-","*"]`), assert ≥K master slots
  DIFFER from their initial descriptions (the missing "population
  refreshed" check that let this stay latent). Keep the existing
  valid+finite+explored asserts.

DOWNSTREAM UNBLOCK (my lane, gated on green cert): flip test_phase44 T7
from the shipping-contract-only stub back to the full parallel steering
assertion; run the PAUSED spawn arms of exp_phase44_spec.py; produce
fig28 (serial vs spawn steering agree).

OPEN QUESTIONS / RISKS: (1) does any fingerprinted run OTHER than
test_phase2mpi exercise atom-free spawn SAMPLING (vs only burn-in)? —
enumerate before re-baselining. (2) burn-in also uses this wire; for
representable depth-1 trees skeleton reproduces the same descriptions,
but a foldable burn-in candidate previously SKIPPED could now be
ACCEPTED — check whether any fingerprinted suite's spawned burn-in ever
ships a foldable tree (if not, burn-in fingerprints hold; if so,
re-baseline as above). (3) NOT in scope: option 2 (canonical-form gate)
— rejected, leaves the representability gap and is semantically lossy.

**RESOLUTION 2026-07-24 — WIRE-FIDELITY FIX SHIPPED + CERTIFIED.**
Implemented variant: NOT the unconditional always-skel of work item 1
but the anticipated SCOPED alternative — `_tree_to_wire` gains a third
optional arg `bint force_skel=False` (pxd: `=*`), passed True ONLY at
the three sampling-wire producers (run_mpi_spawn init_trees ~:2975;
both crossover-child wires ~:3167; mpi_worker._sampler_tree_to_wire)
so run_mpi_spawn ships EVERY tree as ('skel3b', skeleton,
description). Burn-in producers untouched (depth1_data 2-arg call;
_evaluate_tree's call is atom-gated). WHY scoped: (a) test_phase210
[2.10p]:855 HARD-ASSERTS the atom-free default payload stays legacy
("must stay legacy"); (b) the survey answered open question (2)
EMPIRICALLY — test_phase42's spawned burn-in DOES ship a foldable
tree (worker log: rebuilt "exp(a0*x)" != wire "exp(2*x)"), so an
unconditional switch would have moved the pipeline fingerprint;
scoping keeps every certified fingerprint bit-identical BY
CONSTRUCTION while replacing the defective wire. `_tree_from_wire`
needed NO change (its skel branch already gates on the SHIPPED
description; theta-equivalence audit: _install_cached_theta is
line-for-line the legacy plain-param loop, same order/lookup/skip;
from_string('a0') yields a proper registered param leaf; skeleton
rebuild is STRICTLY more faithful — exact topology where
_expr_to_tree re-associates).
**Certification (same-day baseline vs postfix, timed cert script
/tmp/cert_wirefix.sh, full per-suite logs byte-diffed,
/tmp/certw_baseline vs /tmp/certw_postfix):** test_phase23 + 28 diff
= 0 LINES; test_phase210 diff = 0 after normalizing id() memory
addresses in debug prints; test_phase29 identical EXCEPT [2.9c],
which is PRE-EXISTING run-to-run nondeterminism (three distinct
values across runs of the OLD binary alone: Jul-21 TV=0.302/xover
261; baseline 0.029/165; 2.9a/b/d/d2 byte-identical); test_phase2mpi
moved ONLY in [2.12]+[2.10q] — exactly the two run_mpi_spawn sampling
sections = the fix surface ([2.12] xover 4→6, paths 58→68, best now
actually evolves: "x + cos(x)"); [2.12d] serial and BOTH [2.10f]
burn-in halves byte-identical incl. 1.72e-28; test_phase42
byte-identical except one worker-stdout interleave line — post
−64.534224 / nrmse 0.009240 match the reference JSON. Fingerprints
2.10o 5.28e-28, 2.10s 1.72e-28 reproduced. **PERF:** cert totals
550s→566s; serial suites moved BOTH directions within machine noise
(23: 69→57s, 29: 248→255s, 210: 68→73s); the only real mover is
test_phase2mpi 38→49s, which is the FIX WORKING (populations no
longer starve → more surviving trees to score, paths 58→68). The
skeleton payload is sympy-free (no expr pickling / no str(expr) on
the sampling path), so per-tree wire cost went DOWN, not up. Serial
"now it's fast" guarantee untouched (wire code unreachable from
serial runs).
**NEW test_phase212_wire.py (additive) — 27/27 PASS, 19s
(mpiexec -n 1):** W1 the three defect-report structures
(x*(x*x), a0*(x*x), x*x+a0*(x*x)) round-trip under force_skel —
payload skel3b, shipped desc == source desc ('x**3', 'a0*x**2',
'a0*x**2 + x**2'), pickle-safe, exact-shape skeleton equality, a0
value installs from name_values; W2 negative control — the SAME
trees on the DEFAULT payload still fold (Pow/Mul/Add) and still
REJECT (documents the defect mechanism + pins the untouched legacy
path); W3 end-to-end refresh — after run_mpi_spawn(20×20, **-free
vocab) the master REFRESHES 6/6 cold-grove slots and 12/12 total
(pre-fix footprint: 0–2 of 6), all slots valid + finite, and the
refreshed cold population lands on 'x**2' (the true structure). This
is the [2.12]-extension check the handoff demanded ("assert the
master population actually REFRESHES").
**test_phase44 T7 RESTORED (same session): 21/21 PASS.** Fixture
switched to the folding-prone T3 vocabulary (sin/x, +-*) ON PURPOSE;
T7 shipping contract PASS; NEW T7b refresh: 0/6 stale; NEW T7c
steering ON THE MASTER: powers [3.0, 3.0, 3.0, 2.01, 2.97, 3.0],
m3=5 > m2=1, best slot 'a0*x**3' — the Phase 4.4 v2 parallel
steering path is now validated END-TO-END (serial T1 bit-identity
re-confirmed on the fixed build).
**Adversarial review (3 independent lenses over the diff + new test):
verdict CLEAN on all three** — correctness (every wire consumer
checked against the skel payload; W2's rejection mechanism traced:
_expr_to_tree rebuilds Pow exponents as fresh params, 'x**a0' !=
'x**3' → None), bit-identity (generated C proves 2-arg calls compile
to force_skel=0; export/import signature strings identical in both
.so; timestamps prove both rebuilt post-edit), test quality (suite
independently re-executed, 27/27). Two minor findings addressed in
test_phase212_wire.py (W3 try/except fail-loud guard mirroring T7;
coverage note: crossover-child producers fire only on accepted
trans-grove crossovers — structurally covered by W1); re-run 27/27.
Two pre-existing notes recorded, NOT introduced by the fix, left as
future kernel-lane items: (i) worker startup fallback installs the
shipped posterior even when it falls back to the local tree
(mpi_worker ~:456); (ii) worker-side per-round rebuild failure is a
silent keep-old (now failure modes coincide with the master's
pre-check under skel).
**exp_phase44_spec.py DEMO + fig28 — COMPLETE 2026-07-24 (all five
arms, spawn arms now unblocked):** serial_notilt 13.7s powers
[2.0,0.94,1.0,2.0,1.18,2.0] (control: unpinned); serial_tilt_p2
341.4s powers [2.0,2.0,1.89,2.0,2.0,2.0] (agreeing tilt = cleaner);
spawn_notilt 11.2s [1.0,2.0,2.0,0.99,2.0,1.0]; spawn_spec_p2 232.6s
[2.0×5, 0.73] (spec route parallel steering agrees with serial);
spawn_spec_p3 442.6s powers [2.76,2.89,3.0,3.0,3.0,2.97] — the
ADVERSARIAL parallel steering demo on the MASTER (pre-fix: n3=0,
slots stale). All arms best=x**2, in-domain AND extrapolation
NRMSE 0.00e+00. Tilt-ON walls are the documented v1 sympy pricing
cost (feature-off arms 11–14s), not a wire cost. Outputs (new names):
results/phase44_spec_demo.json, results/phase44_spec_demo_report.md,
results/figs/fig28_constraint_demo.png.

## Phase FD — Flexible multivariate domains (branch `flexible_domain`) ✅ KERNEL IMPLEMENTED 2026-07-19
f: ℝᵈ → ℝ with user-defined variable names (default x0..x_{d-1}); legacy
1-D runs bit-identical. Full design (decisions D1–D8), edit inventory,
test plan and implementation notes live in **`flexible_domain_plan.md`**
(the phase's living doc — read §0 and §9 first).
- Kernel: module-level variable registry in real_functionC with
  REGISTRY-FIRST symbol classification (the 'x0'-matches-the-param-regex
  trap: a variable classified as a parameter would be optimized as a
  constant); strict (d, n) coordinate-rows domain layout; from_string
  compiles over all registry variables (d=1 keeps the historical closure
  verbatim); naming guards so new_par/atom slots can never mint a
  registered variable name; compose ('o') gated d=1-only via ONE innermost
  gate in RealFunctionC.compose (compose-at-d>1 is a standing item, plan
  §8.1 — user decision 2026-07-19); d>1 trees ALWAYS ride the skel3b wire
  (raw-expr rebuild is d=1-only); spawned MPI workers install the shipped
  registry before any rebuild.
- Additive contract CERTIFIED on this box: seeded 150-step micro-run
  (bare Grove on the 6-op vocab + Pólya-reinforced Grove + 2-rung Forest
  EMC with swaps/crossovers) BYTE-IDENTICAL pre/post-edit; suites
  test_phase23/28/29, `mpiexec -n 1` test_phase2mpi, test_cython4 all
  green. NB the 2.10 fingerprint suites live in the experiment-side
  checkout — re-run there before any merge to master.
- Tests: `test_phaseFD.py` (FD-a..FD-k): registry/classification incl.
  the variable-as-parameter catastrophe assert, naming guards, (n,)
  shape law, BIC branches + NaN-penalty regression at d=2, d=2
  determinism (accept rate 0.38), Pólya + frozen vocabulary at d=2,
  serial burn-in+EMC recovery of y = 2.5·x0 + sin(x1) (nrmse 5.3e-4),
  wire round-trips (legacy d=1 raw-expr payload byte-contract kept),
  atoms at d=2, live 2-rung run_mpi_spawn at d=2 (`--mpi` flag under
  `mpiexec -n 1`).
- Pre-existing anomalies OBSERVED, deliberately untouched (flagged for
  the test-side instance, plan §9): Tree's σ-ARRAY optimize path passes
  optimize_sigma args in the wrong positional slots (function_treeC
  ~944/981, never exercised by suites); RealFunctionC.is_param() returns
  True/None, never False; module-level `is_parameter` is dead code.

### FD ⋈ master INTEGRATION — MERGED + CERTIFIED 2026-07-28
Branch `integrate-fd` = merge commit **f69fca5**, parents **7e48140**
(master line: Phase 4.4 tilt v1+v2 + the 2026-07-24 wire-fidelity fix)
and **80fa784** (flexible_domain). Merge base dd0f992. FD was fetched
directly from the sibling clone
(`gp/2d_domain/machine-scientist`, local path — no remote round-trip;
80fa784 is NOT yet on origin). The two lines are near-orthogonal: FD
never touched the six tilt acceptance sums, master never touched the
registry/shape machinery, so only **groveC.pyx + mpi_worker.pyx**
conflicted (7 hunks); PLAN.md auto-merged.

**Conflict resolutions — all unions, no semantic compromise:**
(1)+(2) Grove and Forest ctor signatures take the three tilt kwargs
THEN `variables=None`; (3) Forest body keeps the tilt mutual-exclusion
TypeError BEFORE the registry install, so a malformed call cannot
mutate the process-global variable registry; (4) **the wire predicate
is the OR of both fixes** — `force_skel or has_atomic or
real_functionC.n_variables() > 1` — the atomic scan still skipped when
force_skel (master's cheap path) and d>1 still auto-skeletons (FD's
invariant); (5) Forest→Grove forwards all four kwargs; (6) `_ctor`
carries `structural_prior_spec` + `structural_prior_cache` +
`variables`; (7) mpi_worker sampler Grove ctor takes both features'
kwargs. **Ordering property that fell out for free:** FD installs the
registry at mpi_worker ~:414, BEFORE the AtomSpec rebuild and before
Grove construction — and the tilt spec resolves INSIDE the Grove ctor,
i.e. strictly after, so a future d>1 constraint builder may reference
variable names safely.

**Certification (both regimes, one pipeline, 1876s):** rebuild clean
(123s). d=1 — test_phase23 and test_phase28 logs **BYTE-IDENTICAL** to
the pre-merge postfix baseline; test_phase210 identical modulo id()
addresses, fingerprints **2.10o 5.28e-28 / 2.10s 1.72e-28** reproduced;
test_phase2mpi **[2.12] byte-identical** (paths=68, swaps=4, xover=6,
best `x + cos(x)`), [2.12d] serial and [2.10f] spawn burn-in identical
incl. 1.72e-28; test_phase42 pipeline fingerprint **post −64.534224 /
nrmse 0.009240 (matches reference)**. test_phase212_wire **27/27**,
test_phase44 **21/21** (T7b 0/6 stale, T7c m3=5). d>1 — test_phaseFD
**ALL PASSED** both serial (152s) and `--mpi` (159s), including the
live 2-rung `run_mpi_spawn` at d=2 with an atom on board.
**The two movers were PROVEN pre-existing nondeterminism, not merge
effects:** a determinism probe ran test_phase2mpi TWICE on the
identical merged binary → `[2.10q] polya_paths_replayed` **117 vs 113**
(cert had 123 vs 111; same band, PASS either way, mse 0.00e+00 both
runs), and [2.9c] is the long-documented TV-statistic wobble (three
distinct values from the OLD binary alone). Everything deterministic
stayed deterministic across the probe ([2.12], [2.10f], [2.12d]).
**Structural completeness verified:** `git diff 7e48140..HEAD` is
exactly the FD file set, `git diff 80fa784..HEAD` exactly the
master-side set, zero conflict markers, and the no-duplication counts
are exact (`def __init__` 2, `set_variables(_vars)` 2, `cdef tuple
_vars` 2, `cdef bint _flex` 2) — no hunk lost, none duplicated.

**PERF:** no serial-path change (all FD validation is flexible-mode-
only; legacy `variables=None` + 1-D x skips every new branch by
construction) — d=1 walls in the usual environmental band, and FD's
own timing check reports d=1 0.006s vs d=2 0.007s per 2000 evaluates.
NB `structural_prior` + d>1 is UNTESTED and the shipped
`exp_constraints.asymptotic_power` builder is 1-D (x→+∞) by
construction — treat the tilt as a d=1 feature until a multivariate
constraint builder exists (deferred item, not a defect).

---

## Phase MR — Merger-ringdown closed forms (science case OPENED 2026-07-28)
Target: SEOBNRv5HM merger-ringdown ansatz (PhysRevD.108.124035 Sec. IV B,
Eqs. 48–50) — SR on the QNM-factorized residual channels Ã_lm(t), φ̃_lm(t)
(separately, d=1), per-point σ from NR resolution differences; the
oscillatory waveform enters ONLY through the σ weights and as the held-out
waveform-domain judge. Agreed order of work: (1) σ-array kernel route
[DONE below] → (2) transparent atoms → (3) preprocessing pipeline
(codebase-independent, parallelizable via a separate instance).

### MR.1 σ-ARRAY FIX — DONE + CERTIFIED 2026-07-28
The per-point-σ route existed since Phase 2.7(a) but was unreachable.
THREE defect surfaces (first flagged at FD review, flexible_domain_plan §9):
  (i)+(ii) `Tree._update_parameters` / `_update_parameters_and_fitness`
  passed (input_pars, method_str, use_gradients, self.sigma) into
  `Node.optimize_sigma(..., input_pars, SIGMA, method_str, use_gradients)`
  — the str landed in the typed float64 sigma slot → TypeError; the
  optimize path for array σ never ran. (iii) `gp_movesC.parameter_refresh`
  took `double sigma` → TypeError at the coercion on the FIRST PR draw of
  any Grove configured with an array (pr_prob=0.02 ⇒ every EMC run dies).
FIX (additive contract respected):
  * both call sites repaired to declaration order, σ coerced
    `np.ascontiguousarray(...,float64).ravel()` at the typed boundary
    (no-op view for proper arrays; lists/float32 now work);
  * same coercion at the optimize=False scoring boundary
    (`_log_sigma_array_bic` call site);
  * `parameter_refresh` σ `double`→`object`: scalar branch verbatim
    (verified in generated C: same PyFloat_AsDouble → same vtab call, no
    added RNG); array branch routes through NEW ADDITIVE
    `Tree._log_likelihood_at_sigma_arr` (heteroscedastic Gaussian:
    −0.5Σlog(2πσᵢ²) −0.5Σrᵢ²/σᵢ²; −inf on shape mismatch / any σᵢ≤0 or
    non-finite / non-finite wssr; plain np.sum, zero-likelihood semantics).
NORMALIZATION AUDIT: σ applied exactly once on all four scoring paths
(array/scalar × optimize on/off); `_shortcut_bic`'s array branch consumes
the already-weighted mse from `objective_sigma`, scalar branch divides once.
NEW TEST `test_sigma_array.py` (28 checks, no MPI): TypeError regressions
at all three surfaces; analytic scorer identities (fast + slow path);
const-array ≡ scalar (exact + same Powell landing); NaN → penalty-scale on
the array path; PR chains scalar↔const-array step-identical under pinned
RNG; SA-e4/e5 = full numpy replication of a 12-step heteroscedastic PR
chain (AM cold+warm proposals, Robbins–Monro, accept draws, per-point
logL) matched step-for-step — the value-level pin of the only new math;
guard checks (σᵢ≤0, wrong length ⇒ deterministic reject); Grove smoke
(σ ramp, PR ACCEPTED≥1, σ-identity pin on trees, stored post == analytic
weighted BIC, same-seed determinism); d=2×σ-array Grove; Forest
swaps/crossovers. 28/28 PASS.
CERTIFICATION (baseline pre-fix binary vs postfix, per-suite logs
byte-diffed, /tmp/sa_base vs /tmp/sa_post): test_phase23 + 28 diff = 0
LINES; 210 identical modulo memory addresses/tqdm rates, fingerprints
2.10o 5.28e-28 + 2.10s 1.72e-28 reproduced; 29 identical except the
documented [2.9c] TV wobble; 2mpi [2.12] byte-identical (paths=68,
swaps=4, xover=6, best x+cos(x)), only [2.10q] 113→117 = the documented
nondeterministic pair; test_phase42 fingerprint EXACT (post −64.534224,
nrmse 0.009240; identical trajectory lines, diffs = worker stdout
interleave). test_phaseFD ALL PASSED serial (175s) + --mpi (187s);
test_phase212_wire 27/27; test_phase44 21/21 (T3b/T7c steering powers
reproduce). Adversarial review (2 independent lenses over diff + C
output + repo sweep): correctness/bit-identity CLEAN, no new bug in the
diff; test-adequacy majors all addressed in the hardened test.
FOLLOW-UPS RECORDED (kernel lane, none blocking):
  (a) `logdet` finiteness guard in `_log_likelihood_at_sigma_arr`
      (defensively incomplete, provably unreachable-wrong today);
  (b) boundary validation of σ (len==len(y), σ>0, ndim==1) in
      Grove.__init__/Tree.set_sigma — today a wrong-length σ degrades to
      silent −inf posts (BIC) / permanent PR no-op;
  (c) PRE-EXISTING, now numerically relevant under arrays:
      `compute_posteriors(full=True)`/superficial-posterior path stores an
      (N,) array into root_posteriors (legacy path, corrupts state);
      `Grove.set_sigmas` updates trees but not Grove.sigma (PR desync —
      pass σ via ctor); spawn burn-in workers score σ-blind
      (profiled-σ at reference θ — extension of the documented 2.13(j)
      anomaly); rank-parallel burn-in rebuild drops σ (path already
      broken/unused); `Node.optimize_sigma` ignores use_gradients/
      method_str — array path is Powell-only BY DESIGN (the symbolic
      gradient differentiates the unweighted MSE; do not enable);
  (d) pxd hygiene: new cdef was inserted mid-class (vtab-slot shift is
      covered by the full rebuild; prefer appending in future); stale
      samplerC/cpython-38 .so files in-tree are dead, delete some day;
  (e) SA-e2 accept-sequence comparison is fp-portable only per-box
      (deterministic under this compiler/BLAS; documented).

### MR.2 TRANSPARENT ATOMS (Phase FX v1) — IMPLEMENTED 2026-07-29
Kernel lane. Enables the DISCOVERY arm of Phase MR (searchable arguments
inside frozen shapes: tanh(g(t)), log(1+d·e^{−h(t)})) and subsumes
compose-at-d>1 (arity-1 case) without reopening 'o'.

**What shipped.** `real_functionC.FlexAtomSpec(name, template,
slot_symbol='s')`: validated template (name must sympify to an
UndefinedFunction head — never a known function or a variable or the
param pattern; slot-purity: free symbols = slot ∪ placeholders; slots
sorted by `_slot_sort_key`; compiled once via lambdify over
`[slot] + slots`; picklable via `__reduce__` with re-validation). A flex
scaffold is a UNARY Node: `flex_spec` set, `left` = live slot subtree,
`right` = None, `theta` dict for internal values. Entry by SEEDING only:
`function_treeC.make_flex_node` (positional; validates live slot, no
nesting, k distinct legal names), Grove `node_list`, or the new `('F',
head, (names…), slot_skel)` wire tag (`tree_to_skel`/`node_from_skel`;
`_tree_to_wire` always ships flex trees as skel3b; θ travels in
name_values via the extended `_install_cached_theta`). The six 2026-07-15
design decisions resolved as: (1) identity = UNEXPANDED head
`W(c0,…,slot_expr)` (description injective, BIC-cache/Pólya-key safe);
(2) evaluation = numeric composition — `_flex_function` closure feeds the
slot's output into the compiled template, NaN-fills on any failure
(zero-likelihood semantics pinned by FX-b10); (3) frozen interior = never
a move target + full immortality: every outer move that would destroy/
replace/rebuild a scaffold or select slot internals force-rejects
(`_contains_flex` guards; `_flatten_nodes` masks slot descent); split MAY
wrap a scaffold and internal_hoist MAY unwrap (sibling guards allow death
only when no flex node is deleted); slots mutable BOTH ways via ONE new
move `flex_slot_mutation` (uniform flex pick × uniform inner move from
the 7-set on the slot wrapped as a Tree with the outer name pool
injected; correction = the inner move's (lqf, lqr) exactly);
(4) crossover force-rejects when either selected subtree contains flex;
(5) identifiability pruning DEFERRED to the application layer (template
authors' responsibility in v1); (6) selection cardinalities masked
everywhere (`_count_selectable` counts a scaffold as 1;
`_collect_ets`/param-leaf walks guarded); Pólya = ONE template cell then
the slot path continues in the child subtrie (registration invariance via
Forest pre-registration; auto_register fallback for standalone use).
Grove/Forest: `flex_atoms=` kwarg (OFF by default ⇒ bit-identical
legacy); move probs = legacy 8-vector ×0.8 + 0.2 appended (both
birth/death pair equalities preserved ⇒ Green term stays 0); stats key
`flex_slot_mutation` exists ONLY when the feature is on; internal θ
counts 1 each toward BIC k; PR routes flex trees through name_values
coordinates (mirrored predicates in `pr_coord_names` and
`_parameter_refresh`). Burn-in token extension NOT done (refused instead,
see below).

**v1 refusals (fail-fast, all tested):** adapt_moves+flex (logit map is
8-move), burn_in+flex (burn-in constant folding is not flex-aware and
would destroy/crash on scaffolds), run_mpi_spawn+flex (workers lack the
FlexAtomSpec registry; serial EMC fully supported), flex/opaque name
collisions.

**Adversarial review (2026-07-29, 3 inline agents: sampler math /
bit-identity / object lifecycle) + fixes applied:** (a) OUTER depth-cap
post-check added to `_flex_slot_mutation` — the inner caps bound the
slot-as-a-tree only, and unchecked slot growth drove the OUTER tree past
`_MAX_TREE_DEPTH`, where an accepted internal_hoist claims a finite
split-reverse that split's own cap zeroes (one-way probability leak at
the boundary); the post-check closes the only channel into over-cap flex
states (seeds must start in-support — legacy node_list convention,
documented in make_flex_node). (b) `flex_slot_mutation` on a flex-free
tree now returns (0, −inf) hard reject — the earlier (0,0) no-op was a
guaranteed accept that re-reinforced the incumbent's Pólya path on
~p_flex of draws in mixed groves (adaptation bias). Bit-identity lens:
CLEAN — every legacy-path hunk cleared (arange sizing argument-identical,
guards precede RNG draws and are unreachable on legacy trees, key sets/
wire bytes/pxd layouts intact). Lifecycle lens: no chain-corrupting
defect in v1 as shipped.

**Caveats ledger (recorded, deliberately NOT fixed — several would break
legacy bit-identity):** (i) legacy `_add_node` leaf selection uses
depth-biased `select_random_leaf` while its density charges uniform —
PRE-EXISTING legacy defect, now inherited into slots via the inner move;
fixing it changes legacy RNG streams ⇒ own certified lane. (ii)
`rename_prefix` exit-state is stale below depth 1 (pre-existing; live
callers heal via `_compact_param_labels`' `_post_order_update` —
exit-contract comment added). (iii) `_flex_slot_mutation` name-pool
injection relies on inner moves capturing the pool BEFORE their first
tree_update (invariant comment added at the injection site). (iv)
`fetch_fun`/constant-folding/`flag_constants` ancestors are not
flex-threaded (burn-in-only today — hence the burn_in refusal). (v)
Seeding flex trees WITHOUT `flex_atoms=` is a silent-degradation trap
(slots frozen, no slot move); standalone Groves with `flex_atoms` +
their own polya rely on auto_register or pre-registration. (vi)
`sample_construction_path` treats a registered template cell as terminal
(diagnostics only; zero kernel callers). (vii) Node/Grove pickle streams
widened by the new fields (no in-repo pickle path). (viii) O(n)
`_contains_flex`/`tree_has_flex` walks added to some legacy hot paths —
pure CPU, no RNG/bytes.

**Tests:** `test_phaseFX.py` — 71/71 PASS (spec validation/pickle;
composition exactness vs numpy; NaN policy; BIC k analytic pin under a
σ-array; deepcopy isolation; optimizer θ write-back; per-move immortality
guards incl. legal split-wrap/hoist-unwrap; slot-move invariance/motion/
name-discipline/determinism (pin BOTH gp_movesC and function_treeC RNGs)
+ outer-cap pin FX-e7; Pólya path/coverage/KeyError/auto-register; 'F'
wire round trip + missing-spec reject; Grove ctor refusals, 9-vector,
conditional stats key, 300-step immortality + same-seed determinism;
Forest shared-Pólya registration, serial smoke, spawn refusal;
end-to-end: y = 2·tanh(1.5x−1)+ε recovered through a seeded c0·tanh(s)
scaffold, NRMSE ≈ 0.011, scaffold intact). Harness lessons: expr_to_tree
canonicalizes names to the 'a' family (seed slots with canonical names);
Grove(node_list=…) needs compute_posteriors before run.

**Certification (2026-07-29, vs /tmp/sa_post σ-fix baseline, logs in WSL
/tmp/fx_cert2/ + Windows scratchpad backup):** all 12 suites exit 0.
Byte-identical outright: test_sigma_array (28/28), test_phase23,
test_phase28. Identical after normalizing memory addresses
(`id(...)=N` debug prints), tqdm throughput, async MPI print order, and
timing values: test_phase210, test_phase42, test_phaseFD,
test_phaseFD --mpi, test_phase212_wire (27/27), test_phase44 (21/21) —
every RNG-driven value, candidate expression and fingerprint EXACT
([2.10o] mse 5.28e-28, [2.10s] 1.72e-28, [4.2reg] post −64.534224 /
nrmse 0.009240 "fingerprint matches reference", [I.12.2] RECOVERED
nrmse 0.0092 best 269.205/(x·(137.7x − eˣcos x))). Only the two
DOCUMENTED wobbles moved: [2.9c] TV statistic (PASS both sides) and
[2.10q] polya_paths_replayed 117→113 with its companion wire-skip
messages (PASS). Why addresses moved when the σ-cert had them
byte-stable: FX adds transient guard-walk list allocations on legacy
paths, shifting heap layout — the id() debug prints in some suite logs
were never a real invariant (exactly the one legacy-path side effect the
bit-identity review predicted). RNG streams and numerics are unchanged
everywhere. OPS NOTE: suites that MPI-spawn (2mpi, 42, FD --mpi,
212_wire, 44) MUST be launched under the ENV's `mpiexec -n 1` (as
sa_cert.sh did) — a plain-python launch resolves the spawn through the
system /usr/bin/mpiexec and hangs on `-pmi_args`.

Note: Eq. 49/50-class shapes (affine arguments) remain reachable via
opaque atoms; FX scaffolds now make the ARGUMENTS searchable.

### MR.3 FINAL TODO — PREPROCESSING PIPELINE (NOT codebase-dependent)
NR merger-ringdown → regression-ready (t, y, σ) products per (simulation,
mode, channel): SXS acquisition, alignment at t22_peak, remnant/QNM
factorization, mode-mixing removal for (3,2)/(4,3), channel extraction
with late-time-flatness validation, per-point σ from resolution
differences (e^{+σ_I τ} amplification kept — that IS the honest
likelihood), Eq. 49/50 baseline fits, validation gates V1–V5. Pure
numpy/scipy + sxs/qnm — runs anywhere, parallelizable with kernel work.
HANDOFF PROMPT WRITTEN: `PROMPT_PREPROCESSING_PIPELINE.md` (2026-07-28)
— defines the data contract (float64 (n,) t/y/σ, metadata, manifest),
the validation gates, and the house rules for a fresh instance.

### MR.4 MULTI-DATASET JOINT EVIDENCE — DESIGN AGREED 2026-07-31
Spec: `PROMPT_DEV_MULTIDATASET_BIC.md` (written by the preprocessing
instance). Model (settled, not up for redesign): ONE shared skeleton tau,
per-dataset parameters theta_j over datasets D_j = (x_j, y_j, sigma_j),

    log post(tau) ∝ log pi_polya(tau) [− lambda*C(tau)]
                    + SUM_j [ log L_j(theta_hat_j) − (k/2) log n_j ]

Prior / tilt / moves / reinforcement UNCHANGED (they never see data).
Validity strictest-wins. Additive, opt-in via `datasets=`, OFF by default,
bit-identical when off. Feasibility assessed 2026-07-31: FEASIBLE, sized
like Phase FX; no detailed-balance surface (no move or density changes),
but it edits the hottest path (candidate scoring) and has a larger hidden
surface than the spec implies.

**Decisions agreed with the user (2026-07-31), before coding:**

* **D1 — parameter_refresh REFUSED under `datasets`.** PR proposes theta
  and accepts on the raw likelihood of ONE (y, sigma); refreshing only a
  reference theta would leave the chain's accepted state inconsistent
  with the reported joint score. BIC profiles theta anyway, so PR is a
  mixing aid, not a correctness requirement. Raise if `datasets` and
  `pr_prob > 0`. (A J*k-dimensional AMState is the v2 option.)
* **D2 — burn-in REFUSED under `datasets`.** Not one function: the ctor
  path (groveC `if self.burn_in:`) fans out into candidate scoring in
  five places (depth-1 enumeration + serial / MPI-parallel / MPI-spawn
  stage 2 + the public burn_in_combinations / burn_in_depth2 wrappers),
  ALL scoring against the single `self.data` / `self.domain`; two of them
  score inside worker processes that rebuild candidates from string
  tokens, so going joint means shipping J datasets into every worker and
  reworking the gather protocol (spawn-wire risk, cf. Phase 4.4). Silent
  reference-dataset selection is forbidden by the spec, so v1 raises,
  exactly as Phase FX does. Documented alternatives for the user:
  (i) seed with `node_list=` (incumbent ansatz + FX scaffolds — the
  intended workflow anyway); (ii) run single-dataset burn-in on one
  representative simulation and pass the winners as `node_list` (the same
  approximation, but performed visibly in the experiment script);
  (iii) `burn_in=False` and longer chains. Joint burn-in is a clean,
  strictly-additive v2 once the joint scorer exists.
* **D3 — per-dataset (x, sigma) travel as ARGUMENTS.** `_shortcut_bic`
  reads `self.sigma` off the Tree, so a naive loop would call
  `set_sigma_recursive` per dataset per candidate (O(nodes*J) of churn on
  the hot path, plus mid-scoring state mutation — the aliasing trap the
  FX lifecycle review flagged). Add an explicit-argument scoring entry;
  legacy method delegates to it.
* **D4 — combination is a REDUCTION POINT over the per-dataset score
  vector.** The scorer computes [S_1 … S_J] and then reduces.
  `combine='sum'` is the DEFAULT and the scientific claim (it is the
  joint evidence: the Laplace approximation of ∫ prod_j L_j(theta_j)
  dtheta_j with independent theta_j, scale-consistent, covered by the
  existing formalism). `'min'` and `'softmin'` (one beta knob, smoothly
  interpolating mean → min) ship as OPT-IN, OFF by default. The J fits
  dominate the cost, so the reduction is nearly free.
* **D5 — min-of-ALPHAS is REJECTED (invalid MCMC).** Taking the smallest
  acceptance ratio across datasets equals MH with the worst-case score
  difference (min_j min(1,e^{x_j}) = min(1,e^{min_j x_j})), and
  min_j [F_j(tau') − F_j(tau)] is NOT the difference of any function of
  tau, so no stationary distribution exists. Counterexample (checkable):
  three skeletons with per-dataset scores A=(0,0), B=(3,−1), C=(1,1)
  under symmetric proposals imply pi(B)/pi(A)=e^2 and pi(C)=pi(B), yet
  pi(C)/pi(A)=e^1 — inconsistent; probability circulates A→B→C→A. The
  VALID form of the same intent is the min of the SCORES (D4 `'min'`),
  which is a genuine energy.
* **D6 — `'min'` mode requires baseline subtraction.** Raw log L_j scales
  with n_j and 1/sigma_j^2, so an unnormalized min means "largest /
  tightest dataset", not "worst fit". Use per-dataset constants (the
  incumbent ansatz's score on that dataset): U = log pi + min_j [S_j(tau)
  − S_j(tau_incumbent)], still a valid energy (constants are
  tau-independent), scale-free, and literally the paper's question
  ("does it beat the incumbent on EVERY simulation?"). Under `'min'` the
  complexity term stays SUMMED (a min over the full BIC_j would charge
  complexity once instead of J times and bias toward complex forms).
* **D7 — tests:** spec's a–f, PLUS **(g) joint x transparent atoms**
  (one FX-scaffolded skeleton across two datasets, per-dataset theta for
  the flex internals) — that is the actual science configuration and it
  exercises the three-kinds-of-theta install path. Test (b) caveat:
  warm-start each dataset from its OWN previous fit, never from the
  neighbouring dataset's, or `datasets=[D,D]` passes trivially while
  violating "independently optimized".
* **D8 — spawn: `NotImplementedError` in v1** (precedent: raw-callable
  structural_prior, and flex_atoms).

**Representation note (lowers risk):** per-dataset theta = J
`name_values` dicts. `groveC._install_cached_theta` is already exactly
the "install theta_j into this tree" primitive and already covers all
three theta kinds (plain parameter leaves, opaque atom internals, FX
flex internals), so the joint feature composes with Phase FX for free.
BIC cache: keep description keying, store (joint_post, [nv_0 … nv_{J−1}]),
invalidate on the datasets-object identity instead of `_cache_y is not y`,
keep the penalty-exclusion invariant (< −1e90 never cached) PER dataset.

**First-hour check before plumbing:** n_j differ across datasets — verify
parameter/constant leaves broadcast from the PASSED x, not from the
construction-time `self.domain`; otherwise dataset 2 silently inherits
dataset 1's length.

**MR.4 IMPLEMENTATION LOG (2026-07-31, overnight run)**

*Iter 1 — broadcasting probe: PASS (no kernel change needed).* Probe
(scratchpad `md_probe.py`): a Tree built on X1 (n=60) evaluated/fitted
against a foreign X2 (n=41, different range): (P1) expression trees
compute the SAME function on the passed x, output sized (41,) — NB the
first run "failed" only because expr_to_tree canonicalizes parameter
ROLES (the SA-a lesson; probe made role-agnostic); (P2) parameter-leaf-
only trees size np.full output from the PASSED x, not the domain; (P3)
function leaves fine; (P4) `update_parameters(X2, y2)` fits on the
passed data (rmse 0.011 at noise 0.01) — the explicit-x optimizer entry
is joint-ready as-is; (P5) FX flex composition evaluates correctly on
foreign x; (P6, documented) `log_marginal_bic(y_foreign)` returns −inf
(reads self.domain; validity/shape gate) — confirms D3: the joint path
requires a NEW explicit-argument scoring entry and must never route
through the legacy y-only method. Files changed so far: NONE (probe is
scratchpad-only). Backups of all *.pyx/*.pxd/PLAN.md taken to scratchpad
`backup_md_20260731/` (22 files).

*Iter 2 — explicit-argument scoring entry + reduction point (WRITTEN,
build pending).* Files changed: `function_treeC.pyx` — new
`Tree._log_marginal_bic_x(x, y, optimize, use_gradients, method_str)`,
an exact mirror of `_log_marginal_bic` with the PASSED x substituted at
every self.domain site (optimizer x via the already-argument-driven
`_update_parameters*`, both is_valid gates, the three no-optimize scorer
calls; n comes from y inside `_shortcut_bic`); new cpdef
`Tree.score_dataset(x, y, sigma, optimize=True, use_gradients=False,
method_str='Powell')` -> `(post, dict(name_values))` — swaps the ONE
Tree-level `self.sigma` attribute around the call with try/finally
restore (D3: never the node-recursive setter), leaves fitted theta_j
installed for the caller to snapshot. `function_treeC.pxd` — both decls
appended at the END of the Tree block. `groveC.pyx` — module-level
`_md_dataset_penalty(k, n, sigma)` (mirrors _shortcut_bic's k vs k+1
branches) and `_reduce_joint(posts, penalties, combine, softmin_beta,
baselines)` implementing D4/D5/D6: 'sum' (default, joint evidence),
'min' and 'softmin' on the baseline-shifted FIT parts with the
complexity charge kept SUMMED; any non-finite S_j -> -inf; unknown mode
raises. libc imports extended (log, isfinite). `groveC.pxd` — two decls
appended at file end. Legacy scoring paths byte-untouched.

*Iter 2b — entry smoke: PASS.* score_dataset matches the analytic
per-point-sigma BIC to full precision on a foreign dataset, restores the
configured sigma, fits each dataset independently, and reproduces the
legacy log_marginal_bic EXACTLY at J=1 (same Powell trajectory) even
with a deliberately wrong configured sigma (scratchpad
`md_smoke_entry.py`).

*Iter 3 — Grove/Forest joint plumbing (WRITTEN, build pending).* Files
changed: `groveC.pyx` — (1) Grove ctor: `datasets=None, md_combine='sum',
md_softmin_beta=8.0, md_baselines=None` kwargs; validation block after
the pr_prob init (triples coerced to contiguous float64; per-dataset
x/y/sigma length checks; positional (x, y) REQUIRED to equal the
reference dataset datasets[0][:2]; self.sigma forced to datasets[0][2];
combine-mode/beta/baselines validation; burn_in refused (D2); pr_prob
forced 0.0 (D1)); (2) `_score_cand_joint(cand, optimize)` — strictest-
wins validity over every dataset's x, description-keyed cache with the
JOINT value shape (joint_post, [nv_0..nv_{J-1}]), independent per-
dataset fits from the candidate's incoming theta (D7b: start_nv
re-installed before each dataset), penalties via _md_dataset_penalty,
reduction via _reduce_joint, REFERENCE theta (dataset 0) left installed,
per-dataset penalty-exclusion on caching; (3) `_step` scoring block:
one `if self.datasets is not None:` branch routing to the joint scorer
with the legacy block wrapped in the else (behaviourally identical for
legacy Groves; <= -1e290 joint returns hard-reject like the legacy
validity gate); (4) `_score_post`: same route at the top (swaps/
crossovers/lazy fills are automatically joint); (5) `compute_posteriors`:
joint Groves score via _score_cand_joint regardless of `full`;
(6) `_step_parameter_refresh`: raises under datasets (D1) — ctor already
forces pr_prob=0; (7) `joint_thetas(tree)` cpdef introspection (cached
per-dataset theta list for hall-of-fame consumers); (8) cpdef wrappers
`reduce_joint` / `md_dataset_penalty` for tests and the post-hoc
cross-check script; (9) Forest ctor: same four kwargs, pass-through to
every Grove (rungs share one datasets list => one joint posterior),
`self.datasets` stashed, run_mpi_spawn refuses datasets (D8).
`groveC.pxd` — Grove attrs (datasets, md_combine, md_softmin_beta,
md_baselines) + Forest attr (datasets) appended at block ends; method
decls (_score_cand_joint, joint_thetas) appended at method-list end;
module decls (reduce_joint, md_dataset_penalty) at file end.

*Iter 3b — build note:* one Cython error fixed: an explicit `<str>` cast
on method_str at the cross-module score_dataset call trips the
unicode->str coercion rule (the documented atoms-registration trap);
the uncast object converts at the call boundary like the legacy
_log_marginal_bic call. Rebuild clean.

*Iter 4 — test_phaseMD.py (NEW FILE) + first runs.* Suite: MD-a J=1
exactness, MD-b duplicate-dataset identity, MD-c synthetic recovery +
distractor + short joint EMC, MD-d strictest-wins validity + no-cache,
MD-e cache shape/reference-theta install, MD-f sigma-array analytic pin
+ profiled k+1 pin, MD-g joint x FX scaffold (per-dataset internal
thetas; immortality under the joint kernel), MD-h reduction unit
(sum/min/softmin/baselines/beta limits/error paths + penalty helper),
MD-i refusals (burn_in, PR hand-re-enable, spawn, reference-equality,
bad mode/baselines/lengths) + Forest serial EMC smoke + min/softmin
end-to-end. First pass: 34/36 — both failures were TEST issues, kernel
clean: (1) the NORMALIZED softmin tends to min + log(J)/beta from above
(the 1/J term that makes beta->0 the mean); pin corrected to include
the exact offset. (2) DOCUMENTED policy divergence: joint
compute_posteriors POPULATES the BIC cache (deliberate — a hit saves J
fits) while legacy compute_posteriors bypasses it; exact J=1 trajectory
equivalence therefore holds under SYMMETRIC cache states (both groves
initial-scored through run()'s lazy fill, which caches on both paths) —
test restructured accordingly. Second run: 36/36 PASS.

*Iter 5 — adversarial review round 1 (lifecycle/aliasing lens) + fixes.*
Reviewer verdict: design solid (theta snapshots genuine copies; the
name-set invariant makes _install_cached_theta a complete value-only
restore; the sigma swap exception-safe; the feared in-place zero-fix on
shared dataset arrays does NOT exist on any scoring path; MD x FX clean
end to end). Four fixes applied (all joint-only paths): (1) HIGH —
`_init_trees` seeded RANDOM-INIT joint Groves with legacy single-dataset
posts (incommensurable with joint candidate scores; masked by every
node_list/compute_posteriors test) — joint branch added, scoring via
_score_cand_joint; (2) `joint_thetas` now DEEP-copies the per-dataset
dicts (a shallow copy handed callers live cache references — mutation
would silently change what later hits install); (3) every dataset x is
COPIED (never a view of the caller's array) and zero-SCRUBBED like the
legacy domain at ctor validation (else datasets j>0 were never scrubbed
— an exact zero, e.g. t=0 at the aligned peak, would make 1/x-bearing
candidates invalid on that dataset only — and the legacy in-place scrub
of the positional x could leak through a shared view into caller data);
also: empty-dataset refusal; (4) reference-theta re-install moved BEFORE
the reduction call (a raising reduction left the live tree on the last
dataset's theta). Recorded as DOCUMENTED CONTRACTS (no code change):
optimize=False lazy fills can cache unoptimized scores (legacy-parity
hazard, now paper-facing via joint_thetas — experiment layer should
compute_posteriors before swaps); `_md_dataset_penalty` ignores the
Phase 4.3 no-BIC ablation switch, so md_combine='min'/'softmin' plus
set_bic_penalty(False) mis-decomposes L_j (ablation is a diagnostic
lane; combine only with 'sum'); positional sigma is NOT required to
equal datasets[0][2] (self.sigma is force-overridden, by design). New
regressions MD-j1/j2/j3 added (ctor-path duplicate-dataset identity;
joint_thetas mutation immunity; zero-bearing dataset scrub + caller-
array non-mutation). Sampler-math and bit-identity reviews still
running. Also: the `_init_trees` joint scoring POPS its own cache entry
(the legacy ctor never writes the cache; without the pop, ctor-cached
topologies took cache-hit paths later while legacy re-fit, breaking
exact J=1 equivalence in low-order bits — MD-a pins it). MD suite
40/40. FX cross-check on the MD binary exposed a DETERMINISM FLAW in
FX-e1 (my own FX test): it pinned only gp_movesC's RNG while slot
rebuilds also draw from function_treeC's stream (the FX-e6 lesson), so
the finite-proposal count varied across processes (observed 14 vs the
>20 threshold on an identical kernel — gp_movesC/polya/real_functionC
are untouched by MD). Fixed by pinning both streams in FX-e1; FX back
to 71/71.

*Iter 6 — adversarial review round 2 (sampler-math lens) + fixes.*
Reviewer confirms the scoring core ("statistically sound where designed
hardest": the reduction is a genuine per-tree energy in all modes, the
L_j decomposition exact on both sigma branches, strictest-wins
deterministic, warm-start/cache behavior the same documented
approximation class as legacy — not a new detailed-balance defect) and
independently re-derives the _init_trees finding (already fixed, iter
5). NEW fixes applied: (1) MAJOR — size==0 flips self.burn_in on
BEFORE the datasets block, bypassing the D2 refusal (the guard tested
the ctor flag only); guard now tests `burn_in or self.burn_in`;
(2) MAJOR — the PUBLIC burn-in wrappers (burn_in_combinations,
burn_in_depth2) carried no datasets guard: one post-construction call
silently scored everything on ONE dataset and REPLACED the joint
population; both now raise; (3) MAJOR-conditional —
`_md_dataset_penalty` now mirrors the Phase 4.3 ablation switch (new
`function_treeC.get_bic_penalty()` cpdef getter + pxd): without it,
set_bic_penalty(False) + min/softmin decomposed L_j with a phantom P_j
and mis-weighted the argmin by dataset size; (4) unbaselined
min/softmin now emits a ctor warnings.warn (raw fit terms — the
largest/tightest dataset dominates the argmin) and the docstring pins
B_j = the incumbent's FIT term L_j^inc = S_j^inc + P_j^inc, NOT its
full BIC; (5) test defects fixed: MD-c2 was vacuous (`A and B or
True`), now a real per-dataset-theta-difference pin; MD-i9 upgraded
from finiteness to equality with a MANUAL reduce_joint over
independent score_dataset fits; new MD-i10/i11/i12 pin the size==0 and
wrapper refusals. PLAN CORRECTION (finding 7): the earlier
"invalidate on the datasets object identity" cache note is NOT how v1
shipped — datasets are CTOR-FROZEN and the joint cache is never
invalidated (correct for immutable data); the md_* config attributes
are public but MUST NOT be reassigned post-ctor (cache entries would
mix configurations) — hardening deferred to v2. Remaining recorded
non-blocking items (legacy-parity class): profiled-sigma junk landings
can pass the -1e90 cache exclusion (S_j ~ -116*n, finite) — inherited
from legacy, J multiplies exposure; optimize=False lazy-fill caching
(documented iter 5); reference-dataset warm-start privilege (same
first-fit-pinning class as legacy, documented). Re-run after fixes:
test_phaseMD 43/43, test_phaseFX 71/71.

**HANDOFF NOTE (experiment instance — how to run joint evidence).**
Invocation (serial EMC only in v1):
```python
d1 = np.load(".../products/v1/export/SXS_BBH_2325__l2m2__amp.npz")
# ... more products; each gives t/y/sigma float64 (n,)
DS = [(d["t"], d["y"], d["sigma"]) for d in loaded_products]
f = Forest(DS[0][0], DS[0][1], temps, size, depth, funcs,
           sigma=DS[0][2], operations=ops, seed=...,
           node_list=seeds_per_rung,        # incumbent + FX scaffolds
           flex_atoms=[...],                # composes with Phase FX
           datasets=DS)                     # <- activates joint scoring
f.compute_posteriors_all(DS[0][1])          # populate caches + thetas
f.run_all(y=DS[0][1], n) ; f.attempt_swaps(...) ; f.attempt_crossovers(...)
g = f.groves[0]; best = g.best_tree()
thetas_per_dataset = g.joint_thetas(best)   # [nv_0 .. nv_{J-1}] copies
```
Contract points: positional (x, y) MUST equal datasets[0][:2] (the
REFERENCE dataset; is_valid and the single-theta convention live
there); every dataset x is copied+zero-scrubbed at ctor (caller arrays
never mutated); y in run/compute calls is a dead API carrier on joint
Groves. Cost scales ~J x in optimizer time (the J fits dominate; a
cache hit saves all J). md_combine='min'/'softmin' need md_baselines =
the incumbent's FIT term L_j = S_j + P_j per dataset (compute via
Tree.score_dataset on the incumbent + md_dataset_penalty). Do NOT
reassign datasets/md_* attributes post-ctor (cache would mix configs).
min/softmin under set_bic_penalty(False) is safe since iter 6 (the
penalty helper mirrors the ablation switch), but 'sum' remains the
scientific claim; treat min/softmin as robustness reporting. The post-hoc joint-scoring CROSS-CHECK stays
mandatory: for hall-of-fame skeletons, refit externally per dataset
and compare against kernel joint BIC + joint_thetas — groveC exposes
`reduce_joint` / `md_dataset_penalty` / `Tree.score_dataset` so the
external script can reproduce the kernel arithmetic exactly (MD-i9
does precisely this in-suite). Expected smoke scale on tier-A (2,2)
amp products: incumbent-shaped candidates at chi^2/dof ~ O(100) per
dataset (REPORT §12 baseline) — an O(1) result for tanh-class forms
means the sigma plumbing is wrong, not the physics.
Spec deliverables explicitly DEFERRED (documented, not silent):
exp_common additive kwargs (P2) — the experiment scripts consume the
Forest API directly above; spawn wire (P3) — refused with
NotImplementedError per D8; real-data smoke on 2-3 tier-A products —
requires the preprocessing products at runtime, left for the
experiment lane (the synthetic MD-c/f/g pins cover the mechanism).

### MR.5 PINNED PARAMETERS IN OPAQUE ATOMS — Phase PP (2026-08-06)
Spec: `PROMPT_DEV_PINNED_PARAMS.md` (MR-SR science instance). Goal:
`AtomSpec(name, template, pinned={placeholder: float})` — pins are
physics constants the fit must not touch and the BIC must not charge;
a pinned atom is a BASIC FUNCTION whose complexity flows through the
Polya prior (its vocabulary cell), not through k.

**Assessment / design decisions (2026-08-06):**

* **D1 — SUBSTITUTION AT SPEC CONSTRUCTION (the load-bearing choice).**
  Pinned values are substituted into the template's sympy expression in
  `AtomSpec.__init__`; `slots`/`k` become the FREE placeholders only.
  Downstream, pins are numeric literals inside the compiled leaf
  function — structurally incapable of appearing in get_params /
  name_values, the optimizer flat vector, theta dicts, theta0, rename
  maps, PR coordinates, or `_install_cached_theta`. The alternative (a
  mask threaded through the optimizer and every theta consumer) was
  rejected: it would touch the hot path and every theta-handling site,
  each a bit-identity risk; substitution touches ONE constructor.
  Corollary: `instantiate_atom`, `rename_leaf_params`, `_make_atom_leaf`,
  skel 'A' wire, burn-in tokens, and the whole move set needed ZERO
  changes — they operate on the free-slot machinery, which no longer
  contains the pins.
* **D2 — IDENTITY = EXTENDED VOCABULARY SYMBOL.** `self.name` becomes
  `base[c1=v1,c2=v2]` (keys sorted by the natural slot order, values in
  shortest round-trip float repr — deterministic across processes).
  Since every registry in the kernel keys on `spec.name` (Polya cells,
  burn-in '@atom:' tokens, wire atom_specs maps, skel 'A' tags), pin
  identity propagates everywhere with no further code: different pin
  values = different Polya cells (the "basic function through the
  prior" requirement), different cache/fingerprint keys, and a wire
  rebuild against a wrong-pin registry REJECTS instead of silently
  substituting (the REPORT §16.2/16.4 lessons). `base_name` (new field)
  keeps the caller's name; `__reduce__` and every spec RE-DERIVATION
  site rebuilds from (base_name, template, pinned) — rebuilding from
  the extended name would double the suffix.
* **D3 — descriptions are faithful for free:** the instantiated leaf
  description is str(expr with instance names), which now renders pins
  as numeric literals (e.g. `a3*exp(-0.1731*x)*sin(0.0806*x + a4)`).
* **D4 — validation:** pinned keys must match the placeholder pattern
  AND be placeholders of THIS template; values finite floats; pinning
  EVERY placeholder is refused (k=0 — a fully-fixed shape is already
  expressible as a plain function STRING in `funcs`, zero new
  machinery); a pin that annihilates every registered variable (e.g.
  c1=0 in sin(c1*x)) is refused (the atom would be a constant map).
* **D5 — wire config compatibility:** spawn configs ship pinned specs
  as (base_name, template, pinned) TRIPLES; unpinned specs keep the
  historical (name, template) pairs BYTE-IDENTICALLY; mpi_worker
  accepts both lengths. `__reduce__` likewise keeps the legacy 2-tuple
  when unpinned.
* **D6 — `expr_to_tree(..., pin_numbers=False)`:** opt-in flag; True
  renders numeric literals as CONSTANT leaves (never fitted, k
  untouched) via a module-scope flag set/restored by the public wrapper
  (the ~15 recursive call sites stay untouched; same single-threaded
  module-global convention as the RNG handles). Intended for
  scoring/reference trees (tier-1 constrained scoring): constant leaves
  are OUTSIDE the move vocabulary — the standing seeds-in-support
  convention applies if such trees are evolved. Default False =
  bit-identical legacy.
* **D7 — MD interaction (spec v1 rule) holds BY CONSTRUCTION:** pins
  live in the one compiled map shared by all datasets; per-dataset
  theta_j covers free internals only. No refusal needed — per-dataset
  pinning is not expressible.

Files changed: `real_functionC.pyx` (AtomSpec ctor + __reduce__ +
__copy__/__deepcopy__), `real_functionC.pxd` (base_name/pinned +
RealFunctionC.pinned_const appended at block ends), `groveC.pyx`
(2 spec re-derivation sites -> base_name+pinned; 2 spawn config sites
-> conditional triples; flex/pinned base-name collision guard;
pin_numbers seed refusal), `mpi_worker.pyx` (2 config resolution sites
accept pairs and triples), `function_treeC.pyx` (`_PIN_NUMBERS` flag +
literal-branch constant path + expr_to_tree kwarg + Node.is_param
pinned_const check). NEW: `test_phasePP.py` (PP-a..PP-i covering the
brief's T-P1..T-P6 + interactions; the spawn check runs under `--mpi`
with the env mpiexec — the certification battery MUST include that leg).

**Adversarial review (2026-08-06, 2 inline agents: soundness+bit-identity
/ test-adequacy) — verdict + fixes.** Soundness verdict: "a pinned atom
really is indistinguishable from a basic function to the proposal
kernel, the BIC, and the Polya prior" — M+1+A cardinalities count it as
1 in both directions, spec.k never enters any proposal density
(topology-marginal chain), all four scorer k-paths draw from
name_values (pins structurally absent), Polya/burn-in/wire treat the
extended symbol as an opaque string, unpinned paths bit-identical by
inspection at every touched site. FIXES from the round: (1) CRITICAL —
`RealFunctionC.__copy__/__deepcopy__` did not forward the new
pinned_const flag (nor constant_value, which the ctor re-defaults to
1.0): a deepcopied pin_numbers constant re-classified as a fitted
parameter and evaluated as 1.0 — i.e. silent corruption on the FIRST
MCMC proposal; both fields now forwarded (provably identity on legacy:
the ctor is the only legacy assignment site). (2) MAJOR — a ZERO pin
could annihilate a FREE slot ("c0*sin(x)+c1*c2", c2=0 kills c1),
leaving spec.k overcounting: the skel wire ships the actual names while
instantiate_atom demands spec.k -> every MPI exchange of such trees
silently self-loops (the wire-starvation class); ctor now verifies
every free slot survives substitution. (3) MAJOR — the pin_numbers
scoring-only contract is now ENFORCED: Grove refuses node_list seeds
containing pinned constants (outside the move vocabulary => a
destructive move on one claims an unrealizable reverse density).
(4) pins substituted at 17 significant digits (tier-1 attachment values
survive the from_string round trip bit-exactly; was 15). (5) '['/']'
rejected in base names (extended-symbol spoofing). (6) _PIN_NUMBERS
set/restored on both wrapper branches. Test-adequacy fixes: PP-e2
non-vacuity guard (the ONLY mpi_worker coverage could pass with zero
atoms returned); PP-b5 reworked to assert desc + fixed-theta bitwise
equality ON THE FITTED TREE (was a near-tautology on fresh spec
instantiations); PP-d1 oracle upgraded from substring to exact
abs-literal set + atom_symbol identity (substring accepted sub-5th-digit
drift; NB sympy folds sign into literals — magnitudes are the pin
invariant); PP-d2b move-mix counters + PP-d3 direct rename_leaf_params
unit; PP-c6 same-key/wrong-spec registry rejected by the desc-fidelity
gate; PP-f4 analytic joint pin (k=2 per dataset under MD); PP-g4..g7
(free-param eval, deepcopy survival, Integer literals, Grove seed
refusal); PP-i flex x pinned (collision guard + combined-chain smoke).
Recorded caveats (documented, not fixed): pinned-atom identity prints
pins at repr precision in the SYMBOL but 17-digit Float in the EXPR
(consistent doubles, different strings); pin_numbers trees remain
scoring-only (set_trees/set_single_tree bypass the seed refusal — same
seeding-convention class as always); the stale samplerC .so is now
three struct-growths behind (rebuild before any samplerC use).
test_phasePP: 46/46 serial, 47/47 under --mpi.

**CERTIFICATION (2026-08-06 19:35, final PP binary, vs /tmp/sa_post
baseline; logs /tmp/pp_cert/ + run log /tmp/pp_cert_run2.log): ALL 15
RUNS EXIT 0.** test_phasePP 46/46 serial + 47/47 under `mpiexec -n 1
--mpi` (the spawn leg — config triple wire, worker spec rebuild, skel
round trips, pins intact in every returned tree); test_phaseMD 43/43;
test_phaseFX 71/71. Legacy byte-diff: RAW-IDENTICAL
test_sigma_array/23/28; CLEAN after the established normalization
210/FD/FD--mpi/212_wire(27/27)/44(21/21); residuals: the documented
[2.9c] TV line (PASS both sides) and ONE async worker-stderr line
("About to gather") shifted by one position in 2mpi and 42 (the
established flush-timing class; [2.10q] landed on 117 = baseline this
run). Both features' interaction suites green on the same binary.
Phase PP v1: COMPLETE.

**CERTIFICATION (2026-07-31 08:44, final MD binary, vs /tmp/sa_post
σ-fix baseline; logs /tmp/md_cert/ + run log /tmp/md_cert_run.log):**
ALL 13 SUITES EXIT 0. test_phaseMD 43/43; test_phaseFX 71/71 (after
the FX-e1 dual-RNG-pin determinism repair). Legacy byte-diff:
RAW-IDENTICAL test_sigma_array/23/28; CLEAN after the established
normalization (id() addresses, timings, async MPI print order)
210/FD/FD--mpi/212_wire(27/27)/44(21/21); the two DOCUMENTED wobbles
only: [2.9c] TV line (PASS both sides), [2.10q] 117→113 + companion
wire-skip message (PASS). test_phase42: fingerprint EXACT
(post −64.534224 / nrmse 0.009240 "matches reference", identical
recovered expression); residual = addresses + 15 spawned-worker stderr
warning lines present in baseline but absent this run — worker-stderr
forwarding into the parent log is flush/disconnect-timing dependent
(same async class as the [Sampler] disconnected reordering; all
NUMERIC content exact). Adversarial review: 3 lenses; lifecycle +
sampler-math verdicts in (fixes applied, iters 5-6); the bit-identity
agent had not returned at certification time — the raw byte-diff above
IS the empirical bit-identity check, and its verdict will be appended
if/when it lands. Phase MD v1: COMPLETE (P2/P3 deferred as noted).

**ADDENDUM (bit-identity verdict + final fix, 2026-07-31 ~09:00).**
The bit-identity agent returned after certification: "by inspection,
the legacy (datasets=None) path of the current working tree is
bit-identical to the pre-MD baseline" — it audited the LIVE TREE (it
correctly flagged that the review patch had gone stale during the fix
iterations and regenerated its own diff, so the verdict attaches to
the final state); `_step`/`_init_trees` wraps verified char-exact
modulo indent, the log/isfinite cimport-shadowing risk cleared (zero
bare uses pre-MD; all np.log/np.isfinite attribute access), all pxd
appends at true block ends. Its remaining items: (a) PROCESS —
test_phaseFX.py was edited this session (the FX-e1 dual-RNG pin, iter
5b) and test files were not in the pre-MD backup set, so the agent
could not diff it independently; the edit is exactly the 5 lines
documented in iter 5b + PLAN MR.2's harness-lesson note — verify
against the last commit during morning review (git is the user's
lane). (b) PRE-EXISTING: samplerC .so dates to 2025 and lags the
groveC struct growth (FX/σ/MD) — any samplerC import will warn;
rebuild it if that module is ever used. (c) LOW, FIXED NOW: the FX-era
burn-in refusal tested only the ctor flag, so flex_atoms + size==0
could still reach single-dataset burn-in (the exact loophole class MD
finding 2 closed) — guard extended to `burn_in or size <= 0` inside
the flex-only block (unreachable with flex_atoms=None). Post-fix:
rebuild + test_phaseMD 43/43 + test_phaseFX 71/71 re-run, plus a
byte-diff sentinel: test_sigma_array BYTE-IDENTICAL to the baseline on
the final binary (verified via plain diff after a shell-quoting
artifact made the first automated verdict unreliable) — the full
battery was NOT repeated for this one-line flex-block edit; rationale
recorded here. FINAL STATE 09:12: all suites green on the shipped
binary; working tree ready for morning review.

**Penalty-scaling observations (carry into the handoff note):**
  1. Three factors raise the BIC penalty, not one: k (= `len(name_values)`,
     which counts opaque-atom and FX-scaffold internals, +1 in the
     profiled-sigma branch), each n_j (through log), and J (through the
     sum). Under the joint sum a k-parameter form pays k/2*log n_j on
     EVERY dataset, so atoms/scaffolds get ~J times more expensive than
     in a single-dataset run.
  2. Likelihood AND penalty scale with J while the Pólya prior and the
     tilt do NOT, so the structural prior's relative influence falls ~1/J.
     If the tilt is meant to be worth a fixed amount of evidence, lambda
     should be scaled deliberately (DO NOT silently retune defaults;
     measure and document, same discipline as the ladder note).
  3. Practical regime: with the measured incumbent baseline at
     chi^2/dof ~ 111 on n ~ O(10^3) points, likelihood differences are
     O(10^4) per dataset while the penalty is O(10^1–10^2). The BIC
     complexity term is effectively negligible until candidates approach
     the noise floor; it only starts biting near chi^2/dof ~ 1.
  4. CAVEAT to document, not fix here: NR time series are oversampled, so
     adjacent residuals are correlated and n overstates the independent
     information. BIC assumes iid, so the fit term is over-weighted
     relative to log n, making model selection more permissive. Thinning
     to the effective resolution or a correlated-noise likelihood are the
     honest fixes; both are out of scope for this feature.

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
5. ~~(Phase 2.10) Prior-mass treatment for user atomic/opaque primitives.~~
   RESOLVED 2026-07-09/10: DP(c, G₀) with registration = carve-from-
   remainder (DESIGN_2_10_PRIMITIVES.md), implemented through Stage 3c.
6. (PAPER_PLAN §0) THE central formal decision: frame the adaptive Pólya
   prior as (A) fixed (urn reinforcement frozen before inference) or
   (B) self-interacting flagship. NB Stage 3c settles the VOCABULARY half
   of "fixed"; (A)/(B) concerns the γ_t reinforcement and is still open.
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
- ✅ Phase FD kernel (flexible multivariate domains, branch `flexible_domain`,
  2026-07-19) — registry + (d,n) domains + guards + test_phaseFD.py;
  compose-less (standing item). Details: flexible_domain_plan.md §9.


### MR-SR1 FIELD NOTES (2026-08-02, experiment side; preprocessing instance
### running the merger-ringdown SR campaign per MR_SR_REGIMES_PLAN.md)

Two kernel-relevant observations from the SR-1 smoke phase, logged per the
division of labor (observations only; nothing kernel-side was modified):

1. **FX: hand-BUILT slots of depth >= 2 evaluate incorrectly through the
   `make_flex_node` path.** Reproducer (scratchpad `sr1_flex_probe.sh`,
   session 2026-08-02; dataset = mr_preprocessing export
   SXS_BBH_2325__l2m2__amp, n=334, sigma-array):
   - template "c0*tanh(s) + c1", slot expr_to_tree("a0*x") with a0=0.0864:
     fits CORRECTLY (chi2/dof 8763 = the analytic optimum of that family;
     internals + slot params co-optimize fine, k=2 internals fine).
   - SAME template, slot expr_to_tree("a0*x + a1") (depth-2, op '+' inside
     the slot): evaluation is WRONG -- at hand-set theta the prediction
     collapses (pred[-1] = 0.0 where tanh+offset cannot be 0), and every
     fit lands on the CONSTANT-solution optimum (post -7.59e8). Also
     `Tree.get_params()` on such a tree lists ONLY the slot params, and the
     internals c1 stays frozen at 1.0 through score_dataset.
   Suspected surface: slot subtrees ripped from a donor Tree via
   `slot.get_root()` (aliasing/registration), only for nested-op slots;
   slots GROWN by flex_slot_mutation are the certified path and appear
   unaffected (Phase FX 71/71, MD-g). Triage when convenient; SR-1 works
   around it by seeding minimal depth-1 slots and letting the 9th move grow
   the argument.

2. **Legacy burn-in wire + sqrt(x): warning storms and out-of-vocabulary
   rebuilds.** On a '**'-free vocabulary containing sqrt(x), the spawn
   burn-in's legacy dual-format wire canonicalizes sqrt-bearing stage-2
   forms into parametrized powers (x**a0 etc.), emitting one
   "Reconstructed tree description does not match" warning per combo (the
   4.4-incident canonicalization class; burn-in producers deliberately
   stayed legacy). Cosmetically identical warnings appear (and pass) in
   test_phase42; at MR scale (53k combos, n=334) the warning storm through
   the 9p /mnt/c mount dominated wall time and the rebuilt x**a0 forms
   pollute the candidate set's provenance. SR-1 excludes sqrt(x) from all
   vocabularies (physics loses nothing; documented in
   mr_preprocessing/REPORT.md SS16). A burn-in force_skel analog is the
   clean long-term fix, but it is fingerprint-affecting -- kernel
   instance's call.

3. For calibration context: spawn health re-confirmed on this machine
   2026-08-02 via `mpiexec -n 1 test_phase42.py` -> PASS, fingerprint
   post -64.534224 / nrmse 0.009240 intact on the MD-era binaries.
