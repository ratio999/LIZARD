# Flexible-Domain Plan — multivariate input f: ℝᵈ → ℝ (branch `flexible_domain`)

**Goal**: let the machine scientist search expressions of d ≥ 1 independent variables
(f(x0, x1, …)), with user-defined variable names, while the observation vector `y`
stays 1-D and every existing 1-D workflow remains bit-identical.
**Status**: IMPLEMENTED 2026-07-19 (stages F0–F4 + `test_phaseFD.py`; compose-less
per user decision — compose is the §8.1 standing item). CERTIFIED: full WSL
rebuild; the FD-h seeded micro-run (2 Groves × 150 steps incl. 6-op legacy
vocabulary + Pólya-reinforced Grove + 2-rung Forest EMC with swaps/crossovers)
is **byte-identical** to the pre-edit baseline JSON; all five in-repo suites
(test_phase23/28/29, `mpiexec -n 1` test_phase2mpi, test_cython4) pass on the
edited kernel. Implementation notes in §9. Fingerprint suites (2.10o etc.)
live in the experiment-side checkout — run there before merging to master.

---

## §0 Design decisions (settled in discussion, 2026-07-19)

- **D1 — Runtime-general dimension, no compile-time templating.** Cython cannot
  parameterize `cdef class` by a type/dimension (no class templates; fused types
  don't cover attributes), and the hot path contains **no elementwise C loop over
  x** — storage is already `cdef object` (`Node.domain`/`Tree.domain`
  function_treeC.pxd:13/102, `Grove.x`/`Grove.domain` groveC.pxd:17/29) and the
  numerics are NumPy-vectorized lambdify calls + `np.sum` over 1-D `y`/ŷ
  (real_functionC.pyx:355–368). Per-dimension builds would optimize an operation
  that never occurs and would break single-source certification. REJECTED.
- **D2 — Canonical layout**: the domain travels as ONE C-contiguous float64 array
  of shape **(d, n)** — coordinate arrays as rows, points along the last axis.
  `d` is a runtime property (`X.shape[0]`), fixed **per Grove/Forest instance** at
  construction; one compiled binary serves every d. Rationale: `compiled(*X, *θ)`
  unpacks zero-copy contiguous row views straight into the lambdified callable;
  `n = X.shape[-1]` uniformly. Legacy 1-D `(n,)` input is NEVER reshaped — it is
  the d=1 fast path, byte-identical to today.
- **D3 — Variable names are user-defined**, held in a module-level registry in
  `real_functionC` (same pattern as `set_g_tau`/`set_rng` module setters; set by
  the Grove/Forest ctor). Defaults: feature OFF / 1-D x → `('x',)` (legacy
  descriptions must not change — they are BIC-cache keys and Pólya cell labels);
  feature ON with 2-D X and no names given → `('x0', 'x1', …, 'x{d-1}')`.
- **D4 — Registry-first symbol classification (mandatory, not cosmetic).**
  Today `x0` matches the parameter regex `^[a-zA-Z]\d+$`
  (real_functionC.pyx:170–181) → it would be classified as a parameter, and
  `is_param()` (real_functionC.pyx:715, "no variables ⇒ constant") would let the
  optimizer tune the independent variable as a constant. Classification becomes:
  symbol ∈ registry → variable; else current regexes. Complemented by
  generation-side guards (§2.2 item 6) so a parameter can never be minted with a
  variable's name.
- **D5 — `y` is reserved for observations** (never a variable name; registration
  rejects it). Multi-output is OUT OF SCOPE.
- **D6 — compose (`'o'`) is d=1-only.** It substitutes into THE argument
  (real_functionC.pyx:846–868); for d>1 its semantics are undefined. d>1 + `'o'`
  in `operations` (or `mode='Frequency'`) ⇒ hard error at construction, not a
  silent strip. Production configs already run safe-4 without `'o'`.
  **STATUS (user decision 2026-07-19): compose-at-d>1 is a STANDING ITEM for
  later work — see §8. The present implementation is compose-less at d>1.**
- **D7 — Additive contract** (project constitution, PLAN 2.10/tour rule 6): the
  feature is opt-in (new ctor argument), OFF BY DEFAULT, and with it off the
  kernel is bit-identical — same RNG stream, same descriptions, same
  fingerprints. Wherever possible the d=1 branch IS the literal existing code
  path, not a rewritten general case.
- **D8 — MPI wire**: d>1 trees ALWAYS ship as the Stage-3b skeleton payload
  `('skel3b', skeleton, description)`. Reason: the legacy sympy-expr payload is
  rebuilt by `_expr_to_tree`, which reconstructs function applications via
  compose nodes (function_treeC.pyx:1523–1527) — unavailable at d>1 by D6. The
  raw-expr payload remains reserved for d=1 atom-free trees (bit-identical
  bytes on the historical path).

---

## §1 Invariants every edit must respect

1. Output shape law: any prediction/fill produced for input X is **(n,)** with
   `n = X.shape[X.ndim-1]`. (The current `np.full_like(x, …)` idiom breaks this
   at d>1: it would return (d, n) and silently mis-broadcast into residuals.)
2. Signature relaxations only: `cnp.ndarray[cnp.float64_t, ndim=1]` → untyped
   `cnp.ndarray` (or `object`) on domain-carrying arguments. `y`, `sigma` arrays
   and θ vectors STAY `ndim=1` typed.
3. Never call a cpdef function with kwargs across a cimport boundary (tour
   invariant 1) — all new cross-module helpers positional.
4. Scoring semantics untouched: plain `np.sum` + C `isfinite` +
   `_MSE_PENALTY=1e100`; `nanmean`/`nansum` stay banned from scoring paths.
5. Hot-path budget (tour rule 12): d=1 call sequence unchanged; d>1 adds only a
   d-way tuple unpack per evaluate. Any helper on the evaluate path must be
   O(1) dispatch, not per-point Python work.
6. Deterministic naming: registry order = row order of X = lambdify argument
   order. Same seed ⇒ same trajectory at any d.
7. `polya_treeC` is not touched. Cells key on description strings /
   `vocab_symbol`; a d=2 vocabulary is just more cells.
8. Certification gate after every stage (§4): full WSL rebuild + all five
   in-repo suites green (`test_cython4`, `test_phase23`, `test_phase28`,
   `test_phase29`, `mpiexec -n 1 python test_phase2mpi.py`) + the FD-h
   bit-identity micro-run. Fingerprint suites (2.10o mse 5.28e-28, 2.10s/f
   1.72e-28, test_phase42 reference) live in the experiment-side checkout — to
   be run there before merge; deliberate trajectory changes are flagged, never
   silently re-recorded.

---

## §2 Edit inventory (by module, dependency order)

### 2.1 `real_functionC.pyx` / `.pxd` — numeric floor + registry (the bulk)

1. **Variable registry** (new, module-level): `_VARIABLES = ('x',)`,
   `_VAR_SET`, `_VAR_SYMS` (pre-built sympy symbols) + cpdef
   `set_variables(tuple names)` / `get_variables()` (mirrors
   `set_rng`/`set_g_tau`). Validation at set time: names sympifiable single
   symbols; not `'y'`; not matching an active parameter band prefix+digit
   pattern in a way that collides (see guard note in 2.2.6); not shadowing a
   sympy function used by the vocabulary (`sin`, `cos`, `exp`, `log`, `sqrt`);
   no duplicates. Module-global (like the RNGs): two Groves in one process
   share it — last ctor wins; document (same pre-existing property as
   `set_rng`).
2. **`classify_symbols`** (real_functionC.pyx:170): registry-first membership
   test before the two regexes. Also fixes the silent-drop of names matching
   neither regex.
3. **`from_string`** (real_functionC.pyx:752–802): replace the single
   `x_sym = sp.symbols('x')` (754) and the `str(s) != 'x'` param filter (766)
   with registry symbols / registry-exclusion; lambdify over
   `[*var_syms, *param_syms]` (768); the closure builds
   `arg_vals = [*_coords(input_x)] + param_values`, where `_coords` is a new
   tiny helper: 1-D input → `[x]` (identity, d=1 fast path), 2-D → row views.
   NaN-fill error paths (794, 800) → `(n,)`-shaped fill helper.
4. **`evaluate`** (real_functionC.pyx:226–260): parameter-leaf fills
   `np.full_like(x, …)` (252, 253) and the exception fill (259) → output-shape
   helper returning `(n,)` (invariant 1). theta0 merge logic unchanged.
5. **`is_defined`** (real_functionC.pyx:728–737): `full_like(x, False)` (737) →
   `(n,)` boolean; `is_valid` (200–223) needs no change (operates on ŷ).
6. **Domain plumbing**: `__init__` domain arg (51) relaxed; `_domain_fixer`
   (real_functionC.pyx:1057–1065) generalized (zero-shift over a raveled view;
   keep the exact 1-D loop on the legacy branch); `_domain_fixer` return type
   relaxed in `.pxd`.
7. **Arithmetic operators** `__add__/__sub__/__mul__/__truediv__`
   (808–844): only the inner closure signatures relax; logic untouched
   (operates on `(n,)` operands; `safe_div`'s `np.where` shape-agnostic).
8. **`compose`** (846–868): d>1 ⇒ raise (D6). `np.zeros_like(x)` (862) fixed
   anyway for hygiene on the d=1 path.
9. **d=1-only legacy**: `derivative` (740), `time_to_frequency` (193),
   `_arr_param_expr`/`_cfit`/`_cfunc` (dead vegas backend, 131–156) — explicit
   d=1 guards or leave untouched-dead; NOT generalized.
10. **Gradient machinery** (`_build_gradient_functions` 386–410,
    `_symbolic_gradient*`, `_finite_diff_gradient`, `grad`/`symb_grad`,
    `_optimize_with_grad`): effectively disabled on kernel paths
    (`compute_grad=False` at every tree-building `from_string` call; Powell
    default). Guard d=1, defer generalization (cross-ref PLAN 2.4 "fake
    gradient" + the Jacobian-nansum residual-risk note).
11. **`optimize` / `optimize_sigma` / `objective(_sigma)`** (600–696): x_data
    arg relaxations only; the guarded-minimize try/except structure (the
    2026-07-15 Powell fix) is NOT touched.
12. **BIC/fitness family** (`fitness` 355, `posterior_fitness` 337,
    `fitness_sigma` 370, `_log_marginal_bic` 265, `_log_sigma_bic` 277,
    `_log_sigma_array_bic` 287): x arg relaxations; the math reads
    `N = y.shape[0]` already — zero numeric change.
13. **`AtomSpec`** (926–953): validation generalizes `s_str == 'x'` (937) to
    "s ∈ registry"; "must depend on x" (944) → "must depend on ≥1 registered
    variable"; templates are written against the names registered at
    construction time. **`instantiate_atom`** (969–1014): domain arg relax;
    fresh-name check (997) additionally rejects registered variable names.
14. `.pxd`: mirror every relaxed signature; add the two registry cpdefs.

### 2.2 `function_treeC.pyx` / `.pxd`

1. **`Node.fetch_fun` / `update_func` compose dispatch** ('o' at 277, 433):
   reachable only if `'o'` ∈ operations — Grove ctor already rejects that at
   d>1 (D6); add a defensive error message here (cheap, cold path).
2. **`Tree.__init__`** x arg (575) + **optimize chain** (`optimize` 850,
   `_update_parameters` 927, `_update_parameters_and_fitness` 964,
   `update_parameters` 999, Node.`optimize`/`optimize_sigma` 295/302):
   x_data relaxations, pass-through unchanged.
3. **`_shortcut_bic`**: mse/N-based, no x — verify-only, no edit.
4. **`build_tree`** (1181–1305): x arg relax; the leaf `from_string(func_str,
   x)` calls become registry-aware automatically; operations validation stays
   as threaded by 2.13(f).
5. **`_expr_to_tree`** (1356+): the hardcoded variable dispatch
   `if symbol_str == 'x'` (1390) → registry membership; the function-application
   branch that composes (1523–1527) raises `NotImplementedError` at d>1 (D8
   reroutes those trees through the skeleton wire). Number/params bookkeeping
   (1376–1399) unchanged.
6. **Naming guards** (D4 complement): `new_par` (1335–1354) — derived prefix
   set must exclude registered variable prefixes even if `existing_params` is
   poisoned or `user_choice` names a variable prefix (cheap set check);
   `parameter_parser` (1307, band `'a'` hardcoded) — assert `'a'` ∉ variable
   prefixes at registry-set time instead (validation in 2.1.1);
   `rename_prefix`/`rename_prefix_downwards`/`_compact_param_labels` — same
   guard on their prefix arguments ('a'/'b'/'c' bands today).
7. **Skeleton wire** `tree_to_skel`/`node_from_skel` (domain arg at 1148):
   relax; leaves rebuild via `from_string` → registry-aware for free.
8. `.pxd`: mirror relaxations (`build_tree`, `_expr_to_tree`,
   `node_from_skel`, Tree/Node optimize chain).

### 2.3 `gp_movesC.pyx` / `.pxd` — no structural change (verify pass)

- Move machinery counts cardinalities (M+1+A, O) and manipulates tree
  structure; `domain` flows as `object` through `_make_atom_leaf`,
  `_build_subtree_with_params`, `_draw_leaf_et`. **Zero density changes**: a
  d=2 vocabulary is just larger M.
- Only touch points: the three DEAD float32-typed module functions
  (`size_fair_mutation` 1609, `replace_with_subtree` 1621,
  `generate_subtree_with_size` 2213) — leave as-is (Phase 3 dead-code list);
  `parameter_refresh`/AMState are y/θ-only — untouched.
- Deliverable of this stage is a written verify note in this file, not edits.

### 2.4 `polya_treeC` — explicitly untouched (verification only)

- Cells key on leaf description strings / `vocab_symbol`; `'x0'`, `'sin(x1)'`
  are simply new cells. ψ_const unification (`unify_params`) concerns
  parameter leaves only — with D4, a variable leaf is never `is_param()` and
  never maps to `PARAM_CELL_TOKEN`. `freeze_vocab` closes over whatever funcs
  list the Forest registers — dimension-agnostic. Covered by test FD-f.

### 2.5 `groveC.pyx` / `.pxd`

1. **`Grove.__init__`** (145–214): x arg relax (146); DELETE/branch the 1-D
   memoryview coercion `cdef cnp.float64_t[::1] x_view = x` (175); new
   `variables=None` ctor arg → resolves registry per D3 and calls
   `real_functionC.set_variables(...)` (exactly where `set_rng`/`set_g_tau`
   are already called); construction-time validation: X shape ↔ len(registry);
   `'o'` ∈ operations ⇒ error at d>1 — NB the ctor DEFAULT ops list is
   `["+", "-", "*", "/", "**", "o"]` (180), so at d>1 with operations=None the
   default must be rejected/replaced explicitly (decision §6-Q2);
   vocabulary check: every funcs string's free symbols ⊆ registry ∪ param
   pattern (catches `sin(x2)` typos at construction, not mid-chain).
2. `_domain_fixer` call site (208) — flows through the generalized fixer.
3. **Sampler core** (`_choose_move`, `_propose`, `_step`,
   `_step_pair_crossover`, `_score_post`, `run`, caches): y-typed only —
   verify-only, no edits. BIC cache keys (descriptions) unchanged mechanics.
4. **Burn-in machinery**: `_burn_in_primitives` token list, depth-1/stage-2
   enumeration, `_combine_depth1(_trees)` — string/vocabulary-driven,
   dimension-agnostic; name bands 'a'/'b'/'c' interact with guards (2.2.6)
   only. `_burn_in_stage2_parallel` (description-string rebuild path) gains a
   d>1 `NotImplementedError` guard exactly like its existing atoms guard
   (serial and spawn paths are the supported ones).
5. **Wire** `_tree_to_wire`/`_tree_from_wire` (cpdef): payload selection
   condition extends from "atom-bearing" to "atom-bearing OR d>1" (D8);
   domain arg relax; rebuild path sets nothing itself (worker owns registry).
6. **`Forest.__init__`** (2367+): x relax (2368); `variables=` arg forwarded
   to every Grove + stored for the MPI config; freeze_vocab pre-registration
   loop unchanged (registers whatever funcs strings it is given).
7. **`run_mpi_spawn`**: startup config dict gains `('variables', tuple)` and
   ships the (d, n) domain; masters'/workers' rebuild fidelity check
   unchanged (REJECT-the-exchange convention on failure).
8. `.pxd`: ctor-adjacent signatures + any relaxed cpdef.

### 2.6 `mpi_worker.pyx` / `.pxd`

1. Config unpack sites: `worker_main` (`config['domain']` typed pull at 71)
   and `sampler_main` (361–383): relax domain typing; **call
   `real_functionC.set_variables(config['variables'])` FIRST on every spawned
   rank** (before any tree rebuild — replica-identical classification is what
   keeps descriptions and Pólya path indices replica-stable).
2. `_helper` (23, expr→Tree via `_expr_to_tree`) and `_evaluate_tree` (283):
   domain relax; d>1 candidates arrive as skeleton payloads (D8) so `_helper`
   never sees a d>1 raw expr — add the defensive error anyway.
3. Combiner name-band logic ('c'→'b' renaming) untouched (guarded by 2.2.6).

### 2.7 Out of scope / untouched

`setup.py` (no new modules — helpers live inside `real_functionC`),
`loss_function` (compiled but unimported), `samplerC`/`num_wrapC` (dead,
not built), `numerical_backend/`, `lasso.py`/`polya_*.py` prototypes, the
experiment layer (exp_*/results — test-side instance's property).
NOT included: per-tree domains, masking/valid_frac semantics, multi-output,
gradient generalization, compose-at-d>1, flexible-atom scaffolds (PLAN 2.11)
— interaction with variable slots noted there if/when both exist.

---

## §3 API surface after the change (sketch, no code)

- `Forest(x, y, temperatures, …, variables=None)` / `Grove(x, y, …,
  variables=None)`:
  * `variables=None` + 1-D x → legacy: registry `('x',)`, everything
    bit-identical (THE certified path).
  * `variables=None` + 2-D X (d, n) → registry `('x0', …, 'x{d-1}')`.
  * `variables=('f', 'r')` + X (2, n) → user names, row i ↔ name i.
  * Mismatched len(variables) vs X.shape[0], or (n, d)-looking input → hard
    error with a transpose hint (§6-Q1).
- Vocabulary: funcs strings written over the variable names
  (`["x0", "x1", "sin(x0)", "sin(x1)", …]`); atoms templates likewise
  (`AtomSpec("Asin", "c0*sin(c1*x1)")`).
- `real_functionC.set_variables/get_variables` public for tests/tools; not
  meant to be called mid-run (registry changes invalidate every compiled
  leaf closure — document).

---

## §4 Implementation order (each stage ends at the §1.8 gate)

- **F0 — Registry + classification + guards** (real_functionC 2.1.1–2, 2.1.13
  validation half; function_treeC 2.2.6). Feature dormant; default `('x',)`.
  Risk: touches `classify_symbols` on the hot construction path — the
  bit-identity micro-run (FD-h) is the gate.
- **F1 — Numeric floor** (real_functionC 2.1.3–12, 2.1.14): `_coords`/output
  helpers, from_string/evaluate generalization, signature relaxations,
  `_domain_fixer`. d=2 works for a hand-built RealFunctionC.
- **F2 — Tree layer** (function_treeC 2.2.1–5, 2.2.7–8): build_tree/optimize
  chain threading, `_expr_to_tree` registry dispatch + d>1 compose guard.
  d=2 works for hand-built Trees + all moves (gp_movesC verify note 2.3).
- **F3 — Grove/Forest** (2.5): ctor args, validation, serial d=2 end-to-end
  (burn-in serial + EMC loop). Tests FD-a…FD-g runnable.
- **F4 — MPI** (2.5.5/7 + 2.6): wire condition, config variables, worker
  registry set; spawn burn-in + run_mpi_spawn at d=2. Test FD-i.
- **F5 — Docs + closure**: PLAN.md gets the coordination entry (one section,
  cross-ref this file); tour-prompt invariants list gains the registry rule;
  fingerprint certification on the experiment-side checkout; hand the new
  test file over to the test-side instance per the division of labor.

---

## §5 Test plan — `test_phaseFD.py` (plain script, test_phase23 conventions)

**FD-a Registry & classification (unit).** set/get round-trip; default
`('x',)`; `x0` classifies variable iff registered, param otherwise; `a0`
always param; mixed `a0*x0 + sin(x1)` → params {a0}, vars {x0, x1};
**the catastrophe test**: leaf `x0` has `is_param() == False` when registered
(today it would be True — this single assert certifies D4); registration
rejects `'y'`, `'sin'`, duplicates, unparseables.

**FD-b Naming guards (unit).** `new_par` never returns a registered-variable
name: poisoned `existing_params` containing `x3`, and `user_choice='x'`, both
still yield an 'a'/'b'/'c'-band name; `instantiate_atom` rejects a fresh name
equal to a variable; band renames (`rename_prefix`) refuse variable prefixes.

**FD-c Shape/evaluate (unit).** d=1: outputs `(n,)` and VALUES bit-identical
to the pre-change build on a fixed battery (exprs × grids; hash comparison —
this is the numeric half of the additive contract). d=2 on X (2, n): leaf
`x0` returns row 0, `x1` row 1; param leaf returns `(n,)` fill (kills the
`full_like` bug class); `sin(x0)*x1 + a0` matches a NumPy reference; NaN
error paths return `(n,)`; non-contiguous and (n, d)-transposed inputs
produce the defined error; `is_valid`/`is_defined` finite-ness on d=2.

**FD-d BIC scorer at d=2 (unit, mirrors 3.3(d)).** y = a·x0 + b·x1 + ε:
`log_marginal_bic` finite; profiled-σ (k+1) vs fixed-σ vs σ-array branches
all finite and correctly ordered; complexity penalty monotone in k.
**NaN-leak regression at d=2**: a tree non-finite on part of the grid (e.g.
`log(x0)` with x0 crossing ≤0) scores at penalty scale — never better than a
finite competitor; penalty-scale post NOT cached (cache-exclusion property).

**FD-e Moves & densities at d=2 (mirrors 3.3(c)).** All 8 moves propose/accept
on a `('x0','x1')` vocabulary; add_node/split density reconstruction from
(K, M, O, d) with the enlarged M; ETR exact-density from a single-leaf start
(both arities); off-support ⇒ `lqr == -inf` unchanged; determinism: two
Groves, same seed, d=2 → identical descriptions + move_stats (2.5(h) analog).

**FD-f Pólya at d=2.** extract_path/log_prior finite over cells
{x0, x1, sin(x0), …}; update_path moves the prior; freeze_vocab: unknown
symbol raises the guarded KeyError; ψ_const α-equivalence (2.10(r) analog)
still exact with variables present (renaming a-band ↔ b-band trees).

**FD-g Serial end-to-end recovery at d=2 (the success criterion).** Seeded
single-rung (or short-ladder) run, safe-4 ops, σ known, n≈100: recover
y = 2.5·x0 + sin(x1) (and/or x0·x1 + c) to NRMSE < 0.05 on a held-out grid
within a fixed budget; best tree uses only registered variables; finite
posts throughout. Plus an irrelevant-variable control: data depending on x0
only — run completes healthily (observational assert on x1 usage, not a
hard gate).

**FD-h Legacy bit-identity (THE additive-contract gate, run at EVERY stage).**
Feature off: (i) the five in-repo suites green
(`test_cython4`, `test_phase23`, `test_phase28`, `test_phase29`,
`mpiexec -n 1 python test_phase2mpi.py`); (ii) micro-run: fixed seed,
Grove(300 steps) pre/post build → identical `tree_get_description()` per
slot, identical `move_stats()`, `root_posteriors` within 1e-9 (2.5(h) form);
(iii) descriptions still print `'x'` (cache keys / Pólya cells unchanged).
Fingerprint suites (2.10o 5.28e-28, 2.10s/f 1.72e-28, `mpiexec -n 1
test_phase42.py` vs existing reference) run on the experiment-side checkout
before merge; any deliberate deviation is flagged to the test-side instance,
never re-recorded silently.

**FD-i MPI at d=2 (mirrors 2.10(p)/(q) + 2mpi).** (i) Wire round-trip unit:
d=2 tree (atom-free AND atom-bearing) → skel3b payload → rebuild with exact
shape, description, name_values; d=1 atom-free still ships the RAW EXPR
payload (bytes compared — D8's bit-identical clause); missing-registry
rebuild → REJECT convention. (ii) Live 2-rung `run_mpi_spawn` short run at
d=2: valid trees, finite posts, replay fingerprint
`t_after == t_before + paths_replayed`, ≥0 accepted swaps; spawn burn-in
d=2 sized above the ≥50-items-per-worker progress-loop limitation.
(iii) Worker registry: a planted d=2 tree survives a worker round-trip with
identical description (replica-stable classification).

**FD-j Performance guard.** Wall-clock of the FD-h micro-run (feature off)
within the environmental band of the pre-change build — the tour-rule-12
argument in numbers; d=2 evaluate micro-benchmark reported (informational,
no gate).

**FD-k Atoms at d≥2 (unit).** AtomSpec over a registered variable
(`c0*sin(c1*x1)`) constructs, instantiates, pickles (`__reduce__`), births
via point_mutation/ETR with A=1 cardinalities, and round-trips the skeleton
wire. Full atom-recovery experiments stay experiment-side.

---

## §6 Open decisions (need user input before the flagged stage)

- **Q1 — ADOPTED 2026-07-19 (implementation green-lit on recommendation):
  strict (d, n) only**, with a clear error naming the expected shape and a
  transpose hint. Rationale recorded: silent transposes and square-ish data
  are how shape bugs hide; d=n squares are undetectable.
- **Q2 — ADOPTED 2026-07-19: explicit `operations` required at d>1** (the
  legacy default contains `'o'`; mirrors the freeze_vocab explicit-over-
  implicit style). `'o'` present in a user-passed list at d>1 ⇒ hard error.
- **Q3 (before F3)**: variable count cap / burn-in budget note — stage-2
  enumeration grows ~O(M²) and M grows with d (funcs per variable); no code
  change planned, but the experiment protocol should state expected burn-in
  cost at d=2 (and the spawn worker chunking floor).
- **Q4 (before F4)**: keep the module-global registry (RNG precedent, one
  registry per process) or move to per-instance threading through every call
  chain? Global is the plan (zero signature churn beyond domains); flag if
  multi-Forest-different-d in ONE process is ever a real use case.

---

## §7 Risk register

- **R1 — silent shape corruption** (the `full_like` class): closed by
  invariant 1 + FD-c; the dangerous failure is (d,n)-shaped output
  broadcasting "successfully" — hence value tests, not just shape tests.
- **R2 — variable-as-constant misclassification** (the `is_param()` trap):
  closed by D4 + the FD-a catastrophe assert.
- **R3 — description drift at d=1** (cache keys / Pólya cells / fingerprints):
  closed by D3's `('x',)` default + FD-h(iii).
- **R4 — RNG stream desync** (an extra draw anywhere desyncs everything —
  tour invariant 8): no new RNG draws anywhere in this plan; FD-e/FD-h
  determinism asserts are the tripwire.
- **R5 — wire/replica divergence at d>1** (worker classifies differently than
  master): closed by config-shipped registry set FIRST on every rank (2.6.1)
  + FD-i(iii).
- **R6 — cpdef kwargs trap** on new/changed cross-module calls: §1.3;
  reviewed per edit.
- **R7 — hot-path regression**: helpers on evaluate are O(1) dispatch; FD-j
  measures; the NaN-fix bar ("the fix made the hot path faster") is the
  reference standard.

---

## §8 Standing items (deferred, design recorded so the reasoning is not lost)

### 8.1 compose at d>1 (deferred by user decision 2026-07-19)

Compose is NOT impossible at d>1 — it is underdetermined as a single binary
op, and each way of fixing that is its own project:

- **The core problem**: `f ∘ g` substitutes g into THE variable
  (real_functionC.pyx:849 `expr.subs(x_sym, other.expr)`); with
  f = f(x0, x1) there are d candidate slots. Numerical evaluation is easy
  for any GIVEN slot (replace row i of X with g's (n,) output); choosing i
  is the design decision.
- **Option A — d slot operators ∘₀…∘_{d−1}**: well-defined; grows the op
  vocabulary with d (touches move-density cardinality O and Pólya op
  cells — mechanical). FATAL-ISH HAZARD: when the left subtree lacks x_i,
  ∘ᵢ is a function-level identity but not a structural one — a hidden
  g-subtree with parameters in name_values that str(expr) does not show.
  That breaks description-keyed identity (BIC cache collisions between
  trees of different k; Pólya path vs description disagreement). At d=1
  this class CANNOT occur: any x-free left is `is_param()` and compose is
  special-cased to multiplication (real_functionC.pyx:850–853) — a
  special case that does not generalize.
- **Option B — single op, one-distinct-variable-left convention** (the
  recommended candidate): compose legal ONLY when the left subtree uses
  exactly one distinct variable; substitute into that one. Deterministic,
  no slot choice, no hidden-subtree aliasing. Cost: proposal support
  becomes state-dependent (validity depends on the left subtree's variable
  set) → force-reject (lqr = −inf) convention + its own on/off-support
  density argument, in the style of the depth caps. A Phase-2.1-grade
  densities job with tests like 3.3(c).
- **Option C — diagonal convention** f(g, g, …): well-defined, one op, but
  collapses the multivariate structure; semantically poor. Not recommended.
- **Supporting facts**: tuned production configs already exclude `'o'`
  (safe-4 ladder, CBC/Feynman vocabularies); symbolic substitution is the
  documented sympy blow-up operation (build_tree hang class, ~2^(2^(h−1)));
  the deferred flexible-atoms design (PLAN 2.11) is the intended mechanism
  for composition-like structure in the time-domain CBC case. Adding ∘ᵢ or
  Option B later is purely additive (new ops in `operations`, off by
  default) — deferring forecloses nothing.
- **Wire corollary while deferred**: `_expr_to_tree` rebuilds function
  applications AS compositions (function_treeC.pyx:1523–1527), so d>1
  trees must never ride the raw-expr wire — hence D8 (skeleton payload
  always at d>1) and the NotImplementedError guard in `_expr_to_tree`.

### 8.2 Other standing items inherited from the discussion

- Gradient-path generalization (multi-variable lambdified gradients) —
  currently d=1-guarded; finite-difference fallback is dimension-agnostic
  (perturbs θ, not x). Cross-ref PLAN 2.4.
- `mode='Frequency'` / `time_to_frequency`, `derivative`, vegas backend
  (`_cfit`/`_cfunc`) — legacy d=1 utilities, not generalized.
- Flexible atoms (PLAN 2.11) × variables interaction: slot subtrees over a
  chosen variable subset — note kept with the 2.11 sketch.

---

## §9 Implementation notes (2026-07-19, what actually landed)

Edits by module (all on branch `flexible_domain`; every touched signature
mirrored in its `.pxd`):

- **real_functionC**: module registry `_VARIABLES/_VAR_SET/_VAR_SYMS` +
  cpdef `set_variables/get_variables/n_variables/is_variable_name`
  (registration rejects `'y'`, the a/b/c parameter bands, duplicates, and
  anything that does not sympify to a plain Symbol); registry-first
  `classify_symbols`; `_flat_fill` output-shape helper replacing the three
  `full_like` sites in `evaluate` (+ the `is_defined` mask); `from_string`
  compiled over `[*registry_syms, *param_syms]` with TWO closures — the
  d=1 branch is the historical closure verbatim, the d>1 branch unpacks
  the (d, n) coordinate rows; `compose` carries the single innermost d>1
  gate (every 'o' dispatch in the codebase lands there); gradients return
  `{}` at d>1 (finite-diff fallback + Powell); AtomSpec classifies
  registry-first; `instantiate_atom` rejects variable-named fresh slots;
  `_domain_fixer` scrubs zeros in place for (n,) AND (d, n) (shape
  preserved, C-contiguity required at d>1); domain-carrying signatures
  relaxed to untyped `cnp.ndarray`.
- **function_treeC**: optimize chain + Tree/`build_tree`/`node_from_skel`/
  `_expr_to_tree` signature relaxations; `_expr_to_tree` gains the d>1
  NotImplementedError entry guard (D8) + registry-aware Symbol dispatch +
  registry-built primitive key (`'sin(x)'` legacy-identical); `new_par`
  guarded by `_prefix_mints_variable` (prefix filter is a no-op under the
  legacy registry — RNG stream untouched) + a last-resort index bump.
- **groveC**: Grove/Forest ctors gain trailing `variables=None`; registry
  resolution + strict-(d, n) validation + explicit-operations rule at d>1
  + flexible-mode-only vocabulary validation and AtomSpec re-derivation;
  the unused 1-D memoryview coercion of x (old line 175) was REMOVED;
  legacy runs stay on the `variables=None` path end-to-end (Forest passes
  `None` to its Groves unless flexible mode is on); `_tree_to_wire` ships
  the skeleton payload for atom-bearing OR d>1 trees; the burn-in spawn
  config and the `_ctor` stash carry `'variables'`;
  `_burn_in_stage2_parallel` (description-string rebuild) gains the d>1
  guard beside its atoms guard.
- **mpi_worker**: both entry points install the shipped registry FIRST
  (before any wire rebuild / Grove construction); `_evaluate_tree`
  returns skeleton wires for atom-bearing OR d>1 candidates; the worker
  Grove receives the ctor registry (legacy `('x',)` maps back to
  `variables=None` for bit-identity); domain typings relaxed.
- **polya_treeC / gp_movesC**: ZERO edits, as planned — verified
  pass-through by the touch-point sweep (44-finding gp_movesC report:
  domain always flows as `object`; densities count cardinalities only).

Adversarial review (2026-07-19, 5 lenses × 2-skeptic verification, 35
agents): three confirmed defects, ALL FIXED + regression-tested:
- **is_defined except-path OOB read**: `x.shape[-1]` on a buffer-typed
  `cnp.ndarray` compiles to raw C pointer indexing
  (`PyArray_DIMS(x)[-1]`) — an out-of-bounds heap read producing a
  garbage-size mask (reproduced live by a verifier). Fixed with
  `x.shape[x.ndim - 1]`; regression assert in FD-c. (`_flat_fill` was
  verified safe — its parameter is object-typed.)
- **AtomSpec unpickle-before-registry in spawned workers**: templates
  used to cross the MPI config as pickled AtomSpec objects, which
  re-validate via `__init__` DURING the bcast — before the worker
  installs the shipped registry — crashing the worker under any
  non-legacy registry and deadlocking the master's next collective.
  Fixed: configs ship (name, template) STRING PAIRS; workers rebuild
  AtomSpec AFTER `set_variables` (identical objects for legacy
  registries). Regression: FD-i2 now runs the live 2-rung spawn WITH an
  atom at d=2.
- **(1, n) single-variable domain hybrid**: was accepted but rode the
  d=1 code paths as a 2-D array (silent all-NaN composes). Fixed:
  Grove/Forest reject it with a `ravel()` hint; regression in FD-c.
Minor hardenings adopted from the review: `_prefix_mints_variable`
regex-escapes the prefix; `variables=` rejects a bare string (would
split into characters). Declined (documented): registry installed
before ctor validation completes (module-global semantics, same class
as the RNG globals); `set_variables`' non-str branch unreachable
(TypeError fires first — same outcome).

Build traps hit and resolved:
- `function_treeC` compiles under Cython language_level 2 (no pragma):
  f-strings cannot implicitly coerce to `str`-typed cpdef parameters →
  `is_variable_name`/`_prefix_mints_variable` take `object`.
- Generator expressions must not capture `cdef int` loop variables (the
  default-name construction in the ctors uses untyped locals).

Observations recorded for the test-side instance (pre-existing, NOT
touched, bit-identity preserved):
- `Tree._update_parameters`/`_update_parameters_and_fitness` pass
  `(input_pars, method_str, use_gradients, self.sigma)` positionally into
  `Node.optimize_sigma(x, y, input_pars, sigma, method_str,
  use_gradients)` — the σ-ARRAY branch has its arguments in the wrong
  slots (function_treeC.pyx ~944/981). Never exercised by the suites.
- `RealFunctionC.is_param()` returns `True` or `None` (falls through),
  never `False`.
- `is_parameter` (real_functionC.pyx ~1048) has zero callers (dead).

Tests: `test_phaseFD.py` (FD-a registry/classification incl. the
variable-as-parameter catastrophe assert; FD-b naming guards; FD-c shape
law + strict-(d,n) ctor validation; FD-d BIC branches + NaN-penalty
regression at d=2; FD-e determinism/moves at d=2; FD-f Pólya + frozen
vocabulary at d=2; FD-g serial burn-in + EMC recovery of
y = 2.5·x0 + sin(x1) to NRMSE < 0.05; FD-i wire round-trips incl. the
legacy raw-expr payload byte-contract; FD-k atoms at d=2; FD-j
informational timing; FD-i2 live 2-rung `run_mpi_spawn` at d=2, opt-in
via `--mpi` under `mpiexec -n 1`).
