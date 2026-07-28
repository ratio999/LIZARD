# Design 2.10 — Custom Atomic Primitives: the Formal Treatment

Companion to PAPER_PLAN.md §10 and PLAN.md Phase 2.10. This document gives
the *most rigorous* answer to the open question "what prior mass does a
user primitive get in the Pólya node?" — chosen deliberately over the easy
answer (append an α₀ slot), because the rigorous construction (i) unifies
parameter leaves and user primitives into one object class, (ii) exposes and
repairs a latent inconsistency in the current `add_primitive` growth scheme,
and (iii) settles Phase 2.3 and Phase 1.5 as corollaries.

---

## 1. The primitive space

Fix a measurable space $(\Psi, \mathcal{B})$ of *admissible primitives*,
stratified by internal parameter dimension:

$$\Psi \;=\; \bigsqcup_{k \ge 0} \Psi_k, \qquad
\psi \in \Psi_k :\; \psi : X \times \mathbb{R}^k \to \mathbb{R}
\ \text{measurable, with an identifiability convention on } \theta_\psi.$$

- The core function symbols ($\sin x$, $\cos x$, $x$, …) are **atoms of
  $\Psi_0$** (no internal parameters).
- **Unification observation (the payoff):** the existing *parameter leaf* is
  the simplest nontrivial primitive — the constant map
  $\psi_{\mathrm{const}}(x; \theta) = \theta \in \Psi_1$. A user primitive
  like $A\sin(\omega x)$ is an atom of $\Psi_2$. There is exactly ONE object
  class: an atomic leaf symbol carrying $k_\psi \ge 0$ internal parameters.
  The current code's special-casing of "parameter leaves" vs "function
  leaves" is the $k=1$-constant vs $k=0$ split of the same taxonomy.

This unification is what makes §10.4 of PAPER_PLAN ("the extension is a
special case of the §4 prior") a theorem rather than a slogan.

## 2. The node-level prior: a Dirichlet Process, not a growing Dirichlet

At each construction-path node $u$, the choice space is
$C = \Omega \sqcup \Psi$ (an op, or a primitive). Place on it

$$H_u \sim \mathrm{DP}\big(c,\; G_0\big), \qquad
G_0 = \lambda_{\mathrm{op}} \cdot \mathrm{Unif}(\Omega)
    + \lambda_{\mathrm{prim}} \cdot G_\Psi,
\quad \lambda_{\mathrm{op}} + \lambda_{\mathrm{prim}} = 1,$$

where $G_\Psi$ is a fixed probability measure on $\Psi$ — e.g. a geometric
envelope over the parameter dimension times a per-stratum base:
$G_\Psi(\Psi_k) \propto (1-\rho)\rho^k$. The defining property of the DP is
exactly what the implementation needs:

> **For every finite measurable partition** $(B_1, \dots, B_m)$ of $C$,
> $\big(H_u(B_1), \dots, H_u(B_m)\big) \sim
> \mathrm{Dirichlet}\big(c\,G_0(B_1), \dots, c\,G_0(B_m)\big).$

So the finite Dirichlet the code maintains at each Pólya node **is** the
DP's finite-dimensional projection onto the partition

$$\big\{\{\omega_1\}, \dots, \{\omega_O\},\ \{\psi_1\}, \dots, \{\psi_m\},\ R\big\},
\qquad R = \Psi \setminus \{\text{registered atoms}\},$$

with one extra cell the current code does not have: the **remainder cell**
$R$, carrying the base mass $c\,G_0(R)$ of every primitive not yet
registered. The symmetric-α₀ initialisation corresponds to the particular
$G_0$ that is uniform over the initially registered atoms.

## 3. Registration is partition refinement — and the current scheme is wrong

**The rule.** Registering a new primitive $\psi$ must *split the remainder
cell*:

$$\alpha_R \;\longrightarrow\; \big(\alpha_\psi,\ \alpha_{R'}\big)
\quad\text{with}\quad \alpha_\psi = c\,G_0(\{\psi\}),\ \
\alpha_{R'} = \alpha_R - \alpha_\psi,$$

leaving every other cell's parameter — and the total concentration
$c$ — **unchanged**. This is forced by the Dirichlet aggregation property
(merging cells sums parameters; refining must therefore conserve them),
which is the Kolmogorov-consistency condition making all the finite
projections marginals of ONE prior.

**Theorem (registration invariance).** Under the splitting rule, the prior
over trees whose symbols lie in any registered set $V$ is invariant to when
and in what order the primitives of $V$ (or any others) were registered.
*Proof:* all such priors are finite-dimensional projections of the same
$\mathrm{DP}(c, G_0)$; projections commute with refinement by aggregation.
$\square$

**The current code violates this.** `add_primitive` *appends* a fresh
$\alpha_0$ slot and grows `alpha_total` by $\alpha_0$
(`polya_treeC._grow_all_nodes`). Appending changes the predictive
probability of **every existing symbol** at every node
($p_a = \alpha_a / \alpha_{\mathrm{total}}$ with a grown denominator): the
prior itself drifts with vocabulary size, so trees scored before and after
a registration are scored under *different priors*. The 2.13(c) fix made
each single acceptance ratio internally consistent, but the sequence of
priors is still registration-history-dependent. Under the splitting rule
this disappears exactly: registered symbols' masses are untouched, and
Phase 1.5's "growth invariance" — which the plan asked us to *prove* —
becomes true instead of false. (As implemented today it holds only in the
weak sense that indices stay addressable.)

**Consequence for the unvisited-region default.** `log_prior`'s
$-\log(n_{\mathrm{total}})$ default for unvisited nodes should become the
base-measure predictive $\log G_0(\text{cell})$ — a registration-invariant
constant per cell, instead of a quantity that changes globally every time
the vocabulary grows.

## 4. What the sampler proposes: restriction by neutrality

The sampler only ever *proposes* registered symbols; the remainder cell is
never selected during tree construction. Two coherent conventions exist:

- **(chosen) Restrict-and-renormalize:** the effective per-node choice law
  is $H_u$ conditioned on the registered set. By Dirichlet **neutrality**,
  this conditional is again Dirichlet with the same parameters over the
  registered cells — closed form, and the conditioning factor is a
  *per-node-visit constant* absorbed identically into forward and reverse
  path probabilities of the same tree, hence exactly cancelling in every
  prior ratio the sampler computes.
- (rejected) Keep the unrestricted prior and let trees carry a mass deficit
  $H_u(R)$ per node visit — coherent but introduces an implicit
  tree-size penalty entangled with $G_0(R)$; harder to reason about and to
  report.

## 5. The continuous internals: where $k_\psi$ actually enters

The internal parameters $\theta_\psi \in \mathbb{R}^{k_\psi}$ get a prior
$g_\psi$ (unit-information by default, user-overridable) that enters
**only through the evidence**, never through proposal densities — this is
the (a1) topology-marginal discipline, unchanged:

- **Evidence:** the Laplace/BIC identification for a tree $T$ with plain
  parameters $k(T)$ and opaque leaves $\{\psi_j\}$ penalizes the *total*
  continuous dimension:
  $\log p(y \mid T) \approx \log L(y \mid T, \hat\theta)
  - \tfrac12\big(k(T) + \sum_j k_{\psi_j} \,[+1 \text{ if } \sigma
  \text{ profiled}]\big)\log n$. The existing `_shortcut_bic` needs only a
  correct *count* of effective parameters — the opaque leaf reports
  $k_\psi$ instead of 0.
- **Proposals:** birth of an opaque leaf draws $\theta_\psi$ from its
  warm-start law (any fixed density; it integrates out of
  $q_{\mathrm{struct}}$); death discards it; **no $g$ terms in any
  correction** (THEORY_2_13.md §1.5). Jacobian $= 1$ (identity embedding).
- **Refresh:** `parameter_refresh`/AMState treats $\theta_\psi$ like any
  other coordinates (post-a1 it is warm-start jitter only).

## 6. Opacity is a support restriction of the move kernel

"Atomic" means the structural moves never select a node *inside* $\psi$
(there is none — it is a leaf) and never split it: the opaque leaf is
created/destroyed/replaced **as a unit**. Formally this is not a
restriction at all: the leaf has no internal tree structure, so the state
space is unchanged; what changes is only the *leaf-choice cardinalities* in
the proposal densities. With $R$ registered opaque primitives (parameter
leaf included among them, per §1):

- point_mutation leaf choices, ETR's $N_0$, and `_draw_leaf_et` /
  `_build_subtree_with_params` structural factors change from
  $M + 1$ to $M + R'$ (where $R'$ counts the available atomic options
  under the chosen selection scheme), with matched forward/reverse
  bookkeeping — same pattern as every existing density, no new
  correctness argument.
- Irreducibility (THEORY_2_13.md §4) is preserved verbatim: ETR
  creates/destroys opaque leaves as 0-ETs; point_mutation relabels them.

## 7. The canonical extension (named, deferred): hierarchical sharing

The construction above uses independent $H_u \sim \mathrm{DP}(c, G_0)$ per
node with a *shared base*. The canonical way to let a primitive that earns
mass in one context gain prior mass elsewhere is the **Hierarchical DP**
(Teh et al. 2006): $H_u \mid H \sim \mathrm{DP}(c, H)$,
$H \sim \mathrm{DP}(c_0, G_0)$. The urn reinforcement then propagates
through the hierarchy. This is the formally right home for "the data teach
the prior which primitives matter globally" — flagged for the paper's
future-work section, not v1.

## 8. What must be proven in the paper (checklist for §10 / Appendix)

1. **Unification lemma** (§1): parameter leaves are $\Psi_1$ constants;
   the §4-prior over the extended vocabulary is the same construction.
2. **Registration-invariance theorem** (§3): splitting rule ⇒ prior
   independent of registration history (Dirichlet aggregation).
3. **Restriction lemma** (§4): neutrality ⇒ the restricted sampler's prior
   ratios equal the unrestricted ones.
4. **Evidence bookkeeping** (§5): BIC with total continuous dimension
   $k + \sum k_\psi (+1)$; regularity conditions per primitive.
5. **Kernel correctness** (§6): updated leaf-cardinality factors in each
   density; irreducibility carries over.
6. **Propriety** unchanged: finite support under the depth cap; the DP
   layer changes per-node masses, not the support argument.

## 9. Implementation consequences (ordered, for when 2.10 coding starts)

1. ✅ (Stage 1, 2026-07-09) `polya_treeC`: add the remainder cell; change
   `add_primitive` from append-α₀ to split-from-remainder (with `c` and
   `G_0` weights fixed at construction); change the unvisited-node default
   to the $G_0$ predictive. Provide a config for $G_\Psi$ (per-primitive
   base mass). Backward-compat flag for the legacy append behaviour
   (existing tests).
2. ✅ (Stage 2, 2026-07-09) `real_functionC` / `function_treeC`:
   opaque-leaf object — a RealFunctionC built from a user template
   (`AtomSpec` + `instantiate_atom`) with $k_\psi$ named internal
   parameters; `is_param`-style predicate generalized to `atomic leaf with
   k_psi >= 0` (`is_atomic`/`leaf_k`/`leaf_param_names`); effective-
   parameter count exposed to `_shortcut_bic` by absorbing $\theta_\psi$
   into `name_values` (θ lives on the Node, deepcopy-safe; the leaf's
   canonical Pólya key is `vocab_symbol()` = the template symbol).
3. ✅ (Stage 2, 2026-07-09) `gp_movesC`: leaf-choice cardinalities
   ($M + R'$, $R' = 1 + A$) in point_mutation, ETR,
   `_build_subtree_with_params` (and $M{+}A$ in every `_eval_build_density`
   call); opaque leaves selectable as units; densities updated
   symmetrically via a trailing `atoms` argument (None ⇒ $A = 0$ ⇒
   bit-identical legacy); force-reject where a reverse move cannot rebuild
   an atom (subtree_mutation always; death moves when $A = 0$).
4. ◐ (Stages 3a+3b, 2026-07-10) `groveC`/`mpi_worker`: user primitives ARE
   pre-registered at Forest construction (one Pólya cell per template,
   never via mid-run auto_register — the registration-invariance
   requirement); `atoms` threaded uniformly into every atom-aware kernel
   move on master AND workers; AMState built from
   `gp_movesC.pr_coord_names(tree)` (θ_ψ refreshable); BIC-cache hits
   install opaque θ̂; the MPI wire carries atom-bearing trees as
   ('skel3b', skeleton, description) structural payloads
   (`function_treeC.tree_to_skel`/`node_from_skel`) — exact shape, opaque
   leaves reconstructed as units — while atom-free trees keep the
   bit-identical legacy sympy-expr payload; AtomSpec pickles via
   __reduce__ for the worker startup config. ✅ (Stage 3c, 2026-07-10)
   the FULL vocabulary freeze: since no finite name pool covers new_par's
   reachable indices (they ratchet without bound), the freeze is realized
   through §1's own unification — `unify_params` maps every plain
   parameter leaf to the single ψ_const cell, making the vocabulary
   {ops} ∪ {funcs} ∪ {ψ_const} ∪ {templates} finite and CLOSED; Forest's
   `freeze_vocab=True` builds it with auto_register=False (the guarded
   error). The prior becomes fixed from t = 0 AND renaming-invariant
   (α-equivalent trees receive equal mass — the formally correct quotient,
   since parameter names are bound labels). Opt-in; legacy per-name mode
   is the bit-identical default. See PLAN §2.10 Stage 3c.
5. Tests: ✅ registration-invariance unit test (register in two orders,
   assert identical log_priors) — test_phase210 (c); ✅ density tests
   extended to the new cardinalities — test_phase210 (h)–(n); ✅ the
   end-to-end with one $\Psi_2$ primitive — test_phase210 (o): a
   single-rung Forest with the $A\sin(\omega x)$ template and the safe-4
   op kernel (frequency unreachable by composition) recovers
   $2.3\sin(1.3x)$ to machine precision.
