# Theoretical Notes — Phase 2.13 Soundness Fixes

Formal companion to `AUDIT_2_13_FIXES.md` (technical change log) and
`PLAN.md` §2.13 (roadmap). This file collects the *mathematical* content:
precise statements of each soundness issue, the balance arguments, and the
justification of each fix. Material here is written to be liftable into the
thesis/paper theory chapter (Phase 5.1).

---

## 1. Issue (a): proposal densities over θ vs. a profiled (θ-free) target

### 1.1 Setup and notation

Let $T$ denote a tree topology (labelled parse tree) with $K(T)$ parameter
leaves, and $\theta \in \mathbb{R}^{K(T)}$ the vector of values stored in
those leaves. The sampler's *stored* state is the pair $x = (T, \theta)$.

The acceptance ratio in `groveC._step` uses the θ-free target

$$\pi(T) \;\propto\; \exp\!\Big(-\tfrac{1}{2}\,\mathrm{BIC}(T) \;+\; \log \pi_P(T \mid \Theta)\Big),
\qquad \mathrm{BIC}(T) = -2\log L\big(y \mid T, \hat\theta(T)\big) + K(T)\log n,$$

with $\hat\theta(T) = \arg\max_\theta L(y \mid T, \theta)$ obtained by
numerical optimization, and $\pi_P$ the Pólya structural prior. By the
Schwarz/Laplace identification, $-\mathrm{BIC}/2 = \log p(y\mid T) + O(1)$
under a unit-information prior on θ, regularity at $\hat\theta$, iid data —
so $\pi(T)$ is a legitimate (approximate) *marginal* posterior over
topologies. Nothing below questions this; the issue is the proposal
bookkeeping.

For a trans-dimensional (birth/death) move pair, collect every *structural*
factor of the MH ratio into a single constant

$$A \;=\; \frac{\pi(T')}{\pi(T)} \cdot \frac{s_{\mathrm{rev}} \cdot \frac{1}{M}}{s_{\mathrm{fwd}}},$$

where $s_{\mathrm{fwd}}, s_{\mathrm{rev}}$ are node-selection and
kind-choice probabilities and $1/M$ is the reverse uniform function choice
($M = |\mathcal F|$). The pre-fix code (point_mutation cases (c)/(d);
analogous terms in `_eval_build_density` and ETR) accepts:

- **Birth** $T \to T'$: draw $\theta^* \sim g = N(0, \tau^2)$, accept w.p.
  $\alpha_{\mathrm{b}}(\theta^*) = \min\{1,\, A / g(\theta^*)\}$
  ($g$ enters via $\log q_{\mathrm{fwd}}$);
- **Death** $T' \to T$: accept w.p.
  $\alpha_{\mathrm{d}} = \min\{1,\, g(\theta_{\mathrm{stored}}) / A\}$
  ($g$ enters via $\log q_{\mathrm{rev}}$, evaluated at the value the tree
  currently carries).

### 1.2 The construction for which this bookkeeping is exact

**Lemma (pointwise balance under retention).** On the product space with
improper reference measure $\tilde\pi(T, d\theta) = \pi(T)\,d\theta$
(Lebesgue on each θ coordinate), the above acceptances satisfy detailed
balance *pointwise in* $\theta^*$:

$$\underbrace{\pi(T)\, s_{\mathrm{fwd}}\, g(\theta^*)\, \min\{1, A/g(\theta^*)\}}_{\text{forward flow at } \theta^*}
\;=\; \pi(T)\, s_{\mathrm{fwd}} \min\{g(\theta^*), A\}
\;=\; \underbrace{\pi(T')\, s_{\mathrm{rev}}\tfrac1M\, \min\{1, g(\theta^*)/A\}}_{\text{reverse flow from the state storing } \theta^*}.$$

*Proof.* The identity $z\min(1, w/z) = \min(z, w)$ applied with
$z = g(\theta^*)$, $w = A$, and the definition of $A$. $\square$

The lemma has **one binding hypothesis**: the value at which the death move
evaluates $g$ must be the *same* $\theta^*$ the birth drew — i.e. θ must be
*retained* as chain state between the two events. (The RJMCMC Jacobian is
genuinely 1 here — identity embedding — that part of the plan's argument
was and remains correct.)

### 1.3 How the implementation violates retention (two channels)

**(i) Optimizer projection.** Before accept/reject, the scorer
(`_log_marginal_bic(optimize=True)` or the `_bic_cache` re-install) replaces
the proposed $\theta^*$ by $\hat\theta(T')$. The realized topology-marginal
transition probabilities become

$$a_{\mathrm{b}} = \mathbb{E}_{\theta^* \sim g}\big[\min\{1,\, A/g(\theta^*)\}\big]
\qquad\text{vs.}\qquad
a_{\mathrm{d}} = \min\{1,\, g(\hat\theta)/A\},$$

and invariance of $\pi$ would require
$\mathbb{E}_{\theta^*}\big[\min\{g(\theta^*), A\}\big] = \min\{g(\hat\theta), A\}$
— an average against a point evaluation at a data-chosen value. False in
general $\Rightarrow$ net probability flow between $T$ and $T'$
$\Rightarrow$ $\pi$ is not stationary.

This is the *Monte-Carlo-within-Metropolis* error (a fresh random quantity
inside $\min(1,\cdot)$ each visit), **not** a pseudo-marginal scheme:
Andrieu–Roberts (2009) requires an unbiased positive estimator of the target
retained as auxiliary state and reused on the denominator side. The $g$
factors are proposal densities, not target estimates, and nothing is
retained.

**(ii) Kernel composition.** Three kernels act on the θ coordinates with
three different invariant laws: the structural kernel (in its retained-θ
reading) preserves $\pi(T)\otimes\mathrm{Leb}$; the optimizer projection
preserves only $\delta_{\hat\theta(T)}$; `parameter_refresh` preserves
$\propto L(y\mid T,\theta)^{1/T_{\mathrm{temp}}}$. A cycle/mixture of kernels
is $\pi$-invariant only if every component preserves the *same* law; here no
common invariant law exists, so the composite chain targets nothing — in
particular the $T$-marginal need not be $\pi$.

### 1.4 Exact bias in a two-state example

Let $M = 1$, topologies $T_0$ (single function leaf) and $T_1$ (single
parameter leaf), $\pi(T_0) = \pi(T_1)$, symmetric selection, so $A = 1$;
$g = N(0, \tau^2)$, $\tau = 1$, hence $g(\theta) = \varphi(\theta) \le
(2\pi)^{-1/2} < 1$. Then

$$a_{\mathrm{b}} = \mathbb{E}\big[\min\{1, 1/\varphi(\theta^*)\}\big] = 1,
\qquad a_{\mathrm{d}} = \varphi(\hat\theta),$$

so the two-state chain has stationary odds

$$\frac{\hat\pi(T_1)}{\hat\pi(T_0)} = \frac{1}{\varphi(\hat\theta)}
= \sqrt{2\pi}\,\tau\, e^{\hat\theta^2 / 2\tau^2}
\quad\text{instead of } 1.$$

$\hat\theta = 2 \Rightarrow$ odds $\approx 18.5$; $\hat\theta = 4
\Rightarrow$ odds $\approx 7.5\times 10^3$. In general the pre-fix sampler
behaves as if each parameter contributed a spurious
$-\log g(\hat\theta_j) = \tfrac12\log(2\pi\tau^2) + \hat\theta_j^2/2\tau^2$
bonus to the log-evidence: it *hoards* parameters whose fitted values are
far from 0 on the $\tau = \mathrm{std}(y)$ scale (e.g. frequencies,
offsets), since their death probability $\propto g(\hat\theta_j)$ collapses.
The bias is data-dependent and unbounded — not a constant that could cancel.

### 1.5 The fix (a1): a purely topology-marginal chain

Declare the chain state to be $T$ alone, and $\hat\theta(\cdot)$ a
*deterministic map* — operationally enforced by the BIC topology cache,
which freezes the first-visit optimum per topology per dataset. Under this
reading the birth draw $\theta^* \sim g$ is proposal-internal randomness
that does not influence *which topology* results, hence integrates out of
the kernel:

$$q(T' \mid T) \;=\; q_{\mathrm{struct}}(T' \mid T)\int g(\theta^*)\,d\theta^*
\;=\; q_{\mathrm{struct}}(T' \mid T),$$

and the exact MH acceptance is

$$\alpha = \min\Big\{1,\; \frac{\pi(T')\, q_{\mathrm{struct}}(T \mid T')}{\pi(T)\, q_{\mathrm{struct}}(T' \mid T)}\Big\}
\qquad\text{— no } g \text{ anywhere.}$$

Corrected per-move corrections (what the code implements after the fix):

| Move | $\log q_{\mathrm{fwd}}$ | $\log q_{\mathrm{rev}}$ | change |
|---|---|---|---|
| point_mutation BIRTH | $0$ | $-\log M$ | dropped $+\log g(\theta_{\mathrm{new}})$ from fwd |
| point_mutation DEATH | $-\log M$ | $0$ | dropped $+\log g(\theta_{\mathrm{dying}})$ from rev |
| add_node / split (fwd build) | $K_S \log p_{\mathrm{par}} + (2^d\!-\!K_S)\big(\log(1\!-\!p_{\mathrm{par}}) - \log M\big) - (2^d\!-\!1)\log O$ + selection/op/depth/side | unchanged (selection only) | dropped $\sum \log g(\theta)$ from the build density |
| leaf_hoist / internal_hoist (rev build) | unchanged (selection only) | same structural build density as above | dropped $\sum \log g(\theta)$ |
| ETR | $-\log\Omega_i - \log 2 - \log s(o_f)$ | $-\log\Omega_f - \log 2 - \log s(o_i)$ | dropped both $\sum\log g$ terms |
| subtree_mutation | unchanged | unchanged | never had $g$ terms |

Two consistency observations:

1. **In every coherent construction the naked $g$ factors vanish from the
   final ratio.** In (a1) they never enter; in the fully joint alternative
   (a2 below) they cancel against the parameter prior (the classic
   "propose births from the prior" cancellation). Surviving naked
   $\log g$ terms in an acceptance ratio are themselves diagnostic of an
   inconsistency.
2. **`parameter_refresh` becomes harmless.** After (a1) no acceptance
   probability reads θ, so PR only perturbs the optimizer warm start. It
   may be kept (or dropped) with no effect on the topology chain's
   invariant law.

**Residual caveat (path-independence of $\hat\theta$).** With warm starts
and a possibly multimodal likelihood, the optimizer output depends on the
path taken to first reach $T$. The cache freezes the first-visit value, so
the realized target is $\pi$ *defined by the frozen values* — exact
conditional on the cache, with the quality caveat that a badly converged
first visit permanently under-scores that topology. Mitigations
(multi-start, pseudo-marginal) remain Phase-1.6 future work. Cache resets
(new $y$) redefine the target consistently for the new dataset.

### 1.6 The rejected alternative (a2), for the record

A fully explicit product-space sampler: target
$\tilde\pi(T, \theta) \propto L(y \mid T, \theta)\, \prod_j g(\theta_j)\,
\pi_P(T\mid\Theta)$ with $g$ now a genuine prior; births proposed from $g$
cancel it in the acceptance; the likelihood is scored at the *current* θ
(no optimizer inside the kernel); PR retargeted to the tempered
$L \cdot g$. Textbook-correct RJMCMC, but it abandons the BIC commitment
(Phases 1.6/2.7), forbids in-kernel optimization, and mixes far more slowly
in θ. Rejected for v1; documented here because the thesis should state why.

### 1.7 A sharp regression test implied by §1.4

After the fix, the two-topology chain of §1.4 (one function, flat
structural prior, birth/death only) must recover 1:1 occupancy *regardless
of* $\hat\theta$ and $\tau$. This is implementable as a unit test with the
BIC term neutralized (equal-evidence fixtures) and is proposed as
`test_a1_toy_occupancy` in the retest checklist.

---

## 2. Issue (b): within-grove crossover reversibility

### 2.1 The defect

The pre-fix within-grove crossover proposed from the grove state
$x = (T_1,\dots,T_S)$ by picking slot $i$, donor $j \ne i$, deep-copying
$T_j$, swapping subtrees $S_v \leftrightarrow S_w$ between the copies — and
then installing **only** slot $i$'s child on accept, discarding the donor
copy. From the new state $x' = (\dots, T_i', \dots, T_j, \dots)$ the
displaced subtree $S_v$ exists nowhere in the grove, so the true reverse
proposal density is

$$q(x \mid x') = 0,$$

while the bookkeeping still supplied the pair-swap counts
$\log q_{\mathrm{rev}} = -\log n_{\mathrm{int}}(T_a') - \log n_{\mathrm{int}}(T_b')$.
A positive claimed reverse density against a zero true one violates detailed
balance outright (not merely biases it): flow along $x \to x'$ has no
compensating reverse flow.

### 2.2 The fix (b1): a genuine pair move

Both children are proposed, scored, and installed (or rejected) **together**.
Both slots live in the same grove, hence at the same temperature $T$; the
grove's joint tempered target is the product $\prod_s \pi_T(x_s)$, so the
move touches the factor $\pi_T(x_i)\,\pi_T(x_j)$ and the exact acceptance is

$$\log\alpha = \frac{(U_i' - U_i) + (U_j' - U_j)}{T} + (\log q_{\mathrm{rev}} - \log q_{\mathrm{fwd}}),
\qquad U = \text{BIC log-evidence} + \log\pi_P.$$

Selection factors cancel exactly: slot $i$ is uniform ($1/S$), the donor
uniform ($1/(S{-}1)$), and the unordered pair $\{i,j\}$ is reachable through
both orderings with the swap distribution symmetric under
$(v,w) \leftrightarrow (w,v)$ — identically so in the reverse direction. The
changing part (node-selection counts $n_{\mathrm{int}}$) is exactly what
`subtree_crossover`'s $(\mathrm{lqf}, \mathrm{lqr})$ carries. This is the
within-grove specialization of the Forest trans-grove crossover
($T_a = T_b = T$), whose form was already verified correct.

Reinforcement convention (R1): within-grove crossover is mutation-family —
on accept **both** children's construction paths reinforce the shared Θ
(deferred under `polya_defer` in MPI mode).

---

## 3. Issues (c), (e), (f): shorter balance arguments

### 3.1 (c) Auto-registration ordering

`extract_path` may register a fresh symbol, growing every Dirichlet's
support from $n$ to $n{+}1$ choices. Pre-fix, `_step` scored the incumbent's
prior on the $n$-vocabulary and the candidate's on the $n{+}1$-vocabulary
inside the *same* ratio. Each unvisited-region step contributes
$-\log n_{\mathrm{total}}$, so the candidate was penalized by
$\approx \#\{\text{unvisited steps}\}\cdot\log\frac{n+1}{n}$ relative to a
consistent evaluation — a systematic pro-incumbent tilt firing exactly when
the candidate introduces a new symbol. Fix: extract *all* paths first, then
evaluate all priors against the final vocabulary. (Applied in `_step`,
`legacy_step`, the Forest swap/crossover, the pair-crossover helper, and the
MPI-master blocks; in the crossover cases the old paths must be captured
*before* the in-place move mutates the trees.)

### 3.2 (e) Proposal-mixture retry

With `max_attempts = 2`, the realized kernel was
$q_{\mathrm{eff}}(\cdot \mid T) = q(\cdot \mid T) + p_{\mathrm{inv}}(T)\, q(\cdot \mid T)$-mixture,
where $p_{\mathrm{inv}}(T)$ is the state-dependent probability that the first
draw fails validity. The acceptance ratio used the single-attempt densities
of whichever attempt succeeded, so the mismatch
$q_{\mathrm{eff}} \ne q$ enters the ratio unaccounted, and
$p_{\mathrm{inv}}(T) \ne p_{\mathrm{inv}}(T')$ makes it non-cancelling.
Fix: one attempt; an invalid candidate is a rejection (a self-loop is always
target-preserving).

### 3.3 (f) Build-density/vocabulary mismatch

`build_tree` sampled subtree-internal ops uniformly from the hardcoded
6-symbol set while `_eval_build_density` charged $-n_{\mathrm{int}}\log O$
with $O = |\texttt{operations}|$. For a restricted kernel ($O < 6$) every
grown subtree's forward density (add_node/split) and every dropped sibling's
reverse density (hoists) is off by $n_{\mathrm{int}}(S)\,\log(6/O)$. This
error is **not absorbable into a reweighted target**: size changes are also
reachable through correctly-densitied routes (ETR, crossover), so the
Kolmogorov cycle condition fails and the invariant law becomes dependent on
the move-mixture weights — which is precisely what the 2.8c
weight-perturbation test detects (adaptation reweights the kernel and the
size marginal moves).

Fix: `operations` is threaded through `build_tree` (default preserving the
historical order, hence bit-identical RNG streams for default kernels),
`_build_subtree_with_params`, and `subtree_mutation`'s rebuild. **Support
closure at initialization**: `Grove._init_trees` now builds initial trees
from the kernel's own vocabulary — necessary because an out-of-vocabulary op
in the start state could never be legally removed (op-resample and
death-move reverse densities for it are all zero), leaving the chain
permanently off-support. Caveat documented: user-supplied `node_list` trees
must respect the kernel vocabulary (unchecked, caller's contract).

### 3.4 (g) Single-leaf delegation and the move-choice factor

Green's ratio includes the move-selection factor
$p(m_{\mathrm{rev}} \mid T') / p(m \mid T)$, which the kernel cancels by
enforcing pair equality ($p_{\mathrm{add}} = p_{\mathrm{leaf\_hoist}}$,
$p_{\mathrm{split}} = p_{\mathrm{internal\_hoist}}$) and using self-inverse
moves elsewhere. The old single-leaf branch of `subtree_mutation` broke this
silently: the transition was *selected* with $p_{\mathrm{subtree}}$ but
*performed* as an `add_node`, whose reverse (`leaf_hoist`) is selected with
$p_{\mathrm{leaf\_hoist}} \ne p_{\mathrm{subtree}}$ — a non-cancelling
selection factor on exactly the single-leaf → grown-tree transitions. Fix:
the branch is now a forced reject (a self-loop, always target-preserving).
Irreducibility is unaffected because add_node/split/ETR grow single leaves
under their own, correctly paired weights.

### 3.5 (h) Partial-domain optima and the nanmean loophole

The optimizer's objective is $\overline{r^2} = \texttt{nanmean}\,(y - f)^2$:
parameters that push the model into NaN on part of the domain are scored on
the finite subset only, while the BIC penalty and the $N$ in the likelihood
use the full sample — inflated evidence for partially-defined models, and a
basin the optimizer can *actively* settle in. The pre-scoring validity gate
runs at the pre-optimization θ, so it cannot catch this. Fix: re-validate at
$\hat\theta$ (same criterion as the gate: finite everywhere, $|f| \le
10^{12}$) and return $-\infty$ on failure. Note the exactness bonus: if the
re-validation passes, no residual is NaN, so $\texttt{nanmean} =
\texttt{mean}$ and the recorded MSE is the true full-domain MSE — the
loophole closes exactly rather than approximately.

### 3.6 (i) Profiled σ counts as a parameter

With $\sigma$ unknown, profiling gives $\hat\sigma^2 = \mathrm{SSR}/N$ and
$\log L(\hat\theta, \hat\sigma) = -\tfrac{N}{2}\log(2\pi\hat\sigma^2) -
\tfrac{N}{2}$; the Schwarz penalty then counts $k+1$ estimated quantities
(θ plus σ). Since *every* model in the space estimates the same single σ,
the correction is a constant $-\tfrac12\log N$ shift: MH ratios, orderings,
and the sampler's stationary law are all unchanged. What changes is the
*absolute* reported evidence and any comparison between the profiled-σ and
fixed-σ modes — both now honest. (Fixed-σ mode keeps the $k$ penalty:
σ known is not estimated.)

---

## 4. Within-chain irreducibility (Phase 2.9 R2; paper §7.7)

### 4.1 Setting

Let $\mathcal{T}_D$ be the space of parse trees over vocabulary
$(\mathcal{F}, \Omega)$ ($M = |\mathcal{F}|$ functions as leaves, parameter
leaves, $O = |\Omega|$ binary ops) with tree depth $\le D$ (leaf $= 0$).
With the Phase 2.13(f) closure, every move draws ops/functions from this
same vocabulary, so $\mathcal{T}_D$ is closed under the kernel. The discrete
state is the parse structure (leaf kinds and op labels included); parameter
*values* are not part of the state (θ̂ is a deterministic per-topology
statistic under the (a1) topology-marginal reading, §1.5).

Let $Q$ be the within-grove mutation kernel: the mixture of point_mutation,
leaf_hoist, internal_hoist, add_node, split, subtree_mutation and ETR with
strictly positive weights (guaranteed under adaptation by the `adapt_floor`
containment, §9.2 of PAPER_PLAN's map).

### 4.2 Lemma (connectivity of the proposal kernel)

*For any $T, T' \in \mathcal{T}_D$ there is $n$ with $Q^n(T, T') > 0$.*

**Sketch — two moves suffice; the rest only add edges.**

1. **Down to a leaf (ETR).** Any internal node whose two children are both
   leaves is a 2-ET; every finite binary tree of positive depth has at
   least one. ETR replaces a uniformly chosen ET by a fresh ET of arity
   drawn from $\{0, 2\}$; choosing the deepest 2-ET and arity 0 replaces it
   by a single leaf. Each such step has positive probability (selection
   $\ge 1/\Omega_{\mathrm{ET}}(T)$, arity $1/2$, structure $\ge 1/(M{+}1)$),
   strictly decreases the node count, and never violates the depth cap
   (depth is non-increasing). Iterating reduces $T$ to a single leaf in at
   most $\#\mathrm{internal}(T)$ accepted steps.
2. **Up to the target shape (ETR).** Reverse the reduction of $T'$: a
   0-ET → 2-ET replacement at a chosen leaf builds one internal node of
   $T'$'s shape per step, top-down; every intermediate tree has depth
   $\le \mathrm{depth}(T') \le D$, so no step is depth-capped. Positive
   probability as in (1).
3. **Labelling (point_mutation + ETR draws).** With the shape of $T'$ in
   place: internal op labels are resampled uniformly from $\Omega$
   (probability $\ge 1/(|N|\,O)$ per node per step); leaf kinds are
   connected by the birth/death sub-moves (function ↔ parameter, each with
   probability $\ge \tfrac{1}{2}\cdot\tfrac{1}{|N|}\cdot\tfrac{1}{M}$
   structurally); specific function labels via sub-move A ($1/M$).

Composing (1)→(2)→(3) gives a positive-probability path
$T \to \text{leaf} \to \text{shape}(T') \to T'$. $\square$

**Remark (why other moves are not needed but harmless).** add_node/split
propose only perfect-binary grafts and hoists require perfect-binary
dropped siblings, so on their own they conserve certain parities (cf. the
2.9d test's leaf-count-parity note); ETR has no such constraint, which is
why it carries the connectivity argument. The remaining moves strictly add
transitions, which can only improve mixing, never break irreducibility.

### 4.3 Corollary (ergodicity of the fixed-target chain)

For any strictly positive target $\pi$ on the finite $\mathcal{T}_D$
(our tempered targets are: BIC evidence is finite on valid trees, the
Pólya prior is strictly positive on the vocabulary's support), MH
acceptance of an on-support proposal is $\ge \min(1, r) > 0$, so the MH
chain inherits the lemma's connectivity: it is irreducible. Aperiodicity
holds because every state has positive holding probability (rejections and
no-op sub-moves). On a finite space, irreducible + aperiodic ⇒ uniformly
ergodic for $\pi$. Adaptation is then layered on top via diminishing
adaptation + containment (Roberts–Rosenthal 2007), cf. PAPER_PLAN §9.

### 4.4 Empirical companions

* `test_phase29.py::test_2_9d2_full_support_coverage` — the strong check:
  on the micro-space $M{=}1$, $O{=}1$, $D{=}2$ (exactly 38 parse
  structures) the always-accept-on-support kernel chain achieves
  **set equality** visited = support from a single-leaf start, per seed:
  coverage (⊇) is the lemma, closure (⊆, asserted at every step) is the
  depth cap + vocabulary threading. The always-accept driver is faithful
  for reachability because MH accepts every on-support proposal with
  positive probability (Corollary 4.3).
* `test_phase29.py::test_2_9d_within_chain_irreducibility` — the dynamic
  companion on the real Grove (BIC target): up/down lattice connectivity
  and broad exploration, where exhaustive coverage would be flaky by
  design (the target concentrates).

---

## 5. The DP completion of the structural prior (Design 2.10; paper §4.8)

### 5.1 Statement

At each construction-path node $u$, model the choice law as
$H_u \sim \mathrm{DP}(c, G_0)$ on $C = \Omega \sqcup \Psi$, where $\Psi$ is
the measurable primitive space stratified by internal parameter dimension
(DESIGN_2_10_PRIMITIVES.md §1–2). The implementation's finite Dirichlet at
$u$ is the DP's finite-dimensional projection onto the partition
$\{\text{registered atoms}\} \cup \{R\}$, $R$ = the unregistered remainder.

### 5.2 Lemma (mass conservation / projective consistency)

Let registration of $\psi$ carve $\alpha_\psi = c\,G_0(\{\psi\})$ from the
remainder: $\alpha_R \to (\alpha_\psi, \alpha_R - \alpha_\psi)$, all other
cells untouched. Then for every node the total base mass
$\sum_a \alpha_a + \alpha_R = c$ is invariant, and every finite Dirichlet in
play is a marginal of the single $\mathrm{DP}(c, G_0)$. *Proof:* the
Dirichlet aggregation property — if $(p_1,\dots,p_m) \sim
\mathrm{Dir}(a_1,\dots,a_m)$ then merging cells sums parameters — makes
refinement-with-conservation exactly the Kolmogorov-consistency condition
for the DP's finite-dimensional laws. $\square$

### 5.3 Theorem (registration invariance)

Under the carving rule, the prior over trees is invariant to the *order and
timing* of registrations: any two registration histories producing the same
registered set (and the same reinforcement history on atoms) yield
identical construction-path probabilities. *Proof:* both are projections of
the same DP onto the same final partition, by Lemma 5.2; reinforcement
touches atoms only, so it commutes with carving (which touches only $R$ and
the newborn cell). $\square$

**Contrast with the append-α₀ scheme (pre-2.10 code):** appending grows the
total mass, so every registration changes every symbol's predictive at
every node — the "prior" is then a registration-history-indexed *family*,
not a fixed measure, and the old Phase 1.5 "growth invariance" claim is
false in the strict sense. The carving rule makes it a theorem.

### 5.4 Lemma (restriction by neutrality)

The sampler proposes only registered symbols. By Dirichlet neutrality, the
conditional of $H_u$ on the registered set is again Dirichlet with the same
parameters over the registered cells; operationally the predictive is
$\alpha_a / \alpha_{\mathrm{reg}}$ with
$\alpha_{\mathrm{reg}} = \sum_{a\,\mathrm{registered}} \alpha_a$
(remainder excluded from the normaliser). Hence the *implementation formula
is unchanged* from the legacy code — DP mode differs only in the
bookkeeping of $\alpha_R$ and in what `add_primitive` does. Both trees in
any MH ratio are scored under the same restriction set (the 2.13(c)
extract-then-score discipline), so restriction introduces no asymmetry; and
under the freeze-at-construction policy the restricted prior is a fixed
measure throughout sampling.

### 5.4b Remark (the ψ_const quotient / vocabulary freeze; Stage 3c)

Parameter *names* (`a0`, `b3`, …) are bound labels: the likelihood depends
only on the values bound to them, the sampler's `new_par` assigns them
deterministically given the tree, and label compaction already treats
renamed trees as interchangeable. The state space is therefore effectively
the α-equivalence quotient, and a prior with one urn cell **per name** (the
pre-Stage-3c implementation) both weights α-equivalent trees differently
and lives on an unbounded vocabulary — `new_par`'s indices ratchet upward
without bound, so no finite pre-registered pool closes it. The
§1-unification of DESIGN_2_10 resolves this: the parameter leaf is the
single atom $\psi_{\mathrm{const}} \in \Psi_1$, and mapping every
parameter leaf to that ONE cell (`unify_params`) makes the vocabulary
$\Omega \cup F \cup \{\psi_{\mathrm{const}}\} \cup \{\text{templates}\}$
finite and closed. Consequences: (i) the prior is a fixed measure from
$t = 0$ with no mid-run registration at all (the Phase 2.3 freeze,
`auto_register` off); (ii) the prior is renaming-invariant — it descends
to the α-equivalence quotient, where the state space actually lives;
(iii) reinforcement of "place a parameter here" accrues to one urn cell
rather than fragmenting across labels; (iv) MPI replica path indices are
stable by construction. Legacy per-name mode remains the default for
backward compatibility; the paper's runs use the frozen mode.

### 5.5 Remark (does the DP dilute the Pólya-tree story? No — it sharpens it)

(i) The Blackwell–MacQueen predictive of a DP *is* the Pólya urn: "each
node is a DP" = "each node is an infinite-capacity Pólya urn" — the DP is
the urn's de Finetti name, not a rival formalism. (ii) Carving-under-
refinement is precisely the classical Pólya-tree mass-splitting mechanism
(Ferguson/Lavine), now present WITHIN each node in addition to the
derivation trie ACROSS nodes: the object is a tree of urns twice over.
(iii) The novel contributions — the γ_t-reinforced adaptive urn coupled to
RJMCMC, and the context-indexed trie over derivations — are untouched by
the completion; the DP layer supplies the fixed-measure foundation that a
formal-correctness paper requires (Theorem 5.3 repairs what was previously
a genuine vulnerability). (iv) On a frozen vocabulary the DP mode is
computationally invisible: identical predictives, identical reinforcement;
$\alpha_R$ is inert bookkeeping until the first registration.

---

*(Sections for issues (j)–(l) — documentation items — to be folded directly
into the paper per PAPER_PLAN.md §5.5, §9.5, §12.4; see AUDIT_2_13_FIXES.md
for status.)*
