# Paper Plan — A Fully Bayesian Symbolic Regression via an Adaptive Pólya-Tree Structural Prior

**Purpose of this document.** An ordered, consequential list of the sections and
paragraphs the paper must contain, written so that every formal object is
*defined before it is used* and every claim of correctness is *earned*. This is
a writing roadmap, not the paper. It supersedes and expands the thin Phase 5
skeleton in `PLAN.md`. Formal/mathematical content that already exists and is
liftable lives in `THEORY_2_13.md` (RJMCMC reversibility, the topology-marginal
argument) and `PLAN.md` §1 / §2.13; cross-references are given inline as
[THEORY §x] and [PLAN §x].

---

## 0. The thesis and the three things the paper must convince a referee of

**One-sentence thesis.** Symbolic regression can be cast as *posterior inference
over a space of expression trees*, with a structural prior that is (i) a proper,
grammar-consistent probability measure and (ii) *adaptive* — it learns which
sub-expressions the data favour via a Pólya-urn reinforcement — while the whole
scheme remains a formally correct (reversible, ergodic) MCMC targeting a
well-defined posterior.

**The referee must be convinced of three things, in order of difficulty:**
1. **The prior is a genuine probability distribution over expression trees**
   (proper, normalizable, grammar-consistent), and the target
   π(T) ∝ exp(−BIC(T)/2)·π_prior(T) is a legitimate (approximate) marginal
   posterior over *structures*. — Hard-ish; mostly bookkeeping + one propriety
   argument.
2. **The sampler is correct**: each RJMCMC move is reversible with the right
   proposal correction, the chain is irreducible and aperiodic, hence ergodic
   for the fixed-prior target. — Medium; the material largely exists
   [THEORY §1–3], the 2.13 audit closed the gaps.
3. **The *adaptive* prior does not destroy correctness** and converges to a
   well-characterised object. — **THE HARD ONE.** This is the novel formal
   contribution and the single biggest risk in the paper. See §9 below; it must
   be confronted head-on, not hand-waved.

> **CENTRAL FORMAL DECISION (make before writing §4/§9).**
> Is the adaptive Pólya prior (A) a *fixed* Bayesian prior whose posterior we
> sample, with the urn reinforcement demoted to an empirical-Bayes / warm-up
> device that is *frozen* before the inference phase; or (B) an intrinsically
> *self-reinforcing* prior whose "posterior" is the stationary law of a
> self-interacting Markov chain? (A) is clean and defensible with standard
> tools; (B) is more novel but requires a stochastic-approximation / adaptive-
> MCMC argument and must deal with Σγ_t = ∞ (unbounded urn mass; audit (l)).
> The safest and still-novel framing: **present (A) as the rigorous baseline
> and (B) as the flagship extension**, proving for (B) the weaker-but-real
> statement we can actually support (diminishing adaptation + containment ⇒
> ergodic averages converge; the *predictive* prior stabilises), and being
> explicit about what is conjectured vs proved. Resolve this before drafting.

**Framing / venue decision (open, Phase 0).** Methods-first statistics/ML venue
(the formalism is the product) vs. applied venue with the method as apparatus.
Recommendation: write the methods paper as primary; the hero application (§12.3)
is extractable as a companion. This changes emphasis, not content order.

---

## 1. Introduction

1.1 **Motivation.** SR as *inference*, not optimization. What a posterior over
    expressions buys you that a single best-fit expression does not: uncertainty
    over *structure*, principled complexity control, model averaging.
1.2 **The gap.** Genetic programming (Koza; Eureqa; PySR) optimizes a fitness;
    even where "Bayesian" language appears the structural prior is usually
    ad-hoc (parsimony penalty) and the exploration is not a correct MCMC. The
    closest prior art — the Bayesian Machine Scientist (Guimerà et al.) — uses a
    fixed grammar prior estimated offline; contrast sharply (see §2).
1.3 **Contributions (bulleted, precise).** (a) A grammar-consistent Pólya-tree/
    Dirichlet structural prior over expression *construction paths*; (b) its
    *adaptive* urn-reinforced form; (c) a reversible trans-dimensional sampler
    with the correct topology-marginal proposal corrections; (d) a
    parallel-tempered EMC population sampler sharing one prior; (e) extensibility
    to user-defined opaque primitives with proper prior mass; (f) an honest
    ergodicity analysis for both the fixed and adaptive regimes.
1.4 **What "formal correctness" means here** and why it is the target of the
    work: a named target distribution, reversible moves, ergodicity, and a prior
    that is a bona fide probability measure — each stated as a checkable claim.
1.5 **Roadmap.**

## 2. Related work
- RJMCMC and trans-dimensional inference (Green 1995; Green & Hastie).
- Bayesian Machine Scientist (Guimerà & Sales-Pardo 2020) — the primary
  comparator; articulate what we add (adaptive prior; tempered EMC; opaque
  primitives; the correctness treatment).
- GP symbolic regression (Koza; PySR/Cranmer; Eureqa/Schmidt-Lipson).
- Bayesian model selection, BIC, Laplace approximation (Schwarz; Kass-Raftery).
- Grammar priors: PCFGs, adaptor grammars, program-synthesis priors.
- Pólya trees, Dirichlet processes, reinforced/Pólya urns (Ferguson; Lavine;
  Mauldin-Sudderth-Williams; Pemantle's survey of reinforced processes).
- Adaptive & self-interacting MCMC (Roberts-Rosenthal 2007; Andrieu-Thoms 2008;
  Del Moral-Miclo).

## 3. Problem setup and the state space
3.1 **The grammar.** Expressions as labelled binary parse trees: internal nodes
    are binary operations Ω = {+,−,×,÷,∘,^}; leaves are primitives from a set
    F (sin, cos, x, …) *or* parameters. State clearly: functions are LEAVES,
    not unary operators — this is non-standard vs. BMS and matters for every
    later count (e.g. ET arities, §7). [PLAN §2.11 grammar note]
3.2 **The state space 𝒯.** Countable; with the depth bound d_max it is *finite*.
    Give the σ-algebra (discrete) and the reference (counting) measure. The
    depth bound is a modelling choice with two payoffs (propriety §4.3;
    build_tree blow-up control) — introduce it here, honestly, not as a hack.
3.3 **Parses vs equivalence classes.** We put the prior on *parse trees*, not on
    algebraic equivalence classes of functions. State the identifiability
    convention and its consequence (a function has many parses; the posterior is
    over parses; report canonicalised summaries). [PLAN §1.7]
3.4 **The construction-path bijection.** Pre-order DFS ↔ tree: define
    visit(node) = record (kind, choice_idx) then recurse. State and prove the
    bijection between finite trees and finite decision paths. This is the hinge
    the entire prior hangs on. [polya_treeC.extract_path]

## 4. The structural prior (the star of the paper)
4.1 **Grammar → branching prior.** At each decision node of the construction
    path there is a categorical choice over the combined {op ∪ primitive} set.
    Place a Dirichlet(α₀) prior on each node's choice probabilities.
4.2 **The Dirichlet-multinomial predictive (fixed-α₀ prior).** Integrate out the
    per-node probabilities → the predictive path probability is a product of
    Dirichlet-multinomial predictive factors. Write π(T) as a product over the
    path. This is a genuine *exchangeable* prior (de Finetti reading of the urn).
    [polya_treeC.log_prior]
4.3 **Propriety (the one real theorem here).** Prove π is a probability measure
    on 𝒯. Under the depth bound: finite support ⇒ trivially proper *once
    normalised over 𝒯* — but note the branching construction as written assigns
    a mass to each path without an explicit stop/normalisation, so state exactly
    what object is normalised and over what set. Discuss the unbounded case
    (needs a termination probability at each node or a summable tail) and why
    the depth bound sidesteps it. **Do not skip this** — a referee will poke it.
4.4 **What this is and isn't.** Precise naming. It is NOT a classical Pólya tree
    on [0,1] partitions (Ferguson); it is a Dirichlet-multinomial predictive
    over a discrete grammar with lazy, path-indexed nodes — closer to an
    adaptive PCFG with a shared Dirichlet at each context. Be honest about the
    analogy that motivates the "Pólya" name (the urn reinforcement, §4.7) while
    naming the actual object correctly. Reviewers in Bayesian nonparametrics
    will care.
    **REVISED 2026-07-09 (post DESIGN_2_10):** the naming objection is now
    materially weaker and the section should say so. With the DP completion,
    (i) each node's choice law is a DP, whose Blackwell–MacQueen predictive
    IS the Pólya urn (the DP is the urn's de Finetti name, not a rival
    formalism); (ii) registration = partition refinement conserving Dirichlet
    mass — literally the classical Pólya-tree mass-splitting mechanism, now
    present WITHIN each node in addition to the derivation trie ACROSS nodes.
    The object is a tree of urns twice over. Also state the novelty
    hierarchy plainly: (1) adaptive urn-reinforced structural prior coupled
    to RJMCMC — unique; (2) context-indexed trie of urns over derivations —
    the structural idea; (3) DP completion — deliberately standard tools so
    the novel parts rest on checkable foundations. The registration-
    invariance theorem also repairs a genuine vulnerability: pre-2.10 the
    "prior" was a registration-history-dependent family, not a fixed
    measure; now it is one measure with computed finite views.
4.5 **Support and the unvisited-region convention.** Paths that descend into
    never-visited nodes contribute the symmetric default −log(n_total); state
    this as the prior's behaviour on its own support and connect to the lazy-trie
    lemma. [PLAN §1.4; polya_treeC.log_prior]
4.6 **Kind-conditional vs unified nodes.** The implementation uses a unified
    op+prim Dirichlet at every node; discuss the alternative (separate α_op,
    α_prim blocks conditioned on grammar context) and justify the choice. [PLAN §2.6]
4.7 **The adaptive (reinforced) prior — stated, analysed later.** Define the urn
    update α_k ← α_k + γ_t on accepted paths, γ_t = 1/(1+t). Here only DEFINE it
    and give intuition (data-favoured sub-expressions gain prior mass). All
    ergodic/limit claims are deferred to §9 — flag this forward reference
    explicitly so §4 stays about the prior-as-object.
4.8 **The DP completion (extensible vocabularies).** Each node's choice law
    as DP(c, G₀) over ops ⊔ the stratified primitive space Ψ; the finite
    Dirichlets of the implementation as its projections onto
    {registered atoms} ∪ {remainder}; registration = mass-conserving
    partition refinement (carve from remainder). State and prove: the
    conservation/consistency lemma (Dirichlet aggregation), the
    registration-invariance theorem, and the restriction-by-neutrality
    lemma (sampler formula unchanged; remainder excluded from the
    normaliser). Include the §4.4 REVISED naming discussion (the DP is the
    urn's de Finetti name; carving IS classical Pólya-tree mass-splitting —
    tree of urns twice over) and the novelty hierarchy. Note the frozen-
    vocabulary collapse (DP mode computationally invisible until a
    registration). Full derivations: THEORY_2_13.md §5; design:
    DESIGN_2_10_PRIMITIVES.md.

## 5. The likelihood and the marginal likelihood
5.1 **Observation model.** y = f_T(x;θ) + ε, ε ~ N(0, σ²I). σ known (instrument)
    vs profiled (σ̂² = SSR/n). [PLAN §1.6, §2.7(a)]
5.2 **Profiling θ.** θ̂ by numerical optimization; the plug-in. State that θ is a
    per-structure nuisance handled by profiling, not sampled — this sets up §5.4.
5.3 **BIC-as-Laplace.** The Schwarz derivation: under a unit-information prior on
    θ, regularity at θ̂, iid data, log p(y|T) = −BIC(T)/2 + O(1). State the
    regularity conditions as assumptions and where they can fail. [PLAN §1.6]
5.4 **The θ-marginal target and the topology-only chain.** The key formal move
    that makes the sampler consistent: the chain lives on 𝒯 alone, θ̂(T) is a
    deterministic per-structure statistic (frozen first-visit value via the BIC
    cache), and *all* proposal corrections are purely structural. Lift the full
    argument, the pointwise-balance lemma, and the two-state bias example from
    [THEORY §1]. This subsection is the paper's answer to "why is your
    acceptance ratio correct even though you optimize θ inside the kernel?"
5.5 **Honest limitations of the marginal.** Multimodal likelihood (θ̂
    path-dependence), boundary optima, redundant/near-unidentifiable parameters
    (effective vs nominal k), optimizer non-determinism. State each and its
    mitigation or its deferral to future work (full Laplace with −½log|H|,
    pseudo-marginal, multi-start). [PLAN §1.6 deferred upgrades]

## 6. The posterior target
6.1 **The target.** π(T) ∝ exp(−BIC(T)/2)·π_prior(T|Θ); present as an
    approximate marginal posterior over structures (approximate only through the
    BIC step, §5.3). State it once, cleanly; everything downstream samples this.
6.2 **The tempered family.** π_β(T) ∝ [exp(−BIC/2)]^β · π_prior; the role of β,
    and the important convention that tempering acts on the target ONLY, never
    on the proposal correction. [THEORY §1.1; PLAN §2.2]

## 7. Sampling I — the trans-dimensional kernel
7.1 **RJMCMC framework.** Green (1995); the reversibility condition and the
    dimension-matching / Jacobian requirement, specialised to our setting.
7.2 **The move catalogue.** point mutation (incl. birth/death of a parameter),
    add_node ↔ leaf_hoist, split ↔ internal_hoist, subtree mutation, ETR,
    crossover. For each: what it does and its reverse partner. [gp_movesC]
7.3 **Per-move proposal densities and reverse pairing.** The heart of the
    sampler's correctness. Lift the density derivations and the birth/death
    pair-equality argument from [THEORY §1–3] and [PLAN §2.1.*]. One
    subsection or a table per move; full derivations to Appendix A.
7.4 **Jacobian = 1.** Because θ is profiled (identity embedding, not a
    transform), no continuous-parameter Jacobian appears. State and justify.
    [PLAN §1.3, §1.6]
7.5 **Structural (topology-marginal) corrections — no g(θ) terms.** Why the
    freshly born θ ~ g is warm-start randomness that integrates out of the
    proposal density; the pre-fix bug and its resolution. [THEORY §1.5, §3.3]
7.6 **Bounded support as a restriction, not a bias.** The depth cap forces
    log_q_rev = −∞ on over-deep proposals; this restricts 𝒯, preserving
    detailed balance on the restricted support. [THEORY §3.3; PLAN §2.9]
7.7 **Irreducibility and aperiodicity.** The within-chain irreducibility lemma
    (birth/death pairs + subtree replacement span 𝒯 up to the depth bound) and
    an aperiodicity remark ⇒ ergodicity for the *fixed* π. Include the small
    empirical connectivity test as evidence. [PLAN §2.9 R2]

## 8. Sampling II — population, tempering, EMC
8.1 **Parallel tempering.** The ladder, the swap move, and the swap acceptance
    (1/T_a − 1/T_b)(U_b − U_a) with U = BIC-evidence + log-prior on a *shared* Θ.
    Ladder design (geometric; target swap rate 0.2–0.4). [PLAN §2.9, §4.1]
8.2 **EMC crossover.** Within-grove (a genuine pair move; joint acceptance) and
    trans-grove crossover; why both are prior-preserving relabelings; reduction
    to the PT swap for a whole-tree swap. [THEORY §2; PLAN §2.9]
8.3 **The shared prior across the ladder.** One Θ for all temperatures ⇒ the
    population is a tempered ladder of *one* self-consistent posterior. State the
    modelling consequence (the limit is the self-consistent posterior, not the
    fixed-α₀ one) — forward-ref to §9. [PLAN §2.9 R1]
8.4 **Random-scan pair selection** (one consecutive pair per step; reversibility
    per step). [PLAN §2.9 R3]

## 9. Adaptation and ergodicity — the hard formal core
9.1 **Two adaptations, kept separate.** (i) adaptive move-weights (Phase 2.8);
    (ii) the adaptive prior (the urn). State that they must be analysed under one
    umbrella (both make the kernel time-inhomogeneous).
9.2 **Move-weight adaptation.** Diminishing adaptation (γ_t → 0) + containment
    (bounded logits + ε-floor) ⇒ ergodic by Roberts-Rosenthal (2007). The
    birth/death pair-equality keeps the move-selection correction identically
    zero under adaptation — state as a preserved invariant. [PLAN §2.8]
9.3 **The adaptive prior as stochastic approximation / self-interacting MCMC.**
    Frame the urn-reinforced prior as a Robbins-Monro recursion on Θ coupled to
    the tree chain — a self-interacting Markov chain (Del Moral-Miclo; Andrieu et
    al.). This is the novel object.
9.4 **What does it converge to?** The central question (see §0 decision).
    Candidate limits: the fixed-α₀ posterior vs a self-consistent fixed point Θ*
    solving Θ* = 𝔼_{π(·|Θ*)}[reinforcement]. State precisely which is claimed.
9.5 **The Σγ_t = ∞ subtlety.** γ_t = 1/(1+t) ⇒ total urn mass diverges, so Θ_t
    need not converge to a point; but the *predictive* prior may still stabilise
    and, crucially, diminishing adaptation (per-step change → 0) + containment
    still give convergence of ergodic averages. State the theorem we can prove
    and clearly label what remains conjectural. Do NOT overclaim a
    "self-consistent posterior" limit without the argument. [audit (l); PLAN §1.2]
9.6 **The frozen-prior regime.** Freezing Θ after warm-up gives a clean,
    standard Bayesian posterior with the §7 ergodicity — present this as the
    rigorous baseline and as the recommended default for reproducible inference.
    The MPI deterministic-replay mode already realises a delayed/frozen variant.
    [PLAN §2.12]
9.7 **Joint containment.** Sketch why the combined (move-weight + prior +
    tempering) adaptive scheme satisfies diminishing adaptation + containment
    together.

## 10. Extensibility — custom atomic primitives (Phase 2.10)
10.1 **Motivation.** Domain primitives (e.g. A·sin(ωx), a damped mode) as single
     opaque parametric terminals — indivisible basis functions.
10.2 **Grammar extension and opaque-leaf semantics.** A user primitive is one
     leaf carrying ≥1 internal parameter; structural moves treat it as a unit
     (never split/hoist into it); its parameters are tuned only by
     parameter-refresh. [PLAN §2.10]
10.3 **Prior mass for opaque primitives — RESOLVED (2026-07-09), see
     `DESIGN_2_10_PRIMITIVES.md`.** The rigorous construction: each node's
     choice law is a DP(c, G₀) over ops ⊔ a measurable primitive space Ψ
     stratified by parameter dimension; the implementation's finite
     Dirichlets are its finite-dimensional projections onto
     {registered atoms} ∪ {remainder cell}; registration = partition
     refinement CARVING mass c·G₀({ψ}) out of the remainder (NOT appending
     α₀ — the current append scheme makes the prior drift with vocabulary
     size and must be replaced). Registration-invariance theorem via
     Dirichlet aggregation; sampler restriction via neutrality; parameter
     leaves unify as the k=1 constant primitive; k_ψ enters only the BIC
     dimension count. HDP named as the deferred canonical extension.
10.4 **Rigor.** Show the extension is a special case of the §4 prior (a primitive
     set with proper prior mass), so no new correctness argument is needed beyond
     assigning the mass — a clean generality claim.

## 11. Implementation
- Cython module architecture (function_treeC / real_functionC / gp_movesC /
  polya_treeC / groveC / mpi_worker); the BIC topology cache and its role in
  making θ̂ a deterministic statistic [§5.4]; RNG plumbing and determinism
  guarantees; the MPI/EMC spawn model and deterministic replay; numerical
  guards (NaN/inf sentinels, warning suppression). Keep it reproducible-first.

## 12. Experiments
12.1 **Sanity / recovery.** Recover known functions; check posterior
     concentration on the true structure; report posterior over structures, not
     just the MAP. [PLAN §4.1]
12.2 **Ablations (isolate each contribution).** Remove the Pólya prior (flat
     structural prior), remove tempering, replace BIC with raw fit — show each
     hurts, quantifying what the prior/tempering/marginal each buy. [PLAN §4.3]
12.3 **Hero application.** One real problem (CCSNe / chirp-mass / Feynman —
     choose). [PLAN §4.2]
12.4 **Diagnostics.** Swap-acceptance rates per rung; ESS over structures;
     adaptation traces (Θ and move-weights vs t); convergence / R-hat analogues
     for a discrete state space; the two-state occupancy regression check as a
     correctness demonstration [THEORY §1.7].

## 13. Limitations and future work
- Full Laplace (−½log|H|) instead of BIC; pseudo-marginal MCMC for optimizer
  noise; multi-start / cross-tree warm-start; near-unidentifiability and
  effective parameter counts; the gaps in the adaptive-limit theory (§9.5);
  unbounded-depth propriety (§4.3).

## 14. Conclusion
- Restate the three convictions of §0 as delivered; position the adaptive
  structural prior as the transferable idea beyond SR (any grammar-structured
  Bayesian model search).

## Appendices
- **A.** Full per-move proposal-density derivations (lift + complete
  [THEORY §1–3]).
- **B.** BIC-as-Laplace derivation with the regularity conditions.
- **C.** Ergodicity: irreducibility lemma proof; adaptive-MCMC containment
  arguments; the stochastic-approximation statement for the urn.
- **D.** Reproducibility: hyperparameters, seeds, ladder, hardware, BLAS pinning.

---

## Writing order (recommended, ≠ reading order)
1. Nail the **§0 central decision** (fixed vs adaptive framing) — everything
   keys off it.
2. Draft **§3–§7** first (setup → prior → likelihood → target → kernel): this is
   the load-bearing formal spine and most of it exists in THEORY_2_13.md.
3. Then **§9** (the hard adaptive analysis) while the spine is fresh — this is
   where genuinely new theorem-writing (or honest conjecture-labelling) happens.
4. Then **§8, §10** (population moves; extensibility).
5. **§12** experiments once (a)–(e) are verified and Phase 4 has run.
6. **§1, §2, §13, §14, abstract** last.

## Dependencies on the code roadmap (PLAN.md)
- §7 correctness prose is unblocked only after Phase 2.13 (a)–(f) verify.
- §10 needs the Phase 2.10 design decision (10.3) — do that *in this document's
  notation* before implementing.
- §12 is gated on Phase 4, itself gated on 2.13.
- §9.5/§4.3 need the two theory items still open in PLAN §1.2 / audit (l).
