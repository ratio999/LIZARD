# Review findings — §9 two-regime / adaptive-prior exposition

Reviewed 2026-07-14 against `PROMPT_REVIEW_SEC9.md`. Files read in
full: `sections/sec9_adaptation.tex`; swept for regime language:
`sec4_prior.tex`, `sec6_target.tex`, `sec8_population.tex`,
`sec12_results.tex`, `sec2_related.tex`, `sec11_userdoc.tex`,
`paper_preamble.tex`. **Nothing was applied**; every item below is a
quoted-old → proposed-new suggestion for the author.

Verdict key: **MUST-FIX** (contract violation or broken output),
**fix proposed** (correct but improvable / small gap), **OK** (verified,
no finding).

---

## 1. Contract violations (silent borrowing) — 1 must-fix, 2 fixes

### F1 — MUST-FIX: §9.6 attributes the experiments to the frozen regime; they ran adaptive

- **Evidence:** `sec9_adaptation.tex:607–609`:
  > "It is the configuration under which the experiments of
  > \secexperiments\ report their headline results."
  contradicted by `sec12_results.tex:99–101`:
  > "The adaptive prior ran live throughout (no freeze), so these runs
  > also exercise the continuously adaptive regime of \secadaptation."
- **Why it matters:** this is the exact leak the contract forbids, in
  its most damaging direction — it silently wraps adaptive-regime
  experimental numbers in the frozen regime's Theorem
  (`thm:frozen`) + CLT guarantee chain. A referee who reads both
  sections will call it an inconsistency; one who reads only §9 will
  credit the experiments with guarantees they do not have.
- **Proposed rework** (replace the quoted sentence):
  ```latex
  The preliminary experiments of \secexperiments\ deliberately run the
  \emph{continuously adaptive} regime instead --- reinforcement is
  never frozen in those runs --- so their results do not lean on this
  theorem; they exercise the harder case and generate exactly the
  adaptation traces the scoreboard below prescribes monitoring. The
  frozen regime remains the recommended configuration whenever the
  end-to-end guarantee chain is wanted.
  ```

### F2 — fix proposed: §8.3 states the adaptive ladder "is" a family of the self-consistent object

- **Evidence:** `sec8_population.tex:359–362`:
  > "under continuous adaptation, the ladder is a tempered family of
  > the \emph{learned}, self-consistent object, not of the
  > fixed-$\alpha_0$ posterior --- a statement about what the model
  > is, not a defect of the sampler."
- **Why it matters:** "self-consistent object" is §9's *conjectural*
  limit (Proposition `prop:fixedpoint` under Assumptions 1–2 +
  Conjecture). Stating the ladder "is" a tempered family of it, with
  no conditional marker, borrows the conjecture as fact. (The
  "modelling statement" framing mitigates but does not mark the
  epistemic status.)
- **Proposed rework** (minimal insertion):
  ```latex
  under continuous adaptation, the ladder is a tempered family of the
  \emph{learned} prior's posterior --- conjecturally, of the
  self-consistent object of \secadaptation\ (under the assumptions
  stated there), and in no reading of the fixed-$\alpha_0$ posterior
  --- a statement about what the model is, not a defect of the
  sampler.
  ```

### F3 — fix proposed: §8.3 uses "self-consistent" for the *fixed-Θ* ladder (term-of-art collision)

- **Evidence:** `sec8_population.tex:300–301`:
  > "the population is a tempered ladder of one self-consistent
  > object"
  — here the claim is the *unproblematic* fixed-Θ statement (one
  shared urn ⇒ one posterior powered through β), yet it reuses the
  §9 term of art for the conjectural limit. Readers will conflate the
  two.
- **Proposed rework:** replace "one self-consistent object" by
  "a \emph{single} posterior" (the sentence already says exactly
  this in its own words two lines earlier; the term-of-art is
  unnecessary here).

### Sweep result for everything else — OK, no finding

- `sec4_prior.tex:771–781` (the two-regime passage): correctly
  conditional; restates the contract verbatim. Clean.
- `sec6_target.tex:83–89`: "what `sampling the posterior' means is
  precisely the subject of \secadaptation" — clean deferral.
- `sec2_related.tex:355–362`: presents the ledger as
  proved/conjectured split; clean.
- `sec12_results.tex`: no frozen-CLT language applied to the runs;
  "the posterior selects and retains" (scope subsection) is
  descriptive of the pipeline and is disambiguated by the explicit
  regime statement at lines 99–101 two paragraphs earlier.
  Acceptable as is.

---

## 2. The ledger — completeness and labelling

### F4 — fix proposed: "per-step validity" is not a visible ledger item

- **Evidence:** the scoreboard (`sec9_adaptation.tex:649–671`) opens
  its Proved list with fixed-kernel ergodicity; nowhere in §9 is
  "each executed step is exact MH for the *current* target
  π_β(·|Θ_t)" stated as a result. It is true and provable — it lives
  in Proposition `prop:cancel`'s proof parenthetical
  (`sec4_prior.tex:334–341`: both prior evaluations inside one ratio
  use the same Θ_t) plus Theorem `thm:ergodicity` at fixed Θ — but a
  referee reconstructing the ledger from theorem environments will
  not find it.
- **Proposed rework** (prepend one bullet to the Proved list):
  ```latex
  \item \textbf{Proved.} Per-step exactness: at every $t$ the
        executed step is exact Metropolis--Hastings for the current
        target $\pi_\beta(\cdot \mid \Theta_t)$ --- both prior
        evaluations inside one acceptance ratio use the same
        $\Theta_t$ (Proposition~\ref{prop:cancel}), and
        reversibility at fixed $\Theta$ is
        Theorem~\ref{thm:ergodicity}. Further: fixed-kernel
        ergodicity (...continue existing bullet...)
  ```
  (or as its own bullet ahead of the existing one).

### F5 — fix proposed: Assumption `ass:averaging` lacks an explicit "why unproven"

- **Evidence:** `sec9_adaptation.tex:349–368`. The two-time-scale
  rationale (plausibility) is present and quantitative; the reason it
  remains unproven is only implicit ("asserts that this heuristic
  decoupling actually holds").
- **Proposed rework** (append inside the parenthetical):
  ```latex
  Proving it would require ergodicity bounds uniform along the
  realised trajectory --- precisely the containment question of
  \S\ref{sec:gammadivergence} --- which is why it is an assumption
  and not a lemma.)
  ```

### Everything else in the ledger — OK, no finding

- PROVEN items all present, unconditional, correctly typed:
  `lem:da` (with explicit rate O(γ_t), lines 465–526),
  `lem:predrecursion` (244–270), `lem:norawlimit` (433–455, and the
  consequence — predictives, not raw counts, are the objects of
  convergence statements — is drawn *inside the lemma statement*,
  lines 438–442), `thm:mwergodic` (169–216), `thm:frozen` (577–596)
  with the downside-capping framing present at 563–565.
- ASSUMED: `ass:qconv` (335–347) is a numbered assumption with
  interiority, plausibility, and the full "THE unproven step"
  disclosure at 411–417 (Pemantle citation + the three named
  obstacles: state-dependent steps, self-interaction through a
  trans-dimensional kernel, growing index set). Complete.
- CONDITIONAL: `prop:fixedpoint` (370–399) carries its hypotheses in
  the statement. **The section's "soul" sentence survives intact**
  (`sec9:407–411`): "the continuously adaptive sampler is not a noisy
  approximation of the symmetric-prior posterior; it targets
  (conjecturally) a different, arguably more interesting object." No
  weakening edit detected.
- CONJECTURED: `conj:containment` (545–557) is in a conjecture
  environment; the "what stands on each side" inventory (559–569) is
  complete and accurate (proved: DA, recursion, conditional
  fixed-point, frozen cap; unproved: containment + predictive
  convergence; monitored: traces).

---

## 3. Tone calibration — 1 fix, otherwise no finding

### F6 — fix proposed: an unsupported empirical quantifier in the frozen regime's favour

- **Evidence:** `sec9_adaptation.tex:604–607`:
  > "captures most of the adaptive prior's practical benefit while
  > forfeiting only the self-consistency interpretation"
  — "captures most of the practical benefit" is an empirical claim;
  no frozen-vs-live comparison exists anywhere in the project. This
  is the mirror image of the leak the prompt warns about: the
  recommendation must stay guarantees-based.
- **Proposed rework:**
  ```latex
  retains the prior exactly as learned up to the freeze while
  forfeiting only the self-consistency interpretation of
  \S\ref{sec:limitobject}
  ```

### Otherwise — no finding

- No apologetic hedges ("we hope", "heuristically one expects" as a
  substitute for conditional statements) anywhere in §9; the one
  "heuristics say" (line 309) is immediately followed by "We now
  write that statement down precisely", which is the correct move.
- The adaptive regime's proven lemmas are stated in clean,
  disclaimer-free environments; disclaimers sit on the conjecture and
  the assumptions, where they belong. No frozen/adaptive asymmetry in
  register detected.
- The "recommended default" sentences (quote block lines 69–72;
  `sec4:776–777`) are guarantees-based ("every guarantee of §7 holds
  verbatim"), never performance-based — except the F6 sentence above.

---

## 4. The concrete defects of prompt item 4

### F7 — MUST-FIX: the "(Section ??)" broken reference — located, one-line fix

- **Evidence:** `paper_preamble.tex:66`:
  ```latex
  \newcommand{\secfrozen}{Section~\ref{sec:frozen}}
  ```
  The label `sec:frozen` does not exist anywhere; the frozen-regime
  subsection's label is `sec:frozenregime`
  (`sec9_adaptation.tex:573`). Every `\secfrozen` site renders
  "(Section ??)": `sec4_prior.tex:37` and `:777` (the "Two operating
  regimes follow from one definition..." passage named in the
  prompt), `sec6_target.tex:83`, `sec8_population.tex` (two sites),
  `sec11_userdoc.tex:570` and `:1155`.
- **Proposed rework** (single point, fixes all sites):
  ```latex
  \newcommand{\secfrozen}{Section~\ref{sec:frozenregime}}
  ```

### F8 — fix proposed: non-uniqueness / run-dependence of the limit is nowhere acknowledged

- **Evidence:** `prop:fixedpoint` and its caveat paragraph
  (`sec9:401–422`) are silent on uniqueness; the proposition claims
  only that *any* limit is a fixed point, and classic urn behaviour
  makes a random, run-dependent limit plausible. Per the prompt, this
  is the one permitted new-content proposal — **one sentence**, to
  follow the Pemantle caveat at line 417:
  ```latex
  Nor does Proposition~\ref{prop:fixedpoint} assert uniqueness: the
  fixed-point equation may admit several self-consistent families,
  and classic urn behaviour makes a random, run-dependent limit
  entirely plausible --- under Assumption~\ref{ass:qconv} the
  trajectory settles on \emph{a} self-consistent pair, not
  necessarily the same one on every run.
  ```

### F9 — OK: the population / shared-urn caveat is present and consistent

- `sec9_adaptation.tex:417–422` (F read with the temperature mixture)
  and `sec8_population.tex:356–358` (innovation law is a mixture over
  temperatures; F "must be (and is) read with that mixture") agree.
  No finding — modulo F2's conditional-marker fix in the adjacent
  sentence.

### F10 — OK: sec12 states which regime the experiments run

- `sec12_results.tex:99–101`, quoted under F1. Present, correct
  (vocabulary frozen, Θ not — the two senses of "frozen" are kept
  distinct), and it is the sentence F1's fix should be made
  consistent with. No further fix needed in sec12.

---

## Out-of-scope defect noticed during the sweep (reported, not part of §9)

- `sec2_related.tex:350`: `\cite{robbins1951stochastic,
  benaim2006dynamics}` — the second key does not exist; the
  bibliography entry (sec9 header block) is `benaim1999dynamics`.
  Renders as an undefined citation. Proposed fix: change the key to
  `benaim1999dynamics`.

---

## Summary

| # | Verdict | Where | One-line description |
|---|---------|-------|----------------------|
| F1 | MUST-FIX | sec9:607–609 | Experiments falsely attributed to frozen regime |
| F2 | fix proposed | sec8:359–362 | Conjectural limit stated as fact ("is") |
| F3 | fix proposed | sec8:300–301 | Term-of-art collision ("self-consistent" for fixed-Θ ladder) |
| F4 | fix proposed | sec9:649–671 | Per-step exactness missing from the scoreboard |
| F5 | fix proposed | sec9:349–368 | Averaging assumption lacks explicit "why unproven" |
| F6 | fix proposed | sec9:604–607 | Unsupported "captures most of the benefit" quantifier |
| F7 | MUST-FIX | preamble:66 | `\secfrozen` points at nonexistent label → "(Section ??)" |
| F8 | fix proposed | sec9:~417 | Add the one non-uniqueness sentence |
| F9 | OK | sec9/sec8 | Shared-urn mixture caveat present + consistent |
| F10 | OK | sec12:99–101 | Regime of the experiments correctly stated |
| — | out of scope | sec2:350 | `benaim2006dynamics` → `benaim1999dynamics` |

The contract ("prove what we can for the second regime, never
silently borrow") is **structurally kept** — the ledger exists, is
correctly typed, and the soul sentence is intact — with exactly one
substantive violation (F1), one rendering defect (F7), and a handful
of small hardenings. The two must-fixes are one sentence and one
line respectively.
