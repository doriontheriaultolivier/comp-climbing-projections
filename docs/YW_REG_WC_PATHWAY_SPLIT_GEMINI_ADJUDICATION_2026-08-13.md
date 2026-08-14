# Adjudication of the Gemini pathway review

Status: research design decision only. No model, readiness value, interval, app,
or release is promoted by this document.

## Evidence binding

This adjudication covers the single `gemini-3.1-pro-preview` review recorded in
`YW_REG_WC_PATHWAY_SPLIT_GEMINI_PRO_REVIEW_2026-08-13.md`. The receipt binds the
review to prompt SHA-256
`3cc6b940dbcd7f84dc532ed14c83b3a7e8e526e18dd70fc68280efdb6dac6440`
and review SHA-256
`0e28d54f9cb66f1b3198af6bdb5e2910f167566cc18f7a0ed87da4b836e91421`.
The call used 5,758 tokens and has no promotion authority.

## Accepted

1. `YW-IFSC Ready - current category` is a posterior/predictive target against
   a governed reference field. It is not a category-specific raw rating.
2. Candidate B is the smallest defensible first experiment: retain a shared
   Youth Worlds context response, but compute category-specific placement and
   top-k probabilities from a joint field simulation.
3. Direct demonstrated evidence and projected readiness are different product
   concepts. Zero direct starts does not imply zero information.
4. An athlete's shared state persists through an age-category transition; the
   governed target reference field changes at the eligibility boundary.
5. Category-specific latent heads are too fragmented for the current support.
6. A hierarchical youth/senior context model is a worthwhile later challenger,
   but it must earn its complexity chronologically.
7. A global-only pre-2024 initializer is not mathematically invalid merely
   because downstream context offsets are split. Rebuilding remains necessary
   if the identity-correct source rows, routing, or graph edges differ.

## Corrected or rejected

1. **Sparse-prior direction.** The review recommends widening independent
   priors for sparse contexts. That increases weakly identified context motion.
   Sparse contexts must instead shrink more strongly toward a governed parent
   or global response. A wider prior is permitted only as a sensitivity, never
   as the default sparse-data remedy.
2. **Arbitrary support cliffs.** Suggested thresholds of ten events and one
   hundred cross-context athletes were not derived from this panel. Support
   must be reported continuously and the operational threshold preregistered
   from effective competition clusters, connectivity, and stability. No
   threshold may be tuned on 2025 outcomes.
3. **Validation test.** A paired permutation test alone is insufficient because
   fields within a source competition and date are correlated. Proper-score
   deltas must be clustered by whole competition and, as a sensitivity, source
   date. Pair rows never form the resampling unit.
4. **Uncertainty display.** The proposed credible-rank band is not currently
   authorized. The existing absolute covariance calibration failed. Until an
   interval has chronological empirical coverage, the UI may show only an
   evidence/support badge and point/probability sensitivity explicitly labeled
   research-only.
5. **Reference field as fixed semantics.** Eligibility rules are semantic, but
   field composition and weighting are modeling choices. They must be frozen
   using only information available before the target season/event and tested
   for robustness; they are not automatically valid product definitions.
6. **Ready estimand.** Expected rank alone is field-size dependent and can hide
   tail behavior. The primary output should be a coherent placement
   distribution with probabilities such as semifinal/top-k plus placement RPS
   for evaluation. Expected or median rank is a summary, not the estimand.

## Resulting staged design

### Stage 0: support and transition preflight

Before replay, inventory Youth Worlds event-date categories, rule provenance,
competition clusters, athlete overlap, transitions between youth categories,
and connections to youth-regional, senior-regional, and WC+ contexts. Freeze a
reference-field construction protocol using pre-target information only. If a
category cannot be governed or connected, its current-category readiness is
withheld rather than guessed.

### Stage 1: executable Candidate B

- one shared dynamic athlete state and form;
- separate direct routing for `yw_ifsc`, `youth_reg_ifsc`, `reg_ifsc`, and
  `wc_plus`;
- a shared Youth Worlds response, not separate age-category offsets;
- current-category reference fields built under the frozen Stage-0 protocol;
- joint rank simulation yielding a coherent placement distribution and top-k
  probabilities;
- pooled predecessor and global-only control evaluated on identical fields;
- 2024 development, 2025 one-time validation, 2026 descriptive only;
- competition-clustered proper-score deltas and predeclared adult/youth,
  category-transition, source/pathway, evidence, and connectivity guards.

### Stage 2: hierarchical challenger only if Stage 1 exposes residual structure

Test `Global -> {Youth, Senior/Open} -> {Youth Worlds, Youth Regional,
Senior Regional, WC+}` with shrinkage learned only inside chronological
development folds. A smooth age trajectory is added only if within-athlete
transitions and access-aware diagnostics establish support. Age or category may
never be inferred from current survivors alone.

## Product contract retained

The default athlete view should expose one selected target at a time:

- target context and current governed category;
- readiness placement/top-k distribution summary;
- `demonstrated` badge only with direct target-context evidence;
- `projected` badge for graph-connected indirect evidence;
- counts of direct starts, anchored competitions/opponents, and recency;
- `not estimable` for structural disconnection or unsupported target transfer;
- no numeric interval until coverage is validated.

This supports an Amari-like case: no WC+ demonstrated rating is honest, while a
graph-anchored WC+ Ready estimate is useful and must not be blank solely because
direct WC+ starts are zero.
