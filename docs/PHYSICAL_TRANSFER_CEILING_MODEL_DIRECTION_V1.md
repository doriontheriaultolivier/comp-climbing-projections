# Physical capacity → board expression → competition expression

Date: 13 August 2026  
Status: research direction; no individual transfer or pressure estimate released

## Decision

The proposed ceiling interpretation is a useful coaching hypothesis, but it is
not yet an identifiable causal model. Physical tests, board choices, training
volume, injury, style preference, competition access and stress are all partly
selected and partly unobserved. A stochastic-frontier fit on the current sparse
sample would risk relabelling measurement error and selection as athlete
"inefficiency."

The next executable model should therefore be a modest predictive baseline:

1. represent board grades as ordered categories, not equally spaced numbers;
2. predict ordered competition-item outcomes (`no score < Zone < Top`) using
   athlete and problem hierarchy;
3. let the two board indicators enter through separate monotone effects;
4. preserve uncertainty and repeated athlete/problem/event dependence; and
5. call the individual residual a **conditional expression discrepancy**, never
   proof of a pressure, motivation, technical or transfer deficit.

Physical-test features enter only in a later challenger after their repeated
measurement, missingness and reliability are adequately represented.

## Three different questions and clocks

| Question | Information allowed | Honest use |
|---|---|---|
| Pre-event projection | Only measurements, board observations and competition history available before the event | Predict expected field or round performance without knowing future boulders |
| Post-event coaching description | Pre-event athlete evidence plus the observed competition problems, human-reviewed demand tags and outcomes | Ask on which observed terrain the athlete over- or under-expressed relative to comparable evidence |
| Causal training decision | Would require stronger intervention, confounding and adherence information | Not identified by the present observational database |

Problem demand tags created after a competition are valid for retrospective
item description. They must not be backdated into a pre-event forecast. A model
may use tags from earlier competitions to learn general relationships, but it
cannot use the target event's unseen terrain unless the product is explicitly a
post-event analysis.

## What Gemini contributed—and what was changed

One bounded `gemini-3.1-pro-preview` review recommended a monotone ordinal IRT
baseline and rejected a frontier, GAM frontier and causal mediation model for
the present data. That core recommendation is accepted.

Several details are deliberately not adopted:

- "tag 50 problems" is a useful planning scale, not a statistical sufficiency
  threshold; readiness must be reported from the resulting athlete/item/event
  support and registered before scoring;
- physical and board V grades are ordered labels, but the review's claim that
  their physical demand is exponentially spaced was unsupported;
- max hang and pull-up were examples, not justified choices for the two tests to
  standardize; selection should follow reliability, repeatability, coverage and
  coaching relevance;
- problem tags should not automatically become structural priors; independently
  reviewed tags can support interactions and stratified checks;
- an individual "85% stress/transfer" statement is too strong. Even a posterior
  discrepancy remains conditional on observed comparators and selected access.

The call used 5,114 total tokens (962 prompt, 2,004 thinking, 2,148 answer), sent
no athlete names or private rows, and had no promotion authority. Its response
SHA-256 is `23b96fe122f5177849f2935d431a580fc3b8eeac98734d0de1b2f701e5cdfce3`.

## Finite implementation sequence

### Gate 0 — chronology and semantics

- Materialize a row contract that binds each physical/board observation to its
  actual observation date and each outcome to its event/problem identity.
- Keep `50%-flash` and `recent three hardest physical sends` separate.
- Preserve `Zone`, `Top` and certified attempt ordinals without inventing failed
  attempts from unresolved counters.
- Record tag submission time and whether a feature is valid for pre-event or
  post-event use.
- Missing tests remain missing; no zero-fill or weakness label.

### Gate 1 — human problem review

- Continue the existing information-value queue rather than choosing convenient
  examples.
- Require exact item identity and reviewer provenance.
- Report independent-review agreement and support by athlete, problem, event,
  age category and competition level.
- Do not declare model readiness from an item count alone.

### Gate 2 — simple chronology-safe baselines

- Compare an intercept/item-difficulty baseline, a prior-competition baseline,
  and a regularized ordinal board model.
- Use expanding-time or walk-forward evaluation; never random row folds.
- Normalize pair/item contributions within competition and cluster uncertainty
  by competition and athlete.
- Score ordered log loss, Top/Zone Brier scores and calibration; simulate full
  fields only when the item model and target are coherent.
- Pre-register youth/senior, domestic/international, source, evidence-availability
  and missingness guards. Treat these as transport strata, not biology.
- Include future-board-date and shuffled-board negative controls to expose
  leakage or a non-temporal association.

### Gate 3 — physical-capacity challenger

Only after Gate 2 is numerically useful, introduce a latent physical-capacity
layer with metric-specific measurement error, repeated-test drift and partial
pooling. Compare it against the board-only model. A ceiling/frontier challenger
is worth testing only if repeated data can distinguish a one-sided accessibility
process from ordinary symmetric residual variation.

## Coaching output contract

The product may show:

- the dated physical observations;
- the two dated board-expression indicators;
- expected and observed outcomes on reviewed competition terrain;
- a conditional discrepancy distribution with comparator and support counts;
- plausible alternative hypotheses and the next observation that would
  distinguish them.

It must not show a definitive "pressure problem," "poor transfer," "physical
limiter" or prescribed training allocation from this observational model.
When evidence is sparse, the useful result is a data request or competing
hypotheses—not a low athlete score.

## Current action

**GO** for the chronology/semantic panel and continued problem tagging.  
**NO-GO** for an athlete-specific ceiling, transfer or pressure model today.

