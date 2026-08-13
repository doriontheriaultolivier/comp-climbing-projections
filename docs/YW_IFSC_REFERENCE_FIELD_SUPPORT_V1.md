# YW-IFSC reference-field support v1

## What this closes

The proposed `YW-IFSC Ready` product is conceptually useful, but it cannot be
built by treating historical youth labels as permanent age categories. This
audit tests the smallest defensible alternative against the independently
recomputed V4 graph:

1. `shared graph`: pre-event skill plus current form;
2. `shared graph + pooled YW`: the same state plus the athlete-specific pooled
   Youth Worlds context component; and
3. literal prior-season category fields, preserved exactly as published and
   never translated through age or name heuristics.

The probabilities use the V4 link: display scale 400, event-performance SD
155, and the exact covariance of each pre-event component sum. Only
qualification fields are scored. Later rounds are conditional survivor fields
and are not mixed into the entry-readiness target.

## Evidence available

The verified replay contains six Youth Worlds event-pools, but only three
independent competitions: Guiyang 2024, Helsinki 2025 and Arco 2026. Their 14
literal qualification fields contain 837 athletes and 46,876 canonical pairs.
The label structure changes materially:

- 2024: `Youth B`, `Youth A`, and `Juniors`, separately by sex;
- 2025–2026: `U17` and `U19`, separately by sex.

There are 225 athletes observed in more than one Youth Worlds year. Those
transitions are useful empirical continuity evidence. They are not rule
aliases. In particular, a 2025 U17 athlete can appear in either U17 or U19 in
2026, depending on the athlete's age progression.

The dated official youth-rule registry has 54 federation-specific rows but no
IFSC rule rows and no reviewed IFSC event-scoped category alias certificates.
It therefore activates no Youth Worlds label translation. Observed ages are
reported only as diagnostics and never used to manufacture a category.

## Chronological result

Pair losses are averaged within each qualification field and then equally
across fields. Negative deltas favor the pooled Youth Worlds component.

| Evidence year | Role | Log-loss delta | Brier delta | Interpretation |
|---|---|---:|---:|---|
| 2024 | Development / first YW update | -0.000426 | -0.000111 | Descriptive only; the YW component is initially unlearned |
| 2025 | Chronological test | +0.000197 | -0.000102 | Mixed and practically negligible |
| 2026 | Reused descriptive evidence | -0.000909 | -0.000306 | Directionally favorable, but not a fresh lock |

The 2025 result does not justify a public YW-specific adjustment. The 2026
result is enough to retain the pooled component as a research challenger, but
not enough to select it. Pair counts do not replace the three independent
competition clusters.

## Product decision

- Keep `YW-IFSC` as a demonstrated-level head for direct Youth Worlds results.
- Keep `YW-IFSC Ready` as the intended product name and estimand.
- Do not publish a pooled YW adjustment yet.
- Do not create category-specific response parameters from these three events.
- A 2025 literal U17/U19 field may be a reference-field prototype for a 2026
  descriptive reconstruction; it is not validated production evidence.
- Zero direct Youth Worlds starts is not a reason to withhold readiness. A
  future readiness estimate should be available from connected graph evidence,
  just as WC+ Ready is, once the reference-field model is validated.
- Current-category publication requires dated IFSC rule evidence, reviewed
  event aliases, a coherent joint field simulator, and a fresh 2027 test.

The next model should therefore have one shared Youth Worlds context response
with strong shrinkage, while category affects only the governed reference
field and resulting joint placement distribution. It should not fit a separate
latent rating system for each sparse age label.

## Evidence chain

The generated receipt under
`.artifacts/yw-ifsc-reference-field-support-v1/receipt.json` binds the exact V4
prepared rows, replay history, full covariance trace, replay plan, external
dated-rule registry and every generated table. The restricted athlete-level
transition table remains in ignored artifacts; the committed documentation
contains only aggregate evidence.

Status: `RESEARCH_ONLY_CURRENT_CATEGORY_READY_NOT_IDENTIFIED_FOR_PUBLICATION`.
No rating, identity, app, release or deployment artifact is changed.
