# Pathway demonstrated and readiness heads V1

Status: research contract; no app, production, rating, or deployment authority.

## Product vocabulary

- **Overall**: shared connected-graph level.
- **YW-IFSC**: demonstrated Youth World level in the athlete's governed event-date category.
- **YW-IFSC Ready**: projected level in a reference Youth Worlds field for the athlete's current event-date category.
- **REG-IFSC**: demonstrated senior/open regional or continental IFSC level.
- **REG-IFSC Ready**: projected level in a reference senior/open regional IFSC field.
- **WC+**: demonstrated senior World Cup, World Championships, and governed Olympic-pathway level.
- **WC+ Ready**: projected level in a reference WC+ field.

`Elo` remains a technical method name, not a required product label.

## Evidence rules

All governed competition results may update the shared graph. Direct evidence is
stricter: youth results cannot populate REG-IFSC or WC+, and regional results
cannot populate WC+. Prior-category youth results remain indirect graph evidence
after an athlete changes category.

Eligibility and category use the rule applicable on the event date. They are not
derived from a blanket under-18 test, and source labels are not trusted without
the governed age-class and event-tier fields.

## Evidence found in the verified V4 replay

The old `ifsc_non_wc` component contains 72 event-pools: 44 senior/open regional
or continental event-pools and 28 youth regional event-pools. Renaming that
component to REG-IFSC would therefore be incorrect. The replay separately has 6
Youth Worlds event-pools and 34 WC+ event-pools.

The new routing artifact separates all governed events without ambiguity:

- 44 `REG_IFSC`;
- 28 `YOUTH_REG_IFSC_GRAPH_ONLY`;
- 6 `YW_IFSC`;
- 34 `WC_PLUS`;
- 1,156 other shared-graph event-pools.

## Current release gates

The verified graph already yields a connected WC+ readiness coordinate for
8,408 athletes, including 7,810 athletes with zero direct WC+ starts. A zero
start count is no longer a withholding reason. The value is absent only when no
anchored graph connection exists. Absolute intervals remain withheld because
the predecessor covariance calibration gate failed.

Amari Bourbonnais is a concrete test: despite zero direct WC+ starts, her graph
has 10 anchored events, 146 anchored comparisons, 62 unique anchored opponents,
and 34.1 effective anchored weight. Her research-only WC+ Ready coordinate is
2108.6. This demonstrates that her NACS, Senior Nationals, Pan-American and
shared-opponent evidence are usable; the previous blank was a policy artifact,
not an information absence.

YW-IFSC Ready is not released from V4 because V4 pooled all Youth World age
categories within sex. REG-IFSC Ready is not released because V4 pooled youth
and senior/open regional offsets. Both require a category-aware chronological
replay. The pooled Youth Worlds coordinate is retained only as an internal
sensitivity and cannot be displayed as current-category readiness.

## Next mathematical gate

Replay distinct target domains `yw_ifsc`, `youth_reg_ifsc`, `reg_ifsc`, and
`wc+`, while retaining a shared latent skill base. Compare the split model with
the pooled predecessor using rolling chronological evaluation: 2024 development,
2025 validation, and 2026 descriptive reuse. Pairwise log loss and placement RPS
are co-primary. Advancement, calibration, current-category youth, adult/youth,
source, pathway, age-transition, and sparse-evidence cells are harm guards.
Context offsets must shrink toward their parent/shared effect when support is
weak; they are not separate disconnected rankings.
