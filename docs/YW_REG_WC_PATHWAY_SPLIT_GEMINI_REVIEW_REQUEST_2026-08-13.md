# Independent methods review request: YW-IFSC, REG-IFSC and WC+ heads

Status: sanitized research-design packet. It contains no private measurements,
credentials, names, or deployable model parameters. Advice cannot promote a
model.

## Product question

The coach-facing product should distinguish:

- `Overall`: shared connected-graph level;
- `YW-IFSC`: directly demonstrated Youth Worlds level;
- `REG-IFSC`: directly demonstrated senior/open regional or continental IFSC
  level;
- `WC+`: directly demonstrated World Cup / World Championship level; and
- a separate `Ready` estimate for each target context.

Youth and domestic results should still influence every readiness estimate via
the shared graph. They must not be mislabeled as direct senior/open or WC+
evidence. A zero direct-start count must not itself cause withholding when an
athlete has anchored graph connections.

## Evidence defect found

The verified dynamic replay currently has a shared latent skill, form, and
context offsets. Its legacy `ifsc_non_wc` offset mixes:

- 44 senior/open regional or continental event-pools; and
- 28 youth regional or continental event-pools.

It separately contains 6 Youth Worlds event-pools and 34 WC+ event-pools. The
replay covers 2024-2026. Pre-2024 history supplies the global graph initializer.
The event router now separates all 112 governed event-pools with zero ambiguous
age-class cases, but the model has not been refit with split targets.

The existing Youth Worlds offset pools youth categories within sex. It therefore
cannot honestly be labeled `YW-IFSC Ready · current category` without deciding
how category and maturation enter the estimand.

## Current useful result and remaining gates

The independently recomputed WC+ graph has 8,408 anchored athlete states,
including 7,810 with zero direct WC+ starts. This proves indirect readiness is
not structurally impossible. Absolute intervals remain withheld because the
predecessor covariance-calibration gate failed.

The current pair-probability scale is also not fully settled. Any pathway split
must be evaluated against the pooled predecessor, not assumed better because it
is semantically cleaner.

## Candidate estimands for current-category Youth Worlds readiness

Please compare at least these three designs.

### A. Category-specific context heads

Use a separate partially pooled response for each governed Youth Worlds category
and sex. Prior-category evidence enters through shared skill and the hierarchical
parent. The displayed coordinate changes as eligibility changes.

Risk: very sparse event clusters, unstable category labels/rules across years,
and confounding maturation with category difficulty.

### B. Shared YW context response plus category-specific reference field

Estimate one Youth Worlds context response (possibly sex-specific through the
existing pool), then derive `YW-IFSC Ready · current category` from a joint
simulation against a frozen empirical reference field for the athlete's
event-date category. The latent coordinate need not be category-specific; the
placement/probability target is.

Risk: may miss real category-specific transfer effects and requires stable
reference-field construction.

### C. Hierarchical age/category trajectory

Use shared skill plus a weak, smooth age-at-event trajectory and partially
pooled category deviations. Category rules are governed by event date. Prior
category results remain direct historical evidence but readiness is projected
to the current eligible category.

Risk: survivor/access bias, sparse longitudinal transitions, and the temptation
to infer maturation effects from selected international entrants.

## Proposed replay architecture

The minimum challenger retains one shared dynamic latent skill and form. It
routes events into `yw_ifsc`, `youth_reg_ifsc`, `reg_ifsc`, and `wc+`, with all
other competitions supplying shared graph evidence. Context effects must shrink
toward a defensible parent/global value; independent ledgers are prohibited.

The current engine supports arbitrary target domains but only independent
Gaussian context-offset priors, not a full hierarchy. A configuration-only split
is therefore executable but may not be the theoretically best candidate.

## Chronological decision protocol under consideration

- pre-2024: identity-correct global initializer only;
- 2024: development and hyperparameter/architecture selection;
- 2025: untouched validation;
- 2026: reused descriptive evidence only, not a fresh lock;
- pairwise event-balanced log loss and full-field placement RPS co-primary;
- advancement/top-k Brier, calibration, adult/youth, event-date category,
  category-transition, source/pathway and sparse-evidence harm guards;
- whole-competition or source-date clustered uncertainty, never pair bootstrap;
- compare split challenger with pooled predecessor and fully pooled/global
  control;
- no public interval until empirical chronological coverage passes;
- no direct demonstrated label without direct target-context participation;
- `not estimable` only for structural graph disconnection or a predeclared
  target whose transfer is unsupported, not merely zero starts.

## Questions for the reviewer

1. What is the correct estimand for `YW-IFSC Ready · current category`?
2. Which of A/B/C is theoretically strongest and which is executable with the
   present support? Propose a staged design if they differ.
3. Should youth-regional and Youth Worlds share a hierarchical parent? Should
   REG-IFSC and WC+ share another? Specify the partial-pooling structure.
4. How should category-rule changes and age transitions be encoded without
   future leakage or survivor bias?
5. Is a fresh pre-2024 initializer mathematically required, given that the
   initializer contains only global states and the context offsets begin in
   the 2024 replay? Distinguish mathematical necessity from provenance hygiene.
6. What minimum support and chronological gates should decide whether a head is
   estimated, inherited from a parent, or withheld?
7. Critique the proposed 2024/2025/2026 protocol and propose the smallest valid
   experiment that can falsify the split.
8. Give a coach-facing representation that distinguishes demonstrated level,
   readiness, target category, evidence support and uncertainty without
   presenting seven confusing numbers.
9. Identify assumptions that are semantic/product choices versus parameters
   that must be learned from data.

Return a candid verdict, recommended generative/statistical design, executable
first experiment, rejected alternatives, failure conditions, and UI contract.
