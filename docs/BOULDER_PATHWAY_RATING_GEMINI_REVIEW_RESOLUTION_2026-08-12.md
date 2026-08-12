# Resolution of Gemini Pro pathway-rating review

## External review used

Gemini `gemini-3.1-pro-preview` reviewed the sanitized competing designs with
8,406 total tokens (5,610 prompt, 1,380 answer, 1,416 thinking). The full response
is preserved in `BOULDER_PATHWAY_RATING_GEMINI_PRO_REVIEW_2026-08-12.md`.

## Accepted recommendations

- All results inform one shared dynamic athlete state.
- FED, NACS, IFSC-Regional and WC are partially pooled target-context heads,
  not independent Elo ledgers and not input filters.
- Olympics is an exact field/format scenario derived mainly from WC evidence,
  not an independently estimated head.
- The display scale uses two empirically identified central thresholds:
  2000 = 50% semifinal and 3000 = 50% final in the frozen reference field.
- Win probability remains a direct probability with uncertainty.
- Identity closure, strict rolling-origin evaluation, transfer ablations and
  whole-event uncertainty precede any product promotion.

## Recommendation deliberately narrowed

The review says the demonstrated T=3 temperature should calibrate the model.
That is too broad. T=3 improved held-out scores for the current frozen model and
is evidence of its compression. It is not automatically the correct temperature
for a new multi-context architecture. Every candidate must fit calibration only
inside its chronological training window and be evaluated without retuning on
the next events.

## Why two anchors remain useful

Two anchors do not add predictive information. They uniquely set the location
and scale of an otherwise arbitrary latent coordinate and make cross-pathway
readiness legible. This is worthwhile only when:

1. both 50% thresholds lie inside empirical support;
2. the underlying joint probabilities are calibrated;
3. reference fields are event-balanced and frozen; and
4. uncertainty and direct/transfer evidence accompany the number.

Where these conditions fail, probability cards and `not estimable` are better
than a pathway rating.

## Next falsifiable implementation

Fit the simplest shared-skill plus fixed context-offset model first. Compare it
chronologically with:

1. fully pooled global skill;
2. isolated pathway models;
3. shared skill plus partially pooled context response.

Only add athlete-specific context response if it improves event-balanced pair
log loss and placement RPS and is noninferior across adult/youth, evidence and
pathway strata. Complexity is earned by held-out utility, not assumed.
