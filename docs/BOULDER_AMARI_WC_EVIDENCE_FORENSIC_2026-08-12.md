# Amari Bourbonnais WC evidence forensic

Status: model-integrity finding; not an athlete assessment.

## Reported inconsistency

The identity-rebuilt overview reported Amari Bourbonnais (`IFSC:14843`) at
`1942.755474` WC+-ELO and labelled two contests as WC+ evidence. This was only
36.07 points below Madison Richardson (`IFSC:1629`, `1978.825802`), despite
Amari having no senior World Cup or senior World Championship Boulder start in
the bound source history.

## Exact cause

The legacy selector deliberately included Youth World Championships in WC+.
Amari's two selected rows were:

- 2024 IFSC Youth World Championships Guiyang, Boulder qualification, rank 22;
- 2025 IFSC Youth World Championships Helsinki, Boulder qualification, rank 27.

The selected specialist ledger produced:

- raw specialist rating: `1513.565382`;
- selected contests: `2`;
- shared effective evidence: `1.426094`;
- uncertainty: `138.351991`;
- Global prior before display normalization: `1715.328548`.

The specialist-to-Global alignment and sparse-evidence shrinkage produced a
pre-normalization displayed value of `1668.794425`. The legacy women's
single-anchor display then added `273.961050`, yielding `1942.755474`.

Therefore the displayed value was not evidence that Amari had performed at a
1943 senior-WC level. It combined youth evidence, a Global prior, a
cross-ledger alignment and a large additive display offset under a misleading
WC+ label.

## Correction

All legacy WC+ families now require both:

1. a senior World Cup/Series, senior World Championship or Olympic-qualifier
   event; and
2. `rating_context == "Senior / Open"`.

Youth Worlds remain valuable evidence in the shared performance graph, but
they are a separate `IFSC_WORLD_YOUTH` context. Their contribution to a senior
WC forecast must pass through a chronologically validated youth-to-senior
transfer estimate and retain transfer uncertainty.

The product contract should distinguish:

- **direct senior-WC evidence**: observed senior target-context starts;
- **projected senior-WC readiness**: whole-graph forecast, transfer route and
  uncertainty;
- **not estimable**: insufficient graph connection, rather than a fabricated
  or inherited target rating.

## Scale implication

The planned two-anchor presentation scale is not restricted to chess-like
values around 1500–2000. For each supported target context it maps 50%
semifinal probability to 2000 and 50% final probability to 3000. Values below
1000 are valid when the calibrated joint model implies that much separation.
They must emerge from held-out probability calibration, not from manually
stretching ratings or transferring youth values directly onto a senior scale.

This finding is a regression case for the pathway model and product: no athlete
with zero direct senior-WC starts may be presented as having direct WC evidence,
even when a transferred WC projection is available.
