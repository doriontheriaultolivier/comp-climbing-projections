# Physical → board coaching slice V1

Date: 13 August 2026
Status: private descriptive coaching surface; no causal or competition-transfer claim

## What changed

The compact Streamlit athlete profile now loads the governed physical-testing
release artifacts only when the deployment explicitly sets
`PHYSICAL_COACHING_SLICE_ENABLED=true`. For each safely linked athlete it then
shows:

- the dated 50%-flash Kilter-equivalent grade;
- the dated recent-three-hardest-send Kilter-equivalent grade;
- up to five ranked physical hypotheses that passed the existing exploratory
  `Focus candidate` screen; and
- an explicit hold on competition-transfer interpretation until exact,
  human-reviewed competition-item demand tags exist.

The two board indicators stay separate. The interface does not subtract them,
convert their difference into a pressure or transfer score, or treat V-grade
distance as an interval measurement. The first is labelled repeatable,
lower-pressure expression; the second is labelled an exposure-dependent
observed upper tail.

Physical rows are likewise framed as hypotheses to investigate, not training
prescriptions. Anthropometric context and non-focus rows are excluded from the
short list. Missing, unlinked, or inconclusive evidence receives a useful
explanation rather than a weakness label.

The explicit feature flag is a privacy boundary, not row-level security. This
version is for a staff-only private deployment. It must stay disabled in public
or athlete-facing deployments. Before access expands beyond authorized coaches
and HP staff, the project needs identity-aware accounts, row-level access,
consent scope, audit logs, correction/withdrawal and retention controls.

## Current governed coverage

- 22 safely linked athlete profiles;
- 18 athletes with both requested board indicators;
- 1,727 existing athlete/metric priority rows available to the filter; and
- no new model fit, identity decision, source download, or raw physical row in
  this change.

## Verification

- the artifact loader returns 22 profiles and 1,727 priority rows;
- the default loader does not read either physical artifact unless the explicit
  staff-only flag is enabled;
- all 18 complete profiles retain both board observations;
- focused physical, snapshot, app, cache, and end-to-end tests pass;
- the helper fails closed when the athlete identity is absent; and
- a regression test prevents contextual or non-focus metrics from entering the
  coaching shortlist.

The complete pull-request diff received an independent Vertex review from
`gemini-3.1-pro-preview`. The final verdict was `ACCEPT`: it confirmed the
descriptive semantics, missing-data behavior, shortlist filtering and
staff-only cache boundary. Its sole minor display observation was resolved by
converting the stored 0–1 peer percentile to an explicit 0–100 display value.
The public receipt records hashes and token usage without exposing any private
physical measurements.

## Next gate

Join exact CEC/IFSC item outcomes only after the relevant problems receive
human-confirmed demand tags. Then present three visibly separate layers:
physical capacity, board expression, and matched competition expression. Any
statistical transfer model remains a chronological research challenger.
