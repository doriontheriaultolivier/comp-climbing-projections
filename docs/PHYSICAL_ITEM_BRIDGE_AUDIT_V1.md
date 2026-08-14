# Physical/Kilter to exact-item bridge audit V1

Status: research inventory materialized; no model or coaching prescription.

The existing evidence is already sufficient to target tagging efficiently:

- 22 athletes have settled physical/Kilter identities;
- 20 overlap exact federation problem outcomes;
- 2,067 exact ordinary Zone/Top outcomes span 66 competitions; and
- 1,004 outcomes from 18 athletes and 27 competitions have both a latest-prior
  physical profile and Kilter observation no more than 365 days old.

The 1,004 candidates contain real outcome variation (84.3% Zone and 56.2% Top),
but they do not yet contain human-confirmed problem-demand tags. They therefore
remain ineligible for a physical-support or coaching-transfer model.

This changes the efficient tagging strategy. The next task is not broad video
screenshot extraction. It is to deduplicate the candidate competition/problem
identities and tag those items first, because one confirmed problem tag supports
many athlete outcomes. Video may help only where it is the cheapest reliable
way to resolve a missing item-demand tag; it remains a supporting lane rather
than the main product.

`data/physical_item_tagging_priority_v1.csv` contains all 535 items without
athlete identifiers. It is a continuous ordering, not a pass/fail gate. It
sorts first by the number of linked athletes who reuse a tag, then by observed
Zone/Top outcome spread, completeness of the two Kilter measures, physical-test
coverage, and recency. A human can still tag any lower-ranked item when it is
cheap or strategically useful.

The restricted candidate table retains athlete and outcome rows. The committed
report contains counts, input/output hashes, and authority flags only.

## Successor dependency discovered 2026-08-13

The governed physical-result identity bridge now covers two tested athletes
(`IFSC:17186` and `IFSC:599`) that were absent from the older identity-safe
problem bridge, and several athletes gain governed USAC/SAC links. A V2
rematerialization could therefore add legitimate exact-item evidence.

The full adapter closure is now versioned: adapter/test commit `7633959`, plus
its source-semantics and terrain-response dependency closure `b84cbbb` in the
core research repository (cherry-picked here as `902c72d` and `168454a`). The
combined closure passes 65 tests plus 12 subtests from this clean worktree.

The governed V2 overlay does **not** enlarge the calibration-ready item set or
the aggregate physical-athlete coverage. It identifies four conflicts with old
source-node assignments, but zero staged Boulder rows change assignment because
exact frozen round snapshots take precedence; the strict calibration contract
therefore remains unchanged. The two
physical athletes absent from the old snapshot (`IFSC:17186` and `IFSC:599`)
have only Lead rows in the currently staged IFSC problem evidence. These facts
are useful negative evidence: they prevent an unnecessary model rerun and keep
identity inventory separate from rating-state imputation. A future
identity-correct chronological replay could make some reassigned rows eligible,
but this audit never invents that state.
