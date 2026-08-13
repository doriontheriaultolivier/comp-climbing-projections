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

That rerun is not yet reproducible: the required
`scripts/boulder_terrain_problem_adapter.py` exists only as an untracked file in
the core worktree (observed SHA-256
`8da160f257460b0c5aa467baca58f6c23d8161d66bedc449acdbfe12888b1ac4`).
Its tests are also untracked. V1 remains frozen evidence; no V2 counts or queue
replacement should be claimed until the adapter, focused tests and input
closure are versioned and hash-bound. Copying mutable code into this lane would
not close that dependency.
