# Release identity successor V2 — local evidence

The release-compatible successor is built from the reviewed canonical IFSC
identity rebuild, not by joining two public profiles after rating. It therefore
preserves chronological state: IFSC source athlete `18545` is resolved to
`IFSC:14843` before the rating replay, leaving one 25-row Amari Bourbonnais
history, Global-ELO `1989.2895977737599`, and no direct Senior/Open WC+ rating.

The adapter binds and rechecks the exact research receipt and its four candidate
artifacts. It then:

- reconstructs the current 80-column athlete schema without publishing raw
  birth dates;
- retains the legacy WC+ qualification-flash columns as all-missing
  compatibility placeholders because senior WC qualification is onsight and
  the canonical rebuild has no eligible flash evidence;
- recomputes generalized Plackett–Luce contest-performance posteriors for all
  9,075 contests / 179,253 rows from canonical pre-event state; and
- writes a content-addressed local staging directory without modifying release
  `data/`.

The first staged result is
`staged-v2-d9766be1020ef7c9899b975c7325ec5ec0039ca7d5d4a7d402dd3a958981c96f`:

- athlete artifact: 36,978 rows × 80 columns, SHA-256
  `c5be49ac219a14978f240cc6c1d61699bfbd9fc4737941d00f856d043897b7ec`;
- history artifact: 179,253 rows × 23 columns, SHA-256
  `b51d2df74e04362a07b0099d1bf3c8229a1dec9e6e967283c6534f0e3d9d7379`.

On 2026-08-13, two independent output roots reproduced all three files
byte-for-byte under the pinned project Python environment:

- manifest: 2,195 bytes, SHA-256
  `d9766be1020ef7c9899b975c7325ec5ec0039ca7d5d4a7d402dd3a958981c96f`;
- athlete artifact: 3,875,287 bytes, SHA-256
  `c5be49ac219a14978f240cc6c1d61699bfbd9fc4737941d00f856d043897b7ec`;
- history artifact: 16,666,886 bytes, SHA-256
  `b51d2df74e04362a07b0099d1bf3c8229a1dec9e6e967283c6534f0e3d9d7379`.

The app lifecycle guard was also exercised against a disposable copy with
these staged bytes. It now accepts exactly the predecessor state (one
canonical plus one reviewed alias, requiring suppression and rating
quarantine) or the rebuilt state (one canonical and no alias). Other
cardinalities still fail closed. In the rebuilt state the new ratings remain
visible, while the separately frozen current-WC projection row for this
athlete remains withheld until that projection artifact is rebuilt.

Product checks against copied staged bytes passed for the default view plus
Global progression, IFSC Pool, WR Pool, and Towards Olympics, with zero
Streamlit errors/exceptions. The app smoke suite passed 10 tests plus 13
subtests. The current-release focused identity/profile/probability suite passed
28 tests before the staged swap, and the lifecycle transition suite passed 23
tests after the guard update.

This is a whole-model replay, not an Amari-only patch. Relative to the current
release, only 47 otherwise unique profiles change base Global-ELO, but 2,157
shared profiles change WC+-ELO and thousands change specialist-family values.
Those changes therefore require their own chronological predictive validation;
the reviewed identity decision alone is not promotion evidence.

This remains a staged local research successor. The two-root and app-render
gates are complete, but deployment and production rating promotion remain false
until the fresh identity-safe locked adult/youth/pathway pair and placement
calibration completes. Existing locked predictions still contain both sides of
the reviewed split and cannot be relabelled into valid successor evidence.
