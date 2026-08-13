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

Product checks against copied staged bytes passed for the default view plus
Global progression, IFSC Pool, WR Pool, and Towards Olympics, with zero
Streamlit errors/exceptions. The focused release/identity/profile/probability
suite passed 23 tests.

This remains a staged local research successor. Deployment and production
rating promotion remain false until the second-root byte reproduction, locked
adult/youth/pathway pair and placement calibration, and release candidate
acceptance complete.
