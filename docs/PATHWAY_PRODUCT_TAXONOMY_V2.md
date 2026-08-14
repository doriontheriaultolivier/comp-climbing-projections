# Pathway product taxonomy V2

The product vocabulary is now deliberately simpler than the internal source
taxonomy:

- youth: **Y-NAT → Y-REG → YW-IFSC**;
- open: **NAT → REG → WC+**; and
- **OLY** as a conditional scenario after WC+, not a separately fitted rating.

`FED` remains an internal data-source prefix because it identifies the governing
federation. It is not a useful user-facing level name, so the product says
`NAT`. The youth equivalent is explicitly `Y-NAT`. Internally these remain
federation-specific heads—`NAT:CAN`, `NAT:USA`, and so on—not one isolated
worldwide national ledger. Their cross-country comparability comes from the
shared graph and bridge athletes.

NACS and Pan-American competitions both enter the broad `REG` head. They remain
different event subtypes—North America series versus continental
series/championship—so the interface can expose procedure, field and transfer
differences. Creating a separate public rating for each sparse series would
fragment the evidence without improving coaching interpretation.

Each evidence-bearing head has two concepts:

- **demonstrated** means direct governed starts in that context;
- **Ready** means a prediction against a governed reference field, using the
  full connected graph and a validated context-transfer model.

An athlete with no direct WC+ start may therefore have no demonstrated WC+
level while still having a WC+ Ready estimate. Missingness should come from
structural disconnection or an unvalidated reference-field model—not from a
blanket zero-start rule.

Youth results still update the shared time-varying graph but do not become
direct NAT, REG or WC+ evidence. When an athlete ages up, the shared state
persists and the target reference field changes. This preserves useful youth
information without pretending that youth and senior fields are interchangeable.

The existing three-competition Youth Worlds audit remains decisive: the name
`YW-IFSC Ready` is retained, but a current-category numeric version is not yet
released. The `REG` and `WC+` context heads similarly require chronological
pair and placement validation. The proposed 2000/3000 anchors remain a future
display layer only after the relevant semifinal and final thresholds are
identified by a calibrated joint simulator.

This document changes product language only. It does not promote a rating,
probability, interval or app output.
