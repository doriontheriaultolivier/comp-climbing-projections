# Physical test to competition-result identity bridge V1

## Practical result

The 22 current physical/Kilter profiles can be joined to competition results
without fuzzy name matching. The bridge contains 52 governed source links and
covers every profile through its IFSC identity. Twenty profiles also have at
least one governed non-IFSC result identity.

This is identity infrastructure, not a physical ceiling model. It does not
change ratings, canonical identities, athlete profiles, or the coaching app.

## Evidence policy

The governed lane accepts only:

- the IFSC source identity;
- exact normalized name plus exact birth date; or
- an explicit reviewed source-ID override.

Name-only matches never enter the governed lane. Seven such candidates are in
the restricted manual queue with their actual competition evidence and the
decision vocabulary `ACCEPT_SAME`, `KEEP_SEPARATE`, or `DEFER`. Three CEC
candidates have conflicting dates of birth. Four Virtual League candidates
have no published birth date in the current result rows. Every candidate stays
`DEFER` until reviewed.

DigitalRock has no governed identity link for these 22 profiles in the current
ledger. The lossless DigitalRock results remain available for future evidence,
but no name-only DigitalRock link is inferred.

## Reproduction

```powershell
python scripts/materialize_physical_result_identity_bridge_v1.py `
  --profiles data/physical_test_profiles.csv `
  --links data/identity_link_audit.csv.gz `
  --results data/source_results.csv.gz `
  --physical-overrides data/physical_testing_identity_overrides.csv `
  --reviewed-overrides data/reviewed_identity_overrides.csv `
  --output-dir .artifacts/physical-result-identity-bridge-v1
```

The tracked receipt records only counts and hashes. The restricted output keeps
names, source IDs, DOB evidence, and competition context locally.

## Next product gate

Use only governed links to assemble athlete-relative chronological timelines:
physical tests, 50% Kilter flash, three-month hardest Kilter sends, and later
competition performance. Do not fit a linear correlation. The intended model
remains a monotone saturating physical-support ceiling plus separate wall
expression and competition-accessibility layers, with sparse observations kept
missing rather than imputed low.
