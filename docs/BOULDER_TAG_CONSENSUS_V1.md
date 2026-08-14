# Boulder tag consensus V1

This is the deterministic consumer for downloaded schema-v4 tagger records.
It keeps the latest submission from each pseudonymous reviewer for every
Boulder, preserves pre-zone and post-zone tags separately, and reports mean,
standard deviation, range, independent-review count and a descriptive
continuous evidence weight.

There is deliberately no two-review eligibility cliff. A single review remains
available with visibly weak evidence; additional independent reviews increase
the descriptive weight only when their values agree. Neither the raw review nor
the consensus table is authorized as a model input yet. Chronological outcome
evaluation and held-out coaching-value checks are still required.

Run after downloading the shared JSON record history:

```powershell
python scripts/materialize_boulder_tag_consensus_v1.py `
  --records comp_climbing_style_tags.json `
  --priority data/physical_item_tagging_priority_v1_1.csv `
  --inventory data/boulder_problem_inventory.csv.gz `
  --output-dir .artifacts/restricted/boulder-tag-consensus-v1
```
