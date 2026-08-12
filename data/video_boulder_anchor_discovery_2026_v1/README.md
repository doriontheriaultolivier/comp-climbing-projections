# 2026 Boulder anchor discovery packet v1

This private evidence packet freezes the completed shallow discovery pass for
the Madrid, Prague and Innsbruck 2026 Boulder semi-final and final broadcasts.
It was produced by GitHub Actions run
[`30705758644`](https://github.com/doriontheriaultolivier/ifsc-performance-projections/actions/runs/30705758644)
from commit `ac210e0` and merged deterministically offline.

Coverage is complete for the planned discovery grid: 77/77 windows, split 25
for event 1479, 26 for event 1480 and 26 for event 1482. There are no missing or
quarantined windows. The model returned 213 anchor candidates, 172 scene-change
candidates and 185 visible-athlete intervals. These are candidates for a later
60-second, higher-resolution verification pass; they are not verified tactics,
athlete traits, commercial hold identities, scores or Elo inputs.

Safety gates remain closed in every record and summary: no production use,
athlete scoring/comparison or Elo update is allowed. The merge manifest stores
hashes for every frozen output. Validate the packet offline with:

```bash
python scripts/validate_boulder_anchor_discovery_packet.py
```

