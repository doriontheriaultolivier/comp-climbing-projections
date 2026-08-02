# Boulder rating-family contract

**Release:** Comp Climbing Projections v2  
**Decision:** Boulder is the only governed discipline in this interface until
its data, model and coaching language pass review.

## One scale, different evidence

Every displayed rating answers the same practical question: *what Open World
Cup Boulder level does the eligible evidence project?* Youth and senior
athletes therefore do not have separate scales. Men and women retain separate
competition ledgers because they compete on different terrain, but the same
model and display scale are used.

## Interpretable 2000 anchor

On the displayed scale, **2000 means a fitted 50% chance of advancing from
qualification to semifinal at a randomly sampled 2025 IFSC Open World Cup**,
within the athlete's gender pool and assuming participation. The anchor is fit
from ratings frozen before each competition and the observed advancement
outcome: 477 men's starts and 381 women's starts across six World Cups per
pool. The native 50% thresholds were 1704.6 for men and 1686.2 for women, so
the release adds +295.4 and +313.8 respectively.

This is a display translation, not an extra model update. It changes neither
athlete order nor spacing, and it does not mean every athlete at 2000 has
exactly 50% odds in every venue: field, setting and current form still matter.

| Display name | Eligible evidence |
|---|---|
| Global-ELO | All de-duplicated local, national, international, youth and senior Boulder rounds |
| Global-ELO-Onsight | Global rounds with confirmed onsight procedure |
| Global-ELO-Scramble | Global rounds with confirmed scramble/redpoint procedure |
| Global-ELO-Flash | Global rounds with confirmed flash procedure, including all youth qualifications |
| WR-ELO | Rounds in events currently included in the IFSC World Ranking calculation |
| WR-ELO-Qualies / Semies / Finals | Only that round family inside WR events |
| IFSC-ELO | All non-paraclimbing IFSC Boulder rounds in the archive |
| IFSC-ELO-Qualies / Semies / Finals | Only that round family inside the IFSC archive |
| Performance-ELO | The isolated level shown by one round using ratings frozen before that competition |

Confirmed youth qualifications are Flash. Confirmed youth semifinals and
finals are Onsight. Senior/Open IFSC Boulder rounds are Onsight. Other domestic
procedures remain Unknown unless the source explicitly establishes Flash,
Onsight or Scramble; Unknown rounds still inform Global-ELO but cannot move a
procedure specialist.

## Sparse-evidence rule

Raw specialist ledgers start at the same mathematical origin as Global-ELO,
then map directly onto the current Global scale within each competition pool
using `Global = intercept + slope × raw specialist`. The previous inverse
regression compressed sparse WR ledgers and produced an artificial downward
shift, especially in the women's pool. Direct mapping removes that pool-level
bias. The
displayed specialist difference from Global-ELO is proportional to eligible
evidence. This prevents two distortions:

1. a strong new athlete is not shown near the arbitrary initial rating merely
   because only a few specialist rounds exist; and
2. one unusual round cannot be presented as a stable specialist trait.

Fewer than two eligible contests produces no displayed specialist rating.
Evidence status is Provisional at 2–3 contests, Developing at 4–7 and
Established at 8 or more. These labels describe the quantity of rating
evidence, not the quality of the athlete.

## What correlations can and cannot say

The Overview reports a rank correlation between each rating family and
WR-ELO. A larger value means the ordering is more similar to the WR ordering.
It does **not** identify a causal setting or training-environment effect.
Attendance, event access, travel, selection, age, field strength and training
environment remain mixed in the comparison. Any claim about setting
specificity requires a separate, chronologically tested model.

## Governance safeguards

- Cross-source duplicate contests are removed before updates.
- Every pairwise contest update is zero-sum after new entrants are registered.
- Paraclimbing is excluded from these able-bodied competition pools.
- Current ratings may describe current evidence; historical predictions must
  use ratings frozen before the predicted competition.
- Performance-ELO is diagnostic and never silently replaces cumulative Elo.
- Missing evidence is shown as missing, not filled with an unrelated rating.
