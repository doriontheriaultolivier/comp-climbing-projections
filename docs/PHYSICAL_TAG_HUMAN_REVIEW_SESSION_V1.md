# First human Boulder-demand review session

The physical-to-board-to-competition coaching analysis is currently blocked by
one concrete evidence gap: the governed Boulder queue has no completed human
demand tags. The live review tool is already available at
<https://comp-climbing-boulder-tags.streamlit.app/>. This session packet turns
the existing continuous priority order into a manageable first review session;
it does not create a statistical inclusion threshold.

## Session design

Two reviewers independently tag the same 30 governed Boulders in **Coaching
evidence unlocked** order. They use separate pseudonymous reviewer codes and do
not discuss scores until both exports are saved.

- **Wave A, tasks 1-10:** a shared calibration set. Complete the movement
  directions and all three core demands: physical, technical and coordination.
- **Wave B, tasks 11-30:** the next high-unlock tasks. Keep the same core scope;
  detailed tags remain optional and should be completed only when the reviewer
  can see the entire Boulder well enough to score them.
- Use `Low` confidence when footage or the wall view is incomplete. Uncertainty
  is evidence and should not be replaced by a confident guess.
- Save every review to the shared backend. Also download the session JSON before
  closing the tab, so the work remains recoverable.

The operational roster is
[`physical_tag_human_review_session_v1.csv`](../data/physical_tag_human_review_session_v1.csv).
It contains no athlete identity or physical-test value. The app will normally
open directly on this 30-task session. The complete coaching-ranked and generic
489-task queues remain available as alternate review orders.

## What happens after both reviews

Download the full shared review history and run the existing consensus
materializer. It retains each reviewer's latest score, keeps start-to-Zone and
Zone-to-Top separate, and preserves disagreement instead of forcing a single
label. There is no arbitrary two-review model-eligibility cliff: this first
session is an agreement and workflow audit.

The session-bound consumer is:

```powershell
python scripts/materialize_physical_tag_review_session_v1.py `
  --records comp_climbing_shared_style_tags.json `
  --priority data/physical_item_tagging_priority_v1_1.csv `
  --inventory data/boulder_problem_inventory.csv.gz `
  --session data/physical_tag_human_review_session_v1.csv `
  --session-receipt data/physical_tag_human_review_session_v1.json `
  --output-dir .artifacts/restricted/physical-tag-review-consensus-v1
```

It binds the exact session hash, keeps each reviewer separately, inventories
direction disagreements, and reports `not_started`, `single_review_complete`
or `independently_double_reviewed` for every task. Even a complete session stays
research-only until chronological coaching and forecast gates are passed.

If task identity, saving and independent-review accounting all work, continue
down the same continuous queue. The physical challenger remains inactive until
reviewed tags have enough support for the preregistered chronological analysis.
Target-event tags may support post-event coaching only; they are never
backdated into a pre-event forecast.

## Expected outcome

The session should produce 60 independent review records across 30 exact
Boulders. The immediate deliverable is a disagreement-preserving consensus and
an honest estimate of review speed and ambiguity. It is not a weakness label,
a psychological diagnosis, a training prescription or a model release.
