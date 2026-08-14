# Physical-tag review consensus V1

This is the deterministic handoff between the live 30-task review session and
the physical → board → competition coaching analysis. It does not estimate a
model and does not create a tag when a human review is absent.

The consumer binds the exact session CSV and receipt, validates all six required
core segment scores and both movement directions, and keeps only the latest
whole record for each pseudonymous reviewer and Boulder. Replacing a record
therefore also removes optional detail that the reviewer chose not to resubmit.

Four outputs stay separate:

- `session_task_progress.csv` records zero, one or at least two independent
  reviewers for each of the 30 exact tasks;
- `direction_distributions.csv` retains categorical disagreement and tied modes;
- `latest_reviewer_tags.csv` retains each reviewer's latest numeric observations;
- `tag_consensus.csv` reports means, ranges, standard deviations and descriptive
  evidence weights without hiding single-review evidence.

Wave A is operationally complete when its first 10 tasks each have two distinct
reviewer codes. The full session is operationally complete at 60 reviewer-task
records. Neither condition authorizes model input, model fitting or a coaching
prescription. Those decisions require a later chronological outcome and
coaching-value audit.

The consumer fails closed if the session hash or wave sizes drift, if a reviewer
code is not pseudonymous, if a required field is missing, or if a 0–3 score is
invalid. Records outside the 30-task session are counted but do not affect its
completion status.
