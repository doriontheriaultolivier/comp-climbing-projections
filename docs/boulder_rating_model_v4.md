# Boulder rating model v4 decision record

## Decision

Production uses one shared Boulder ability estimate and keeps two questions
separate:

1. **Which evidence is included?** Global-ELO uses every usable,
   de-duplicated result; IFSC-ELO and WC+-ELO are restricted-evidence
   diagnostics.
2. **Where are we projecting the athlete?** The shared Global evidence can be
   projected into a Canadian-event or WC+ context. A learned context effect is
   not presented as a separate athlete skill and is not labelled as a cause.

This avoids an unmanageable matrix such as `WC+-ELO(global)`,
`CAN-ELO(global)`, and `NACS-ELO(global)` while retaining the useful distinction
between evidence and target terrain.

## Newcomers

An athlete is provisional until four independent competitions agree and the
robust provisional uncertainty falls to 100 Elo or less. Provisional athletes
do not transfer rating mass to established athletes. Their starting estimate
combines a chronological pool, age-band and country prior with capped round
signals. The signal is measured relative to that round's frozen field median,
weighted by event transfer and information quality, and capped at 300 Elo.

This is deliberately not a universal 1,500/1,800 cold start. It also prevents
a win in one small local field from manufacturing a world-class rating. Once
promoted, normal symmetric pairwise updates resume.

Population growth is allowed to add rating mass through a new athlete's prior.
Established-athlete pair updates remain zero-sum. The public scale is re-anchored
after the replay so that 2,000 retains its fixed interpretation: fitted 50%
semifinal probability at a randomly sampled 2025 Open World Cup.

## Performance-ELO

The raw inverse-logistic round estimate remains in the audit data as
`raw_performance_elo`. The public `performance_elo` is an empirical-Bayes
posterior: the raw signal is shrunk toward the athlete's frozen pre-round
Cumulative-ELO. Larger, reliable fields move it more. Direct WC+ evidence keeps
the full observed round signal, while local/provincial evidence is pulled farther
toward the cumulative estimate because its field and setting transfer are less
certain. This makes the signal readable without hiding the original result, and
it does not feed back as a second cumulative update.

## Locked forward comparison

Models were tuned on 2022–2024 first-round forecasts and evaluated on 2025+
competitions. The promoted 300-cap challenger was compared with the former
quality-weighted model and adjacent 200/400 caps.

| 2025+ target | Metric | Former model | v4 (300 cap) |
|---|---:|---:|---:|
| Open WC+ | Spearman field order | 0.795 | 0.764 |
| Open WC+ | pairwise Brier, calibrated | 0.137 | 0.144 |
| Open WC+ | top-8 Brier | 0.124 | 0.125 |
| CEC national | Spearman field order | 0.626 | 0.628 |
| CEC national | pairwise Brier, calibrated | 0.191 | 0.186 |
| CEC national | top-8 Brier | 0.182 | 0.159 |
| CEC provincial/local | Spearman field order | 0.601 | 0.588 |
| CEC provincial/local | pairwise Brier, calibrated | 0.214 | 0.202 |
| CEC provincial/local | top-8 Brier | 0.222 | 0.200 |

The former model remains slightly better at ordering the middle of an Open WC+
field. V4 is materially better for the advancement and domestic decisions the
Global model must also support, and removes the demonstrably invalid cold-start
compression. This trade-off is shown rather than concealed. Probability
temperature is selected on the earlier window for each evaluation domain; it is
not assumed to be the same as the Elo update scale.

## Known limits

- Context differences mix setting, format, attendance, travel and selection;
  they are not evidence of a psychological cause.
- A provisional rating remains an estimate with visible uncertainty.
- Specialist round/format ratings shrink toward Global-ELO when evidence is
  sparse.
- Model promotion should be revisited when more 2026+ competitions and tagged
  boulder-level evidence are available.
