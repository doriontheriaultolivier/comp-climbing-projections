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

The public `performance_elo` is the mean of a posterior distribution over WC
performance. Frozen Cumulative-ELO is its prior mean, with a 250-Elo form
variance added so one round can differ materially from stable ability. Every
beat/lost-to pairing contributes a Bradley-Terry likelihood against the frozen
opponents' Global/Cumulative-ELO—as though those matchups occurred on WC
terrain. WC+ uses the full likelihood. Lower events use a power
likelihood based on their information quality and transfer to WC terrain. The
posterior standard deviation is published as uncertainty; the unregularized
inverse-logistic estimate remains auditable as `raw_performance_elo`. This
signal does not feed back as a second cumulative update.

The beat/lost-to terms form a composite likelihood: several pairings come from
the same round and therefore are not fully independent observations. The mean
is the requested WC-equivalent summary, but its standard deviation should be
read as model uncertainty conditional on this approximation, not as a complete
measure of all competition volatility. A Plackett-Luce ranking likelihood is a
planned challenger rather than an untested production replacement.

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
- Performance-ELO uncertainty does not include every source of dependence
  between pairings from the same round.
- Model promotion should be revisited when more 2026+ competitions and tagged
  boulder-level evidence are available.
