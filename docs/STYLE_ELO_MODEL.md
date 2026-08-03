# Boulder inventory and style-Elo contract

## Unit of analysis

The stable hierarchy is:

`event → round → gender terrain → age terrain → boulder → segment`

Each boulder has one stable `boulder_uid`. It has two non-overlapping segment
IDs:

- `pre_zone_segment_uid`: start to zone;
- `post_zone_segment_uid`: zone to top.

Canadian Youth A/U19 and Junior/U21 terrains may share a boulder identity.
Youth B/U17 and Youth C/U15 never share that identity with each other or with
A/Junior. At Youth Worlds, every age category remains separate.

## Governed boulder counts

`data/boulder_round_inventory.csv` is built from normalized results. A count is:

- `source-confirmed` when one positive `n_routes` value is reported;
- `source-conflict` when the source contains more than one value;
- `format-assumption` when the source is empty and the normal round default is
  proposed;
- `contributor-proposed` when a tagger corrects an unconfirmed proposal;
- `unknown` when no count evidence exists.

The public form locks source-confirmed counts and exposes every weaker status.

## Segment outcomes

For athlete *i* and boulder *b*:

- start→zone is successful when the athlete controls the zone or tops;
- zone→top is successful when the athlete tops, conditional on first reaching
  zone;
- when the athlete never reaches zone, zone→top is censored—not a failure.

Attempts can refine ordering inside each segment when source data exposes them.

## Style-specific update

For style `s ∈ {physical, technical, coordination}`:

```text
ΔR(i,e,s) = q(e) K(i,e) / B(e)
             × Σ_b [wZ d(b,Z,s) r(i,b,Z) + wT d(b,T,s) r(i,b,T)]
```

`d` is consensus demand from 0 to 1, `r` is observed-minus-expected paired
evidence, `q` is event-quality weight and `B` is the round's governed boulder
count. `wZ + wT = 1`; the initial challenger uses 0.5/0.5. Symmetric pairwise
updates keep each field zero-sum. Dividing by `B` controls total round weight.

## Release gate

Style Elo is not a production selection metric until all are true:

1. athlete problem outcomes join cleanly to stable boulder IDs;
2. enough boulders have independent tags and acceptable inter-rater agreement;
3. segment weights and event-quality weights are chosen only on training data;
4. frozen next-event tests show incremental value over Global-Elo;
5. calibration, subgroup error and missing-data sensitivity are published.

