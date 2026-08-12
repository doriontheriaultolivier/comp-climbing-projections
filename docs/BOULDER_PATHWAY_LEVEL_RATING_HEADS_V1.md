# Boulder pathway-level rating heads V1

## Constructive disposition

The proposed FED / NACS / IFSC-Regional / WC / OLYM decomposition is more useful
for coaching than a single `Global-ELO`, provided it is implemented as multiple
context heads over one shared athlete graph. It should not become five isolated
Elo ledgers.

The shared graph answers “what has this athlete demonstrated anywhere?” Each
context head answers “how does that evidence transfer to this competitive
environment?” This makes pathway gaps interpretable without discarding bridge
athletes who compete at multiple levels.

## Recommended heads

| Head | Direct evidence | Transfer treatment | Primary coaching use |
|---|---|---|---|
| `FED:<code>` | within one federation's governed competitions | partially pooled federation offset/temperature; bridge athletes connect to higher levels | domestic dominance, depth and promotion readiness |
| `INTERFED:North-America` | North American Cup Series | direct NACS plus learned FED→inter-federation→continental/world transfer | cross-border readiness below the full Pan America continental layer |
| `CONT:<Africa|Asia|Europe|Oceania|Pan-America>` | recognised Continental Series and Championships | region-specific head, partially pooled across continents through the shared graph | continental readiness and pathway comparison |
| `WC` | World Cups and World Championships | direct WC/WCH likelihood with lower-level transfer estimated, never assumed | elite comparison and target-event forecasting |
| `IFSC-World-Youth` | Youth World Championships | separate transition context, partially pooled through shared skill and estimated WC transfer | youth ordering and adult-WC transition without treating youth results as adult WC results |
| `OLYM-scenario` | fixed Olympic field/format scenario | derived primarily from WC state and exact Olympic format | conditional Olympic placement/readiness, not a standalone Elo |

`OLYM` should remain a scenario rather than a fitted independent rating until
there are enough comparable Olympic events. One event every four years cannot
separately identify form, field, format and athlete skill.

Every competition family also has a youth variant when a distinct youth event
exists. Youth is not a suffix that merely changes the label: it is a separate
context response connected to adult contexts through the athlete's shared
time-varying skill and empirically estimated transition. Youth World results do
not directly update the adult World head.

## How the full graph influences every head

No result is assigned exclusively to one rating. For athlete `i`, time `t`, and
target context `c`, the predictive readiness is conceptually

`eta(i,t,c) = shared_skill(i,t) + context_response(i,t,c)`.

Every valid result updates `shared_skill`. A result directly observed in context
`d` also updates `context_response(d)`. Its effect on another target context `c`
is learned from the historical competition graph rather than imposed by a
manual level multiplier. Candidate transfer structures must use only
pre-event evidence and may include:

- shared-athlete connectivity between contexts;
- recency and number of bridge athletes;
- uncertainty in both source and target context states;
- empirically identified context-transition residuals;
- later, governed terrain/item-distribution distance when Boulder Tags provides
  adequate route-style evidence.

The effect should shrink smoothly as connection evidence weakens. It must not
be `result_weight = 1 / hand-chosen graph distance`: graph distance is partly a
missing-data property, highly connected elite athletes are selectively sampled,
and terrain difficulty/style may mediate rather than merely attenuate transfer.
Compare at least these chronological candidates:

1. fully pooled shared skill (no context response);
2. independent Gaussian context offsets with strong shrinkage;
3. hierarchical context offsets grouped by federation/inter-federation/
   continental/world and youth/adult family;
4. graph-kernel or multilevel transition effects estimated from bridge athletes;
5. the same structures with terrain distance only after the tagging evidence is
   frozen and available at each historical forecast date.

Weakly connected contexts remain forecastable through shared skill but should
show wider uncertainty and stronger shrinkage. They are not automatically set
to zero and are not promoted solely because they cross a fixed event-count
threshold.

## The two-anchor display scale

For pathway context `c`, simulate event-balanced reference fields using only
past eligible competitions. Find latent abilities:

- `s_semi,c`: average `P(reach semifinal | start, reference field)=0.50`;
- `s_final,c`: average `P(reach final | start, reference field)=0.50`.

Then display:

`R_c(s) = 2000 + 1000 * (s - s_semi,c) / (s_final,c - s_semi,c)`.

This gives the proposed interpretations exactly:

- 2000 = 50% semifinal probability;
- 3000 = 50% final probability.

The scale is intentionally unbounded. Athletes may appear below 1000 when the
validated probability model supports that distance below semifinal level; do
not impose a chess-like floor or manually widen the distribution. Conversely,
an athlete with no direct target-context starts may still receive a transferred
projection, but it must be labelled as projected, show its transfer path and
uncertainty, and never be counted as direct evidence. The legacy Amari
Bourbonnais WC+ case is the regression example: two Youth Worlds were
incorrectly counted as senior WC+ evidence and then heavily shifted by the
display normalization.

It is a good communication layer because the distance now has a pathway-specific
meaning. It does **not** by itself fix calibration: any affine transform can
spread numbers. The underlying joint ranking distribution must first predict
held-out pair and placement outcomes correctly.

## Reference competition definition

Do not use “the last eligible event” as the default; one unusually strong or
weak field would move every athlete's scale. Freeze an event-balanced empirical
reference distribution, initially 2025, with exact:

- eligible events and source hashes;
- field sizes and semifinal cut sizes;
- gender, round and procedure;
- attendance/selection conditionality;
- chronology and refresh rule.

On refresh, use a rolling multi-event reference or a partially pooled yearly
reference. Preserve the old anchors so changes can be decomposed into athlete
change versus reference-field change.

The original 50%-win proposal is rejected as the display anchor: it is often a
tail extrapolation, moves sharply with field size and dominant outliers, and is
least stable in the sparse contexts where interpretability matters most. Win
probability remains a direct model output with uncertainty. If 50%-final is not
identified inside supported ability for a pathway, the product must show
`upper anchor not estimable` rather than manufacture a number.

## Model structure

A practical first candidate is:

`ability_i,t,c = shared_skill_i,t + context_response_i,c + context_offset_c`

with:

- dynamic shared skill and uncertainty;
- strongly regularized athlete-by-context response;
- federation/region offsets and temperatures;
- explicit event field strength, format and round cut;
- age-at-event as a weak dynamic prior, not a direct youth bonus;
- a coherent Plackett–Luce/random-utility joint ranking simulator;
- source and identity uncertainty propagated or quarantined.

FED observations compare athletes inside their federation directly. Cross-level
and cross-federation transfer is learned only through shared athletes and
connected competitions. A disconnected federation can still have an internally
valid FED rating, but its numerical comparison to WC must be labelled
`unconnected / not estimable`.

## Why this helps coaching

The profile can show both level and transfer:

- high FED, uncertain NACS: dominate locally; next information is a continental start;
- high NACS, lower WC head: demonstrated continental skill but incomplete WC transfer;
- similar shared skill but divergent Flash/Onsight or item-style responses: targeted competition-process work;
- strong physical/Kilter ceiling but lower FED/NACS/WC realization: investigate wall transfer, tactics and pressure rather than simply adding strength.

The difference between heads is not automatically a deficit. It may reflect
small evidence, access/selection, terrain, travel, age category or disconnected
graphs. Always display uncertainty, direct event count and transfer route.

## Required validation before product use

1. Canonical identity histories rebuilt first.
2. Strict rolling-origin fits; no target event in training or anchor estimation.
3. Event-balanced pair log loss and placement RPS from the same joint draws.
4. Calibration by probability, rating gap, strong/middle/weak spectrum, age,
   evidence, federation/region and procedure.
5. Whole-competition bootstrap intervals.
6. Direct-vs-transfer ablations for every head.
7. Adult and under-18 reporting separately; unsupported cells are `not estimable`.
8. Anchor stability across reference years and leave-one-event-out analyses.

## Decision

Explore and implement the four evidence-bearing heads. Keep `OLYM` as a
conditional scenario. Use 2000=50% semifinal and 3000=50% final as an
interpretable display contract only after both thresholds are empirically
identified and probability calibration passes. Show win probability separately.
No current rating is promoted by this design document.
