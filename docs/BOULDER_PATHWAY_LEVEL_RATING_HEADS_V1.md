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
| `NACS` | North American Cup Series | direct NACS plus learned FED→NACS and NACS→WC transfer | continental readiness and field selection |
| `IFSC-REG:<region>` | IFSC regional/continental series | region-specific head, partially pooled across regions | regional pathway readiness |
| `WC` | World Cups and World Championships | direct WC/WCH likelihood with lower-level transfer estimated, never assumed | elite comparison and target-event forecasting |
| `OLYM-scenario` | fixed Olympic field/format scenario | derived primarily from WC state and exact Olympic format | conditional Olympic placement/readiness, not a standalone Elo |

`OLYM` should remain a scenario rather than a fitted independent rating until
there are enough comparable Olympic events. One event every four years cannot
separately identify form, field, format and athlete skill.

## The two-anchor display scale

For pathway context `c`, simulate event-balanced reference fields using only
past eligible competitions. Find latent abilities:

- `s_semi,c`: average `P(reach semifinal | start, reference field)=0.50`;
- `s_win,c`: average `P(win | start, reference field)=0.50`.

Then display:

`R_c(s) = 2000 + 1000 * (s - s_semi,c) / (s_win,c - s_semi,c)`.

This gives the proposed interpretations exactly:

- 2000 = 50% semifinal probability;
- 3000 = 50% win probability.

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

If a context never crosses 50% win probability inside supported ability, 3000
is extrapolative. The product must show `win anchor not estimable` rather than
manufacture a number. FED heads with small fields and the Olympic scenario are
especially vulnerable.

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
conditional scenario. Use the 2000/3000 anchors as an interpretable display
contract after probability calibration, not as the mechanism that creates
separation. No current rating is promoted by this design document.
