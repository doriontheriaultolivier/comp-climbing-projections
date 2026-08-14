# Physical and board sparse-intake audit V1

## Outcome

The existing restricted snapshot is structurally usable for continued manual
data entry, but it is not large or complete enough to identify a physical
ceiling model. The audit therefore freezes coverage and modeling semantics
without fitting a model or changing the product.

Exact current coverage:

- 464 valid physical observations from 22 athletes;
- 21 physical protocols across seven capacity dimensions;
- 171 Kilter-equivalent observations from the same 22 athletes;
- 97 flash-grade observations from 21 athletes, 15 with repeats;
- 74 recent-hardest-send observations from 19 athletes, 14 with repeats;
- 97 athlete-date overlaps between physical and board observations.

Coverage is very uneven. `rate_of_force_development` currently covers seven
athletes and `lower_body_power` eleven, while two declared protocols cover one
athlete each. Missing tests must remain missing—not be interpreted as low
capacity. Low-count protocols should be prioritized only when their expected
coaching information exceeds athlete burden and protocol noise.

## Model boundary

The intended chain has three distinct layers:

1. physical tests provide uncertain, movement-demand-dependent support;
2. 50%-flash Kilter-equivalent grade measures repeatable low-pressure wall
   expression, while recent hardest sends provide an exposure-dependent
   upper-tail observation;
3. competition Zone and Top-given-Zone outcomes measure access to that support
   under time, attempt, terrain, and event context.

This is not a deterministic ceiling. Alternative beta, morphology, technique,
measurement error, and unmeasured capacities can all bypass an apparent weak
link. A raw linear correlation is descriptive and cannot authorize coaching
advice.

The next estimable model should be chronological and hierarchical: ordered
Zone then Top-given-Zone hurdles; monotone, saturating physical-support terms;
protocol-specific measurement error; partial pooling for sparse tests; explicit
missingness indicators and sensitivity analysis; and exact terrain/item tags.
Attempt efficiency is a separate conditional outcome, not a replacement for
the Zone/Top hurdle.

## Next data priorities

- keep stable athlete IDs, observation dates, units, protocol versions, and
  validity reasons for every new manual row;
- record board exposure/session volume where possible, especially for the
  three-hardest-send measure;
- retain repeats rather than replacing prior values;
- link only chronologically prior tests to competition outcomes;
- prioritize physical tagging of exact boulders already linked to these 22
  athletes before expanding a high-dimensional model.

The generated report is counts-only and hash-binds the two restricted input
tables. It authorizes no model fit, prescription, Streamlit change, or identity
merge.
