# Resolution of Gemini Pro pathway-graph review

## Accepted

- Use worldwide context families: federation, North-American inter-federation
  (NACS), five recognised continental councils, World, Youth World, and an
  Olympic scenario.
- Keep one shared time-varying athlete state; never create isolated Elo ledgers.
- Estimate transfer through chronologically observed bridge athletes and
  hierarchical shrinkage. Do not hand-code graph-distance weights.
- Keep NACS distinct below Pan America because it is a meaningful developmental
  circuit and currently has 20 Boulder event contexts plus bridge evidence.
- Keep Youth World distinct from adult World and estimate the transition.
- Defer terrain-mediated transfer until Boulder Tags is frozen and historically
  backfilled.
- Compare fully pooled, independent-offset and hierarchical candidates using
  event-balanced pair log loss, placement RPS and whole-competition resampling.
- Retain 2000=50% semifinal and 3000=50% final as display anchors; show win
  probability separately.

## Important correction to the review

Gemini proposed
`eta(i,t,c)=shared_skill(i,t)+context_offset(c)+athlete_context_effect(i,c)` and
called the hierarchical population context offset the first implementation
stage. A population-wide `context_offset(c)` is constant for all athletes inside
one target event and therefore cancels from pairwise ordering and placement
probabilities. It helps compare display locations across contexts, but it cannot
by itself repair athlete ordering or target-context prediction.

The predictive hierarchy must ultimately shrink athlete-context deviations (or
context-specific loading of athlete traits) across related contexts. The
minimal existing-core independent-offset candidate remains useful as an
ablation, not as the recommended final model. The next new model core should
estimate an athlete-by-context low-rank or hierarchical response with uncertainty
and with population-level context calibration kept separate.

## Evidence-dependent implementation

The support audit is continuous, not a binary ten-events rule. Sparse heads
borrow more strength and carry more uncertainty. Product rendering may abstain
when an interval is too wide for the requested decision, but an arbitrary fixed
width must not become a universal validity threshold.

Current implementation sequence:

1. freeze/audit taxonomy, pool normalization and context support;
2. run the existing fully pooled versus independent shrunk-offset ablation;
3. implement a separate hierarchical athlete-context response candidate;
4. compare both chronologically without reopening locked model-selection years;
5. add terrain transfer only after time-correct Boulder Tags evidence exists.

Gemini usage for this advisory review was 7,423 prompt, 1,129 answer and 1,576
thinking tokens (10,128 total), using `gemini-3.1-pro-preview` on Vertex.
