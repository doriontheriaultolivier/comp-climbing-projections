# Independent review request: pathway ratings and two-anchor scales

## Decision under review

The project currently exposes Global-ELO, IFSC-ELO and WC+-ELO. Live evidence
shows identity fragmentation and severe compression in the raw probability
link: higher-rated athletes win much more often than the raw probabilities
imply. The owner proposed pathway-specific ratings:

- FED, initially federation-specific;
- NACS;
- IFSC regional/continental circuits;
- World Cups and World Championships;
- Olympics;

All competition results would help predict every target context through graph
connections and learned transfer. The context is the target being predicted,
not an input-row exclusion rule.

The owner also proposed two outcome anchors per context:

- 2000 = 50% probability of reaching a semifinal in a randomly sampled frozen
  reference qualifying competition;
- 3000 = 50% probability of winning that reference competition.

The local proposed design uses a shared dynamic latent athlete state plus
partially pooled context heads and one coherent joint ranking simulator. It is
skeptical of a standalone Olympic head and of a 50%-win anchor that may lie far
outside observed support.

## Questions requiring an honest critical answer

1. What is the statistically strongest architecture for using all results to
   predict FED/NACS/IFSC-regional/WC target-context ability without creating
   independent fragmented ledgers or unjustified transfer?
2. Does a two-anchor affine display transform add anything beyond
   interpretability? Clearly separate prediction/calibration benefits from UI
   benefits.
3. Is 50% semifinal plus 50% win a defensible pair of anchors? Analyze tail
   extrapolation, field-size sensitivity, yearly drift, and small-context data.
4. Compare alternatives: 50% semifinal + 50% final; 50% semifinal + 10% win;
   empirically supported latent quantiles; or displaying outcome probabilities
   and threshold lines without forcing two fixed numeric anchors.
5. How should reference fields be constructed: single last event, event-balanced
   2025 fields, rolling multi-year empirical fields, or standardized synthetic
   fields? Address attendance conditionality and qualification format.
6. Should Olympics be a target scenario derived from the WC head or an
   independently estimated context? What evidence threshold would justify a
   separate head?
7. How should the model treat under-18 athletes, federation-only athletes,
   disconnected graphs, sparse transfer bridges and uncertainty?
8. Give a concrete chronological evaluation and promotion protocol, including
   pair and placement metrics, event-clustered uncertainty, transfer ablations,
   anchor stability tests and failure conditions.
9. Recommend the actual best product representation for coaches. It need not
   preserve either the owner's or the local proposal.

Please be direct. Identify what is wrong or under-specified. Do not reward
complexity for its own sake. Prefer the simplest model that can be falsified and
that produces coherent named-opponent and placement probabilities.
