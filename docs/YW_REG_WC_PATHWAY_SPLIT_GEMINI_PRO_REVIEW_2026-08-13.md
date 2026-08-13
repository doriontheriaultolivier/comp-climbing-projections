# Gemini Pro independent review: YW-IFSC / REG-IFSC / WC+ split

- Model version: `gemini-3.1-pro-preview`
- Usage metadata: `{"candidatesTokenCount": 1551, "candidatesTokensDetails": [{"modality": "TEXT", "tokenCount": 1551}], "promptTokenCount": 2510, "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 2510}], "thoughtsTokenCount": 1697, "totalTokenCount": 5758, "trafficType": "ON_DEMAND"}`
- Scope: sanitized advisory methods review
- Authority: advice does not promote a research or production model

# Independent Advisory Review: YW/REG/WC+ Pathway Split

**Status:** Independent Methodological Review
**Reviewer Persona:** Senior Bayesian Sports Statistician / High-Performance Director

## 1. Candid Verdict

The legacy `ifsc_non_wc` offset is a semantic and statistical failure. Mixing youth regional and senior continental events confounds maturation, selection bias, and structural competition difficulty. The proposed split into `YW-IFSC`, `REG-IFSC`, and `WC+` is semantically necessary for high-performance coaching. 

However, the current engine's limitation to independent Gaussian context-offset priors (lacking a true hierarchical tree) poses a severe identifiability risk. If you split sparse youth categories without partial pooling, you will overfit to small event clusters. The product definitions of "Demonstrated" vs. "Ready" are excellent, provided "Ready" is strictly defined as a posterior predictive simulation against a frozen reference field, not a raw latent parameter.

## 2. Answers to Methodological Questions

**1. Correct estimand for `YW-IFSC Ready · current category`?**
The estimand is the expected placement (or probability of advancing/top-K) derived from the posterior predictive distribution of the athlete's performance against a *frozen, empirical reference field* composed of athletes currently eligible for that specific category, simulated at the target context's difficulty offset. It is a predictive quantity, not a latent coordinate.

**2. A vs. B vs. C and Staged Design?**
*   **Theoretical Best:** **C** (Hierarchical age/category trajectory). It separates the biological/maturation trajectory from the structural difficulty of the event.
*   **Executable Now:** **B** (Shared YW context + reference field). 
*   **Staged Design:** Implement **B** immediately. The current engine only supports independent Gaussian offsets. Use a single `YW_IFSC` offset to capture the structural difficulty of Youth Worlds relative to the global graph. Handle the category-specific logic entirely in the simulation step (the reference field). Build **C** for the next major engine upgrade. Reject **A** entirely; it will shatter the data and destroy predictive validity.

**3. Hierarchical Structure & Partial Pooling?**
Ideally, `Youth-Reg` and `YW` share a `Youth` parent; `REG-IFSC` and `WC+` share a `Senior/Open` parent. Because your current engine lacks full hierarchical support, you cannot natively do this. 
*Workaround:* Center the independent Gaussian priors for the new offsets on the global mean, but widen the prior variance for sparse contexts. You must upgrade the engine to support a covariance tree (`Global -> {Youth, Senior} -> {Specific Contexts}`) to properly shrink weak evidence.

**4. Encoding Category Rules and Age Transitions?**
Strictly use the governed rule *as of the event date*. Do not backcast current rules onto historical events. When an athlete ages up, their latent skill (shared graph) remains intact, but the UI dynamically swaps the reference field used for their `YW-IFSC Ready` simulation to their new category. This prevents survivor bias because historical results remain valid indirect evidence without polluting the new category's structural definition.

**5. Fresh pre-2024 Initializer?**
*Mathematically:* Not required. If the pre-2024 initializer only contains global states (no context offsets), the global skill is invariant to the downstream context split. 
*Hygiene:* Highly recommended. If the event router changed the fundamental graph topology (e.g., fixing ambiguous age-class cases), the pre-2024 graph might have subtle edge differences. Re-run it to guarantee provenance.

**6. Minimum Support and Chronological Gates?**
*   **Estimated:** A context offset requires a minimum number of anchored events (e.g., $N \ge 10$ events, $M \ge 100$ cross-context athletes) in the 2024-2026 window to be identifiable.
*   **Inherited/Shrunk:** If below the threshold, the offset prior must heavily dominate, effectively shrinking the context difficulty to the global/parent mean.
*   **Withheld:** Only withhold "Ready" estimates if the athlete is structurally disconnected from the global graph (infinite variance). Zero direct starts is *not* a valid reason to withhold.

**7. Critique of Protocol & Falsification Experiment?**
The 2024 (dev) / 2025 (val) / 2026 (descriptive) protocol is rigorous. 
*   **Falsification Experiment:** Fit the Split model (B) and the Pooled model (V4) on pre-2025 data. Predict the full-field placements (RPS) and pairwise log-loss for all 2025 `YW_IFSC` and `WC+` events. 
*   **Falsification Condition:** If the Split model's RPS on 2025 YW events is statistically worse (via paired permutation test) than the Pooled model, the split is overfitting the sparse youth data. Abort the split.

**8. Coach-Facing UI Contract?**
Do not show 7 numbers. 
*   **Primary View:** A single dropdown/toggle for the **Target Context** (e.g., "World Cup", "Youth Worlds - Juniors").
*   **Display:** A single "Readiness Gauge" (e.g., Projected Rank or Probability of Top 20) for the selected context.
*   **Evidence Badge:** A visual icon next to the gauge. 
    *   *Solid/Gold:* "Demonstrated" (Has direct starts in this context).
    *   *Hollow/Dashed:* "Projected" (Based on indirect graph evidence).
*   **Uncertainty:** A shaded credible interval around the projected rank, or a text label ("High Evidence" vs. "Sparse Evidence") based on the posterior variance.

**9. Semantic vs. Learned Assumptions?**
*   **Semantic/Product (Fixed):** Category age-eligibility rules, the composition of the frozen reference fields, the definition of "Ready" (e.g., Top 50% vs Top 10%), and the event routing taxonomy.
*   **Learned (Data-driven):** Latent athlete skill, dynamic form/volatility, the structural difficulty offsets of YW vs. WC+, and the posterior uncertainty of the athlete's state.

## 3. Summary of Recommendations

1.  **Recommended Design:** Execute Candidate B immediately. Use a single `YW_IFSC` offset and handle categories via reference-field simulations.
2.  **Rejected Alternatives:** Candidate A (Category-specific heads) is rejected due to sparsity and identifiability failure. Candidate C is rejected for the current sprint due to engine limitations, but remains the long-term target.
3.  **Failure Conditions:** If the 2025 out-of-sample RPS for the split model degrades compared to the pooled V4 model, the independent Gaussian priors are failing to regularize the split contexts. You must halt release until a hierarchical tree is implemented.
