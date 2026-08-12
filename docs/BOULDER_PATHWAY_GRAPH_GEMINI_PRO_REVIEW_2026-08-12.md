# Gemini Pro independent review: worldwide pathway graph transfer

- Model version: `gemini-3.1-pro-preview`
- Usage metadata: `{"candidatesTokenCount": 1129, "candidatesTokensDetails": [{"modality": "TEXT", "tokenCount": 1129}], "promptTokenCount": 7423, "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 7423}], "thoughtsTokenCount": 1576, "totalTokenCount": 10128, "trafficType": "ON_DEMAND"}`
- Scope: sanitized advisory methods review
- Authority: advice does not promote a research or production model

Here is the critical review of the proposed Boulder pathway graph and transfer model, evaluated from the perspective of a Bayesian ranking researcher and high-performance director.

### Verdict
The conceptual architecture—a single shared dynamic athlete graph with partially pooled context heads—is the correct theoretical approach for climbing's fragmented pathway. However, the proposed Python implementation (`fixed_independent_shrunk_context_offsets`) contradicts the design document's call for partial pooling. Independent offsets will fail catastrophically in sparse regions (Africa, Oceania). The design is approved for staging, provided the generative model is upgraded to a true hierarchical structure and terrain-based transfer is strictly deferred until Boulder Tags data is available.

### Recommended Generative Model
**Direct Answer to Q1 & Q5:** Shared time-varying skill plus partially pooled context response is the best practical model, but the current code's independent offsets are insufficient. 
The model must be: `eta(i,t,c) = shared_skill(i,t) + context_offset(c) + athlete_context_effect(i,c)`.
*   **Hierarchy:** `context_offset(c)` must be drawn from a hierarchical prior (e.g., Federation offsets drawn from their parent Continental distribution, which is drawn from a Global distribution). 
*   **Youth Modeling:** Youth contexts (e.g., `ifsc_world_youth`) must be treated as distinct context heads `c`, not as a covariate on the athlete. This prevents age leakage and stops Youth World results from directly inflating adult WC ratings, while still allowing the shared `shared_skill(i,t)` to learn the transition via athletes who compete in both.

### Transfer Mechanism
**Direct Answer to Q2:** Cross-context transfer must currently rely *exclusively* on bridge athletes and the covariance of the hierarchical context offsets. 
*   Do not use manual pathway distances or arbitrary graph-edge weights; graph distance is heavily confounded by selection bias (only the best athletes travel). 
*   Future terrain distribution (Boulder Tags) is highly promising for transfer but must be excluded until the tagging evidence is frozen and historically backfilled. Attempting to guess terrain difficulty now will introduce unidentifiable bias.

### Identifiability Limits
**Direct Answer to Q3 & Q4:** 
*   **Sparse Regions:** With only 3 bridge athletes for Africa and 22 for Oceania, independent context offsets are unidentifiable and will either explode or collapse to zero depending on the prior. They can only be estimated if they borrow strength via a hierarchical Continental prior. 
*   **NACS:** NACS has 69 bridge athletes and 20 events. This is sufficient to identify it as its own inter-federation context, provided it sits hierarchically below the Pan-American continental prior. It should not be merged into Pan-America, as it serves a distinct developmental coaching purpose.

### Staged Implementation
Do not build the full athlete-by-context response matrix immediately.
1.  **Stage 1 (Current):** Implement shared dynamic skill + *hierarchical* (not independent) context offsets.
2.  **Stage 2:** Introduce athlete-specific context responses (`gamma(i,c)`) only for athletes with sufficient evidence in multiple contexts, heavily regularized toward zero.
3.  **Stage 3 (Future):** Integrate Boulder Tags to model transfer based on governed terrain/item-distribution distance.

### Validation Protocol
**Direct Answer to Q6:** The minimum credible test requires strict chronological, rolling-origin evaluation.
*   **Ablations:** Compare (1) fully pooled global skill, (2) independent context offsets (the current script), and (3) hierarchical context offsets.
*   **Metrics:** Event-balanced pairwise log loss and placement Ranked Probability Score (RPS) on held-out target events.
*   **Diagnostics:** Whole-competition bootstrap intervals to prove that sparse contexts (like Africa) correctly output wide uncertainty bands rather than falsely confident point estimates.

### Rejected Alternatives
**Direct Answer to Q7:** 
*   **The 50% Win Anchor:** Rejected. The audit proves this threshold (2628 for Men, 2683 for Women) lies outside the observed skill support. Using it requires dangerous tail extrapolation.
*   **Independent Elo Ledgers:** Rejected. Splitting the pathway into five isolated ledgers destroys the bridge-athlete information necessary to evaluate pathway readiness.
*   **OLYM as a Standalone Head:** Rejected. The Olympics is a format scenario, not a statistically identifiable rating head.

### Coach-Facing Representation
The 2000 (50% Semifinal) and 3000 (50% Final) anchors are mathematically sound and empirically supported by the 2025 WC+ reference fields. However, for high-performance directors, a single number is dangerous. 
*   The UI must explicitly state the reference field (e.g., "WC 2025 Reference").
*   Ratings for disconnected or sparse federations must be masked as `not estimable` if the credible interval exceeds a predefined width.
*   The profile must visualize the *transfer route* (e.g., showing that an athlete's high FED rating is driving their WC projection via shared skill, but highlighting the lack of direct WC evidence).
