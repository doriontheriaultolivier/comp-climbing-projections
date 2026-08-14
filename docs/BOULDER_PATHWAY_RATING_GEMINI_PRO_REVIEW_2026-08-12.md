# Gemini Pro independent review: pathway ratings and two-anchor scales

- Model version: `gemini-3.1-pro-preview`
- Usage metadata: `{"candidatesTokenCount": 1380, "candidatesTokensDetails": [{"modality": "TEXT", "tokenCount": 1380}], "promptTokenCount": 5610, "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 5610}], "thoughtsTokenCount": 1416, "totalTokenCount": 8406, "trafficType": "ON_DEMAND"}`
- Scope: sanitized advisory methods review
- Authority: advice does not promote a research or production model

# Independent Review: Pathway Ratings and Two-Anchor Scales

## Verdict

The proposal to move from a fragmented Elo system to a pathway-aware model is conceptually sound and highly valuable for high-performance coaching. However, the owner’s specific proposal conflates **probability calibration** with **display scaling**, relies on an unstable 50%-win anchor, and attempts to fit an unidentifiable Olympic head. 

The local proposal’s architecture—a shared dynamic latent athlete state with partially pooled context heads—is the statistically correct approach. It solves the identity and transfer problems without creating isolated ledgers. However, no rating scale should be published until the underlying identity fragmentation (e.g., the Amari Bourbonnais split) is resolved and the raw probability link is calibrated via the demonstrated temperature scaling (T=3.0).

***Disclaimer:** This advice is advisory. It does not authorize the promotion of any specific model to production.*

---

## Core Distinctions

To prevent future architectural confusion, the system must strictly separate three concepts:

1. **Predictive Validity (The Joint Simulator):** Does the model correctly forecast who beats whom, and who places where? The current raw probabilities are severely compressed. The T=3.0 temperature audit proves that scaling the latent location improves pair log-loss and placement RPS. This is where calibration happens.
2. **Identifiability (The Architecture):** Can the model mathematically distinguish an athlete's skill from the field's strength and the event's format? An Olympic head cannot be identified because one event every four years cannot separate form, field, and format. 
3. **Display Interpretability (The Anchors):** The 2000/3000 affine transform. This adds **zero** predictive power. It is purely a UI translation layer to make latent logits meaningful to a coach. It must only be applied *after* predictive validity is secured.

---

## Recommended Design

**1. Statistical Architecture:**
Implement `ability_i,t,c = shared_skill_i,t + context_response_i,c + context_offset_c`. 
*   **Shared Skill:** A dynamic latent state updated by all valid competition evidence.
*   **Context Heads:** FED, NACS, IFSC-REG, and WC. These are partially pooled. Transfer is learned via bridge athletes, never assumed.
*   **Youth/Under-18:** Modeled as a transfer problem with age-trajectory uncertainty, not a direct youth bonus.

**2. The Display Scale (Modified Anchors):**
*   **2000 = 50% probability of reaching Semifinal** (Top 20).
*   **3000 = 50% probability of reaching Final** (Top 6).
*   *Why change the upper anchor?* A 50% win probability is highly vulnerable to tail extrapolation. In eras of extreme dominance (e.g., Janja Garnbret), the latent skill required for a generic athlete to have a 50% win probability may lie far outside observed support. 50% Final is empirically dense, stable across years, and highly relevant to high-performance funding metrics.

**3. Reference Fields:**
Use a frozen, event-balanced empirical reference distribution (e.g., 2025 fields). Do not use "the last event," which introduces severe field-size and attendance volatility. Update via a rolling multi-year empirical field to smooth out yearly drift.

---

## Rejected Alternatives

*   **Standalone Olympic Head:** Rejected. N=1 every four years is statistically unidentifiable. The Olympics must be treated as a **target scenario** derived from the WC head, conditioned on the exact Olympic field and format.
*   **50% Win Anchor:** Rejected. Highly sensitive to field size, dominant outliers, and small-context data (FED). It will force the model to manufacture fake numbers (extrapolation) for lower-tier pathways.
*   **Independent Elo Ledgers:** Rejected. Fragmenting the graph destroys the ability to evaluate pathway readiness for athletes moving from NACS to WC.
*   **Stretching Elo before Identity Closure:** Rejected. Applying an affine transform to a broken identity graph simply makes wrong identities look more confidently wrong.

---

## Validation Protocol

Before any product representation is shown to users, the following chronological evaluation must be passed:

1.  **Identity Closure:** Rebuild canonical identity histories (resolving the 1,136 normalized-name groups and known splits).
2.  **Chronological Rolling-Origin Fit:** Train strictly on past events. No target event or anchor estimation in the training set.
3.  **Calibration Metrics:** 
    *   Event-balanced pair log-loss and placement RPS (must beat the raw v4 baseline).
    *   Calibration curves by probability, rating gap, and age (Adult vs. U18).
4.  **Transfer Ablations:** Prove that the `shared_skill + context_response` model outperforms both a fully pooled global model and fully isolated context models.
5.  **Failure Conditions:** If a pathway lacks sufficient bridge athletes to link to the global graph, or if an athlete lacks minimum evidence, the output must strictly return `not estimable`.

---

## Coach-Facing Product Representation

The UI must prioritize honesty over precision. Coaches need to understand *readiness* and *transfer*, not just a single number.

**The Athlete Profile Card:**
*   **Primary Display:** The athlete's rating in their *most proven* context (e.g., `NACS: 2450 ± 40`).
*   **Pathway Ladder:** A visual spectrum showing the athlete's projected rating across all contexts.
    *   *Example:* `FED: 2800` (Dominant) ➔ `NACS: 2450` (Competitive) ➔ `WC: 1900` (Developmental).
*   **Uncertainty & Evidence:** Explicitly show the 95% credible interval and the direct event count in that specific context.
*   **Transfer Diagnostics:** If an athlete has high shared physical skill but a low WC context response, flag this as a "Transfer Deficit" (indicating a need for tactical/pressure work rather than physical strength).
*   **Quarantine States:** Use explicit UI states for `linked, minimum evidence not met` and `unconnected graph`. Blank cards or manufactured extrapolations are prohibited.
