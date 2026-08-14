"""Request one bounded Gemini design review for the physical-transfer model.

Only aggregate coverage and the research question leave the local machine.  No
athlete names, test values, competition outcomes, or private rows are included.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
MODEL = "gemini-3.1-pro-preview"
PROJECT = "ifsc-performance-infra-2026"
ENDPOINT = (
    f"https://aiplatform.googleapis.com/v1/projects/{PROJECT}/locations/global/"
    f"publishers/google/models/{MODEL}:generateContent"
)
GCLOUD = (
    Path.home()
    / "AppData"
    / "Local"
    / "Google"
    / "Cloud SDK"
    / "google-cloud-sdk"
    / "bin"
    / "gcloud.cmd"
)
ARTIFACT_DIR = ROOT / ".artifacts" / "physical-transfer-ceiling-design-v1"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def build_prompt() -> str:
    return """You are an independent senior Bayesian statistician, sport scientist,
competition-climbing researcher, causal-inference skeptic, and coaching product
designer. Design the smallest honest model family and validation programme for
this question:

How do physical-test capacities constrain an athlete's achievable performance,
how much of that capacity becomes accessible on a lower-pressure training board,
and how much board performance transfers to physically tagged competition
boulders under competition context and pressure?

PROJECT FACTS (aggregate only; no private rows are transmitted)
- Competition climbing is the pilot for a reusable sport-performance analytics company.
- The private governed data currently has 22 safely linked athlete profiles; 18
  have both a dated 50%-flash Kilter-equivalent grade and a dated average of the
  last three months' hardest physical Kilter-equivalent sends.
- Physical tests are irregular, repeated, sparse by metric, and not all athletes
  take all tests. Missing means unobserved, not weak.
- A governed queue contains 535 competition problems prioritized for human demand
  tagging. The first 10 would connect 57 athlete-item links, but tags are not yet
  complete. Per-problem outcomes can preserve Zone/Top and attempt ordinals where
  the source semantics are certified.
- The current release keeps physical capacity, board expression, and competition
  expression separate and makes no transfer or causal claim.
- Youth and senior fields have different access, category, terrain and selection
  processes. Youth results may inform senior ability indirectly, but the project
  has already found that naive youth-to-senior and Canada-to-world transport can
  create serious miscalibration.
- Competition participation is selected. Training-board sends are also selected
  by attempts, access, style choice and reporting. V grades are ordinal and must
  not silently be treated as interval measurements.
- The intended coaching output is not a universal talent score. It should help
  distinguish plausible limiting layers: physical capacity, training-terrain
  expression, unfamiliar-terrain transfer, or competition accessibility/stress.
- Chronology matters: predictors must be available before the target event.
- Data scarcity is expected for years; partial pooling is desirable, but confident
  individual labels are not.

OWNER'S HYPOTHESIS TO CRITIQUE
Physical tests behave like a ceiling; board performance is an intermediate
lower-pressure accessibility layer; lack of transfer or competition-stress
management reduces access to that ceiling. Therefore linear correlations are
not the right final model. The owner may be directionally right but wants the
best model, not confirmation.

DELIVERABLE
1. State the estimands precisely. Separate prediction, description and causal
   claims. Explain what is and is not identifiable with the present data.
2. Compare at least these candidates: hierarchical stochastic-frontier/ceiling,
   monotone latent-variable or IRT mediation, ordinal cumulative-link models,
   Bayesian GAM/quantile frontier, and a simple regularized baseline. Reject any
   whose assumptions are untenable.
3. Recommend one smallest executable V1 and one more ambitious successor. Give
   equations or a generative description, likelihoods for ordinal grades and
   Zone/Top/attempt outcomes, hierarchy/priors, handling of missing tests, repeated
   athletes/items/events, selection and measurement error.
4. Specify a chronology-safe evaluation: folds, scoring rules, baselines,
   competition/athlete clustering, posterior predictive checks, calibration,
   subgroup harm/transport guards and negative controls. Do not invent arbitrary
   pass thresholds; explain how thresholds should be registered.
5. Explain how to communicate individual coaching outputs without turning
   uncertainty into false limiter labels. Include useful outputs when evidence is
   insufficient.
6. Give an incremental data-collection plan ranked by information value, including
   which competition problems to tag and which physical/board observations matter.
7. Identify failure modes, leakage risks and reasons this project could falsely
   conclude that an athlete has a transfer or pressure problem.
8. End with a concise GO / REVISE / NO-GO recommendation for implementing V1 now,
   and the exact prerequisites.

Be critical and technically explicit. Prefer a simpler falsifiable model over a
beautiful unidentifiable one. Do not propose using age, federation or pathway as
causal biological effects; they may be descriptive/transport strata. Do not
fabricate evidence, sample-size rules, effect sizes or citations."""


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    prompt = build_prompt()
    prompt_bytes = prompt.encode("utf-8")
    token = subprocess.check_output(
        [str(GCLOUD), "auth", "print-access-token", "--quiet"],
        text=True,
        encoding="utf-8",
    ).strip()
    if not token or any(character.isspace() for character in token):
        raise RuntimeError("Vertex access token unavailable")

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.12,
            "maxOutputTokens": 10_000,
            "thinkingConfig": {"thinkingBudget": 6_000},
        },
    }
    request_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        ENDPOINT,
        data=request_bytes,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            response_bytes = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Vertex HTTP {error.code}: {detail}") from error

    body = json.loads(response_bytes.decode("utf-8"))
    candidate = body.get("candidates", [{}])[0]
    if candidate.get("finishReason") != "STOP":
        raise RuntimeError(f"Incomplete response: {candidate.get('finishReason')}")
    review = "\n".join(
        part["text"]
        for part in candidate.get("content", {}).get("parts", [])
        if isinstance(part.get("text"), str)
    ).strip()
    if not review:
        raise RuntimeError("Vertex returned no design review")

    (ARTIFACT_DIR / "prompt.txt").write_bytes(prompt_bytes)
    (ARTIFACT_DIR / "request.json").write_bytes(request_bytes)
    (ARTIFACT_DIR / "response.json").write_bytes(response_bytes)
    (ARTIFACT_DIR / "review.md").write_text(review + "\n", encoding="utf-8")
    usage = body.get("usageMetadata", {})
    receipt = {
        "schema": "physical-transfer-ceiling-design-review-v1",
        "model_requested": MODEL,
        "model_version": body.get("modelVersion", MODEL),
        "traffic_type": usage.get("trafficType"),
        "usage_metadata": usage,
        "prompt_sha256": sha256_bytes(prompt_bytes),
        "request_sha256": sha256_bytes(request_bytes),
        "response_sha256": sha256_bytes(response_bytes),
        "review_sha256": sha256_bytes((review + "\n").encode("utf-8")),
        "private_rows_transmitted": False,
        "athlete_names_transmitted": False,
        "promotion_authority": False,
    }
    (ARTIFACT_DIR / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
