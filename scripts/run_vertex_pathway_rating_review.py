"""Send the sanitized pathway-rating design to Vertex Gemini Pro."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
GCLOUD = (
    Path.home() / "AppData" / "Local" / "Google" / "Cloud SDK"
    / "google-cloud-sdk" / "bin" / "gcloud.cmd"
)
MODEL = "gemini-3.1-pro-preview"
ENDPOINT = (
    "https://aiplatform.googleapis.com/v1/projects/"
    "ifsc-performance-infra-2026/locations/global/publishers/google/models/"
    f"{MODEL}:generateContent"
)
INPUTS = (
    Path("docs/BOULDER_PATHWAY_RATING_GEMINI_REVIEW_REQUEST_2026-08-12.md"),
    Path("docs/BOULDER_PATHWAY_LEVEL_RATING_HEADS_V1.md"),
    Path("docs/ATHLETE_IDENTITY_AND_SKILL_SCALE_REPAIR_V1.md"),
    Path("docs/BOULDER_JOINT_TEMPERATURE_LOCKED_AUDIT_2026-08-12.md"),
    Path("scripts/pathway_level_rating_scale_v1.py"),
)
OUTPUT = ROOT / "docs/BOULDER_PATHWAY_RATING_GEMINI_PRO_REVIEW_2026-08-12.md"


def build_prompt() -> str:
    sections = []
    for relative in INPUTS:
        body = (ROOT / relative).read_text(encoding="utf-8", errors="strict")
        sections.append(
            f"\n\n===== BEGIN {relative.as_posix()} =====\n{body}"
            f"\n===== END {relative.as_posix()} =====\n"
        )
    return (
        "This is an explicitly authorized, privacy-sanitized methods review. "
        "No private athlete measurements, credentials, or local paths are in "
        "the packet. Act as an independent senior Bayesian statistician, "
        "ranking-model researcher, climbing high-performance director, and "
        "skeptical product designer. Do not merely endorse the proposal. "
        "Distinguish predictive validity, identifiability, and display "
        "interpretability. Return Markdown with a verdict, recommended design, "
        "rejected alternatives, validation protocol, and coach-facing product "
        "representation. Advice is advisory and cannot promote a model."
        + "".join(sections)
    )


def main() -> int:
    token = subprocess.check_output(
        [str(GCLOUD), "auth", "print-access-token", "--quiet"],
        text=True,
        encoding="utf-8",
    ).strip()
    if not token or any(character.isspace() for character in token):
        raise RuntimeError("Vertex access token unavailable")
    payload = {
        "contents": [{"role": "user", "parts": [{"text": build_prompt()}]}],
        "generationConfig": {
            "temperature": 0.15,
            "maxOutputTokens": 16_384,
            "thinkingConfig": {"thinkingBudget": 8_192},
        },
    }
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Vertex request failed with HTTP {error.code}") from error
    candidate = body.get("candidates", [{}])[0]
    if candidate.get("finishReason") != "STOP":
        raise RuntimeError("Vertex response incomplete")
    review = "\n".join(
        part["text"]
        for part in candidate.get("content", {}).get("parts", [])
        if isinstance(part.get("text"), str)
    ).strip()
    if not review:
        raise RuntimeError("Vertex returned no review text")
    usage = body.get("usageMetadata", {})
    OUTPUT.write_text(
        "# Gemini Pro independent review: pathway ratings and two-anchor scales\n\n"
        f"- Model version: `{body.get('modelVersion', MODEL)}`\n"
        f"- Usage metadata: `{json.dumps(usage, sort_keys=True)}`\n"
        "- Scope: sanitized advisory methods review\n"
        "- Authority: advice does not promote a research or production model\n\n"
        + review + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(OUTPUT), "model": body.get("modelVersion"), "usage": usage}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
