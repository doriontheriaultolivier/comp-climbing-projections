"""Run one bounded Vertex Gemini Pro review of the pathway split."""

from __future__ import annotations

import hashlib
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
REQUEST = ROOT / "docs/YW_REG_WC_PATHWAY_SPLIT_GEMINI_REVIEW_REQUEST_2026-08-13.md"
CONTRACT = ROOT / "docs/PATHWAY_DEMONSTRATED_AND_READY_HEADS_V1.md"
OUTPUT = ROOT / "docs/YW_REG_WC_PATHWAY_SPLIT_GEMINI_PRO_REVIEW_2026-08-13.md"
RECEIPT = ROOT / "docs/YW_REG_WC_PATHWAY_SPLIT_GEMINI_PRO_REVIEW_2026-08-13.receipt.json"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def build_prompt() -> str:
    sections = []
    for path in (REQUEST, CONTRACT):
        sections.append(
            f"\n\n===== BEGIN {path.name} =====\n"
            + path.read_text(encoding="utf-8")
            + f"\n===== END {path.name} =====\n"
        )
    return (
        "Act as an independent senior Bayesian sports statistician, dynamic "
        "ranking-model researcher, youth-development methodologist, climbing "
        "high-performance director, and skeptical product designer. This is a "
        "privacy-sanitized advisory review. Be critical: do not assume the "
        "owner's or Codex's proposed taxonomy or validation protocol is optimal. "
        "Separate identifiability, predictive validity, semantic correctness, "
        "and UI usefulness. Do not fabricate empirical results. Advice cannot "
        "authorize model promotion. Return concise but technically detailed Markdown."
        + "".join(sections)
    )


def main() -> int:
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
            "temperature": 0.1,
            "maxOutputTokens": 12_000,
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
            body_bytes = response.read()
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Vertex request failed with HTTP {error.code}") from error
    body = json.loads(body_bytes.decode("utf-8"))
    candidate = body.get("candidates", [{}])[0]
    if candidate.get("finishReason") != "STOP":
        raise RuntimeError(f"Vertex response incomplete: {candidate.get('finishReason')}")
    review = "\n".join(
        part["text"]
        for part in candidate.get("content", {}).get("parts", [])
        if isinstance(part.get("text"), str)
    ).strip()
    if not review:
        raise RuntimeError("Vertex returned no review text")
    usage = body.get("usageMetadata", {})
    output_text = (
        "# Gemini Pro independent review: YW-IFSC / REG-IFSC / WC+ split\n\n"
        f"- Model version: `{body.get('modelVersion', MODEL)}`\n"
        f"- Usage metadata: `{json.dumps(usage, sort_keys=True)}`\n"
        "- Scope: sanitized advisory methods review\n"
        "- Authority: advice does not promote a research or production model\n\n"
        + review
        + "\n"
    )
    OUTPUT.write_text(output_text, encoding="utf-8")
    receipt = {
        "schema": "yw-reg-wc-pathway-gemini-review-v1",
        "status": "COMPLETE_ADVISORY_NO_PROMOTION_AUTHORITY",
        "model": body.get("modelVersion", MODEL),
        "usage": usage,
        "prompt_sha256": sha256_bytes(prompt_bytes),
        "request_sha256": sha256_bytes(REQUEST.read_bytes()),
        "contract_sha256": sha256_bytes(CONTRACT.read_bytes()),
        "raw_response_sha256": sha256_bytes(body_bytes),
        "review_sha256": sha256_bytes(output_text.encode("utf-8")),
        "authority": {"research_promotion": False, "production": False, "app": False},
    }
    RECEIPT.write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
