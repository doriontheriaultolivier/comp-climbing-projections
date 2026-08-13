"""Run one bounded Vertex Gemini Pro review of the coaching-card diff."""

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
ARTIFACT_DIR = ROOT / ".artifacts" / "physical-board-coaching-slice-gemini-review-v1"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def build_prompt() -> str:
    diff = subprocess.check_output(
        [
            "git",
            "diff",
            "--stat",
            "--patch",
            "origin/comp-climbing-v2...HEAD",
        ],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
    )
    contract = (ROOT / "docs" / "PHYSICAL_BOARD_COACHING_SLICE_V1.md").read_text(
        encoding="utf-8"
    )
    return f"""You are an independent senior sports scientist, applied statistician,
climbing coach-product reviewer, and skeptical software reviewer. Review this
private athlete-profile coaching slice. No private measurements or raw testing
rows are included. The product goal is to separate physical capacity, lower-
pressure board expression, and later matched competition expression.

Check especially:
1. whether the semantics of 50% flash versus recent three-hardest-send are honest;
2. whether any wording turns ordinal V grades into an interval, pressure score,
   causal limiter, or prescription;
3. whether filtering to precomputed Focus candidate rows is useful and safe;
4. missing/unlinked/inconclusive athlete behavior;
5. privacy and identity implications of adding the two governed CSVs to the
   compact private app;
6. Streamlit usability and code/test defects;
7. the smallest material improvements required before merge.

Return: VERDICT (ACCEPT / REVISE / REJECT), P0/P1/P2 findings with exact evidence,
specific fixes, and what should remain explicitly withheld. Be critical; do not
invent facts or arbitrary evidence thresholds.

===== CONTRACT =====
{contract}
===== COMMIT DIFF =====
{diff}
"""


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    prompt = build_prompt()
    prompt_bytes = prompt.encode("utf-8")
    token = subprocess.check_output(
        [str(GCLOUD), "auth", "print-access-token", "--quiet"],
        text=True,
        encoding="utf-8",
    ).strip()
    if not token or any(char.isspace() for char in token):
        raise RuntimeError("Vertex access token unavailable")
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.15,
            "maxOutputTokens": 8_192,
            "thinkingConfig": {"thinkingBudget": 4_096},
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
        with urllib.request.urlopen(request, timeout=600) as response:
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
        raise RuntimeError("Vertex returned no review")

    (ARTIFACT_DIR / "prompt.txt").write_bytes(prompt_bytes)
    (ARTIFACT_DIR / "request.json").write_bytes(request_bytes)
    (ARTIFACT_DIR / "response.json").write_bytes(response_bytes)
    (ARTIFACT_DIR / "review.md").write_text(review + "\n", encoding="utf-8")
    receipt = {
        "schema": "physical-board-coaching-slice-gemini-review-v1",
        "model_requested": MODEL,
        "model_version": body.get("modelVersion", MODEL),
        "traffic_type": body.get("usageMetadata", {}).get("trafficType"),
        "usage_metadata": body.get("usageMetadata", {}),
        "prompt_sha256": sha256_bytes(prompt_bytes),
        "request_sha256": sha256_bytes(request_bytes),
        "response_sha256": sha256_bytes(response_bytes),
        "review_sha256": sha256_bytes((review + "\n").encode("utf-8")),
        "private_measurements_transmitted": False,
        "promotion_authority": False,
    }
    (ARTIFACT_DIR / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
