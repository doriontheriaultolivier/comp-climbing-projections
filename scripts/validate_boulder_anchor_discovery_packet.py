#!/usr/bin/env python3
"""Validate the frozen 2026 Boulder anchor-discovery packet offline."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "data/video_boulder_anchor_discovery_2026_v1"
EXPECTED_EVENT_WINDOWS = {1479: 25, 1480: 26, 1482: 26}
OUTPUT_HASH_KEYS = {
    "coverage_summary_sha256": "coverage_summary.json",
    "coverage_windows_sha256": "coverage_windows.csv",
    "merged_failed_windows_sha256": "merged_failed_windows.jsonl",
    "merged_review_windows_sha256": "merged_review_windows.jsonl",
    "verification_window_plan_sha256": "verification_window_plan.csv",
}
SAFETY_GATES = (
    "production_use_allowed",
    "athlete_scoring_allowed",
    "athlete_comparison_allowed",
    "elo_update_allowed",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(packet: Path = PACKET) -> list[str]:
    errors: list[str] = []
    required = {
        "coverage_summary.json",
        "coverage_windows.csv",
        "merged_failed_windows.jsonl",
        "merged_review_windows.jsonl",
        "merge_manifest.json",
        "verification_window_plan.csv",
    }
    missing = sorted(name for name in required if not (packet / name).is_file())
    if missing:
        return ["missing Boulder discovery packet files: " + ", ".join(missing)]

    try:
        summary = json.loads((packet / "coverage_summary.json").read_text(encoding="utf-8"))
        manifest = json.loads((packet / "merge_manifest.json").read_text(encoding="utf-8"))
        reviews = [
            json.loads(line)
            for line in (packet / "merged_review_windows.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        with (packet / "coverage_windows.csv").open(encoding="utf-8", newline="") as handle:
            coverage = list(csv.DictReader(handle))
    except (OSError, json.JSONDecodeError, csv.Error) as exc:
        return [f"invalid Boulder discovery packet: {exc}"]

    if summary.get("planned_window_count") != 77 or summary.get("completed_window_count") != 77:
        errors.append("Boulder discovery coverage must remain 77/77")
    if summary.get("missing_window_count") != 0 or summary.get("quarantined_window_count") != 0:
        errors.append("Boulder discovery packet must have zero missing and quarantined windows")
    observed_events = {
        int(row["event_id"]): int(row["completed"])
        for row in summary.get("coverage_by_event", [])
    }
    if observed_events != EXPECTED_EVENT_WINDOWS:
        errors.append("Boulder discovery event coverage changed")
    if len(reviews) != 77 or len(coverage) != 77:
        errors.append("Boulder discovery review and coverage rows must both equal 77")
    if (packet / "merged_failed_windows.jsonl").read_bytes() != b"":
        errors.append("Boulder discovery failed-window file must remain empty")

    window_ids = [str(row.get("window_id", "")) for row in reviews]
    if not all(window_ids) or len(window_ids) != len(set(window_ids)):
        errors.append("Boulder discovery window IDs must be present and unique")
    for row in reviews:
        if any(row.get(gate) is not False for gate in SAFETY_GATES):
            errors.append("Boulder discovery review safety gate opened")
            break
    if any(summary.get(gate) is not False for gate in SAFETY_GATES):
        errors.append("Boulder discovery summary safety gate opened")

    outputs = manifest.get("outputs", {})
    for key, filename in OUTPUT_HASH_KEYS.items():
        if outputs.get(key) != _sha256(packet / filename):
            errors.append(f"Boulder discovery output hash mismatch: {filename}")
    source_events = [row.get("event_id") for row in manifest.get("sources", [])]
    if source_events != [1479, 1480, 1482]:
        errors.append("Boulder discovery source events changed")
    if manifest.get("verification_executes_model") is not False:
        errors.append("Boulder discovery merge must not execute the verification model")
    return errors


def main() -> int:
    errors = validate()
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("Boulder anchor discovery packet valid: 77/77 windows; zero missing or quarantined; safety gates closed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
