"""Fail-closed validation for a Boulder anchor-discovery output directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_boulder_anchor_discovery import (  # noqa: E402
    _load_failed_windows,
    _load_records,
    _validate_cached,
    _window_from_row,
)
from video_boulder_anchor_discovery import PIPELINE_VERSION  # noqa: E402
from video_boulder_segmentation import sha256_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    required = {
        "plan": args.output_dir / "window_plan.csv",
        "contract": args.output_dir / "checkpoint_contract.json",
        "checkpoint": args.output_dir / "review_windows.jsonl",
        "failed": args.output_dir / "failed_windows.jsonl",
        "run": args.output_dir / "run_manifest.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise SystemExit("Missing output files: " + ", ".join(missing))
    contract = json.loads(required["contract"].read_text(encoding="utf-8"))
    if contract.get("pipeline_version") != PIPELINE_VERSION:
        raise SystemExit("Pipeline version mismatch")
    if contract.get("window_plan_sha256") != sha256_file(required["plan"]):
        raise SystemExit("Window-plan hash mismatch")
    windows = [_window_from_row(row) for _, row in pd.read_csv(required["plan"]).iterrows()]
    by_id = {window.window_id: window for window in windows}
    errors: list[str] = []
    records = _load_records(required["checkpoint"])
    for record in records:
        window = by_id.get(str(record["window_id"]))
        if window is None:
            errors.append(f"{record['window_id']}: not in plan")
            continue
        errors.extend(
            f"{window.window_id}: {error}"
            for error in _validate_cached(
                record, window, model=str(contract["model"]),
                pass_name=str(contract["pass_name"]),
            )
        )
    if errors:
        raise SystemExit("Validation failed:\n" + "\n".join(errors[:50]))
    completed_ids = {str(record["window_id"]) for record in records}
    failed = _load_failed_windows(required["failed"])
    if not completed_ids:
        errors.append(
            "executed checkpoint contains no valid windows; quarantines are evidence "
            "of failure, not a successful review"
        )
    for record in failed:
        window_id = str(record["window_id"])
        if window_id not in by_id:
            errors.append(f"{window_id}: quarantined window is not in plan")
        if window_id in completed_ids:
            errors.append(f"{window_id}: window cannot be completed and quarantined")
        for key, expected in (
            ("pipeline_version", PIPELINE_VERSION),
            ("pass_name", contract["pass_name"]),
            ("model", contract["model"]),
            ("structured_output_profile", contract["structured_output_profile"]),
        ):
            if record.get(key) != expected:
                errors.append(f"{window_id}: quarantine metadata mismatch: {key}")
    if errors:
        raise SystemExit("Validation failed:\n" + "\n".join(errors[:50]))
    print(
        f"Validated {len(records)} checkpointed and {len(failed)} quarantined windows "
        f"against {len(windows)} planned windows."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
