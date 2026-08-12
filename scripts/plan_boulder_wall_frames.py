"""Create an immutable, review-only frame plan from verified anchor evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from video_boulder_wall_frame_plan import plan_frame_candidates, plan_records  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _records(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"line {line_number} must be a JSON object")
        rows.append(value)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verification-records", type=Path, required=True)
    parser.add_argument("--required-verification-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-clearance-seconds", type=float, default=2.0)
    args = parser.parse_args()
    if not args.verification_records.is_file():
        raise SystemExit("verification records are missing")
    actual_sha = _sha256(args.verification_records)
    if actual_sha != args.required_verification_sha256:
        raise SystemExit("verification record hash mismatch")
    candidates = plan_frame_candidates(
        _records(args.verification_records),
        minimum_clearance_seconds=args.minimum_clearance_seconds,
    )
    payload = {
        "pipeline_version": "ifsc-boulder-wall-frame-plan-v1",
        "verification_records_sha256": actual_sha,
        "minimum_clearance_seconds": args.minimum_clearance_seconds,
        "candidate_count": len(candidates),
        "records": plan_records(candidates),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(f"Planned {len(candidates)} review-only Boulder-wall frame candidates; no media was downloaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
