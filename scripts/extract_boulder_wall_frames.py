"""Extract bounded review-only frame candidates from authorized local media.

The input must be a plan emitted by ``plan_boulder_wall_frames.py``.  This
script never discovers or downloads a URL.  It only calls a supplied ffmpeg
binary against files named ``<video_id>.mp4`` in a local media directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_plan(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("pipeline_version") != "ifsc-boulder-wall-frame-plan-v1":
        raise ValueError("invalid wall-frame plan")
    records = value.get("records")
    if not isinstance(records, list):
        raise ValueError("wall-frame plan has no records")
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("wall-frame plan record is invalid")
        if record.get("candidate_status") != "REQUIRES_VISUAL_EMPTY_WALL_REVIEW" or record.get("empty_wall_verified") is not False:
            raise ValueError("wall-frame plan record is not review-only")
        if any(record.get(gate) is not False for gate in (
            "production_use_allowed", "athlete_scoring_allowed", "athlete_comparison_allowed", "elo_update_allowed"
        )):
            raise ValueError("wall-frame plan record has an open safety gate")
    return value


def _candidate_command(ffmpeg: str, media: Path, output: Path, seconds: float) -> list[str]:
    return [
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-ss", f"{seconds:.3f}",
        "-i", str(media), "-frames:v", "1", "-vf", "scale='min(1920,iw)':-2", "-q:v", "2", "-y", str(output),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--required-plan-sha256", required=True)
    parser.add_argument("--media-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--max-frames", type=int, default=24)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.max_frames <= 24:
        raise SystemExit("max-frames must be between 1 and 24")
    if not args.plan.is_file() or _sha256(args.plan) != args.required_plan_sha256:
        raise SystemExit("wall-frame plan hash mismatch")
    plan = _load_plan(args.plan)
    records = list(plan["records"])
    if len(records) > args.max_frames:
        records = records[:args.max_frames]
    if not args.execute:
        print(f"Dry run planned {len(records)} local frame extractions; no media was read or written.")
        return 0
    if shutil.which(args.ffmpeg) is None:
        raise SystemExit("ffmpeg executable is unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipt_rows: list[dict[str, object]] = []
    for record in records:
        video_id = str(record["video_id"])
        media = args.media_dir / f"{video_id}.mp4"
        if not media.is_file():
            raise SystemExit(f"required local media is missing for video {video_id}")
        candidate_id = str(record["candidate_id"])
        output = args.output_dir / f"{candidate_id}.jpg"
        if output.exists():
            raise SystemExit(f"refusing to overwrite existing frame {candidate_id}")
        command = _candidate_command(args.ffmpeg, media, output, float(record["frame_seconds"]))
        subprocess.run(command, check=True)
        if not output.is_file() or output.stat().st_size == 0:
            raise SystemExit(f"ffmpeg did not create a non-empty frame {candidate_id}")
        receipt_rows.append({
            "candidate_id": candidate_id, "video_id": video_id,
            "source_media_sha256": _sha256(media), "frame_seconds": record["frame_seconds"],
            "frame_filename": output.name, "frame_sha256": _sha256(output), "frame_bytes": output.stat().st_size,
            "candidate_status": "REQUIRES_VISUAL_EMPTY_WALL_REVIEW", "empty_wall_verified": False,
            "production_use_allowed": False, "athlete_scoring_allowed": False,
            "athlete_comparison_allowed": False, "elo_update_allowed": False,
        })
    receipt = {"plan_sha256": _sha256(args.plan), "extracted_frame_count": len(receipt_rows), "frames": receipt_rows}
    (args.output_dir / "frame_extraction_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Extracted {len(receipt_rows)} review-only frame candidates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
