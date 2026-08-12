#!/usr/bin/env python3
"""Merge the three event-scoped Boulder anchor-verification checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from video_boulder_anchor_verification_merge import merge_verification_checkpoints  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = merge_verification_checkpoints(args.input_dir, args.output_dir)
    print(
        f"Merged {summary['completed_window_count']}/{summary['planned_window_count']} "
        "anchor-verification windows; quarantines remain retryable and all safety gates are closed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
