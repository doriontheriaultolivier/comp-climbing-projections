from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from video_boulder_wall_frame_plan import plan_frame_candidates, plan_records
from scripts.extract_boulder_wall_frames import _candidate_command, _load_plan


def _record(*, status: str = "supported", intervals=None, gate: bool = False):
    return {
        "pass_name": "verification", "window_id": "BAD26-V-test", "event_id": 1479,
        "category_round_id": 10670, "event": "Madrid", "gender": "Women", "round": "Final",
        "video_id": "abc123", "youtube_url": "https://www.youtube.com/watch?v=abc123",
        "window_start_seconds": 100, "window_end_seconds": 160, "source_candidate_id": "anchor-1",
        "source_candidate_json": json.dumps({"candidate_id": "anchor-1", "boulder_number": "M2", "broadcast_seconds": 120}),
        "production_use_allowed": gate, "athlete_scoring_allowed": False,
        "athlete_comparison_allowed": False, "elo_update_allowed": False,
        "response": {"verification": {"source_candidate_id": "anchor-1", "status": status}, "visible_intervals": intervals or []},
    }


class BoulderWallFramePlanTests(unittest.TestCase):
    def test_selects_largest_interval_free_gap_and_keeps_review_gate_closed(self):
        candidates = plan_frame_candidates([_record(intervals=[
            {"boulder_number": "M2", "start_seconds": 110, "end_seconds": 122},
            {"boulder_number": "M2", "start_seconds": 130, "end_seconds": 140},
        ])])
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].frame_seconds, 151.0)
        output = plan_records(candidates)[0]
        self.assertFalse(output["empty_wall_verified"])
        self.assertFalse(output["production_use_allowed"])
        self.assertEqual(output["candidate_status"], "REQUIRES_VISUAL_EMPTY_WALL_REVIEW")

    def test_rejects_open_gates_and_ignores_unsupported_or_unknown_slot(self):
        with self.assertRaisesRegex(ValueError, "open safety gate"):
            plan_frame_candidates([_record(gate=True)])
        self.assertEqual(plan_frame_candidates([_record(status="conflicts")]), [])
        unknown = _record()
        unknown["source_candidate_json"] = json.dumps({"candidate_id": "anchor-1", "boulder_number": "unknown", "broadcast_seconds": 120})
        self.assertEqual(plan_frame_candidates([unknown]), [])

    def test_rejects_unbound_source_and_interval_escape(self):
        record = _record()
        record["response"]["verification"]["source_candidate_id"] = "other"
        with self.assertRaisesRegex(ValueError, "binding mismatch"):
            plan_frame_candidates([record])
        record = _record(intervals=[{"boulder_number": "M2", "start_seconds": 99, "end_seconds": 120}])
        with self.assertRaisesRegex(ValueError, "escapes verification window"):
            plan_frame_candidates([record])

    def test_extractor_rejects_promoted_plan_and_uses_bounded_single_frame_command(self):
        payload = {
            "pipeline_version": "ifsc-boulder-wall-frame-plan-v1",
            "records": plan_records(plan_frame_candidates([_record()])),
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "plan.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(_load_plan(path)["records"][0]["candidate_status"], "REQUIRES_VISUAL_EMPTY_WALL_REVIEW")
            payload["records"][0]["empty_wall_verified"] = True
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not review-only"):
                _load_plan(path)
        command = _candidate_command("ffmpeg", Path("input.mp4"), Path("output.jpg"), 123.5)
        self.assertIn("-frames:v", command)
        self.assertEqual(command[command.index("-frames:v") + 1], "1")
        self.assertIn("-nostdin", command)


if __name__ == "__main__":
    unittest.main()
