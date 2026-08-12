from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts.run_boulder_anchor_discovery import (
    WindowReviewFailure,
    _atomic_jsonl,
    _is_transient_exception,
    _load_failed_windows,
    _load_records,
    _review_window,
)
from video_boulder_anchor_discovery import (
    AnchorWindow,
    build_discovery_plan,
    build_prompt,
    build_response_json_schema,
    build_verification_plan,
    build_video_part,
    validate_response,
)


ROOT = Path(__file__).resolve().parents[1]


class _FileData:
    def __init__(self, **kwargs):
        self.values = kwargs


class _VideoMetadata:
    def __init__(self, **kwargs):
        self.values = kwargs


class _Part:
    def __init__(self, **kwargs):
        self.values = kwargs


class _GenerateContentConfig:
    def __init__(self, **kwargs):
        self.values = kwargs


class _Types:
    FileData = _FileData
    VideoMetadata = _VideoMetadata
    Part = _Part
    GenerateContentConfig = _GenerateContentConfig


class _Response:
    def __init__(self, parsed):
        self.parsed = parsed
        self.text = ""


class _Models:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return _Response(outcome)


class _Client:
    def __init__(self, outcomes):
        self.models = _Models(outcomes)


def _window(pass_name: str = "discovery") -> AnchorWindow:
    return AnchorWindow(
        window_id="W1", pass_name=pass_name, event_id=1480, category_round_id=10675,
        event="Prague", gender="Men", round="Semi-final", video_id="abc",
        youtube_url="https://www.youtube.com/watch?v=abc", start_seconds=100,
        end_seconds=1600, fps=0.25, source_candidate_id="A1" if pass_name == "verification" else "",
        source_candidate_json=(
            '{"anchor_type":"athlete_nameplate_graphic","candidate_id":"A1",'
            '"graphic_text":"Athlete Example"}' if pass_name == "verification" else ""
        ),
    )


def _payload(window: AnchorWindow) -> dict[str, object]:
    anchor_time = window.start_seconds + 10
    scene_time = window.start_seconds + 20
    payload: dict[str, object] = {
        "window_id": window.window_id,
        "window_notes": ["Four-panel broadcast view."],
        "anchor_candidates": [{
            "candidate_id": "A1", "broadcast_seconds": anchor_time,
            "evidence_start_seconds": anchor_time - 2, "evidence_end_seconds": anchor_time + 2,
            "anchor_type": "competition_clock_graphic", "clock_text": "04:57",
            "graphic_text": "", "boulder_number": "M2", "identity_cues": [{
                "cue_type": "graphic_name", "cue_text": "A. ATHLETE", "confidence_0_1": 0.8,
            }], "reference_point_hint": "turn_start", "confidence_0_1": 0.9,
            "observable_evidence_note": "Clock and name graphic are visible.",
            "uncertainty_note": "Bib is hidden.",
        }],
        "scene_candidates": [{
            "candidate_id": "S1", "broadcast_seconds": scene_time,
            "evidence_start_seconds": scene_time - 1, "evidence_end_seconds": scene_time + 1,
            "scene_type": "camera_view_switch", "boulder_number": "unknown",
            "confidence_0_1": 0.75, "observable_evidence_note": "View changes to split screen.",
            "uncertainty_note": "No round graphic is visible.",
        }],
        "visible_intervals": [{
            "interval_id": "I1", "start_seconds": window.start_seconds + 5, "end_seconds": window.start_seconds + 25,
            "boulder_number": "M2", "identity_cues": [{
                "cue_type": "country_code", "cue_text": "CAN", "confidence_0_1": 0.6,
            }], "confidence_0_1": 0.7, "uncertainty_note": "Two climbers are visible.",
        }, {
            "interval_id": "I2", "start_seconds": window.start_seconds + 8, "end_seconds": window.start_seconds + 28,
            "boulder_number": "M3", "identity_cues": [],
            "confidence_0_1": 0.5, "uncertainty_note": "Face is not visible.",
        }],
    }
    if window.pass_name == "verification":
        payload["verification"] = {
            "source_candidate_id": "A1", "status": "supported", "confidence_0_1": 0.8,
            "observable_evidence_note": "The same clock graphic is visible.",
            "uncertainty_note": "The name is partially occluded.",
        }
    return payload


class AnchorDiscoveryTests(unittest.TestCase):
    def test_remote_protocol_disconnect_is_retryable(self):
        RemoteProtocolError = type("RemoteProtocolError", (RuntimeError,), {})
        self.assertTrue(
            _is_transient_exception(
                RemoteProtocolError("Server disconnected without sending a response.")
            )
        )

    def test_discovery_plan_covers_twelve_sources_in_20_to_30_minute_windows(self):
        manifest = pd.read_csv(ROOT / "data/video_2026_source_manifest.csv")
        windows = build_discovery_plan(manifest)
        self.assertEqual(len({window.video_id for window in windows}), 12)
        self.assertTrue(all(1200 <= window.end_seconds - window.start_seconds <= 1800 for window in windows))
        self.assertTrue(all(window.fps == 0.25 for window in windows))
        for video_id in {window.video_id for window in windows}:
            source = sorted((w for w in windows if w.video_id == video_id), key=lambda w: w.start_seconds)
            self.assertEqual(source[0].start_seconds, 0)
            self.assertTrue(all(left.end_seconds == right.start_seconds for left, right in zip(source, source[1:])))
        self.assertEqual([w.window_id for w in windows], [w.window_id for w in build_discovery_plan(manifest)])

    def test_event_scoped_plans_cover_exactly_four_broadcasts(self):
        manifest = pd.read_csv(ROOT / "data/video_2026_source_manifest.csv")
        for event_id in (1479, 1480, 1482):
            with self.subTest(event_id=event_id):
                windows = build_discovery_plan(manifest, event_id=event_id)
                self.assertEqual({window.event_id for window in windows}, {event_id})
                self.assertEqual(len({window.video_id for window in windows}), 4)
                self.assertEqual(
                    {(window.gender, window.round) for window in windows},
                    {("Men", "Semi-final"), ("Men", "Final"),
                     ("Women", "Semi-final"), ("Women", "Final")},
                )

    def test_schema_is_shallow_and_closed(self):
        schema = build_response_json_schema("discovery")
        self.assertFalse(schema["additionalProperties"])
        rendered = json.dumps(schema).lower()
        for forbidden in ("tactics", "affect", "emotion", "elo", "training", "beta", "movement"):
            self.assertNotIn(forbidden, rendered)
        prompt = build_prompt(_window()).lower()
        self.assertIn("do not return movement", prompt)
        source = (ROOT / "video_boulder_anchor_discovery.py").read_text(encoding="utf-8")
        for heavy_import in ("import google", "import cv2", "import torch"):
            self.assertNotIn(heavy_import, source)

    def test_valid_payload_preserves_simultaneous_visible_intervals(self):
        window = _window()
        payload = _payload(window)
        self.assertEqual(validate_response(payload, window), [])
        self.assertEqual(len(payload["visible_intervals"]), 2)

    def test_validator_fails_closed(self):
        window = _window()
        payload = _payload(window)
        payload["elo"] = 2100
        self.assertTrue(validate_response(payload, window))
        payload = _payload(window)
        payload["anchor_candidates"][0]["clock_text"] = ""
        self.assertTrue(any("literal clock" in error for error in validate_response(payload, window)))
        payload = _payload(window)
        payload["visible_intervals"][0]["end_seconds"] = 1700
        self.assertTrue(any("outside" in error for error in validate_response(payload, window)))
        payload = _payload(window)
        payload["visible_intervals"][0]["interval_id"] = "A1"
        self.assertTrue(any("unique" in error for error in validate_response(payload, window)))
        payload = _payload(window)
        payload["scene_candidates"][0]["free_form_analysis"] = "unsupported"
        self.assertTrue(any("shallow schema" in error for error in validate_response(payload, window)))

    def test_verification_plan_and_validation(self):
        source = _window()
        record = {
            **source.__dict__, "video_duration_seconds": 2000, **_payload(source),
        }
        first = build_verification_plan([record])
        second = build_verification_plan([record])
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertTrue(all(window.end_seconds - window.start_seconds == 60 for window in first))
        target = next(window for window in first if window.source_candidate_id == "A1")
        payload = _payload(target)
        self.assertEqual(validate_response(payload, target), [])
        payload["verification"]["status"] = "confirmed_fact"
        self.assertTrue(any("status" in error for error in validate_response(payload, target)))

    def test_verification_rejects_obvious_broadcast_graphic_misclassification(self):
        base = _window("verification")
        cases = [
            ({
                "candidate_id": "A1", "anchor_type": "competition_clock_graphic",
                "clock_text": "23°C", "graphic_text": "World Climbing Series Prague 2026",
                "identity_cues": [],
            }, "competition clock"),
            ({
                "candidate_id": "A1", "anchor_type": "athlete_nameplate_graphic",
                "clock_text": "unknown", "graphic_text": "MEN'S BOULDER CLIMBING SERIES RANKING 2026",
                "identity_cues": [
                    {"cue_type": "graphic_name", "cue_text": "Athlete One"},
                    {"cue_type": "graphic_name", "cue_text": "Athlete Two"},
                ],
            }, "nameplates"),
            ({
                "candidate_id": "A1", "anchor_type": "round_transition_graphic",
                "clock_text": "unknown", "graphic_text": "WORLD CLIMBING CALENDAR 2026",
                "identity_cues": [],
            }, "round transitions"),
        ]
        for source_candidate, expected in cases:
            with self.subTest(expected=expected):
                window = AnchorWindow(
                    **{
                        **base.__dict__,
                        "source_candidate_json": json.dumps(source_candidate),
                    }
                )
                payload = _payload(window)
                errors = validate_response(payload, window)
                self.assertTrue(any(expected in error for error in errors), errors)
                payload["verification"]["status"] = "conflicts"
                self.assertEqual(validate_response(payload, window), [])

    def test_direct_youtube_part_has_exact_sampling_bounds(self):
        part = build_video_part(_Types, _window())
        self.assertEqual(part.values["file_data"].values["file_uri"], _window().youtube_url)
        metadata = part.values["video_metadata"].values
        self.assertEqual(metadata, {"start_offset": "100s", "end_offset": "1600s", "fps": 0.25})

    def test_atomic_checkpoint_round_trip_and_duplicate_rejection(self):
        records = [{"window_id": "W1", "response": {}}, {"window_id": "W2", "response": {}}]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "review_windows.jsonl"
            _atomic_jsonl(path, records)
            self.assertEqual(_load_records(path), records)
            path.write_text('\n'.join(json.dumps(records[0]) for _ in range(2)), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                _load_records(path)

    def test_validation_retry_replaces_bad_response(self):
        window = _window()
        invalid = _payload(window)
        invalid["anchor_candidates"][0]["clock_text"] = ""
        client = _Client([invalid, _payload(window)])
        payload = _review_window(
            client, _Types, window, model="gemini-3.1-flash-lite", seed=10,
            max_validation_retries=1, max_transient_retries=0, sleep_fn=lambda _: None,
        )
        self.assertEqual(payload["window_id"], window.window_id)
        self.assertEqual(len(client.models.calls), 2)
        self.assertIn("previous response was rejected", client.models.calls[1]["contents"][1])

    def test_exhausted_validation_is_quarantined_fail_closed(self):
        window = _window()
        invalid = _payload(window)
        invalid["anchor_candidates"][0]["graphic_text"] = ""
        invalid["anchor_candidates"][0]["clock_text"] = ""
        client = _Client([invalid, invalid])
        with self.assertRaises(WindowReviewFailure) as raised:
            _review_window(
                client, _Types, window, model="gemini-3.1-flash-lite", seed=10,
                max_validation_retries=1, max_transient_retries=0, sleep_fn=lambda _: None,
            )
        failure = raised.exception.failure
        self.assertEqual(failure["failure_kind"], "validation_exhausted")
        self.assertEqual(failure["total_attempts"], 2)
        self.assertEqual(failure["validation_failures"], 2)
        self.assertTrue(failure["quarantined"])
        for gate in (
            "production_use_allowed", "athlete_scoring_allowed",
            "athlete_comparison_allowed", "elo_update_allowed",
        ):
            self.assertFalse(failure[gate])

    def test_transient_retry_is_bounded(self):
        window = _window()
        delays = []
        client = _Client([ConnectionError("503 unavailable"), _payload(window)])
        payload = _review_window(
            client, _Types, window, model="gemini-3.1-flash-lite", seed=10,
            max_validation_retries=0, max_transient_retries=1, sleep_fn=delays.append,
        )
        self.assertEqual(payload["window_id"], window.window_id)
        self.assertEqual(delays, [1.0])
        self.assertEqual(len(client.models.calls), 2)

        client = _Client([TimeoutError("deadline exceeded"), TimeoutError("deadline exceeded")])
        with self.assertRaises(WindowReviewFailure) as raised:
            _review_window(
                client, _Types, window, model="gemini-3.1-flash-lite", seed=10,
                max_validation_retries=0, max_transient_retries=1, sleep_fn=lambda _: None,
            )
        self.assertEqual(raised.exception.failure["failure_kind"], "transient_exhausted")
        self.assertEqual(raised.exception.failure["total_attempts"], 2)

    def test_failed_window_manifest_is_atomic_and_rejects_unsafe_records(self):
        window = _window()
        invalid = _payload(window)
        invalid["anchor_candidates"][0]["clock_text"] = ""
        invalid["anchor_candidates"][0]["graphic_text"] = ""
        with self.assertRaises(WindowReviewFailure) as raised:
            _review_window(
                _Client([invalid]), _Types, window, model="gemini-3.1-flash-lite",
                seed=10, max_validation_retries=0, max_transient_retries=0,
                sleep_fn=lambda _: None,
            )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "failed_windows.jsonl"
            _atomic_jsonl(path, [raised.exception.failure])
            self.assertEqual(_load_failed_windows(path)[0]["window_id"], window.window_id)
            unsafe = dict(raised.exception.failure)
            unsafe["elo_update_allowed"] = True
            _atomic_jsonl(path, [unsafe])
            with self.assertRaisesRegex(ValueError, "unsafe"):
                _load_failed_windows(path)

    def test_cloud_workflow_is_event_scoped_bounded_and_shallow(self):
        path = ROOT / ".github/workflows/video-2026-boulder-anchor-discovery.yml"
        workflow = path.read_text(encoding="utf-8")
        self.assertIn("workflow_dispatch:", workflow)
        self.assertNotIn("\n  push:", workflow)
        self.assertNotIn("\n  schedule:", workflow)
        self.assertIn("event_id: [1479, 1480, 1482]", workflow)
        self.assertIn("max-parallel: 3", workflow)
        self.assertIn("max_windows_per_event", workflow)
        self.assertIn('"$MAX_WINDOWS" -gt 30', workflow)
        self.assertIn("--event-id ${{ matrix.event_id }}", workflow)
        self.assertGreaterEqual(workflow.count("--event-id ${{ matrix.event_id }}"), 2)
        self.assertGreaterEqual(workflow.count("--max-windows \"$MAX_WINDOWS\""), 2)
        self.assertGreaterEqual(workflow.count("--fps 0.25"), 2)
        self.assertGreaterEqual(workflow.count("--model gemini-3.1-flash-lite"), 2)
        self.assertGreaterEqual(workflow.count("--max-validation-retries 1"), 2)
        self.assertGreaterEqual(workflow.count("--max-transient-retries 2"), 2)
        self.assertGreaterEqual(workflow.count("--request-timeout-ms 900000"), 2)
        self.assertIn('google-genai==1.75.0', workflow)
        self.assertIn("--continue-on-window-failure", workflow)
        self.assertLess(
            workflow.index("Mandatory event-scoped dry run"),
            workflow.index("Discover bounded event anchors"),
        )
        self.assertIn("if: inputs.execute", workflow)
        self.assertIn("boulder-anchor-discovery-v4", workflow)
        self.assertNotIn("boulder-anchor-discovery-v3-", workflow)
        self.assertIn("actions/cache/restore@caa296126883cff596d87d8935842f9db880ef25", workflow)
        self.assertIn("actions/cache/save@caa296126883cff596d87d8935842f9db880ef25", workflow)
        self.assertIn("actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a", workflow)
        self.assertIn(
            "path: artifacts/boulder-anchor-discovery-v4/event-${{ matrix.event_id }}/",
            workflow,
        )
        self.assertNotIn("--pass verification", workflow)
        self.assertNotIn("pro-escalation", workflow.lower())
        self.assertNotIn("merge-events", workflow)
        self.assertIn("contents: read", workflow)

    def test_runner_keeps_all_downstream_safety_gates_false(self):
        runner = (ROOT / "scripts/run_boulder_anchor_discovery.py").read_text(encoding="utf-8")
        for gate in (
            "production_use_allowed", "athlete_scoring_allowed",
            "athlete_comparison_allowed", "elo_update_allowed",
        ):
            self.assertIn(f'"{gate}"', runner)
        self.assertIn("{gate: False for gate in SAFETY_GATES}", runner)


if __name__ == "__main__":
    unittest.main()
