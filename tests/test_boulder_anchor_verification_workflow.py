from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts.run_boulder_anchor_discovery import (
    REVIEWER_ID,
    SAFETY_GATES,
    _atomic_json,
    _atomic_jsonl,
    _failure_record,
    _load_records,
    _prepare_verification_records,
    _require_bounded_execution,
    _window_from_row,
)
from video_boulder_anchor_discovery import (
    PIPELINE_VERSION,
    build_verification_plan,
    plan_frame,
)
from video_boulder_anchor_gemini_compat import SCHEMA_PROFILE
from video_boulder_anchor_verification_merge import (
    CONTRACT_NAMESPACE,
    EXPECTED_EVENT_WINDOWS,
    FROZEN_DISCOVERY_CHECKPOINT_SHA256,
    FROZEN_SOURCE_MANIFEST_SHA256,
    merge_verification_checkpoints,
)
from video_boulder_segmentation import sha256_file


ROOT = Path(__file__).resolve().parents[1]
DISCOVERY_CHECKPOINT = (
    ROOT / "data/video_boulder_anchor_discovery_2026_v1/merged_review_windows.jsonl"
)
MANIFEST = ROOT / "data/video_2026_source_manifest.csv"


def _verification_response(window) -> dict[str, object]:
    return {
        "window_id": window.window_id,
        "window_notes": ["Shallow verification evidence only."],
        "anchor_candidates": [],
        "scene_candidates": [],
        "visible_intervals": [],
        "verification": {
            "source_candidate_id": window.source_candidate_id,
            "status": "conflicts",
            "confidence_0_1": 0.8,
            "observable_evidence_note": "Synthetic fixture does not promote a source label.",
            "uncertainty_note": "No athlete trait is inferred.",
        },
    }


def _write_event_checkpoint(folder: Path, manifest: pd.DataFrame, event_id: int) -> None:
    records = _prepare_verification_records(
        _load_records(DISCOVERY_CHECKPOINT), manifest,
        event_id=event_id, window_seconds=1500,
    )
    windows = build_verification_plan(
        [{**record, **record["response"]} for record in records],
        context_seconds=60,
        fps=1.0,
        candidate_fields=("anchor_candidates",),
    )
    folder.mkdir(parents=True)
    plan_path = folder / "window_plan.csv"
    plan_frame(windows).to_csv(plan_path, index=False, lineterminator="\n")
    contract = {
        "pipeline_version": PIPELINE_VERSION,
        "reviewer_id": REVIEWER_ID,
        "pass_name": "verification",
        "model": "gemini-3.1-flash-lite",
        "source_manifest_sha256": sha256_file(MANIFEST),
        "event_id_filter": event_id,
        "source_discovery_checkpoint_sha256": sha256_file(DISCOVERY_CHECKPOINT),
        "window_plan_sha256": sha256_file(plan_path),
        "fps": 1.0,
        "window_seconds": 1500,
        "verification_seconds": 60,
        "verification_candidate_source": "anchors-only",
        "contract_namespace": CONTRACT_NAMESPACE,
        "max_validation_retries": 1,
        "max_transient_retries": 2,
        "request_timeout_ms": 900000,
        "structured_output_profile": SCHEMA_PROFILE,
    }
    _atomic_json(folder / "checkpoint_contract.json", contract)
    _atomic_json(folder / "run_manifest.json", {
        **contract,
        "status": "BOUNDED PARTIAL",
        **{gate: False for gate in SAFETY_GATES},
    })
    first = windows[0]
    _atomic_jsonl(folder / "review_windows.jsonl", [{
        "window_id": first.window_id,
        "response": _verification_response(first),
        "pipeline_version": PIPELINE_VERSION,
        "pass_name": "verification",
        "reviewer_id": REVIEWER_ID,
        "reviewer_type": "AI",
        "model": "gemini-3.1-flash-lite",
        "structured_output_profile": SCHEMA_PROFILE,
        "youtube_url": first.youtube_url,
        "video_id": first.video_id,
        "window_start_seconds": first.start_seconds,
        "window_end_seconds": first.end_seconds,
        "fps": first.fps,
        "event_id": first.event_id,
        "category_round_id": first.category_round_id,
        "source_candidate_id": first.source_candidate_id,
        "source_discovery_window_id": first.source_discovery_window_id,
        "reviewed_at_utc": "2026-08-01T00:00:00+00:00",
        "media_download_required": False,
        **{gate: False for gate in SAFETY_GATES},
    }])
    _atomic_jsonl(folder / "failed_windows.jsonl", [])


class BoulderAnchorVerificationWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = pd.read_csv(MANIFEST)
        cls.discovery_records = _load_records(DISCOVERY_CHECKPOINT)

    def test_runner_validates_complete_packet_before_trusted_event_filter(self):
        expected_discovery_rows = {1479: 25, 1480: 26, 1482: 26}
        expected_anchor_windows = EXPECTED_EVENT_WINDOWS
        for event_id in expected_discovery_rows:
            with self.subTest(event_id=event_id):
                selected = _prepare_verification_records(
                    self.discovery_records,
                    self.manifest,
                    event_id=event_id,
                    window_seconds=1500,
                )
                self.assertEqual(len(selected), expected_discovery_rows[event_id])
                self.assertEqual({row["event_id"] for row in selected}, {event_id})
                windows = build_verification_plan(
                    [{**row, **row["response"]} for row in selected],
                    context_seconds=60,
                    fps=1.0,
                    candidate_fields=("anchor_candidates",),
                )
                self.assertEqual(len(windows), expected_anchor_windows[event_id])
                self.assertTrue(all(window.event_id == event_id for window in windows))
                self.assertTrue(all(window.fps == 1.0 for window in windows))
                self.assertTrue(all(window.end_seconds - window.start_seconds == 60 for window in windows))

    def test_spoofed_discovery_event_provenance_fails_before_filtering(self):
        records = [dict(record) for record in self.discovery_records]
        records[0]["source_event_id"] = 1482
        with self.assertRaisesRegex(ValueError, "provenance event mismatch"):
            _prepare_verification_records(
                records, self.manifest, event_id=1479, window_seconds=1500
            )

    def test_anchor_only_plan_excludes_scene_candidates(self):
        selected = _prepare_verification_records(
            self.discovery_records, self.manifest,
            event_id=1479, window_seconds=1500,
        )
        flattened = [{**row, **row["response"]} for row in selected]
        anchors = build_verification_plan(
            flattened, context_seconds=60, fps=1.0,
            candidate_fields=("anchor_candidates",),
        )
        all_candidates = build_verification_plan(
            flattened, context_seconds=60, fps=1.0,
        )
        self.assertEqual(len(anchors), 73)
        self.assertGreater(len(all_candidates), len(anchors))
        for window in anchors:
            source_candidate = json.loads(window.source_candidate_json)
            self.assertEqual(source_candidate["candidate_id"], window.source_candidate_id)
            self.assertIn("anchor_type", source_candidate)
        with self.assertRaisesRegex(ValueError, "candidate fields"):
            build_verification_plan(flattened, candidate_fields=("visible_intervals",))

    def test_historical_global_prefix_plan_is_madrid_only_and_not_used(self):
        historical = pd.read_csv(
            ROOT / "data/video_boulder_anchor_discovery_2026_v1/verification_window_plan.csv"
        )
        self.assertEqual(historical["event_id"].value_counts().to_dict(), {1479: 100})
        workflow = (
            ROOT / ".github/workflows/video-2026-boulder-anchor-verification.yml"
        ).read_text(encoding="utf-8")
        self.assertNotIn(
            "data/video_boulder_anchor_discovery_2026_v1/verification_window_plan.csv",
            workflow,
        )

    def test_runner_enforces_verification_execution_bound(self):
        _require_bounded_execution("verification", execute=True, max_windows=12)
        with self.assertRaisesRegex(SystemExit, "between 1 and 12"):
            _require_bounded_execution("verification", execute=True, max_windows=13)
        with self.assertRaisesRegex(SystemExit, "between 1 and 30"):
            _require_bounded_execution("discovery", execute=True, max_windows=31)

    def test_event_checkpoints_merge_with_closed_gates_and_frozen_counts(self):
        self.assertEqual(sha256_file(DISCOVERY_CHECKPOINT), FROZEN_DISCOVERY_CHECKPOINT_SHA256)
        self.assertEqual(sha256_file(MANIFEST), FROZEN_SOURCE_MANIFEST_SHA256)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = []
            for event_id in EXPECTED_EVENT_WINDOWS:
                folder = root / f"event-{event_id}"
                _write_event_checkpoint(folder, self.manifest, event_id)
                inputs.append(folder)
            summary = merge_verification_checkpoints(
                list(reversed(inputs)), root / "merged"
            )
            self.assertEqual(summary["source_event_ids"], [1479, 1480, 1482])
            self.assertEqual(summary["planned_window_count"], 213)
            self.assertEqual(summary["completed_window_count"], 3)
            self.assertEqual(summary["pending_window_count"], 210)
            self.assertEqual(summary["quarantined_retryable_window_count"], 0)
            for gate in SAFETY_GATES:
                self.assertFalse(summary[gate])
            coverage = pd.read_csv(root / "merged/coverage_windows.csv")
            self.assertEqual(coverage["status"].value_counts().to_dict(), {
                "pending_unreviewed": 210,
                "completed": 3,
            })
            manifest = json.loads((root / "merged/merge_manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["source_verification_model_executed"])
            self.assertFalse(manifest["merge_executes_model"])
            merged_reviews = _load_records(root / "merged/merged_review_windows.jsonl")
            self.assertEqual(len(merged_reviews), 3)
            for record in merged_reviews:
                source_candidate = json.loads(record["source_candidate_json"])
                self.assertEqual(
                    source_candidate["candidate_id"], record["source_candidate_id"]
                )
                self.assertIn("anchor_type", source_candidate)

    def test_merger_rejects_missing_or_mismatched_exact_candidate_claim(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = []
            for event_id in EXPECTED_EVENT_WINDOWS:
                folder = root / f"event-{event_id}"
                _write_event_checkpoint(folder, self.manifest, event_id)
                inputs.append(folder)
            plan_path = inputs[0] / "window_plan.csv"
            plan = pd.read_csv(plan_path, keep_default_na=False)
            source_candidate = json.loads(plan.loc[0, "source_candidate_json"])
            source_candidate["candidate_id"] = "wrong-candidate"
            plan.loc[0, "source_candidate_json"] = json.dumps(source_candidate)
            plan.to_csv(plan_path, index=False, lineterminator="\n")
            contract_path = inputs[0] / "checkpoint_contract.json"
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            contract["window_plan_sha256"] = sha256_file(plan_path)
            _atomic_json(contract_path, contract)
            run_path = inputs[0] / "run_manifest.json"
            run = json.loads(run_path.read_text(encoding="utf-8"))
            run["window_plan_sha256"] = contract["window_plan_sha256"]
            _atomic_json(run_path, run)
            with self.assertRaisesRegex(ValueError, "source candidate does not match"):
                merge_verification_checkpoints(inputs, root / "merged")

    def test_merger_rejects_opened_production_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = []
            for event_id in EXPECTED_EVENT_WINDOWS:
                folder = root / f"event-{event_id}"
                _write_event_checkpoint(folder, self.manifest, event_id)
                inputs.append(folder)
            run_path = inputs[0] / "run_manifest.json"
            run = json.loads(run_path.read_text(encoding="utf-8"))
            run["elo_update_allowed"] = True
            run_path.write_text(json.dumps(run), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsafe verification run-manifest gate"):
                merge_verification_checkpoints(inputs, root / "merged")

    def test_merger_rejects_dry_run_status_and_run_contract_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = []
            for event_id in EXPECTED_EVENT_WINDOWS:
                folder = root / f"event-{event_id}"
                _write_event_checkpoint(folder, self.manifest, event_id)
                inputs.append(folder)
            run_path = inputs[0] / "run_manifest.json"
            run = json.loads(run_path.read_text(encoding="utf-8"))
            run["status"] = "DRY RUN"
            run_path.write_text(json.dumps(run), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "status is not mergeable"):
                merge_verification_checkpoints(inputs, root / "merged-status")
            run["status"] = "BOUNDED PARTIAL"
            run["model"] = "changed-model"
            run_path.write_text(json.dumps(run), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "run contract mismatch"):
                merge_verification_checkpoints(inputs, root / "merged-contract")

    def test_merger_rejects_quarantine_provenance_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = []
            for event_id in EXPECTED_EVENT_WINDOWS:
                folder = root / f"event-{event_id}"
                _write_event_checkpoint(folder, self.manifest, event_id)
                inputs.append(folder)
            plan = pd.read_csv(inputs[0] / "window_plan.csv", keep_default_na=False)
            window = _window_from_row(plan.iloc[1])
            failure = _failure_record(
                window,
                model="gemini-3.1-flash-lite",
                failure_kind="validation_exhausted",
                exc=ValueError("synthetic invalid response"),
                total_attempts=2,
                validation_failures=2,
                transient_failures=0,
                validation_errors=["synthetic invalid response"],
                attempt_history=[],
            )
            failure["source_candidate_id"] = "wrong-candidate"
            _atomic_jsonl(inputs[0] / "failed_windows.jsonl", [failure])
            with self.assertRaisesRegex(ValueError, "quarantine metadata mismatch"):
                merge_verification_checkpoints(inputs, root / "merged")

    def test_quarantine_and_pending_counts_form_an_exact_partition(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = []
            for event_id in EXPECTED_EVENT_WINDOWS:
                folder = root / f"event-{event_id}"
                _write_event_checkpoint(folder, self.manifest, event_id)
                inputs.append(folder)
            plan = pd.read_csv(inputs[0] / "window_plan.csv", keep_default_na=False)
            window = _window_from_row(plan.iloc[1])
            failure = _failure_record(
                window,
                model="gemini-3.1-flash-lite",
                failure_kind="validation_exhausted",
                exc=ValueError("synthetic invalid response"),
                total_attempts=2,
                validation_failures=2,
                transient_failures=0,
                validation_errors=["synthetic invalid response"],
                attempt_history=[],
            )
            _atomic_jsonl(inputs[0] / "failed_windows.jsonl", [failure])
            summary = merge_verification_checkpoints(inputs, root / "merged")
            self.assertEqual(summary["completed_window_count"], 3)
            self.assertEqual(summary["quarantined_retryable_window_count"], 1)
            self.assertEqual(summary["pending_window_count"], 209)
            self.assertEqual(
                summary["planned_window_count"],
                summary["completed_window_count"]
                + summary["quarantined_retryable_window_count"]
                + summary["pending_window_count"],
            )
            madrid = next(
                row for row in summary["coverage_by_event"] if row["event_id"] == 1479
            )
            self.assertEqual(madrid, {
                "event_id": 1479,
                "planned": 73,
                "completed": 1,
                "quarantined_retryable": 1,
                "pending": 71,
            })

    def test_workflow_is_manual_bounded_resumable_and_mergeable(self):
        workflow = (
            ROOT / ".github/workflows/video-2026-boulder-anchor-verification.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("workflow_dispatch:", workflow)
        self.assertNotIn("\n  push:", workflow)
        self.assertNotIn("\n  schedule:", workflow)
        for value in (
            "event_id: 1479", "event_slug: madrid",
            "event_id: 1480", "event_slug: prague",
            "event_id: 1482", "event_slug: innsbruck",
        ):
            self.assertIn(value, workflow)
        self.assertIn('"$MAX_WINDOWS" -gt 12', workflow)
        self.assertGreaterEqual(workflow.count("--pass verification"), 2)
        self.assertGreaterEqual(workflow.count("--verification-seconds 60"), 2)
        self.assertGreaterEqual(workflow.count("--fps 1.0"), 2)
        self.assertGreaterEqual(workflow.count("--verification-candidate-source anchors-only"), 2)
        self.assertGreaterEqual(workflow.count(
            "--required-discovery-checkpoint-sha256 "
            + FROZEN_DISCOVERY_CHECKPOINT_SHA256
        ), 2)
        self.assertGreaterEqual(workflow.count(
            "--required-source-manifest-sha256 "
            + FROZEN_SOURCE_MANIFEST_SHA256
        ), 2)
        self.assertGreaterEqual(workflow.count("--event-id ${{ matrix.event_id }}"), 2)
        self.assertIn("boulder-anchor-verification-v4-frozen77-anchors1fps", workflow)
        self.assertEqual(
            CONTRACT_NAMESPACE,
            "boulder-anchor-verification-v4-frozen77-anchors1fps",
        )
        self.assertIn("actions/cache/restore@caa296126883cff596d87d8935842f9db880ef25", workflow)
        self.assertIn("actions/cache/save@caa296126883cff596d87d8935842f9db880ef25", workflow)
        self.assertIn("--continue-on-window-failure", workflow)
        self.assertIn("merge-checkpoints:", workflow)
        self.assertEqual(workflow.count("actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c"), 3)
        self.assertIn("merge_boulder_anchor_verification_checkpoints.py", workflow)
        self.assertIn("contents: read", workflow)

    def test_merger_contains_no_model_or_cloud_execution(self):
        source = (
            ROOT / "video_boulder_anchor_verification_merge.py"
        ).read_text(encoding="utf-8").lower()
        script = (
            ROOT / "scripts/merge_boulder_anchor_verification_checkpoints.py"
        ).read_text(encoding="utf-8").lower()
        self.assertNotIn("google.genai", source + script)
        self.assertNotIn("generate_content", source + script)
        self.assertNotIn("--execute", source + script)


if __name__ == "__main__":
    unittest.main()
