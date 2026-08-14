from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import pandas as pd

from streamlit.testing.v1 import AppTest

PATH = Path(__file__).parents[1] / "style_tagging_app.py"
SPEC = importlib.util.spec_from_file_location("style_tagging_app", PATH)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


PROBLEM = {
    "source_scope": "IFSC", "source_event_ids": "123", "source_round_ids": "456",
    "event_date": "2026-08-11", "event_name": "Test Open", "round_group": "Final",
    "round_uid": "round-1234567890abcdef1234", "gender": "Women", "terrain_group": "Open / Senior",
    "boulder_number": 2, "boulder_count": 4, "boulder_count_status": "source-confirmed",
    "boulder_uid": "round-1234567890abcdef1234-b2",
    "pre_zone_segment_uid": "round-1234567890abcdef1234-b2-pre-zone",
    "post_zone_segment_uid": "round-1234567890abcdef1234-b2-post-zone",
}


class StyleTaggingAppTest(unittest.TestCase):
    def test_standalone_entrypoint_renders_with_governed_inventory(self) -> None:
        app = AppTest.from_file(str(PATH)).run(timeout=120)
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertEqual(app.title[0].value, "Comp Climbing Boulder Tags")
        self.assertEqual(
            [control.label for control in app.selectbox[:4]],
            ["Competition", "Round", "Terrain", "Boulder"],
        )
        coverage_captions = [caption.value for caption in app.caption]
        self.assertTrue(
            any(
                "Wave A: the same first 10 boulders" in caption
                for caption in coverage_captions
            )
        )
        self.assertIn("Save style-tag proposal", [button.label for button in app.button])
        self.assertEqual(app.text_input[0].label, "Reviewer code")
        self.assertTrue(str(app.selectbox[3].value).startswith("Task 1/30 ·"))
        self.assertEqual(app.radio[0].label, "Review order")
        self.assertEqual(app.radio[0].value, "30-task coach session")
        self.assertTrue(
            any("Wave A calibration: 0/10" in caption for caption in coverage_captions)
        )

    def test_governed_human_review_session_is_exact_and_identity_free(self) -> None:
        module.human_review_session.clear()
        session = module.human_review_session()
        self.assertEqual(len(session), 30)
        self.assertTrue(session["boulder_uid"].is_unique)
        self.assertEqual(session["session_task_order"].tolist(), list(range(1, 31)))
        self.assertEqual(
            session["review_wave"].value_counts().to_dict(),
            {
                "B_high_unlock_extension": 20,
                "A_same_tasks_independent_calibration": 10,
            },
        )
        self.assertTrue(session["requested_independent_reviewers"].eq(2).all())
        forbidden = {"athlete_id", "athlete_name", "test_value", "metric_value"}
        self.assertFalse(forbidden.intersection(session.columns))

    def test_governed_session_selects_and_orders_exact_inventory_tasks(self) -> None:
        session = pd.DataFrame([
            {
                "boulder_uid": "b-2", "session_task_order": 2,
                "review_wave": "B_high_unlock_extension",
                "requested_independent_reviewers": 2,
                "required_review_scope": "directions_and_three_core_demands; detailed_tags_optional",
                "task_status": "human_review_pending",
            },
            {
                "boulder_uid": "b-1", "session_task_order": 1,
                "review_wave": "A_same_tasks_independent_calibration",
                "requested_independent_reviewers": 2,
                "required_review_scope": "directions_and_three_core_demands; detailed_tags_optional",
                "task_status": "human_review_pending",
            },
        ])
        inventory = pd.DataFrame([
            {"boulder_uid": "b-1", "evidence": 11},
            {"boulder_uid": "b-2", "evidence": 22},
            {"boulder_uid": "b-3", "evidence": 33},
        ])
        selected = module.apply_human_review_session(inventory, session)
        self.assertEqual(selected["boulder_uid"].tolist(), ["b-1", "b-2"])
        self.assertEqual(selected["evidence"].tolist(), [11, 22])
        with self.assertRaises(ValueError):
            module.apply_human_review_session(inventory.iloc[[0]], session)

    def test_session_progress_keeps_independent_reviewers_distinct(self) -> None:
        session = pd.DataFrame([
            {"boulder_uid": "b-1", "review_wave": "A_same_tasks_independent_calibration"},
            {"boulder_uid": "b-2", "review_wave": "B_high_unlock_extension"},
        ])
        records = [
            {"boulder_uid": "b-1", "contributor": "reviewer-a"},
            {"boulder_uid": "b-1", "contributor": "reviewer-a"},
            {"boulder_uid": "b-1", "contributor": "reviewer-b"},
            {"boulder_uid": "b-2", "contributor": "reviewer-b"},
        ]
        progress = module.review_session_progress(
            session, records, reviewer_code="reviewer-a"
        )
        self.assertEqual(
            progress,
            {
                "tasks": 2,
                "reviewed_any": 2,
                "reviewed_by_current": 1,
                "double_reviewed": 1,
                "wave_a_double_reviewed": 1,
            },
        )

    def test_full_coaching_queue_remains_available(self) -> None:
        app = AppTest.from_file(str(PATH)).run(timeout=120)
        app.radio[0].set_value("Coaching evidence unlocked").run(timeout=120)
        self.assertFalse(app.exception)
        self.assertEqual(app.radio[0].value, "Coaching evidence unlocked")
        self.assertTrue(str(app.selectbox[3].value).startswith("Coaching 1 ·"))
        self.assertTrue(
            any(
                "489 governed Boulder tagging tasks cover all 535" in caption.value
                for caption in app.caption
            )
        )
        self.assertTrue(
            any(
                "exact both-board Top-given-Zone comparisons" in caption.value
                for caption in app.caption
            )
        )

    def test_standalone_tagger_owns_its_data_path(self) -> None:
        self.assertEqual(module.DATA, PATH.parents[0] / "data")
        self.assertNotIn("from comp_climbing_app import", PATH.read_text(encoding="utf-8"))

    def test_canonical_broadcast_route_ids(self) -> None:
        self.assertEqual(module.canonical_boulder_label(" w-3 "), "W3")
        self.assertEqual(module.canonical_boulder_label("M 1"), "M1")

    def test_rejects_non_broadcast_route_ids(self) -> None:
        with self.assertRaises(ValueError):
            module.canonical_boulder_label("B5")

    def test_builds_a_schema_v4_record_bound_to_a_problem(self) -> None:
        record = module.build_record(PROBLEM, confidence="High", pre_zone_direction="Up", post_zone_direction="Diagonal", core_values={field: (2, 1) for field in module.CORE_TAG_LABELS}, detailed_values={"crimp_edge_0_3": (3, 2)}, optional_tags_completed=True, reviewer_code="reviewer-1")
        schema = json.loads((Path(__file__).parents[1] / "schemas" / "boulder_style_tag_schema_v4.json").read_text())
        self.assertEqual(record["schema_version"], "4.0")
        self.assertTrue(set(schema["required"]).issubset(record))
        self.assertEqual(record["boulder"], "B2")
        self.assertEqual(record["pre_zone_crimp_edge_0_3"], 3)
        self.assertEqual(record["post_zone_crimp_edge_0_3"], 2)
        self.assertEqual(record["contributor"], "reviewer-1")

    def test_frame_receipt_matches_exact_round_and_boulder_only(self) -> None:
        receipt = {"frames": [
            {"candidate_id": "BWF-0123456789abcdef", "category_round_id": 456, "boulder_slot": "M2", "candidate_status": "REQUIRES_VISUAL_EMPTY_WALL_REVIEW", "empty_wall_verified": False, "frame_seconds": 12.5},
            {"candidate_id": "BWF-fedcba9876543210", "category_round_id": 999, "boulder_slot": "M2", "candidate_status": "REQUIRES_VISUAL_EMPTY_WALL_REVIEW", "empty_wall_verified": False, "frame_seconds": 13.5},
        ]}
        self.assertEqual([row["candidate_id"] for row in module.matching_frame_receipts(receipt, PROBLEM)], ["BWF-0123456789abcdef"])

    def test_shared_record_endpoint_preserves_existing_query_parameters(self) -> None:
        self.assertEqual(module.shared_records_url("https://example.test/exec", 25), "https://example.test/exec?action=list&limit=25")
        self.assertEqual(module.shared_records_url("https://example.test/exec?token=public", 25), "https://example.test/exec?token=public&action=list&limit=25")

    def test_shared_loader_requests_full_independent_review_capacity(self) -> None:
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return b'{"ok":true,"records":[]}'

        requested: list[str] = []

        def open_stub(url: str, timeout: int):
            requested.append(url)
            self.assertEqual(timeout, 15)
            return Response()

        module.load_shared_records.clear()
        with patch.object(module.urlrequest, "urlopen", side_effect=open_stub):
            records, error = module.load_shared_records("https://example.test/exec")
        self.assertEqual(records, [])
        self.assertEqual(error, "")
        self.assertEqual(
            requested,
            ["https://example.test/exec?action=list&limit=3000"],
        )

    def test_shared_history_has_a_download_path(self) -> None:
        source = PATH.read_text(encoding="utf-8")
        self.assertIn('"Download full shared review history"', source)
        self.assertIn('"comp_climbing_shared_style_tags.json"', source)

    def test_completed_boulders_are_hidden_without_reordering_pending_work(self) -> None:
        inventory = pd.DataFrame(
            [
                {"boulder_uid": "round-a-b1", "priority_rank": 1},
                {"boulder_uid": "round-a-b2", "priority_rank": 2},
                {"boulder_uid": "round-a-b3", "priority_rank": 3},
            ]
        )
        records = [
            {"boulder_uid": "round-a-b1"},
            {"boulder_uid": "round-a-b1"},
            {"unrelated": "record"},
        ]
        pending = module.pending_review_inventory(inventory, records)
        self.assertEqual(pending["boulder_uid"].tolist(), ["round-a-b2", "round-a-b3"])
        complete = module.pending_review_inventory(
            inventory,
            records,
            include_completed=True,
        )
        self.assertEqual(complete["boulder_uid"].tolist(), inventory["boulder_uid"].tolist())
        self.assertEqual(module.reviewed_boulder_uids(records), {"round-a-b1"})

    def test_reviewer_specific_completion_allows_independent_second_review(self) -> None:
        inventory = pd.DataFrame([
            {"boulder_uid": "round-a-b1", "priority_rank": 1},
            {"boulder_uid": "round-a-b2", "priority_rank": 2},
        ])
        records = [
            {"boulder_uid": "round-a-b1", "contributor": "reviewer-a"},
            {"boulder_uid": "round-a-b1", "contributor": "reviewer-a"},
            {"boulder_uid": "round-a-b1", "contributor": "reviewer-b"},
        ]
        pending_a = module.pending_review_inventory(
            inventory, records, reviewer_code="reviewer-a"
        )
        pending_c = module.pending_review_inventory(
            inventory, records, reviewer_code="reviewer-c"
        )
        self.assertEqual(pending_a["boulder_uid"].tolist(), ["round-a-b2"])
        self.assertEqual(pending_c["boulder_uid"].tolist(), ["round-a-b1", "round-a-b2"])
        coverage = module.independent_review_coverage(records)
        self.assertEqual(int(coverage.iloc[0]["independent_reviewers"]), 2)

    def test_reviewer_code_is_pseudonymous_and_bounded(self) -> None:
        self.assertEqual(module.normalize_reviewer_code(" reviewer_01 "), "reviewer_01")
        with self.assertRaises(ValueError):
            module.normalize_reviewer_code(" !!! ")
        with self.assertRaises(ValueError):
            module.normalize_reviewer_code("ab@example.com")
        with self.assertRaises(ValueError):
            module.normalize_reviewer_code("ab")

    def test_core_tag_agreement_uses_latest_independent_review(self) -> None:
        base = {
            "boulder_uid": "round-a-b1",
            **{
                f"{segment}_{field}": 1
                for segment in ("pre_zone", "post_zone")
                for field in module.CORE_TAG_LABELS
            },
        }
        records = [
            {**base, "contributor": "a", "submitted_at_utc": "2026-01-01T00:00:00Z",
             "pre_zone_physical_0_3": 3},
            {**base, "contributor": "a", "submitted_at_utc": "2026-01-02T00:00:00Z"},
            {**base, "contributor": "b", "submitted_at_utc": "2026-01-01T00:00:00Z",
             "post_zone_coordination_0_3": 2},
        ]
        result = module.independent_core_tag_agreement(records)
        physical = result.loc[result["Core tag"].eq("pre_zone_physical_0_3")].iloc[0]
        coordination = result.loc[
            result["Core tag"].eq("post_zone_coordination_0_3")
        ].iloc[0]
        self.assertEqual(physical["Exact agreement"], 1.0)
        self.assertEqual(coordination["Mean reviewer range (0-3)"], 1.0)
        self.assertEqual(int(coordination["Double-reviewed boulders"]), 1)

    def test_exact_round_priority_is_attached_without_excluding_other_items(self) -> None:
        inventory = pd.DataFrame([
            {
                "source_scope": "CEC", "source_event_id": 224,
                "source_round_ids": "4418|4420", "boulder_number": 3,
                "event_date": "2026-05-14", "event_name": "Youth Nationals",
                "round_group": "Qualification", "gender": "Men",
            },
            {
                "source_scope": "CEC", "source_event_id": 224,
                "source_round_ids": "4418|4420", "boulder_number": 4,
                "event_date": "2026-05-14", "event_name": "Youth Nationals",
                "round_group": "Qualification", "gender": "Men",
            },
        ])
        priority = pd.DataFrame([
            {
                "source_scope": "CEC", "source_event_id": 224,
                "source_round_id": 4420, "boulder_number": 3,
                "priority_rank": 2, "linked_athletes": 7, "linked_outcomes": 7,
                "board_linked_outcomes": 5,
                "top_given_zone_discordant_pairs": 4,
                "zone_discordant_pairs": 2,
            },
            {
                "source_scope": "CEC", "source_event_id": 224,
                "source_round_id": 4418, "boulder_number": 3,
                "priority_rank": 5, "linked_athletes": 5, "linked_outcomes": 5,
                "board_linked_outcomes": 3,
                "top_given_zone_discordant_pairs": 2,
                "zone_discordant_pairs": 1,
            },
        ])
        result = module.apply_tagging_priority(inventory, priority)
        self.assertEqual(result.iloc[0]["priority_rank"], 2)
        self.assertEqual(result.iloc[0]["priority_source_items"], 2)
        self.assertEqual(result.iloc[0]["priority_linked_athletes"], 7)
        self.assertEqual(result.iloc[0]["priority_linked_outcomes"], 12)
        self.assertEqual(result.iloc[0]["priority_board_linked_outcomes"], 8)
        self.assertEqual(result.iloc[0]["priority_top_given_zone_pairs"], 6)
        self.assertEqual(result.iloc[0]["priority_zone_pairs"], 3)
        self.assertEqual(result.iloc[1]["priority_status"], "General governed inventory")
        self.assertEqual(len(result), 2)

    def test_coaching_unlock_priority_is_separate_and_defaults_first(self) -> None:
        inventory = pd.DataFrame([{
            "source_scope": "CEC", "source_event_id": 224,
            "source_round_ids": "4420", "boulder_number": 3,
            "event_date": "2026-05-14", "event_name": "Youth Nationals",
            "round_group": "Qualification", "gender": "Men",
            "boulder_uid": "round-youth-national-b3",
        }])
        priority = pd.DataFrame([{
            "source_scope": "CEC", "source_event_id": 224,
            "source_round_id": 4420, "boulder_number": 3,
            "priority_rank": 20, "linked_athletes": 7, "linked_outcomes": 7,
            "coaching_unlock_rank": 1, "coaching_athletes_unlocked": 8,
            "coaching_observations_unlocked": 145,
            "physical_observations_unlocked": 103,
            "board_observations_unlocked": 42,
        }])
        row = module.apply_tagging_priority(inventory, priority).iloc[0]
        self.assertEqual(row["priority_rank"], 20)
        self.assertEqual(row["coaching_unlock_rank"], 1)
        self.assertEqual(row["coaching_unlock_observations"], 145)
        self.assertTrue(module.problem_display(row).startswith("Coaching 1 ·"))
        self.assertTrue(
            module.problem_display(row, prefer_coaching=False).startswith("Priority 20 ·")
        )

    def test_app_coaching_unlock_file_contains_no_athlete_identity_or_values(self) -> None:
        path = Path(__file__).parents[1] / "data" / "physical_item_tag_unlock_app_v1.csv"
        queue = pd.read_csv(path)
        self.assertEqual(len(queue), 535)
        self.assertTrue(queue["problem_id"].is_unique)
        forbidden = {
            "athlete_id", "athlete_name", "metric_id", "value", "test_result"
        }
        self.assertFalse(forbidden.intersection(queue.columns))

    def test_ranked_review_milestones_report_cumulative_information(self) -> None:
        priority = pd.read_csv(
            Path(__file__).parents[1]
            / "data"
            / "physical_item_tagging_priority_v1_1.csv"
        )
        result = module.tagging_coverage_milestones(priority)
        self.assertEqual(result["Reviewed items"].tolist(), [10, 25, 50, 100])
        self.assertEqual(
            result["Top|Zone discordant pairs"].tolist(),
            [62, 103, 130, 130],
        )
        self.assertEqual(result["Zone discordant pairs"].tolist(), [0, 18, 31, 108])
        self.assertEqual(result["Athlete-item links"].tolist(), [57, 130, 189, 393])

    def test_tagger_renders_incremental_review_table(self) -> None:
        app = AppTest.from_file(str(PATH)).run(timeout=120)
        app.radio[0].set_value("Coaching evidence unlocked").run(timeout=120)
        self.assertFalse(app.exception)
        tables = [item.value for item in app.dataframe]
        milestone = next(
            table for table in tables if "Reviewed items" in table.columns
        )
        self.assertEqual(milestone["Reviewed items"].tolist(), [10, 25, 50, 100])
        self.assertEqual(int(milestone.iloc[2]["Top|Zone discordant pairs"]), 130)
        self.assertTrue(
            any(
                "not independent competitions" in str(item.value)
                for item in app.caption
            )
        )


if __name__ == "__main__":
    unittest.main()
