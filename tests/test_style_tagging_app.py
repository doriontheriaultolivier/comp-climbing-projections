from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest

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
        self.assertIn("Save style-tag proposal", [button.label for button in app.button])

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
        record = module.build_record(PROBLEM, confidence="High", pre_zone_direction="Up", post_zone_direction="Diagonal", core_values={field: (2, 1) for field in module.CORE_TAG_LABELS}, detailed_values={"crimp_edge_0_3": (3, 2)}, optional_tags_completed=True)
        schema = json.loads((Path(__file__).parents[1] / "schemas" / "boulder_style_tag_schema_v4.json").read_text())
        self.assertEqual(record["schema_version"], "4.0")
        self.assertTrue(set(schema["required"]).issubset(record))
        self.assertEqual(record["boulder"], "B2")
        self.assertEqual(record["pre_zone_crimp_edge_0_3"], 3)
        self.assertEqual(record["post_zone_crimp_edge_0_3"], 2)

    def test_frame_receipt_matches_exact_round_and_boulder_only(self) -> None:
        receipt = {"frames": [
            {"candidate_id": "BWF-0123456789abcdef", "category_round_id": 456, "boulder_slot": "M2", "candidate_status": "REQUIRES_VISUAL_EMPTY_WALL_REVIEW", "empty_wall_verified": False, "frame_seconds": 12.5},
            {"candidate_id": "BWF-fedcba9876543210", "category_round_id": 999, "boulder_slot": "M2", "candidate_status": "REQUIRES_VISUAL_EMPTY_WALL_REVIEW", "empty_wall_verified": False, "frame_seconds": 13.5},
        ]}
        self.assertEqual([row["candidate_id"] for row in module.matching_frame_receipts(receipt, PROBLEM)], ["BWF-0123456789abcdef"])

    def test_shared_record_endpoint_preserves_existing_query_parameters(self) -> None:
        self.assertEqual(module.shared_records_url("https://example.test/exec", 25), "https://example.test/exec?action=list&limit=25")
        self.assertEqual(module.shared_records_url("https://example.test/exec?token=public", 25), "https://example.test/exec?token=public&action=list&limit=25")


if __name__ == "__main__":
    unittest.main()
