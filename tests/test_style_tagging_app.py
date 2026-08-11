from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest


PATH = Path(__file__).parents[1] / "style_tagging_app.py"
SPEC = importlib.util.spec_from_file_location("style_tagging_app", PATH)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


class StyleTaggingAppTest(unittest.TestCase):
    def test_canonical_broadcast_route_ids(self) -> None:
        self.assertEqual(module.canonical_boulder_label(" w-3 "), "W3")
        self.assertEqual(module.canonical_boulder_label("M 1"), "M1")

    def test_rejects_non_broadcast_route_ids(self) -> None:
        with self.assertRaises(ValueError):
            module.canonical_boulder_label("B5")

    def test_limits_known_rounds_to_the_source_boulder_count(self) -> None:
        self.assertEqual(module.boulder_options("Women", 4), ["W1", "W2", "W3", "W4"])
        self.assertEqual(module.boulder_options("Men", "5.0"), ["M1", "M2", "M3", "M4", "M5"])
        self.assertEqual(module.boulder_options("Mixed / unknown", 4), [])

    def test_shared_record_endpoint_preserves_existing_query_parameters(self) -> None:
        self.assertEqual(
            module.shared_records_url("https://example.test/exec", 25),
            "https://example.test/exec?action=list&limit=25",
        )
        self.assertEqual(
            module.shared_records_url("https://example.test/exec?token=public", 25),
            "https://example.test/exec?token=public&action=list&limit=25",
        )

    def test_loads_the_governed_route_demand_fields(self) -> None:
        fields = module.route_fields()
        self.assertIn("three_dimensionality_0_3", fields)
        self.assertNotIn("volume_macro_density_0_3", fields)

    def test_builds_a_schema_v2_record(self) -> None:
        record = module.build_record(
            competition="2026 Test Open", competition_date="2026-08-11", round_name="Final",
            category="Women", boulder="W2", confidence="High", top_direction="Up",
            zone_direction="Diagonal", core_values={field: (2, 1) for field in module.CORE_TAG_LABELS},
            detailed_values={"crimp_edge_0_3": (3, 2)}, optional_tags_completed=True,
        )
        schema = json.loads((Path(__file__).parents[1] / "schemas" / "boulder_style_tag_schema_v2.json").read_text())
        self.assertEqual(record["schema_version"], "2.0")
        self.assertTrue(set(schema["required"]).issubset(record))
        self.assertNotIn("route_tags", record)
        self.assertEqual(record["top_crimp_edge_0_3"], 3)
        self.assertEqual(record["zone_crimp_edge_0_3"], 2)


if __name__ == "__main__":
    unittest.main()
