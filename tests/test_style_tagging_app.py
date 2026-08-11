from __future__ import annotations

import importlib.util
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

    def test_loads_the_governed_route_demand_fields(self) -> None:
        fields = module.route_fields()
        self.assertIn("three_dimensionality_0_3", fields)
        self.assertNotIn("volume_macro_density_0_3", fields)


if __name__ == "__main__":
    unittest.main()
