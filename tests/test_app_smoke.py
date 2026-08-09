from pathlib import Path
import unittest

import pandas as pd
from streamlit.testing.v1 import AppTest

from comp_climbing_app import plain_key, selected_rows


ROOT = Path(__file__).resolve().parents[1]


class AppSmokeTests(unittest.TestCase):
    def test_default_canadian_pilot(self) -> None:
        app = AppTest.from_file(str(ROOT / "streamlit_app.py"))
        app.run(timeout=120)
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertEqual(app.title[0].value, "Comp Climbing Projections")
        self.assertIn(
            "Canadian performance benchmark pilot",
            [item.value for item in app.header],
        )
        metrics = {item.label: item.value for item in app.metric}
        self.assertEqual(metrics["Oscar Baudrand"], "57.8%")
        self.assertEqual(metrics["Matthew Rodriguez"], "14.4%")
        self.assertEqual(metrics["DORVAL Hugo"], "6.7%")
        self.assertIn("Canadian Pool", [item.value for item in app.subheader])
        self.assertGreaterEqual(len(app.dataframe), 1)

    def test_progression_view(self) -> None:
        app = AppTest.from_file(str(ROOT / "streamlit_app.py"))
        app.run(timeout=120)
        next(
            item
            for item in app.segmented_control
            if item.label == "Overview section"
        ).set_value("Global progression").run(timeout=120)
        self.assertFalse(app.exception)
        self.assertGreaterEqual(len(app.get("plotly_chart")), 2)

    def test_pool_views_and_format_transform(self) -> None:
        app = AppTest.from_file(str(ROOT / "streamlit_app.py"))
        app.run(timeout=120)

        next(
            item for item in app.selectbox if item.label == "Round format"
        ).select("Onsight").run(timeout=120)
        self.assertFalse(app.exception)

        for section in ["IFSC Pool", "WR Pool", "Towards Olympics"]:
            overview = next(
                item
                for item in app.segmented_control
                if item.label == "Overview section"
            )
            overview.set_value(section).run(timeout=120)
            self.assertFalse(app.exception, section)

    def test_selected_rows_uses_global_id_not_duplicate_display_name(self) -> None:
        frame = pd.DataFrame(
            {
                "pool": ["Boulder_Women", "Boulder_Women"],
                "global_id": ["IFSC:1629", "USAC:999"],
                "name_key": [plain_key("Madison Richardson")] * 2,
                "athlete_name": ["Madison Richardson"] * 2,
            }
        )
        chosen = selected_rows(frame, ["Boulder_Women::IFSC:1629"])
        self.assertEqual(chosen["global_id"].tolist(), ["IFSC:1629"])

    def test_canadian_shared_youth_terrain_has_source_count_six(self) -> None:
        inventory = pd.read_csv(ROOT / "data" / "boulder_round_inventory.csv")
        rows = inventory.loc[
            inventory["event_name"].str.contains(
                "2025-26 Youth Nationals - Boulder", case=False, na=False
            )
            & inventory["terrain_group"].eq(
                "Youth A + Junior (shared Canadian terrain)"
            )
            & inventory["gender"].eq("Men")
            & inventory["round_group"].eq("Qualification")
        ]
        self.assertEqual(set(rows["boulder_count"]), {6})
        self.assertEqual(set(rows["boulder_count_status"]), {"source-confirmed"})
        self.assertEqual(set(rows["category"].astype(str)), {"U19 Male", "U21 Male"})

    def test_youth_world_categories_never_use_canadian_shared_terrain(self) -> None:
        inventory = pd.read_csv(ROOT / "data" / "boulder_round_inventory.csv")
        youth_worlds = inventory.loc[
            inventory["source_scope"].eq("IFSC")
            & inventory["event_name"].str.contains(
                "Youth Championship", case=False, na=False
            )
        ]
        self.assertFalse(
            youth_worlds["terrain_group"]
            .eq("Youth A + Junior (shared Canadian terrain)")
            .any()
        )


if __name__ == "__main__":
    unittest.main()
