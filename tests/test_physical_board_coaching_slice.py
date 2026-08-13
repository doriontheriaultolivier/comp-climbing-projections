from __future__ import annotations

from pathlib import Path
import unittest

import pandas as pd
from streamlit.testing.v1 import AppTest

from comp_climbing_app import coaching_profile_rows, read_data


ROOT = Path(__file__).resolve().parents[1]


class PhysicalBoardCoachingSliceTests(unittest.TestCase):
    def test_read_data_loads_coaching_inputs(self) -> None:
        read_data.clear()
        data = read_data()
        self.assertEqual(len(data["physical_profiles"]), 22)
        self.assertEqual(len(data["physical_priorities"]), 1727)
        self.assertEqual(len(data["physical_tagging_queue"]), 535)
        self.assertIn("boulder_grade_50pct_flash_v", data["physical_profiles"])
        self.assertIn(
            "boulder_grade_3x_physical_sends_last_3_months_v",
            data["physical_profiles"],
        )

    def test_join_is_pool_and_name_scoped(self) -> None:
        profiles = pd.DataFrame(
            {
                "athlete_name": ["BOIVIN Louka", "BOIVIN Louka"],
                "pool": ["Boulder_Men", "Boulder_Women"],
                "test_sessions": [10, 99],
            }
        )
        priorities = pd.DataFrame(
            {
                "athlete_name": ["Louka Boivin", "Louka Boivin"],
                "pool": ["Boulder_Men", "Boulder_Women"],
                "recommendation": ["Focus candidate", "Focus candidate"],
            }
        )
        selected = pd.DataFrame(
            {"athlete_name": ["Louka Boivin"], "pool": ["Boulder_Men"]}
        )
        joined_profiles, joined_priorities = coaching_profile_rows(
            profiles, priorities, selected
        )
        self.assertEqual(joined_profiles["test_sessions"].tolist(), [10])
        self.assertEqual(joined_priorities["pool"].tolist(), ["Boulder_Men"])

    def test_default_app_explains_missing_profile_without_error(self) -> None:
        app = AppTest.from_file(str(ROOT / "streamlit_app.py"))
        app.run(timeout=120)
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertIn(
            "Capacity → board expression",
            [item.value for item in app.header],
        )
        self.assertTrue(
            any(
                "Missing tests are missing evidence" in str(item.value)
                for item in app.info
            )
        )

    def test_linked_athlete_shows_both_board_indicators(self) -> None:
        app = AppTest.from_file(str(ROOT / "streamlit_app.py"))
        app.run(timeout=120)
        next(
            item for item in app.selectbox if item.label == "Main athlete"
        ).select("BOIVIN Louka").run(timeout=120)
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        metrics = {item.label: item.value for item in app.metric}
        self.assertEqual(metrics["50%-flash Kilter equivalent"], "V11.3")
        self.assertEqual(metrics["Recent 3-send Kilter equivalent"], "V12.5")
        self.assertEqual(metrics["Physical test sessions"], "10")
        self.assertTrue(
            any(
                "535 current tasks" in str(item.value)
                for item in app.warning
            )
        )


if __name__ == "__main__":
    unittest.main()
