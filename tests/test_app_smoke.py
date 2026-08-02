from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


class AppSmokeTests(unittest.TestCase):
    def test_default_canadian_pool(self) -> None:
        app = AppTest.from_file(str(ROOT / "streamlit_app.py"))
        app.run(timeout=120)
        self.assertFalse(app.exception)
        self.assertEqual(app.title[0].value, "Comp Climbing Projections")
        self.assertIn("Canadian Pool", [item.value for item in app.subheader])
        self.assertEqual(len(app.get("plotly_chart")), 1)

    def test_progression_view(self) -> None:
        app = AppTest.from_file(str(ROOT / "streamlit_app.py"))
        app.run(timeout=120)
        app.segmented_control[2].set_value("Global progression").run(timeout=120)
        self.assertFalse(app.exception)
        self.assertEqual(len(app.get("plotly_chart")), 2)

    def test_pool_views_and_format_transform(self) -> None:
        app = AppTest.from_file(str(ROOT / "streamlit_app.py"))
        app.run(timeout=120)

        round_format = next(
            item for item in app.selectbox if item.label == "Round format"
        )
        round_format.select("Onsight").run(timeout=120)
        self.assertFalse(app.exception)
        self.assertEqual(
            next(item for item in app.selectbox if item.label == "Round format").value,
            "Onsight",
        )

        for section in ["IFSC Pool", "WR Pool"]:
            overview = next(
                item for item in app.segmented_control
                if item.label == "Overview section"
            )
            overview.set_value(section).run(timeout=120)
            self.assertFalse(app.exception, section)
            self.assertIn(section, [item.value for item in app.subheader])


if __name__ == "__main__":
    unittest.main()
