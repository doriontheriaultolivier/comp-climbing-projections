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


if __name__ == "__main__":
    unittest.main()
