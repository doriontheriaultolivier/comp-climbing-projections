from pathlib import Path
import unittest

import pandas as pd
import plotly.graph_objects as go
from streamlit.testing.v1 import AppTest

from comp_climbing_app import add_outcome_thresholds, physical_transfer_figure


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
        next(
            item for item in app.segmented_control
            if item.label == "Overview section"
        ).set_value("Global progression").run(timeout=120)
        self.assertFalse(app.exception)
        self.assertEqual(len(app.get("plotly_chart")), 2)

    def test_pool_views_and_format_transform(self) -> None:
        app = AppTest.from_file(str(ROOT / "streamlit_app.py"))
        app.run(timeout=120)

        round_evidence = next(
            item for item in app.selectbox if item.label == "Round evidence"
        )
        round_evidence.select("Qualification").run(timeout=120)
        self.assertFalse(app.exception)
        next(
            item for item in app.selectbox
            if item.label == "Qualification procedure"
        ).select("Onsight").run(timeout=120)
        self.assertFalse(app.exception)

        for section in ["IFSC Pool", "WR Pool"]:
            overview = next(
                item for item in app.segmented_control
                if item.label == "Overview section"
            )
            overview.set_value(section).run(timeout=120)
            self.assertFalse(app.exception, section)
            self.assertIn(section, [item.value for item in app.subheader])

    def test_physical_and_maths_workspaces(self) -> None:
        app = AppTest.from_file(str(ROOT / "streamlit_app.py"))
        app.run(timeout=120)
        workspace = next(
            item for item in app.segmented_control if item.label == "Workspace"
        )
        workspace.set_value("Maths behind").run(timeout=120)
        self.assertFalse(app.exception)
        self.assertIn("Maths behind", [item.value for item in app.header])

        workspace = next(
            item for item in app.segmented_control if item.label == "Workspace"
        )
        workspace.set_value("Physical Strength").run(timeout=120)
        self.assertFalse(app.exception)
        self.assertIn(
            "Physical Strength and Training Priorities",
            [item.value for item in app.header],
        )
        self.assertGreaterEqual(len(app.get("plotly_chart")), 1)

        workspace = next(
            item for item in app.segmented_control if item.label == "Workspace"
        )
        workspace.set_value("Tag Boulder Styles").run(timeout=120)
        self.assertFalse(app.exception)
        self.assertIn("Boulder Style Tagging", [item.value for item in app.header])

    def test_four_governed_outcome_reference_lines(self) -> None:
        calibration = pd.read_csv(ROOT / "data" / "boulder_elo_calibration.csv")
        figure = go.Figure()
        add_outcome_thresholds(figure, calibration)
        self.assertEqual(len(figure.layout.shapes), 4)
        levels = sorted(float(shape.y0) for shape in figure.layout.shapes)
        self.assertAlmostEqual(levels[0], 2000.0, places=6)
        self.assertTrue(levels == sorted(levels))

    def test_standalone_style_tagger_uses_paired_optional_scores(self) -> None:
        app = AppTest.from_file(str(ROOT / "style_tagging_app.py"))
        app.run(timeout=120)
        self.assertFalse(app.exception)
        self.assertEqual(app.title[0].value, "Comp Climbing Boulder Tags")
        slider_labels = {item.label for item in app.slider}
        self.assertIn("🟠 Top - Physical", slider_labels)
        self.assertIn("🔵 Zone - Physical", slider_labels)
        self.assertNotIn("Up to 3 moves before zone", {item.label for item in app.text_area})

        optional = next(
            item for item in app.checkbox
            if item.label == "I would like to help further and identify tags"
        )
        optional.set_value(True).run(timeout=120)
        self.assertFalse(app.exception)
        slider_labels = {item.label for item in app.slider}
        self.assertIn("🟠 Top - Slopers", slider_labels)
        self.assertIn("🔵 Zone - Dyno", slider_labels)

    def test_physical_transfer_plot_has_fitted_line_and_residual_tags(self) -> None:
        frame = pd.DataFrame({
            "athlete_name": [f"Athlete {index}" for index in range(14)],
            "value": list(range(14)),
            "Global-ELO": [1800 + 25 * index for index in range(14)],
        })
        frame.loc[13, "Global-ELO"] += 250
        figure, rho, evidence = physical_transfer_figure(
            frame, "value", "Global-ELO", "Test transfer"
        )
        self.assertGreater(rho, 0.2)
        self.assertGreater(evidence["probability_positive"], 0.5)
        trace_names = {trace.name for trace in figure.data}
        self.assertIn("Rating expected from this test alone", trace_names)
        self.assertIn("Possible opportunity: performance ahead of test", trace_names)


if __name__ == "__main__":
    unittest.main()
