from pathlib import Path
import unittest

import pandas as pd
import plotly.graph_objects as go
from streamlit.testing.v1 import AppTest

from comp_climbing_app import (
    _governed_boulder_count,
    _saturation_cv_comparison,
    add_outcome_thresholds,
    physical_sufficiency_table,
    physical_transfer_figure,
    plain_key,
    rating_radar_figure,
    read_data,
)


ROOT = Path(__file__).resolve().parents[1]


class AppSmokeTests(unittest.TestCase):
    def test_default_canadian_pool(self) -> None:
        app = AppTest.from_file(str(ROOT / "streamlit_app.py"))
        app.run(timeout=120)
        self.assertFalse(app.exception)
        self.assertEqual(app.title[0].value, "Comp Climbing Projections")
        self.assertIn("Canadian Pool", [item.value for item in app.subheader])
        self.assertGreaterEqual(len(app.get("plotly_chart")), 2)

    def test_progression_view(self) -> None:
        app = AppTest.from_file(str(ROOT / "streamlit_app.py"))
        app.run(timeout=120)
        next(
            item for item in app.segmented_control
            if item.label == "Overview section"
        ).set_value("Global progression").run(timeout=120)
        self.assertFalse(app.exception)
        self.assertGreaterEqual(len(app.get("plotly_chart")), 3)

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
        self.assertIn("After zone Physical", slider_labels)
        self.assertIn("Before zone Physical", slider_labels)
        self.assertNotIn("Up to 3 moves before zone", {item.label for item in app.text_area})

        optional = next(
            item for item in app.checkbox
            if item.label == "I would like to help further and identify holds and movements"
        )
        optional.set_value(True).run(timeout=120)
        self.assertFalse(app.exception)
        slider_labels = {item.label for item in app.slider}
        self.assertIn("After zone Slopers", slider_labels)
        self.assertIn("Before zone Dyno", slider_labels)
        self.assertIn("Before zone Crimps / edges · 12–30 mm", slider_labels)
        self.assertIn("After zone Small crimps / edges · <12 mm", slider_labels)
        self.assertIn("Before zone No-texture footholds", slider_labels)
        self.assertIn("Before zone Fight a barn door", slider_labels)
        self.assertIn("After zone Overhead press", slider_labels)
        self.assertIn("Before zone Small sideways opposition", slider_labels)
        self.assertIn("After zone Smeary heel hook", slider_labels)
        self.assertNotIn("Before zone Crimps", slider_labels)
        self.assertNotIn("Before zone Toe hook", slider_labels)

    def test_governed_boulder_count_prefers_exact_source_metadata(self) -> None:
        rows = pd.DataFrame({
            "boulder_count": [6, 6],
            "boulder_count_status": ["source-confirmed", "source-confirmed"],
            "boulder_count_source": ["normalized results n_routes"] * 2,
        })
        result = _governed_boulder_count(
            rows, pd.Series([5]), round_name="Qualification"
        )
        self.assertEqual(result["count"], 6)
        self.assertEqual(result["status"], "source-confirmed")
        self.assertFalse(result["editable"])

    def test_rating_radar_keeps_round_evidence_in_hover(self) -> None:
        families = [
            "Global-ELO", "Global-ELO-Qualies", "Global-ELO-Qualies-Flash",
            "Global-ELO-Qualies-Onsight", "Global-ELO-Semies", "Global-ELO-Finals",
        ]
        rows = []
        for athlete_index, athlete in enumerate(["Oscar Baudrand", "Colin Duffy"]):
            for family_index, family in enumerate(families):
                rows.append({
                    "Athlete": athlete, "Rating family": family,
                    "Elo": 1950 + athlete_index * 100 + family_index * 10,
                    "Included rounds": 2 + family_index * 3,
                    "Historical outcome estimate": "Make semifinal: 50% (fit 38%)",
                })
        figure = rating_radar_figure(
            pd.DataFrame(rows), ["Oscar Baudrand", "Colin Duffy"],
            "All-competition profile",
        )
        self.assertEqual(len(figure.data), 3)
        self.assertIn("Included rounds", figure.data[1].hovertemplate)
        self.assertNotEqual(figure.data[1].marker.size[0], figure.data[1].marker.size[1])
        self.assertLessEqual(figure.layout.polar.radialaxis.range[0], 2000)
        self.assertGreaterEqual(figure.layout.polar.radialaxis.range[1], 2000)

    def test_governed_boulder_count_exposes_conflict(self) -> None:
        rows = pd.DataFrame({
            "boulder_count": [4, 5],
            "boulder_count_status": ["source-confirmed", "source-confirmed"],
            "boulder_count_source": ["normalized results n_routes"] * 2,
        })
        result = _governed_boulder_count(rows, round_name="Final")
        self.assertEqual(result["status"], "source-conflict")
        self.assertTrue(result["editable"])

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
            youth_worlds["terrain_group"].eq(
                "Youth A + Junior (shared Canadian terrain)"
            ).any()
        )

    def test_style_tagger_save_and_next_boulder(self) -> None:
        app = AppTest.from_file(str(ROOT / "style_tagging_app.py"))
        app.run(timeout=120)
        next(item for item in app.button if item.label == "Save and continue").click().run(
            timeout=120
        )
        self.assertFalse(app.exception)
        self.assertIn("Next boulder ->", [item.label for item in app.button])
        boulder = next(
            item for item in app.segmented_control if item.label == "Boulder"
        )
        self.assertTrue(any(
            option.startswith("B1 · 1S/0T") for option in boulder.options
        ))
        next(item for item in app.button if item.label == "Next boulder ->").click().run(
            timeout=120
        )
        self.assertFalse(app.exception)
        boulder = next(
            item for item in app.segmented_control if item.label == "Boulder"
        )
        self.assertTrue(boulder.value.startswith("B2"))

    def test_physical_transfer_plot_has_saturation_curve_and_sufficiency_tags(self) -> None:
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
        self.assertIn("Pooled diminishing-return estimate", trace_names)
        self.assertIn("Possibly below estimated sufficiency", trace_names)
        self.assertTrue(evidence["groups"])
        self.assertGreater(evidence["groups"][0]["threshold"], 5)

    def test_grade_curve_detects_group_direction_conflict(self) -> None:
        frame = pd.DataFrame({
            "athlete_name": [f"M {i}" for i in range(8)] + [f"W {i}" for i in range(8)],
            "pool": ["Boulder_Men"] * 8 + ["Boulder_Women"] * 8,
            "grade": list(range(8)) + list(range(8)),
            "Global-ELO": [1800 + 20 * i for i in range(8)]
            + [2100 - 20 * i for i in range(8)],
        })
        comparison = _saturation_cv_comparison(frame, "grade", "Global-ELO")
        self.assertEqual(comparison["choice"], "Gender-specific")
        self.assertIn(
            comparison["reason"],
            {"Lower held-out prediction error", "Pooled line hides opposite group directions"},
        )

    def test_louka_half_crimp_has_no_deficit_signal(self) -> None:
        data = read_data()
        latest = data["physical_latest"].copy()
        latest["name_key"] = latest["athlete_name"].map(plain_key)
        evidence_columns = [
            column for column in [
                "Global-ELO evidence", "IFSC-ELO evidence", "WR-ELO evidence"
            ] if column in data["athletes"]
        ]
        lookup = (
            data["athletes"][["pool", "name_key", *evidence_columns]]
            .sort_values(evidence_columns[0], ascending=False)
            .drop_duplicates(["pool", "name_key"])
        )
        latest = latest.merge(lookup, on=["pool", "name_key"], how="left")
        screen = physical_sufficiency_table(latest, "Louka Boivin")
        half_crimp = screen.loc[screen["Test"].eq("20mm HC Semi-Isolé")].iloc[0]
        self.assertGreaterEqual(half_crimp["Peer percentile"], 95)
        self.assertIn("no deficit signal", half_crimp["Current reading"].lower())


if __name__ == "__main__":
    unittest.main()
