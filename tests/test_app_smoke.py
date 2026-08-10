import hashlib
import json
from pathlib import Path
import unittest

import pandas as pd
from streamlit.testing.v1 import AppTest

from comp_climbing_app import integer_observation, plain_key, selected_rows


ROOT = Path(__file__).resolve().parents[1]


def _stable_snapshot_value(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_stable_snapshot_value(item) for item in value]
    return str(value)


def _element_tree_sha256(app: AppTest) -> str:
    manifest = []
    for position, element in enumerate(app._tree):
        proto = getattr(element, "proto", None)
        serialize = getattr(proto, "SerializeToString", None)
        if serialize is None:
            continue
        element_type = str(getattr(element, "type", type(element).__name__))
        normalized = proto
        if element_type == "dataframe" and hasattr(proto, "arrow_data"):
            normalized = type(proto)()
            normalized.CopyFrom(proto)
            styler = normalized.arrow_data.styler
            styler_uuid = str(styler.uuid)
            if styler_uuid:
                styler.styles = str(styler.styles).replace(
                    f"#T_{styler_uuid}", "#T_STABLE"
                )
                styler.uuid = "STABLE"
        payload = normalized.SerializeToString(deterministic=True)
        manifest.append(
            {
                "position": position,
                "type": element_type,
                "key": _stable_snapshot_value(getattr(element, "key", None)),
                "proto_type": str(proto.DESCRIPTOR.full_name),
                "proto_sha256": hashlib.sha256(payload).hexdigest(),
                "proto_bytes": len(payload),
            }
        )
    encoded = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _olympics_fixture_app(athletes, selected) -> None:
    import pandas as fixture_pd

    from comp_climbing_app import render_olympics

    render_olympics(athletes, selected, fixture_pd.DataFrame())


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

    def test_projection_interpretation_fields_and_named_regressions(self) -> None:
        app = AppTest.from_file(str(ROOT / "streamlit_app.py"))
        app.run(timeout=120)
        self.assertFalse(app.exception)
        self.assertFalse(app.error)

        metrics = {item.label: item.value for item in app.metric}
        self.assertEqual(
            {
                "Oscar Baudrand": metrics["Oscar Baudrand"],
                "Matthew Rodriguez": metrics["Matthew Rodriguez"],
                "DORVAL Hugo": metrics["DORVAL Hugo"],
            },
            {
                "Oscar Baudrand": "57.8%",
                "Matthew Rodriguez": "14.4%",
                "DORVAL Hugo": "6.7%",
            },
        )

        for control, selection_id in zip(
            app.selectbox[:3],
            (
                "Boulder_Men::CEC:3253",
                "Boulder_Men::IFSC:18284",
                "Boulder_Men::IFSC:1682",
            ),
        ):
            control.set_value(selection_id)
        app.run(timeout=120)
        self.assertFalse(app.exception)
        selected_metrics = {item.label: item.value for item in app.metric}
        self.assertEqual(selected_metrics["SANTOPRETE Leonardo"], "8.2%")
        self.assertEqual(selected_metrics["ARTEAU Nicolas"], "8.7%")
        self.assertEqual(selected_metrics["DORVAL Hugo"], "6.7%")

        projection_tables = [
            element.value
            for element in app.dataframe
            if "Representative semifinal" in element.value.columns
        ]
        selected = next(table for table in projection_tables if len(table) == 3)
        all_current = next(table for table in projection_tables if len(table) == 82)
        self.assertEqual(
            selected.columns[2:5].tolist(),
            [
                "Projection confidence",
                "Rating-state sensitivity",
                "Evidence route",
            ],
        )
        self.assertEqual(
            all_current.columns[4:7].tolist(),
            [
                "Projection confidence",
                "Rating-state sensitivity",
                "Evidence route",
            ],
        )
        self.assertIn("Graph connectivity", selected.columns)
        self.assertIn("Graph connectivity", all_current.columns)
        self.assertNotIn("Connected evidence", selected.columns)
        self.assertNotIn("Connected evidence", all_current.columns)

        by_name = all_current.set_index("Athlete")
        hugo = by_name.loc["DORVAL Hugo"]
        self.assertEqual(hugo["Representative semifinal"], "6.7%")
        self.assertEqual(hugo["Rating-state sensitivity"], "2.6%–16.3%")
        self.assertEqual(int(hugo["Direct Senior/Open WC+ comps"]), 9)
        self.assertIn("Higher target evidence", hugo["Projection confidence"])
        self.assertNotIn("Indirect-to-WC", hugo["Graph connectivity"])

        leonardo = by_name.loc["SANTOPRETE Leonardo"]
        self.assertEqual(leonardo["Representative semifinal"], "8.2%")
        self.assertEqual(leonardo["Rating-state sensitivity"], "1.0%–45.1%")
        self.assertEqual(int(leonardo["Direct Senior/Open WC+ comps"]), 0)
        self.assertIn("Very low absolute certainty", leonardo["Projection confidence"])
        self.assertIn("Indirect-to-WC", leonardo["Graph connectivity"])
        self.assertIn("provisional graph", leonardo["Graph connectivity"])

        nicolas = by_name.loc["ARTEAU Nicolas"]
        self.assertEqual(nicolas["Representative semifinal"], "8.7%")
        self.assertEqual(nicolas["Rating-state sensitivity"], "2.0%–31.3%")
        self.assertEqual(int(nicolas["Direct Senior/Open WC+ comps"]), 0)
        self.assertIn("Very low absolute certainty", nicolas["Projection confidence"])
        self.assertIn("Indirect-to-WC", nicolas["Graph connectivity"])
        self.assertIn("established graph", nicolas["Graph connectivity"])

        captions = [str(item.value) for item in app.caption]
        self.assertIn("Rating-state sensitivity: 1.0%–45.1%", captions)
        self.assertIn("Rating-state sensitivity: 2.0%–31.3%", captions)
        self.assertIn("Rating-state sensitivity: 2.6%–16.3%", captions)
        self.assertEqual(captions.count("Prior Senior/Open WC+: 0 competitions"), 2)
        self.assertIn("Prior Senior/Open WC+: 9 competitions", captions)
        self.assertTrue(
            any(
                caption.startswith(
                    "Graph connectivity: Indirect-to-WC · provisional graph"
                )
                for caption in captions
            )
        )
        self.assertTrue(
            any(
                caption.startswith(
                    "Graph connectivity: Indirect-to-WC · established graph"
                )
                for caption in captions
            )
        )
        self.assertTrue(
            any(
                "not head-to-head win probabilities or a firm ordering"
                in caption
                for caption in captions
            )
        )

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

    def test_olympics_all_athlete_modes_and_default_tree_parity(self) -> None:
        expected_captions = {
            "Compare 3": [
                "Current World Ranking: 27 · starts/365d: 4",
                "Current World Ranking: 72 · starts/365d: 6",
                "Current World Ranking: 54 · starts/365d: 6",
            ],
            "EEQ": [
                "Current World Ranking: not recorded · starts/365d: not recorded",
            ] * 3,
            "YNT Tier 1": [
                "Current World Ranking: not recorded · starts/365d: not recorded",
            ] * 3,
            "Canadian National Team proxy": [
                "Current World Ranking: 27 · starts/365d: 4",
                "Current World Ranking: not recorded · starts/365d: not recorded",
                "Current World Ranking: not recorded · starts/365d: not recorded",
            ],
        }
        for mode, expected in expected_captions.items():
            with self.subTest(mode=mode):
                app = AppTest.from_file(str(ROOT / "streamlit_app.py"))
                app.run(timeout=120)
                if mode == "Compare 3":
                    self.assertEqual(
                        _element_tree_sha256(app),
                        "76a54551901e12a39c900708741236236a900923f0152add22bbf555e2f6acff",
                    )
                else:
                    next(
                        item
                        for item in app.segmented_control
                        if item.label == "Athlete set"
                    ).set_value(mode).run(timeout=120)
                next(
                    item
                    for item in app.segmented_control
                    if item.label == "Overview section"
                ).set_value("Towards Olympics").run(timeout=120)
                self.assertFalse(app.exception, mode)
                self.assertFalse(app.error, mode)
                captions = [
                    str(item.value)
                    for item in app.caption
                    if "Current World Ranking:" in str(item.value)
                ]
                self.assertEqual(captions, expected)

    def test_olympics_malformed_or_missing_integer_observations(self) -> None:
        athletes = pd.DataFrame(
            {
                "pool": ["Boulder_Men"] * 3,
                "global_id": ["TEST:1", "TEST:2", "TEST:3"],
                "athlete_name": ["Missing", "Malformed", "Valid Zero"],
                "Global-ELO": [1500.0, 1501.0, 1502.0],
                "IFSC-ELO": [1400.0, 1401.0, 1402.0],
                "WC+-ELO": [1300.0, 1301.0, 1302.0],
                "world_event_rank": [None, "inf", "bad"],
                "starts_365": [None, "bad", "0"],
                "momentum": [0.0, 0.0, 0.0],
            }
        )
        selected = [
            "Boulder_Men::TEST:1",
            "Boulder_Men::TEST:2",
            "Boulder_Men::TEST:3",
        ]
        app = AppTest.from_function(
            _olympics_fixture_app,
            default_timeout=30,
            args=(athletes, selected),
        ).run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        captions = [
            str(item.value)
            for item in app.caption
            if "Current World Ranking:" in str(item.value)
        ]
        self.assertEqual(
            captions,
            [
                "Current World Ranking: not recorded · starts/365d: not recorded",
                "Current World Ranking: not recorded · starts/365d: not recorded",
                "Current World Ranking: not recorded · starts/365d: 0",
            ],
        )

    def test_integer_observation_rejects_non_counts(self) -> None:
        cases = [
            (None, "not recorded"),
            (float("nan"), "not recorded"),
            (float("inf"), "not recorded"),
            ("bad", "not recorded"),
            (-1, "not recorded"),
            (1.5, "not recorded"),
            (True, "not recorded"),
            (0, "0"),
            ("4.0", "4"),
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(integer_observation(value), expected)
        self.assertEqual(integer_observation(0, minimum=1), "not recorded")
        self.assertEqual(integer_observation(1, minimum=1), "1")

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
