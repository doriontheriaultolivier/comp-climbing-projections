from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from comp_climbing_app import (
    CURVE_ORDER_UNAVAILABLE,
    SPARSE_WIN_UNAVAILABLE,
    _projection_triplet_is_nested,
    athlete_selector_frame,
    conditional_outcome_probability,
    conditional_outcome_projection,
    format_probability_sensitivity,
    projection_benchmark_labels,
    quarantine_obvious_fixture_exposure,
    selected_rows,
    wc_semifinal_rating_evidence,
)


class CanadianPilotProjectionTests(unittest.TestCase):
    @staticmethod
    def _calibration() -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for pool in ("Boulder_Men", "Boulder_Women"):
            row: dict[str, object] = {
                "pool": pool,
                "calibration_season": 2025,
                "calibration_competition": "IFSC Open World Cups",
            }
            for index, outcome in enumerate(("semifinal", "final", "podium", "win")):
                row[f"display_elo_at_50pct_{outcome}"] = 2000.0 + 100.0 * index
                row[f"{outcome}_logistic_slope_per_100_native_elo"] = 1.0
                row[f"{outcome}_achievers"] = 100 - 20 * index
            rows.append(row)
        return pd.DataFrame(rows)

    def test_exact_gender_pool_midpoint_is_one_half(self) -> None:
        calibration = self._calibration()
        self.assertAlmostEqual(
            conditional_outcome_probability(
                2000.0, calibration, "Boulder_Men", "semifinal"
            ),
            0.5,
        )
        self.assertIsNone(
            conditional_outcome_probability(
                2000.0, calibration, "Boulder_All", "semifinal"
            )
        )

    def test_rating_state_sensitivity_is_ordered_and_bounded(self) -> None:
        projection = conditional_outcome_projection(
            2050.0, 75.0, self._calibration(), "Boulder_Women", "semifinal"
        )
        self.assertIsNotNone(projection)
        assert projection is not None
        central, lower, upper = projection
        self.assertLessEqual(lower, central)
        self.assertLessEqual(central, upper)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in projection))
        self.assertEqual(
            format_probability_sensitivity((0.003, 0.001, 0.008)),
            "0.3% (0.1%–0.8%)",
        )

    def test_crossing_sparse_win_curve_is_suppressed(self) -> None:
        nested = {
            "semifinal": (0.80, 0.70, 0.90),
            "final": (0.60, 0.50, 0.70),
            "podium": (0.30, 0.20, 0.40),
            "win": (0.35, 0.25, 0.45),
        }
        self.assertTrue(
            _projection_triplet_is_nested(
                nested, ("semifinal", "final", "podium")
            )
        )
        self.assertFalse(
            _projection_triplet_is_nested(nested, ("podium", "win"))
        )
        labels = projection_benchmark_labels(nested)
        self.assertNotEqual(labels["semifinal"], CURVE_ORDER_UNAVAILABLE)
        self.assertEqual(labels["win"], SPARSE_WIN_UNAVAILABLE)

    def test_every_deployed_rating_is_ordered_or_suppressed(self) -> None:
        athletes = pd.read_parquet("data/boulder_overview_athletes.parquet")
        calibration = pd.read_csv("data/boulder_elo_calibration.csv")
        eligible = athletes.dropna(
            subset=["Global-ELO", "Global-ELO uncertainty", "pool"]
        )
        self.assertGreater(len(eligible), 0)
        checked = 0
        outcomes = ("semifinal", "final", "podium", "win")
        for pool in ("Boulder_Men", "Boulder_Women"):
            subset = eligible.loc[eligible["pool"].eq(pool)]
            rating = subset["Global-ELO"].to_numpy(dtype=float)
            rating_sd = subset["Global-ELO uncertainty"].to_numpy(dtype=float)
            curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
            calibration_row = calibration.loc[calibration["pool"].eq(pool)].iloc[0]
            for outcome in outcomes:
                threshold = float(
                    calibration_row[f"display_elo_at_50pct_{outcome}"]
                )
                slope = float(
                    calibration_row[
                        f"{outcome}_logistic_slope_per_100_native_elo"
                    ]
                )

                def logistic(values: np.ndarray) -> np.ndarray:
                    exponent = np.clip(slope * (values - threshold) / 100.0, -700, 700)
                    return 1.0 / (1.0 + np.exp(-exponent))

                curves[outcome] = (
                    logistic(rating),
                    logistic(rating - rating_sd),
                    logistic(rating + rating_sd),
                )
            for index in range(len(subset)):
                projections = {
                    outcome: tuple(float(values[index]) for values in curves[outcome])
                    for outcome in outcomes
                }
                labels = projection_benchmark_labels(projections)
                first_three_nested = _projection_triplet_is_nested(
                    projections, ("semifinal", "final", "podium")
                )
                win_nested = first_three_nested and _projection_triplet_is_nested(
                    projections, ("podium", "win")
                )
                for outcome in ("semifinal", "final", "podium"):
                    self.assertEqual(
                        labels[outcome] == CURVE_ORDER_UNAVAILABLE,
                        not first_three_nested,
                    )
                self.assertEqual(
                    labels["win"] == SPARSE_WIN_UNAVAILABLE,
                    not win_nested,
                )
                checked += 1
        self.assertEqual(checked, len(eligible))

    def test_duplicate_names_select_only_the_stable_requested_record(self) -> None:
        frame = pd.DataFrame(
            {
                "pool": ["Boulder_Men", "Boulder_Men"],
                "global_id": ["CEC:595", "VL-GLOBAL:18197"],
                "athlete_name": ["ALLORA KLINKER", "Allora Klinker"],
                "name_key": ["alloraklinker", "alloraklinker"],
                "country": ["CAN", ""],
                "nationality": ["CAN", ""],
                "Global-ELO": [1864.93, 1666.17],
                "Global-ELO uncertainty": [72.76, 260.14],
                "Global-ELO evidence": [18.0, 2.0],
                "Global-ELO status": ["Established", "Provisional"],
            }
        )
        selectors = athlete_selector_frame(frame)
        self.assertEqual(len(selectors), 2)
        self.assertTrue(selectors["_selection_label"].str.contains("CEC:595", regex=False).any())
        chosen = selected_rows(frame, ["Boulder_Men::CEC:595"])
        self.assertEqual(chosen["global_id"].tolist(), ["CEC:595"])
        self.assertEqual(selected_rows(frame, ["CEC:595"])["global_id"].tolist(), ["CEC:595"])

    def test_fixture_exposure_is_quarantined_per_identity_not_whole_field(self) -> None:
        athletes = pd.DataFrame(
            {
                "global_id": ["CEC:1", "CEC:2"],
                "cnr_rank": [1.0, 2.0],
            }
        )
        history = pd.DataFrame(
            {
                "global_id": ["CEC:1", "CEC:1", "CEC:2"],
                "event_name": ["Bouldering Test", "Real Nationals", "Real Nationals"],
                "source_scope": ["CEC", "CEC", "CEC"],
                "source_event_id": ["fixture", "real", "real"],
            }
        )
        safe_athletes, safe_history, audit = quarantine_obvious_fixture_exposure(
            athletes, history
        )
        self.assertEqual(safe_athletes["global_id"].tolist(), ["CEC:2"])
        self.assertEqual(safe_history["global_id"].unique().tolist(), ["CEC:2"])
        self.assertEqual(int(audit.iloc[0]["withheld_athlete_ids"]), 1)
        self.assertEqual(int(audit.iloc[0]["withheld_canadian_rows"]), 1)

    def test_deployed_fixture_guard_has_exact_closed_effect(self) -> None:
        athletes = pd.read_parquet("data/boulder_overview_athletes.parquet")
        history = pd.read_parquet("data/boulder_overview_history.parquet")
        safe_athletes, safe_history, audit = quarantine_obvious_fixture_exposure(
            athletes, history
        )
        row = audit.iloc[0]
        self.assertEqual(int(row["fixture_event_rows"]), 2041)
        self.assertEqual(int(row["fixture_source_events"]), 94)
        self.assertEqual(int(row["withheld_athlete_ids"]), 731)
        self.assertEqual(int(row["withheld_canadian_rows"]), 4)
        self.assertFalse(
            safe_history["event_name"].astype(str).str.contains(
                r"(?i)\b(?:test|mock|demo|dummy|sandbox|hidden)\b", regex=True
            ).any()
        )
        self.assertEqual(len(athletes) - len(safe_athletes), 731)

    def test_deployed_colliding_names_are_exact_stable_choices(self) -> None:
        athletes = pd.read_parquet("data/boulder_overview_athletes.parquet")
        for selection_id, expected_global_id in (
            ("Boulder_Women::CEC:595", "CEC:595"),
            ("Boulder_Men::CEC:20", "CEC:20"),
            ("Boulder_Men::USAC:610", "USAC:610"),
        ):
            chosen = selected_rows(athletes, [selection_id])
            self.assertEqual(chosen["global_id"].tolist(), [expected_global_id])

    def test_named_wc_regression_uses_target_qualification_not_all_source(self) -> None:
        athletes = pd.read_parquet("data/boulder_overview_athletes.parquet")
        calibration = pd.read_csv("data/boulder_elo_calibration.csv")
        by_id = athletes.set_index("global_id")
        oscar = by_id.loc["IFSC:11847"]
        matthew = by_id.loc["IFSC:14842"]
        oscar_rating, oscar_family, oscar_evidence, _ = wc_semifinal_rating_evidence(oscar)
        matthew_rating, matthew_family, matthew_evidence, _ = wc_semifinal_rating_evidence(matthew)
        self.assertEqual(oscar_family, "WC+-ELO-Open")
        self.assertEqual(matthew_family, "WC+-ELO-Open")
        self.assertGreaterEqual(oscar_evidence, 8)
        self.assertGreaterEqual(matthew_evidence, 8)
        self.assertGreater(oscar_rating - matthew_rating, 75.0)
        self.assertLess(float(oscar["Global-ELO"]), float(matthew["Global-ELO"]))
        oscar_projection = conditional_outcome_projection(
            oscar_rating,
            float(oscar["Global-ELO uncertainty"]),
            calibration,
            str(oscar["pool"]),
            "semifinal",
        )
        matthew_projection = conditional_outcome_projection(
            matthew_rating,
            float(matthew["Global-ELO uncertainty"]),
            calibration,
            str(matthew["pool"]),
            "semifinal",
        )
        assert oscar_projection is not None and matthew_projection is not None
        self.assertGreater(oscar_projection[0], 2.0 * matthew_projection[0])


if __name__ == "__main__":
    unittest.main()
