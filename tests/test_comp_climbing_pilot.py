from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from comp_climbing_app import (
    CURVE_ORDER_UNAVAILABLE,
    SPARSE_WIN_UNAVAILABLE,
    _projection_triplet_is_nested,
    conditional_outcome_probability,
    conditional_outcome_projection,
    format_probability_sensitivity,
    projection_benchmark_labels,
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


if __name__ == "__main__":
    unittest.main()
