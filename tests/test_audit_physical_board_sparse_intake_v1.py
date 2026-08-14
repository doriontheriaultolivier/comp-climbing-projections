from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

import pandas as pd

from scripts.audit_physical_board_sparse_intake_v1 import build_report


class SparsePhysicalBoardIntakeTest(unittest.TestCase):
    def test_sparse_metrics_remain_sparse_and_are_not_imputed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            physical = pd.DataFrame(
                [
                    {"observation_id": "p1", "athlete_id": "a", "observed_on": "2026-01-01", "canonical_metric_id": "pull", "capacity_dimension": "strength", "protocol_id": "pull-v1", "value": 10, "unit": "kg", "valid_result": True, "invalid_reason": None, "source_revision": "x"},
                    {"observation_id": "p2", "athlete_id": "a", "observed_on": "2026-02-01", "canonical_metric_id": "pull", "capacity_dimension": "strength", "protocol_id": "pull-v1", "value": 11, "unit": "kg", "valid_result": True, "invalid_reason": None, "source_revision": "x"},
                    {"observation_id": "p3", "athlete_id": "b", "observed_on": "2026-01-01", "canonical_metric_id": "jump", "capacity_dimension": "power", "protocol_id": "jump-v1", "value": 20, "unit": "cm", "valid_result": True, "invalid_reason": None, "source_revision": "x"},
                ]
            )
            board = pd.DataFrame(
                [{"observation_id": "b1", "athlete_id": "a", "observed_on": "2026-01-01", "board_metric": "flash", "value": 8, "grade_scale": "V", "reporting_window_days": None, "source_revision": "x", "access_class": "research"}]
            )
            physical_path, board_path = root / "physical.csv", root / "board.csv"
            physical.to_csv(physical_path, index=False)
            board.to_csv(board_path, index=False)
            report = build_report(physical_path, board_path)
            self.assertEqual(report["coverage"]["physical_athletes"], 2)
            self.assertEqual(report["coverage"]["athletes_with_physical_and_board"], 1)
            self.assertFalse(report["model_contract"]["missing_test_imputed_as_low"])
            strength = next(row for row in report["capacity_dimensions"] if row["capacity_dimension"] == "strength")
            self.assertEqual(strength["athletes_with_repeats"], 1)

    def test_invalid_boolean_text_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            physical = pd.DataFrame(
                [{"observation_id": "p1", "athlete_id": "a", "observed_on": "2026-01-01", "canonical_metric_id": "pull", "capacity_dimension": "strength", "protocol_id": "pull-v1", "value": 10, "unit": "kg", "valid_result": "maybe", "source_revision": "x"}]
            )
            board = pd.DataFrame(
                [{"observation_id": "b1", "athlete_id": "a", "observed_on": "2026-01-01", "board_metric": "flash", "value": 8, "grade_scale": "V", "reporting_window_days": None, "source_revision": "x"}]
            )
            physical_path, board_path = root / "physical.csv", root / "board.csv"
            physical.to_csv(physical_path, index=False)
            board.to_csv(board_path, index=False)
            with self.assertRaisesRegex(ValueError, "valid_result"):
                build_report(physical_path, board_path)


if __name__ == "__main__":
    unittest.main()
