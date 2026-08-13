from __future__ import annotations

import unittest
from pathlib import Path

from scripts.materialize_physical_competition_timeline_v1 import build_timeline


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT.parent / "ifsc-performance-projections"
SNAPSHOT = (
    CORE / ".artifacts/restricted/physical-transfer-sheet-snapshot-v1/"
    "5011bd3dd7bdd767027fbdecfe65a46f6ca7b6f9fb31169e314810998b3c647d"
)


class PhysicalCompetitionTimelineTest(unittest.TestCase):
    def test_real_timeline_is_chronological_and_descriptive(self) -> None:
        timeline, receipt = build_timeline(
            SNAPSHOT / "physical_observations.csv",
            SNAPSHOT / "board_observations.csv",
            ROOT / ".artifacts/physical-result-identity-bridge-v1/governed_links.csv",
            ROOT / "data/source_results.csv.gz",
            ROOT / "data/pathway_context_event_taxonomy_v1.csv",
            horizon_days=365,
        )
        self.assertGreater(len(timeline), 0)
        self.assertTrue(timeline["days_to_competition"].gt(0).all())
        self.assertFalse(timeline["model_input_authorized"].any())
        self.assertFalse(receipt["claims"]["linear_ceiling_model_fit"])
        self.assertFalse(receipt["claims"]["current_rating_used_as_historical_target"])
        self.assertGreater(receipt["coverage"]["fixture_result_rows_quarantined"], 0)
        self.assertNotIn("TEST (Quota management)", set(timeline["event_name"]))
        self.assertNotIn("Test speed", set(timeline["event_name"]))
        self.assertIn("discipline", timeline.columns)


if __name__ == "__main__":
    unittest.main()
