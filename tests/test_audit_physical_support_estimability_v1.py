from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from scripts.audit_physical_support_estimability_v1 import build_estimability


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT.parent / "ifsc-performance-projections"
SNAPSHOT = (
    CORE / ".artifacts/restricted/physical-transfer-sheet-snapshot-v1/"
    "5011bd3dd7bdd767027fbdecfe65a46f6ca7b6f9fb31169e314810998b3c647d"
)


class PhysicalSupportEstimabilityTest(unittest.TestCase):
    def test_real_profiles_are_continuous_and_non_authorizing(self) -> None:
        profiles, receipt = build_estimability(
            pd.read_csv(SNAPSHOT / "physical_observations.csv"),
            pd.read_csv(SNAPSHOT / "board_observations.csv"),
            pd.read_csv(
                ROOT / ".artifacts/physical-athlete-event-round-vector-v1/"
                "athlete_event_round_vectors.csv.gz",
                low_memory=False,
            ),
            pd.read_csv(ROOT / "data/physical_item_tagging_priority_v1_1.csv"),
        )
        self.assertEqual(receipt["coverage"]["measurement_profiles"], len(profiles))
        self.assertEqual(set(profiles["observation_family"]), {"physical", "board"})
        self.assertFalse(profiles["model_input_authorized"].any())
        self.assertTrue(profiles["event_effective_clusters"].gt(0).all())
        self.assertTrue(profiles["max_event_cluster_share"].between(0, 1).all())
        self.assertTrue(profiles["linked_observation_fraction"].between(0, 1).all())
        self.assertTrue(
            profiles.loc[profiles["boulder_round_rows"].gt(0),
                         "pending_item_candidate_coverage"].between(0, 1).all()
        )
        self.assertTrue(
            profiles["evidence_state"].eq("continuous_profile_no_eligibility_cliff").all()
        )
        self.assertFalse(receipt["semantics"]["hard_sample_cutoff_used"])
        self.assertFalse(receipt["semantics"]["pending_item_candidate_is_completed_tag"])


if __name__ == "__main__":
    unittest.main()
