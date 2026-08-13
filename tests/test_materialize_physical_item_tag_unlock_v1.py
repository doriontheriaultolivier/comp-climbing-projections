from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from scripts.materialize_physical_item_tag_unlock_v1 import build_unlock


ROOT = Path(__file__).resolve().parents[1]
BUNDLES = ROOT / ".artifacts/physical-athlete-event-round-vector-v1/athlete_event_round_vectors.csv.gz"
CANDIDATES = ROOT / "data/physical_item_tagging_priority_v1_1.csv"


class PhysicalItemTagUnlockTest(unittest.TestCase):
    def test_real_unlock_is_pending_and_keeps_gaps(self) -> None:
        bundles = pd.read_csv(BUNDLES, low_memory=False)
        candidates = pd.read_csv(CANDIDATES, low_memory=False)
        unlock, gaps, receipt = build_unlock(bundles, candidates)
        self.assertGreater(len(unlock), 0)
        self.assertGreater(len(gaps), 0)
        self.assertTrue(unlock["problem_id"].is_unique)
        self.assertEqual(set(unlock["tag_status"]), {"human_demand_tags_needed"})
        self.assertEqual(receipt["claims"]["completed_human_demand_tags"], 0)
        self.assertTrue(receipt["claims"]["candidate_status_is_not_a_tag"])
        self.assertFalse(receipt["claims"]["style_imputed_for_uncovered_events"])
        self.assertFalse(receipt["claims"]["model_fit"])
        self.assertGreater(receipt["coverage"]["candidate_items_linked_to_coaching_timeline"], 0)


if __name__ == "__main__":
    unittest.main()
