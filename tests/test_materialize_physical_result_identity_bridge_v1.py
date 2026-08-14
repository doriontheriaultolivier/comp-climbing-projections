from __future__ import annotations

import unittest
from pathlib import Path

from scripts.materialize_physical_result_identity_bridge_v1 import build_bridge


ROOT = Path(__file__).resolve().parents[1]
REAL_INPUTS = (
    ROOT / "data/physical_test_profiles.csv",
    ROOT / "data/identity_link_audit.csv.gz",
    ROOT / "data/source_results.csv.gz",
)


class PhysicalResultIdentityBridgeTest(unittest.TestCase):
    @unittest.skipUnless(
        all(path.exists() for path in REAL_INPUTS),
        "restricted real-data inputs are not present in this clean worktree",
    )
    def test_real_bridge_is_fail_closed_and_preserves_sparse_sources(self) -> None:
        governed, review, receipt = build_bridge(
            ROOT / "data/physical_test_profiles.csv",
            ROOT / "data/identity_link_audit.csv.gz",
            ROOT / "data/source_results.csv.gz",
            ROOT / "data/physical_testing_identity_overrides.csv",
            ROOT / "data/reviewed_identity_overrides.csv",
        )
        self.assertEqual(receipt["coverage"]["physical_profiles"], 22)
        self.assertEqual(receipt["coverage"]["manual_review_candidates"], 7)
        self.assertEqual(receipt["coverage"]["digitalrock_governed_links"], 0)
        self.assertEqual(receipt["coverage"]["profiles_without_governed_non_ifsc_link"], 2)
        self.assertFalse(governed["model_input_authorized"].any())
        self.assertFalse(review["model_input_authorized"].any())
        self.assertTrue(review["decision"].eq("DEFER").all())
        self.assertTrue(review["competition_evidence_rows"].gt(0).all())
        self.assertFalse(receipt["review_policy"]["auto_merge_by_name"])


if __name__ == "__main__":
    unittest.main()
