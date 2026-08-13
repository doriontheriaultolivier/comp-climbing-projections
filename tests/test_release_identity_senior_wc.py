from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import pandas as pd

from release_identity_senior_wc import (
    identity_rebuild_status,
    load_reviewed_identity_overrides,
    senior_wc_direct_evidence_mask,
    suppress_reviewed_alias_profile,
)
from scripts.audit_release_identity_senior_wc_v1 import verify
from scripts.audit_release_identity_senior_wc_v1 import canonical_text_sha256


ROOT = Path(__file__).resolve().parents[1]


class ReleaseIdentitySeniorWcTests(unittest.TestCase):
    def test_reviewed_alias_is_hidden_without_merging_or_dropping_history(self) -> None:
        athletes = pd.DataFrame(
            {
                "global_id": ["IFSC:14843", "IFSC:18545", "IFSC:123"],
                "rating": [2009.0, 1744.0, 1900.0],
            }
        )
        overrides = load_reviewed_identity_overrides(
            ROOT / "data" / "reviewed_identity_overrides.csv"
        )
        safe, audit = suppress_reviewed_alias_profile(athletes, overrides)
        self.assertEqual(safe["global_id"].tolist(), ["IFSC:14843", "IFSC:123"])
        self.assertEqual(float(safe.iloc[0]["rating"]), 2009.0)
        self.assertEqual(audit["suppressed_profile_count"], 1)
        self.assertEqual(audit["history_rows_changed"], 0)
        self.assertFalse(audit["ratings_merged"])
        self.assertTrue(audit["requires_producer_rebuild"])

    def test_override_hash_is_portable_across_line_endings(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary)
            lf = root / "lf.csv"
            crlf = root / "crlf.csv"
            lf.write_bytes(b"a,b\n1,2\n")
            crlf.write_bytes(b"a,b\r\n1,2\r\n")
            self.assertEqual(canonical_text_sha256(lf), canonical_text_sha256(crlf))

    def test_reviewed_ledger_is_exactly_one_known_mapping(self) -> None:
        ledger = load_reviewed_identity_overrides(ROOT / "data" / "reviewed_identity_overrides.csv")
        self.assertEqual(ledger[["source_scope", "athlete_source_id", "ifsc_athlete_id"]].values.tolist(), [["IFSC", "18545", "14843"]])

    def test_direct_senior_wc_excludes_youth_and_pan_america(self) -> None:
        history = pd.DataFrame({
            "source_scope": ["IFSC", "IFSC", "IFSC", "CEC"],
            "event_name": [
                "IFSC World Cup Bern 2026",
                "IFSC Youth World Championships Helsinki 2025",
                "World Climbing Pan America Series Salt Lake City 2026",
                "IFSC World Cup Bern 2026",
            ],
        })
        self.assertEqual(senior_wc_direct_evidence_mask(history).tolist(), [True, False, False, False])

    def test_split_identity_is_rebuild_required_not_runtime_merged(self) -> None:
        athletes = pd.DataFrame({"global_id": ["IFSC:14843", "IFSC:18545"]})
        history = pd.DataFrame({"global_id": ["IFSC:14843", "IFSC:18545", "IFSC:18545"]})
        status = identity_rebuild_status(athletes, history)
        self.assertTrue(status["requires_producer_rebuild"])
        self.assertFalse(status["runtime_merge_performed"])
        self.assertEqual(status["alias_history_rows"], 2)

    def test_bound_release_audit_preserves_80_column_contract(self) -> None:
        result = verify(ROOT)
        self.assertEqual(result["status"], "PASS_REBUILD_REQUIRED_NOT_DEPLOYABLE")
        self.assertEqual(result["identity"]["canonical_profile_count"], 1)
        self.assertEqual(result["identity"]["alias_profile_count"], 1)


if __name__ == "__main__":
    unittest.main()
