from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.materialize_physical_athlete_event_bundle_v1 import (
    GROUP_KEYS,
    build_bundles,
    write_deterministic_gzip_csv,
)


ROOT = Path(__file__).resolve().parents[1]
TIMELINE = ROOT / ".artifacts/physical-competition-timeline-v1/timeline.csv.gz"


class PhysicalAthleteEventBundleTest(unittest.TestCase):
    def test_real_bundle_preserves_round_vector_without_scalar_outcome(self) -> None:
        timeline = pd.read_csv(TIMELINE, low_memory=False)
        bundles, receipt = build_bundles(timeline)
        self.assertFalse(bundles.duplicated(GROUP_KEYS).any())
        self.assertEqual(int(bundles["round_count"].sum()), len(timeline))
        self.assertGreater(receipt["coverage"]["multi_round_bundles"], 0)
        self.assertEqual(set(bundles["discipline"]), {"Boulder", "Lead", "Speed"})
        self.assertFalse(bundles["model_input_authorized"].any())
        self.assertNotIn("event_score", bundles.columns)
        sample = json.loads(bundles.loc[bundles["round_count"].gt(1), "round_vector_json"].iloc[0])
        self.assertGreater(len(sample), 1)
        self.assertTrue(all("round_stage" in row for row in sample))
        self.assertFalse(receipt["claims"]["synthetic_event_score_created"])

    def test_deterministic_gzip_bytes(self) -> None:
        frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            first = Path(temp) / "one.csv.gz"
            second = Path(temp) / "two.csv.gz"
            write_deterministic_gzip_csv(frame, first)
            write_deterministic_gzip_csv(frame, second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(gzip.open(first, "rt").read(), "a,b\n1,x\n2,y\n")


if __name__ == "__main__":
    unittest.main()
