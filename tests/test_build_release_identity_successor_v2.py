import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "build_release_identity_successor_v2.py"
SPEC = importlib.util.spec_from_file_location("release_identity_v2", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


class ReleaseIdentitySuccessorV2Tests(unittest.TestCase):
    def test_joint_performance_respects_rank_order(self):
        means, sds = module.joint_performance(
            np.array([1700.0, 1800.0, 1900.0]),
            np.array([1.0, 2.0, 3.0]),
            np.array([1700.0, 1800.0, 1900.0]),
            np.square(np.array([100.0, 100.0, 100.0])),
            1.0,
            grid_step=20.0,
        )
        self.assertGreater(means[0], means[1])
        self.assertGreater(means[1], means[2])
        self.assertTrue(np.isfinite(sds).all())

    def test_athlete_adapter_withholds_wc_flash_and_raw_birth_dates(self):
        candidate = pd.DataFrame(
            {
                "global_id": ["IFSC:14843"],
                "age": [16.2],
                "birthday": ["2009-07-09"],
                "birth_date_analysis_value": ["2009-07-09"],
                "birth_date_uncertainty_days": [10.0],
                "Canada projection — all evidence": [1888.0],
            }
        )
        baseline = [
            "global_id",
            "age",
            "birth_date_uncertainty_days",
            "canada_projection_all_evidence",
            "WC+-ELO-Qualies-Flash",
            "WC+-ELO-Qualies-Flash evidence",
            "age_lower_years",
            "age_upper_years",
            "age_precision_status",
        ]
        result = module.build_athletes(candidate, baseline)
        self.assertEqual(list(result.columns), baseline)
        self.assertTrue(pd.isna(result.loc[0, "WC+-ELO-Qualies-Flash"]))
        self.assertNotIn("birthday", result)
        self.assertEqual(result.loc[0, "canada_projection_all_evidence"], 1888.0)


if __name__ == "__main__":
    unittest.main()
