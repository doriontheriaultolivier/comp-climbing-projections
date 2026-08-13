"""Verify the release inputs against the reviewed identity/WC correction gate.

The audit is intentionally non-mutating.  It binds the current compact bundle
to its bytes, verifies the one reviewed override, and fails closed if someone
tries to treat a split identity as a rebuilt rating.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from release_identity_senior_wc import (
    identity_rebuild_status,
    load_reviewed_identity_overrides,
    senior_wc_direct_evidence_mask,
)


BASELINE = ROOT / "data" / "release_identity_senior_wc_baseline_v1.json"
FILES = {
    "athletes": "boulder_overview_athletes.parquet",
    "history": "boulder_overview_history.parquet",
    "overrides": "reviewed_identity_overrides.csv",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root: Path = ROOT) -> dict[str, object]:
    baseline = json.loads((root / "data" / BASELINE.name).read_text(encoding="utf-8"))
    expected = baseline["files"]
    actual = {name: sha256(root / "data" / filename) for name, filename in FILES.items()}
    if actual != expected:
        raise ValueError("release identity baseline binding mismatch")
    overrides = load_reviewed_identity_overrides(root / "data" / FILES["overrides"])
    athletes = pd.read_parquet(root / "data" / FILES["athletes"])
    history = pd.read_parquet(root / "data" / FILES["history"])
    if len(athletes.columns) != 80:
        raise ValueError("release athlete schema is not the required 80-column contract")
    status = identity_rebuild_status(athletes, history)
    if status["canonical_profile_count"] != 1 or status["alias_profile_count"] != 1:
        raise ValueError("reviewed identity profiles no longer match the bound release state")
    alias = history.loc[history["global_id"].astype(str).eq("IFSC:18545")]
    if len(alias) != 2 or senior_wc_direct_evidence_mask(alias).any():
        raise ValueError("alias evidence changed or was incorrectly called direct senior WC")
    return {
        "schema": "release-identity-senior-wc-audit-v1",
        "status": "PASS_REBUILD_REQUIRED_NOT_DEPLOYABLE",
        "files": actual,
        "override_rows": len(overrides),
        "identity": status,
        "direct_senior_wc_alias_rows": 0,
        "authority": {
            "runtime_identity_merge": False,
            "production_rating_promotion": False,
            "deployment": False,
        },
        "next_gate": "rebuild the rating family from raw chronology with the reviewed override, then rerun locked adult/youth pair and placement calibration",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(verify(args.repo_root.resolve()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
