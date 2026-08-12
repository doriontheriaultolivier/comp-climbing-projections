"""Materialize the unfitted pathway hierarchy and illustrative shrinkage."""

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

from scripts.pathway_dynamic_rating_candidate_v1 import pathway_hierarchy_audit


DATA = ROOT / "data"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(support_path: Path, output_path: Path, receipt_path: Path) -> dict:
    support = pd.read_csv(support_path)
    hierarchy = pathway_hierarchy_audit(support)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hierarchy.to_csv(output_path, index=False, lineterminator="\n")
    receipt = {
        "schema": "pathway-hierarchy-audit-v1",
        "status": "UNFITTED_HIERARCHY_RESEARCH_ONLY",
        "support_sha256": sha256(support_path),
        "rows": int(len(hierarchy)),
        "hyperparameters_selected": False,
        "binary_validity_threshold": False,
        "interpretation": (
            "Illustrative continuous parent borrowing only; event and bridge "
            "half-saturation values require nested chronological selection."
        ),
        "authority": {"fit": False, "promotion": False, "deployment": False},
    }
    with receipt_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--support", type=Path, default=DATA / "pathway_head_support_v1.csv"
    )
    parser.add_argument(
        "--output", type=Path, default=DATA / "pathway_hierarchy_v1.csv"
    )
    parser.add_argument(
        "--receipt", type=Path, default=DATA / "pathway_hierarchy_audit_v1.json"
    )
    args = parser.parse_args()
    print(json.dumps(run(args.support, args.output, args.receipt), sort_keys=True))


if __name__ == "__main__":
    main()
