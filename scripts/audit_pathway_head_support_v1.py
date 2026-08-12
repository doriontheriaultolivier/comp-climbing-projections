"""Measure continuous evidence and athlete bridges for pathway context heads."""

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

from scripts.pathway_dynamic_rating_candidate_v1 import attach_pathway_model_domains


DATA = ROOT / "data"
SCHEMA = "pathway-head-support-audit-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_support(rows: pd.DataFrame, *, recent_start: str = "2021-01-01") -> pd.DataFrame:
    required = {
        "global_id", "pool", "event_date", "event_name", "source_scope",
        "event_tier", "rating_context", "round_group",
    }
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"rows missing support columns: {sorted(missing)}")
    labelled = attach_pathway_model_domains(rows)
    labelled = labelled.loc[
        labelled["pathway_input_eligible"]
        & labelled["pool"].astype(str).isin({"Boulder_Men", "Boulder_Women"})
    ].copy()
    labelled["event_date"] = pd.to_datetime(labelled["event_date"], errors="raise")
    labelled["event_key"] = (
        labelled["event_date"].dt.strftime("%Y-%m-%d") + "|"
        + labelled["source_scope"].astype(str) + "|"
        + labelled["event_name"].astype(str) + "|"
        + labelled["rating_context"].astype(str) + "|"
        + labelled["event_tier"].astype(str) + "|"
        + labelled["pool"].astype(str)
    )
    athlete_domains = labelled[["global_id", "pathway_target_domain"]].drop_duplicates()
    wc_athletes = set(
        athlete_domains.loc[athlete_domains["pathway_target_domain"].eq("wc"), "global_id"]
    )
    output: list[dict[str, object]] = []
    for domain, group in labelled.groupby("pathway_target_domain", sort=True):
        direct_athletes = set(group["global_id"].astype(str))
        event_count = int(group["event_key"].nunique())
        recent = group.loc[group["event_date"].ge(pd.Timestamp(recent_start))]
        bridge_count = int(len(direct_athletes & wc_athletes))
        output.append({
            "pathway_target_domain": domain,
            "direct_event_contexts": event_count,
            "recent_event_contexts": int(recent["event_key"].nunique()),
            "direct_athletes": int(len(direct_athletes)),
            "athletes_also_observed_in_wc": bridge_count,
            "wc_bridge_fraction": (
                bridge_count / len(direct_athletes) if direct_athletes else 0.0
            ),
            "pools": int(group["pool"].nunique()),
            "first_event_date": group["event_date"].min().date().isoformat(),
            "last_event_date": group["event_date"].max().date().isoformat(),
            "support_interpretation": (
                "continuous evidence measure; controls shrinkage/uncertainty and never "
                "constitutes a binary model-promotion gate"
            ),
        })
    return pd.DataFrame(output).sort_values(
        ["direct_event_contexts", "pathway_target_domain"],
        ascending=[False, True],
    ).reset_index(drop=True)


def run(input_path: Path, output_csv: Path, receipt_path: Path) -> dict:
    rows = pd.read_csv(
        input_path,
        compression="infer",
        low_memory=False,
        usecols=[
            "global_id", "pool", "event_date", "event_name", "source_scope",
            "event_tier", "rating_context", "round_group",
        ],
    )
    support = build_support(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    support.to_csv(output_csv, index=False, lineterminator="\n")
    receipt = {
        "schema": SCHEMA,
        "status": "RESEARCH_SUPPORT_AUDIT_ONLY",
        "input_sha256": sha256_file(input_path),
        "input_rows": int(len(rows)),
        "heads": int(len(support)),
        "total_direct_event_contexts": int(support["direct_event_contexts"].sum()),
        "policy": {
            "binary_support_threshold_used": False,
            "unsupported_head_promotion_allowed": False,
            "evidence_controls_shrinkage_and_uncertainty": True,
        },
        "authority": {"model_fit": False, "promotion": False, "deployment": False},
    }
    with receipt_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATA / "athlete_age_progression.csv.gz")
    parser.add_argument("--output", type=Path, default=DATA / "pathway_head_support_v1.csv")
    parser.add_argument("--receipt", type=Path, default=DATA / "pathway_head_support_audit_v1.json")
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output, args.receipt), sort_keys=True))


if __name__ == "__main__":
    main()
