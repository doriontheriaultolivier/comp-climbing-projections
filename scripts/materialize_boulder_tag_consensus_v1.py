"""Materialize continuous, uncertainty-preserving Boulder tag summaries.

This consumer accepts downloaded schema-v4 tag records.  It keeps only the
latest record per pseudonymous reviewer and Boulder, preserves both pre-zone
and post-zone segments, and summarizes each 0-3 demand field without declaring
an arbitrary minimum-review eligibility cliff.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


SCHEMA = "boulder-tag-consensus-v1"
TAG_RE = re.compile(r"^(pre_zone|post_zone)_([a-z0-9_]+_0_3)$")
REVIEWER_RE = re.compile(r"^[A-Za-z0-9_-]{3,24}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_records(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        payload = payload["records"]
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError("tag download must be a JSON list or {records: [...]} object")
    return payload


def materialize_consensus(
    records: list[dict[str, object]], priority: pd.DataFrame, inventory: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if "problem_id" not in priority or priority["problem_id"].duplicated().any():
        raise ValueError("priority inventory requires unique problem_id")
    required_inventory = {
        "boulder_uid", "source_scope", "source_event_id",
        "source_round_ids", "boulder_number",
    }
    if missing := required_inventory.difference(inventory.columns):
        raise ValueError(f"Boulder inventory missing {sorted(missing)}")
    expanded = inventory[list(required_inventory)].copy()
    expanded["source_round_id"] = expanded["source_round_ids"].astype(str).str.split("|")
    expanded = expanded.explode("source_round_id")
    keys = ["source_scope", "source_event_id", "source_round_id", "boulder_number"]
    queue = priority[[*keys, "problem_id"]].copy()
    for column in keys:
        expanded[column] = expanded[column].astype(str).str.strip()
        queue[column] = queue[column].astype(str).str.strip()
    identity = expanded.merge(queue, on=keys, how="inner")[["boulder_uid", "problem_id"]]
    identity = identity.drop_duplicates()
    if identity["problem_id"].duplicated().any():
        raise ValueError("one exact source problem maps to multiple governed Boulders")
    boulder_to_problems = {
        str(boulder_uid): tuple(sorted(rows["problem_id"].astype(str)))
        for boulder_uid, rows in identity.groupby("boulder_uid", sort=True)
    }
    normalized: list[dict[str, object]] = []
    out_of_priority_records = 0
    for record in records:
        if str(record.get("schema_version", "")) != "4.0":
            raise ValueError("only schema-v4 records are accepted")
        reviewer = str(record.get("contributor", "")).strip()
        if REVIEWER_RE.fullmatch(reviewer) is None:
            raise ValueError("invalid pseudonymous reviewer code")
        boulder_uid = str(record.get("boulder_uid", ""))
        if not boulder_uid:
            raise ValueError("boulder_uid is required")
        if boulder_uid not in boulder_to_problems:
            out_of_priority_records += 1
            continue
        submitted = pd.to_datetime(record.get("submitted_at_utc"), utc=True, errors="coerce")
        if pd.isna(submitted):
            raise ValueError("submitted_at_utc must be a valid timestamp")
        for key, value in record.items():
            match = TAG_RE.fullmatch(str(key))
            if match is None:
                continue
            numeric = pd.to_numeric(value, errors="coerce")
            if pd.isna(numeric) or float(numeric) not in {0.0, 1.0, 2.0, 3.0}:
                raise ValueError(f"invalid 0-3 tag value: {key}")
            for problem_id in boulder_to_problems[boulder_uid]:
                normalized.append(
                    {
                        "boulder_uid": boulder_uid,
                        "problem_id": problem_id,
                        "reviewer_code": reviewer,
                        "submitted_at_utc": submitted,
                        "segment": match.group(1),
                        "tag": match.group(2),
                        "value": float(numeric),
                        "confidence": str(record.get("confidence", "")),
                    }
                )
    long = pd.DataFrame(normalized)
    if long.empty:
        empty = pd.DataFrame(
            columns=["boulder_uid", "problem_id", "segment", "tag", "mean_0_3"]
        )
        report = {
            "schema": SCHEMA,
            "status": "READY_NO_REVIEWS",
            "coverage": {
                "submitted_records": len(records),
                "consensus_rows": 0,
                "priority_source_items": int(len(priority)),
                "priority_items_resolved_to_boulder_uid": int(len(identity)),
                "distinct_governed_boulder_tasks": int(identity["boulder_uid"].nunique()),
                "review_records_outside_physical_priority": int(out_of_priority_records),
            },
            "semantics": {
                "latest_record_per_reviewer_boulder": True,
                "pre_zone_and_post_zone_separate": True,
                "single_review_discarded": False,
                "minimum_review_cliff": None,
                "disagreement_preserved": True,
                "descriptive_weight_is_model_authority": False,
            },
        }
        return empty, long, report
    long = long.sort_values("submitted_at_utc", kind="stable").drop_duplicates(
        ["boulder_uid", "problem_id", "reviewer_code", "segment", "tag"], keep="last"
    )
    grouped = long.groupby(
        ["boulder_uid", "problem_id", "segment", "tag"], sort=True, as_index=False
    ).agg(
        independent_reviewers=("reviewer_code", "nunique"),
        mean_0_3=("value", "mean"),
        sd_0_3=("value", lambda values: float(np.std(values, ddof=1)) if len(values) > 1 else np.nan),
        minimum_0_3=("value", "min"),
        maximum_0_3=("value", "max"),
        latest_review_utc=("submitted_at_utc", "max"),
    )
    grouped["reviewer_range_0_3"] = grouped["maximum_0_3"] - grouped["minimum_0_3"]
    # Evidence weight is descriptive, continuous, and never used as a gate.
    grouped["review_count_weight"] = grouped["independent_reviewers"] / (
        grouped["independent_reviewers"] + 2.0
    )
    grouped["agreement_weight"] = np.where(
        grouped["independent_reviewers"].eq(1),
        0.5,
        1.0 / (1.0 + grouped["reviewer_range_0_3"]),
    )
    grouped["descriptive_evidence_weight"] = (
        grouped["review_count_weight"] * grouped["agreement_weight"]
    )
    grouped["model_input_authorized"] = False
    grouped["eligibility_threshold_applied"] = False
    report = {
        "schema": SCHEMA,
        "status": "RESEARCH_TAG_SUMMARY_READY_NO_MODEL_INPUT",
        "coverage": {
            "submitted_records": int(len(records)),
            "latest_reviewer_boulder_tag_rows": int(len(long)),
            "consensus_rows": int(len(grouped)),
            "boulders": int(grouped["boulder_uid"].nunique()),
            "reviewers": int(long["reviewer_code"].nunique()),
            "priority_items_resolved_to_boulder_uid": int(len(identity)),
            "distinct_governed_boulder_tasks": int(identity["boulder_uid"].nunique()),
            "review_records_outside_physical_priority": int(out_of_priority_records),
            "boulders_with_two_or_more_reviewers": int(
                grouped.loc[grouped["independent_reviewers"].ge(2), "boulder_uid"].nunique()
            ),
        },
        "semantics": {
            "latest_record_per_reviewer_boulder": True,
            "pre_zone_and_post_zone_separate": True,
            "single_review_discarded": False,
            "minimum_review_cliff": None,
            "disagreement_preserved": True,
            "descriptive_weight_is_model_authority": False,
        },
    }
    return grouped, long, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--priority", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    records = load_records(args.records)
    priority = pd.read_csv(args.priority, low_memory=False)
    inventory = pd.read_csv(args.inventory, low_memory=False)
    summary, latest, report = materialize_consensus(records, priority, inventory)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "tag_consensus.csv"
    latest_path = args.output_dir / "latest_reviewer_tags.csv"
    summary.to_csv(summary_path, index=False, lineterminator="\n")
    latest.to_csv(latest_path, index=False, lineterminator="\n")
    report["bindings"] = {
        "records_sha256": sha256(args.records),
        "priority_sha256": sha256(args.priority),
        "inventory_sha256": sha256(args.inventory),
    }
    report["outputs"] = {
        summary_path.name: sha256(summary_path),
        latest_path.name: sha256(latest_path),
    }
    (args.output_dir / "receipt.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
