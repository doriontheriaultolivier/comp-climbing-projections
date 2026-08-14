"""Audit the governed 30-task human review and materialize its consensus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.materialize_boulder_tag_consensus_v1 import (
    latest_records_for_boulders,
    load_records,
    materialize_consensus,
    sha256,
)


SCHEMA = "physical-tag-human-review-consensus-v1"
WAVE_A = "A_same_tasks_independent_calibration"
WAVE_B = "B_high_unlock_extension"
CORE_FIELDS = tuple(
    f"{segment}_{demand}_0_3"
    for segment in ("pre_zone", "post_zone")
    for demand in ("physical", "technical", "coordination")
)
DIRECTION_FIELDS = ("pre_zone_direction", "post_zone_direction")
VALID_DIRECTIONS = {"Up", "Diagonal", "Sideways", "Mixed / unclear"}


def validate_session(session: pd.DataFrame) -> pd.DataFrame:
    required = {
        "session_task_order", "review_wave", "boulder_uid",
        "requested_independent_reviewers", "required_review_scope", "task_status",
    }
    if missing := required.difference(session.columns):
        raise ValueError(f"session is missing {sorted(missing)}")
    rows = session.copy()
    rows["session_task_order"] = pd.to_numeric(
        rows["session_task_order"], errors="raise"
    ).astype(int)
    if (
        len(rows) != 30
        or not rows["boulder_uid"].is_unique
        or rows["session_task_order"].tolist() != list(range(1, 31))
        or rows["review_wave"].value_counts().to_dict()
        != {WAVE_B: 20, WAVE_A: 10}
        or not rows["requested_independent_reviewers"].eq(2).all()
        or not rows["task_status"].eq("human_review_pending").all()
        or not rows["required_review_scope"].eq(
            "directions_and_three_core_demands; detailed_tags_optional"
        ).all()
    ):
        raise ValueError("the 30-task governed review session has drifted")
    return rows


def audit_session_records(
    records: list[dict[str, object]], session: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Retain exact disagreement and report operational completion separately."""

    session = validate_session(session)
    session_uids = set(session["boulder_uid"].astype(str))
    latest, outside_session = latest_records_for_boulders(records, session_uids)
    normalized: list[dict[str, object]] = []
    for record in latest:
        missing = [field for field in (*DIRECTION_FIELDS, *CORE_FIELDS) if field not in record]
        if missing:
            raise ValueError(
                f"latest schema-v4 session record is missing required fields: {missing}"
            )
        invalid_directions = [
            field for field in DIRECTION_FIELDS
            if str(record[field]) not in VALID_DIRECTIONS
        ]
        if invalid_directions:
            raise ValueError(f"invalid movement direction: {invalid_directions}")
        normalized.append({
            "boulder_uid": str(record["boulder_uid"]),
            "reviewer_code": str(record["contributor"]),
            "submitted_at_utc": record["_submitted_at_utc"],
            **{field: record[field] for field in (*DIRECTION_FIELDS, *CORE_FIELDS)},
        })
    latest_frame = pd.DataFrame(normalized)
    counts = (
        latest_frame.groupby("boulder_uid")["reviewer_code"].nunique().to_dict()
        if not latest_frame.empty else {}
    )
    progress = session[
        ["session_task_order", "review_wave", "boulder_uid", "event_date",
         "event_name", "round_group", "gender", "boulder_number"]
    ].copy()
    progress["independent_reviewers"] = (
        progress["boulder_uid"].map(counts).fillna(0).astype(int)
    )
    progress["requested_independent_reviewers"] = 2
    progress["review_status"] = progress["independent_reviewers"].map(
        lambda count: (
            "not_started" if count == 0
            else "single_review_complete" if count == 1
            else "independently_double_reviewed"
        )
    )

    direction_rows: list[dict[str, object]] = []
    if not latest_frame.empty:
        for boulder_uid, group in latest_frame.groupby("boulder_uid", sort=True):
            for field in DIRECTION_FIELDS:
                counts_by_value = group[field].astype(str).value_counts().sort_index()
                maximum = int(counts_by_value.max())
                modes = sorted(counts_by_value.loc[counts_by_value.eq(maximum)].index)
                direction_rows.append({
                    "boulder_uid": boulder_uid,
                    "segment": field.removesuffix("_direction"),
                    "independent_reviewers": int(group["reviewer_code"].nunique()),
                    "direction_counts_json": json.dumps(
                        {key: int(value) for key, value in counts_by_value.items()},
                        sort_keys=True,
                    ),
                    "modal_directions": "|".join(modes),
                    "unanimous": bool(len(counts_by_value) == 1),
                })
    directions = pd.DataFrame(direction_rows, columns=[
        "boulder_uid", "segment", "independent_reviewers",
        "direction_counts_json", "modal_directions", "unanimous",
    ])
    wave_a = progress.loc[progress["review_wave"].eq(WAVE_A)]
    wave_b = progress.loc[progress["review_wave"].eq(WAVE_B)]
    wave_a_complete = bool(wave_a["independent_reviewers"].ge(2).all())
    session_complete = bool(progress["independent_reviewers"].ge(2).all())
    status = (
        "RESEARCH_SESSION_COMPLETE_NO_MODEL_INPUT"
        if session_complete
        else "RESEARCH_WAVE_A_COMPLETE_SESSION_IN_PROGRESS_NO_MODEL_INPUT"
        if wave_a_complete
        else "HUMAN_REVIEW_IN_PROGRESS_NO_MODEL_INPUT"
        if len(latest)
        else "READY_NO_REVIEWS"
    )
    report = {
        "schema": SCHEMA,
        "status": status,
        "coverage": {
            "downloaded_records": int(len(records)),
            "latest_session_reviewer_boulder_records": int(len(latest)),
            "records_outside_session": int(outside_session),
            "session_tasks": 30,
            "expected_independent_review_records": 60,
            "distinct_reviewers": int(
                latest_frame["reviewer_code"].nunique() if not latest_frame.empty else 0
            ),
            "tasks_not_started": int(progress["independent_reviewers"].eq(0).sum()),
            "tasks_singly_reviewed": int(progress["independent_reviewers"].eq(1).sum()),
            "tasks_double_reviewed": int(progress["independent_reviewers"].ge(2).sum()),
            "wave_a_tasks_double_reviewed": int(
                wave_a["independent_reviewers"].ge(2).sum()
            ),
            "wave_b_tasks_double_reviewed": int(
                wave_b["independent_reviewers"].ge(2).sum()
            ),
        },
        "semantics": {
            "latest_whole_record_per_reviewer_boulder": True,
            "directions_disagreement_preserved": True,
            "core_tag_disagreement_preserved": True,
            "wave_a_operational_completion_requires_two_reviewers": True,
            "two_reviews_is_a_model_eligibility_cliff": False,
            "model_input_authorized": False,
            "model_fit_authorized": False,
            "coaching_prescription_authorized": False,
        },
    }
    return progress, directions, report


def verify_session_receipt(session_path: Path, receipt_path: Path) -> dict[str, object]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("schema") != "physical-tag-human-review-session-v1"
        or receipt.get("status") != "HUMAN_REVIEW_SESSION_READY_NO_TAGS_COMPLETED"
        or receipt.get("output", {}).get("sha256") != sha256(session_path)
        or receipt.get("output", {}).get("rows") != 30
        or receipt.get("claims", {}).get("model_input_authorized") is not False
        or receipt.get("claims", {}).get("model_fit_authorized") is not False
    ):
        raise ValueError("the governed session receipt is invalid")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--priority", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--session-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    verify_session_receipt(args.session, args.session_receipt)
    records = load_records(args.records)
    priority = pd.read_csv(args.priority, low_memory=False)
    inventory = pd.read_csv(args.inventory, low_memory=False)
    session = pd.read_csv(args.session, low_memory=False)
    tag_consensus, latest_tags, consensus_report = materialize_consensus(
        records, priority, inventory
    )
    progress, directions, session_report = audit_session_records(records, session)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "tag_consensus.csv": tag_consensus,
        "latest_reviewer_tags.csv": latest_tags,
        "session_task_progress.csv": progress,
        "direction_distributions.csv": directions,
    }
    for filename, frame in outputs.items():
        frame.to_csv(args.output_dir / filename, index=False, lineterminator="\n")
    receipt = {
        **session_report,
        "tag_consensus_status": consensus_report["status"],
        "tag_consensus_coverage": consensus_report["coverage"],
        "bindings": {
            "records_sha256": sha256(args.records),
            "priority_sha256": sha256(args.priority),
            "inventory_sha256": sha256(args.inventory),
            "session_sha256": sha256(args.session),
            "session_receipt_sha256": sha256(args.session_receipt),
        },
        "outputs": {
            filename: sha256(args.output_dir / filename) for filename in outputs
        },
    }
    (args.output_dir / "receipt.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
