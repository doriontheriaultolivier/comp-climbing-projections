"""Materialize the first identity-free human Boulder-tag review session."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd


SCHEMA = "physical-tag-human-review-session-v1"
DEFAULT_TASKS = 30
OUTPUT_COLUMNS = [
    "session_task_order",
    "review_wave",
    "boulder_uid",
    "event_date",
    "event_name",
    "round_group",
    "gender",
    "boulder_number",
    "priority_rank",
    "coaching_unlock_rank",
    "coaching_unlock_source_items",
    "coaching_unlock_athletes",
    "coaching_unlock_observations",
    "coaching_unlock_physical_observations",
    "coaching_unlock_board_observations",
    "priority_top_given_zone_pairs",
    "priority_zone_pairs",
    "requested_independent_reviewers",
    "required_review_scope",
    "task_status",
]
FORBIDDEN_COLUMN_FRAGMENTS = (
    "athlete_id",
    "athlete_name",
    "physical_value",
    "test_value",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_session(tasks: pd.DataFrame, task_count: int = DEFAULT_TASKS) -> pd.DataFrame:
    """Return a deterministic, operational batch without creating an evidence cliff."""
    required = {
        "boulder_uid",
        "event_date",
        "event_name",
        "round_group",
        "gender",
        "boulder_number",
        "priority_rank",
        "coaching_unlock_rank",
    }
    missing = required.difference(tasks.columns)
    if missing:
        raise ValueError(f"task inventory is missing columns: {sorted(missing)}")
    if task_count < 1:
        raise ValueError("task_count must be positive")

    candidates = tasks.loc[
        tasks["priority_rank"].notna() & tasks["coaching_unlock_rank"].notna()
    ].copy()
    if candidates["boulder_uid"].duplicated().any():
        raise ValueError("governed boulder_uid must be unique")
    candidates = candidates.sort_values(
        [
            "coaching_unlock_rank",
            "priority_rank",
            "event_date",
            "event_name",
            "round_group",
            "gender",
            "boulder_number",
            "boulder_uid",
        ],
        ascending=[True, True, False, True, True, True, True, True],
        na_position="last",
        kind="stable",
    ).head(task_count)
    if len(candidates) < task_count:
        raise ValueError(
            f"only {len(candidates)} governed coaching-priority tasks are available"
        )

    candidates.insert(0, "session_task_order", range(1, len(candidates) + 1))
    candidates["review_wave"] = candidates["session_task_order"].le(10).map(
        {True: "A_same_tasks_independent_calibration", False: "B_high_unlock_extension"}
    )
    candidates["requested_independent_reviewers"] = 2
    candidates["required_review_scope"] = (
        "directions_and_three_core_demands; detailed_tags_optional"
    )
    candidates["task_status"] = "human_review_pending"
    for column in OUTPUT_COLUMNS:
        if column not in candidates.columns:
            candidates[column] = pd.NA
    result = candidates[OUTPUT_COLUMNS].copy()
    lowered = [str(column).lower() for column in result.columns]
    if any(fragment in column for column in lowered for fragment in FORBIDDEN_COLUMN_FRAGMENTS):
        raise ValueError("review session contains a forbidden identity or test-value column")
    return result


def materialize(output_dir: Path, task_count: int) -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from style_tagging_app import problem_inventory

    output_dir.mkdir(parents=True, exist_ok=True)
    session = build_session(problem_inventory(), task_count=task_count)
    csv_path = output_dir / "physical_tag_human_review_session_v1.csv"
    session.to_csv(csv_path, index=False, lineterminator="\n")

    bindings = {}
    for relative in (
        "data/boulder_problem_inventory.csv.gz",
        "data/physical_item_tagging_priority_v1_1.csv",
        "data/physical_item_tag_unlock_app_v1.csv",
    ):
        path = root / relative
        bindings[relative] = sha256_file(path)

    report: dict[str, object] = {
        "schema": SCHEMA,
        "status": "HUMAN_REVIEW_SESSION_READY_NO_TAGS_COMPLETED",
        "bindings": bindings,
        "producer_sha256": sha256_file(Path(__file__)),
        "output": {
            "path": csv_path.name,
            "rows": int(len(session)),
            "unique_boulders": int(session["boulder_uid"].nunique()),
            "events": int(session["event_name"].nunique()),
            "sha256": sha256_file(csv_path),
        },
        "operational_design": {
            "wave_a_tasks": int(session["review_wave"].str.startswith("A_").sum()),
            "wave_b_tasks": int(session["review_wave"].str.startswith("B_").sum()),
            "requested_independent_reviewers_per_task": 2,
            "pseudonymous_reviewers": True,
            "reviewers_must_work_independently": True,
            "core_demands_required": [
                "physical_0_3",
                "technical_0_3",
                "coordination_0_3",
            ],
            "detailed_tags_optional": True,
        },
        "claims": {
            "contains_athlete_identity": False,
            "contains_physical_test_values": False,
            "completed_human_tags": 0,
            "minimum_review_eligibility_cliff": None,
            "model_input_authorized": False,
            "model_fit_authorized": False,
            "coaching_prescription_authorized": False,
        },
        "next_gate": (
            "Complete the same tasks independently, download the shared records, "
            "and materialize disagreement-preserving tag consensus."
        ),
    }
    receipt_path = output_dir / "physical_tag_human_review_session_v1.json"
    receipt_path.write_bytes(
        (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--task-count", type=int, default=DEFAULT_TASKS)
    args = parser.parse_args()
    report = materialize(args.output_dir, args.task_count)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
