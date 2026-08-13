"""Counts-only readiness audit for sparse physical and board observations.

The audit prepares the infrastructure for additional manual sheet rows.  It
does not fit a physical ceiling, infer missing tests, or issue prescriptions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


SCHEMA = "physical-board-sparse-intake-audit-v1"
REQUIRED_PHYSICAL = {
    "observation_id", "athlete_id", "observed_on", "canonical_metric_id",
    "capacity_dimension", "protocol_id", "value", "unit", "valid_result",
    "source_revision",
}
REQUIRED_BOARD = {
    "observation_id", "athlete_id", "observed_on", "board_metric", "value",
    "grade_scale", "reporting_window_days", "source_revision",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _coverage(frame: pd.DataFrame, group: str, metric: str) -> list[dict[str, object]]:
    output = []
    for (group_value, metric_value), rows in frame.groupby([group, metric], sort=True):
        per_athlete = rows.groupby("athlete_id").size()
        output.append(
            {
                group: str(group_value),
                metric: str(metric_value),
                "observations": int(len(rows)),
                "athletes": int(rows["athlete_id"].nunique()),
                "athletes_with_repeats": int((per_athlete >= 2).sum()),
                "median_observations_per_athlete": float(per_athlete.median()),
                "first_observed_on": rows["observed_on"].min().date().isoformat(),
                "last_observed_on": rows["observed_on"].max().date().isoformat(),
            }
        )
    return output


def build_report(physical_path: Path, board_path: Path) -> dict[str, object]:
    physical = pd.read_csv(physical_path)
    board = pd.read_csv(board_path)
    if missing := REQUIRED_PHYSICAL.difference(physical.columns):
        raise ValueError(f"physical input missing {sorted(missing)}")
    if missing := REQUIRED_BOARD.difference(board.columns):
        raise ValueError(f"board input missing {sorted(missing)}")
    if physical["observation_id"].duplicated().any() or board["observation_id"].duplicated().any():
        raise ValueError("observation IDs must be unique within each input")
    physical["observed_on"] = pd.to_datetime(physical["observed_on"], errors="raise")
    board["observed_on"] = pd.to_datetime(board["observed_on"], errors="raise")
    if not pd.api.types.is_bool_dtype(physical["valid_result"]):
        normalized = physical["valid_result"].astype(str).str.strip().str.lower()
        if not normalized.isin({"true", "false"}).all():
            raise ValueError("valid_result must contain only booleans")
        physical["valid_result"] = normalized.eq("true")
    valid = physical.loc[physical["valid_result"]].copy()

    physical_dates = valid[["athlete_id", "observed_on"]].drop_duplicates()
    board_dates = board[["athlete_id", "observed_on"]].drop_duplicates()
    same_day = physical_dates.merge(board_dates, on=["athlete_id", "observed_on"])
    physical_athletes = set(valid["athlete_id"].astype(str))
    board_athletes = set(board["athlete_id"].astype(str))

    dimension_summary = []
    for dimension, rows in valid.groupby("capacity_dimension", sort=True):
        per_athlete = rows.groupby("athlete_id").size()
        dimension_summary.append(
            {
                "capacity_dimension": str(dimension),
                "protocols": int(rows["protocol_id"].nunique()),
                "metrics": int(rows["canonical_metric_id"].nunique()),
                "observations": int(len(rows)),
                "athletes": int(rows["athlete_id"].nunique()),
                "athletes_with_repeats": int((per_athlete >= 2).sum()),
            }
        )

    protocol_rows = _coverage(valid, "capacity_dimension", "protocol_id")
    board_rows = _coverage(board, "grade_scale", "board_metric")
    weakest_protocols = sorted(
        protocol_rows,
        key=lambda row: (row["athletes"], row["athletes_with_repeats"], row["observations"]),
    )[:10]
    return {
        "schema": SCHEMA,
        "status": "COUNTS_ONLY_INTAKE_READY_NO_CEILING_MODEL_FIT",
        "bindings": {
            "physical_sha256": sha256(physical_path),
            "board_sha256": sha256(board_path),
        },
        "coverage": {
            "physical_observations": int(len(valid)),
            "physical_athletes": int(len(physical_athletes)),
            "board_observations": int(len(board)),
            "board_athletes": int(len(board_athletes)),
            "athletes_with_physical_and_board": int(len(physical_athletes & board_athletes)),
            "same_day_physical_board_sessions": int(len(same_day)),
            "capacity_dimensions": int(valid["capacity_dimension"].nunique()),
            "physical_protocols": int(valid["protocol_id"].nunique()),
        },
        "capacity_dimensions": dimension_summary,
        "protocol_coverage": protocol_rows,
        "board_coverage": board_rows,
        "next_observation_priority": {
            "method": "lowest distinct-athlete and repeat coverage first; coaching value and athlete burden must still be adjudicated",
            "weakest_protocols_by_coverage": weakest_protocols,
            "not_a_prescription": True,
        },
        "model_contract": {
            "physical_support_is_literal_ceiling": False,
            "kilter_flash_role": "low-pressure repeatable wall expression",
            "kilter_hardest_role": "exposure-dependent observed upper-tail wall expression",
            "competition_accessibility_role": "separate uncertain transfer/context layer",
            "missing_test_imputed_as_low": False,
            "linear_correlation_authorizes_model": False,
            "required_future_model": "chronological hierarchical Zone and Top-given-Zone hurdle with monotone saturating physical support and explicit sparse-observation likelihood",
        },
        "limitations": [
            "counts do not establish protocol comparability or coaching causality",
            "hardest-send exposure is unidentified without session and attempt history",
            "same-day physical and board overlap does not establish temporal transfer to competition",
            "terrain-demand tags and later competition outcomes are required for the planned support model",
        ],
        "authority": {
            "model_fit": False,
            "training_prescription": False,
            "app_change": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical", type=Path, required=True)
    parser.add_argument("--board", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.physical, args.board)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "coverage": report["coverage"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
