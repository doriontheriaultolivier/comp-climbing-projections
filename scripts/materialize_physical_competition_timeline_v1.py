"""Create chronological physical/board to later-competition research pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


SCHEMA = "physical-competition-chronological-timeline-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_timeline(
    physical_path: Path,
    board_path: Path,
    governed_links_path: Path,
    results_path: Path,
    event_taxonomy_path: Path,
    *,
    horizon_days: int = 365,
) -> tuple[pd.DataFrame, dict[str, object]]:
    physical = pd.read_csv(physical_path)
    board = pd.read_csv(board_path)
    links = pd.read_csv(governed_links_path)
    results = pd.read_csv(results_path, low_memory=False)
    taxonomy = pd.read_csv(event_taxonomy_path, low_memory=False)
    physical["observed_on"] = pd.to_datetime(physical["observed_on"], errors="raise")
    board["observed_on"] = pd.to_datetime(board["observed_on"], errors="raise")
    results["event_date"] = pd.to_datetime(results["event_date"], errors="coerce")
    taxonomy["event_date"] = pd.to_datetime(taxonomy["event_date"], errors="raise")
    links["athlete_source_id"] = links["athlete_source_id"].astype(str)
    results["athlete_source_id"] = results["athlete_source_id"].astype(str)

    if links["model_input_authorized"].astype(bool).any():
        raise ValueError("governed identity bridge cannot authorize model input")
    resolved = results.merge(
        links[["global_id", "source_scope", "athlete_source_id"]].drop_duplicates(),
        on=["source_scope", "athlete_source_id"],
        how="inner",
        validate="many_to_many",
    )
    resolved = resolved.loc[resolved["event_date"].notna()].copy()
    quarantine = taxonomy.loc[
        taxonomy["direct_context_head"].astype(str).eq("QUARANTINE"),
        ["event_date", "event_name", "source_scope"],
    ].drop_duplicates()
    quarantine["fixture_quarantine"] = True
    resolved = resolved.merge(
        quarantine,
        on=["event_date", "event_name", "source_scope"],
        how="left",
        validate="many_to_one",
    )
    fixture_mask = resolved["fixture_quarantine"].eq(True)
    fixture_rows = int(fixture_mask.sum())
    resolved = resolved.loc[~fixture_mask].copy()
    resolved = resolved.sort_values(
        ["global_id", "event_date", "source_scope", "source_event_id", "round_name"],
        kind="stable",
    )
    resolved = resolved.drop_duplicates(
        ["global_id", "event_date", "source_scope", "source_event_id", "pool",
         "round_name", "category"],
        keep="first",
    )

    long_observations = pd.concat(
        [
            physical.assign(
                observation_family="physical",
                metric_id=physical["canonical_metric_id"],
                grade_scale="",
                reporting_window_days=float("nan"),
            ),
            board.assign(
                observation_family="board",
                metric_id=board["board_metric"],
                capacity_dimension="on_wall_expression",
                protocol_id="owner_sheet_board_metric_v1",
                unit=board["grade_scale"],
                valid_result=True,
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    long_observations = long_observations.loc[
        long_observations["athlete_id"].astype(str).isin(set(links["global_id"]))
    ].copy()

    joined = long_observations.merge(
        resolved,
        left_on="athlete_id",
        right_on="global_id",
        how="inner",
        suffixes=("_observation", "_competition"),
        validate="many_to_many",
    )
    joined["days_to_competition"] = (
        joined["event_date"] - joined["observed_on"]
    ).dt.days
    joined = joined.loc[
        joined["days_to_competition"].between(1, int(horizon_days), inclusive="both")
    ].copy()
    joined["chronology_status"] = "observation_strictly_before_competition"
    joined["descriptive_only"] = True
    joined["model_input_authorized"] = False
    keep = [
        "observation_id", "athlete_id", "observation_family", "metric_id",
        "capacity_dimension", "protocol_id", "observed_on", "value", "unit",
        "reporting_window_days", "source_revision", "days_to_competition",
        "event_date", "source_scope_competition", "source_event_id", "event_name",
        "discipline", "pool_competition", "category", "round_name", "rank_numeric", "n_athletes",
        "rank_pct", "tops", "zones", "top_attempts", "zone_attempts",
        "format_identifier", "chronology_status", "descriptive_only",
        "model_input_authorized",
    ]
    joined = joined.rename(columns={
        "source_scope": "source_scope_competition",
        "pool": "pool_competition",
    })
    keep = [column for column in keep if column in joined.columns]
    joined = joined[keep].sort_values(
        ["athlete_id", "observed_on", "event_date", "source_event_id", "round_name"],
        kind="stable",
    ).reset_index(drop=True)
    if joined.empty:
        raise ValueError("no chronological observation-to-competition pairs")
    if joined["days_to_competition"].le(0).any():
        raise ValueError("future or same-day result leaked into an observation")
    if joined["model_input_authorized"].any():
        raise ValueError("timeline cannot authorize model input")

    report = {
        "schema": SCHEMA,
        "status": "DESCRIPTIVE_CHRONOLOGICAL_TIMELINE_READY_NO_MODEL_FIT",
        "bindings": {
            "physical_observations_sha256": sha256(physical_path),
            "board_observations_sha256": sha256(board_path),
            "governed_links_sha256": sha256(governed_links_path),
            "source_results_sha256": sha256(results_path),
            "event_taxonomy_sha256": sha256(event_taxonomy_path),
        },
        "coverage": {
            "pairs": int(len(joined)),
            "athletes": int(joined["athlete_id"].nunique()),
            "observations": int(joined["observation_id"].nunique()),
            "competition_fields": int(
                joined[["source_scope_competition", "source_event_id", "pool_competition", "round_name"]]
                .drop_duplicates().shape[0]
            ),
            "physical_pairs": int(joined["observation_family"].eq("physical").sum()),
            "board_pairs": int(joined["observation_family"].eq("board").sum()),
            "horizon_days": int(horizon_days),
            "fixture_result_rows_quarantined": fixture_rows,
            "source_counts": {
                str(key): int(value)
                for key, value in joined["source_scope_competition"].value_counts().items()
            },
        },
        "claims": {
            "observation_strictly_precedes_result": True,
            "current_rating_used_as_historical_target": False,
            "linear_ceiling_model_fit": False,
            "missing_values_imputed_low": False,
            "training_prescription_authorized": False,
            "competition_prediction_authorized": False,
        },
        "next_gate": (
            "Aggregate later competition outcomes by athlete-event without duplicating "
            "rounds, add reviewed boulder demand tags, then compare predeclared monotone "
            "support/transfer candidates under athlete-and-event grouped chronology."
        ),
    }
    return joined, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical", type=Path, required=True)
    parser.add_argument("--board", type=Path, required=True)
    parser.add_argument("--links", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--event-taxonomy", type=Path, required=True)
    parser.add_argument("--horizon-days", type=int, default=365)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    timeline, report = build_timeline(
        args.physical, args.board, args.links, args.results, args.event_taxonomy,
        horizon_days=args.horizon_days,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "timeline.csv.gz"
    timeline.to_csv(output, index=False, compression="gzip", lineterminator="\n")
    report["output"] = {"rows": len(timeline), "sha256": sha256(output)}
    (args.output_dir / "receipt.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["coverage"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
