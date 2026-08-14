"""Quantify continuous support for physical/board transfer candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


SCHEMA = "physical-support-continuous-estimability-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def round_stage(name: object) -> str:
    label = str(name).strip().casefold()
    if "qualif" in label:
        return "qualification"
    if "semi" in label:
        return "semifinal"
    if "final" in label:
        return "final"
    return "unresolved"


def effective_clusters(values: pd.Series) -> float:
    counts = values.astype(str).value_counts(normalize=True)
    return float(1.0 / counts.pow(2).sum()) if len(counts) else 0.0


def max_cluster_share(values: pd.Series) -> float:
    shares = values.astype(str).value_counts(normalize=True)
    return float(shares.max()) if len(shares) else 0.0


def observation_catalog(physical: pd.DataFrame, board: pd.DataFrame) -> pd.DataFrame:
    physical = physical.loc[physical["valid_result"].astype(bool)].copy()
    p = physical.assign(
        observation_family="physical",
        metric_id=physical["canonical_metric_id"].astype(str),
        measurement_protocol=physical["protocol_id"].astype(str),
    )[[
        "observation_id", "athlete_id", "observed_on", "observation_family",
        "metric_id", "capacity_dimension", "measurement_protocol",
    ]]
    b = board.assign(
        observation_family="board",
        metric_id=board["board_metric"].astype(str),
        capacity_dimension="on_wall_expression",
        measurement_protocol=(
            "owner_sheet_board_metric_v1:" + board["board_metric"].astype(str)
            + ":" + board["grade_scale"].astype(str)
        ),
    )[[
        "observation_id", "athlete_id", "observed_on", "observation_family",
        "metric_id", "capacity_dimension", "measurement_protocol",
    ]]
    result = pd.concat([p, b], ignore_index=True)
    if result["observation_id"].duplicated().any():
        raise ValueError("observation IDs collide across inputs")
    result["observed_on"] = pd.to_datetime(result["observed_on"], errors="raise")
    return result


def candidate_round_contexts(candidates: pd.DataFrame) -> pd.DataFrame:
    frame = candidates.copy()
    frame["source_event_id"] = frame["source_event_id"].astype(str)
    frame["round_stage"] = frame["round_name"].map(round_stage)
    return frame[
        ["source_scope", "source_event_id", "category", "round_stage"]
    ].drop_duplicates()


def explode_bundle_rounds(bundles: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for bundle in bundles.to_dict(orient="records"):
        for round_record in json.loads(str(bundle["round_vector_json"])):
            rows.append({
                "observation_id": bundle["observation_id"],
                "athlete_id": bundle["athlete_id"],
                "observation_family": bundle["observation_family"],
                "metric_id": bundle["metric_id"],
                "capacity_dimension": bundle["capacity_dimension"],
                "measurement_protocol": bundle["protocol_id"],
                "observed_on": bundle["observed_on"],
                "event_date": bundle["event_date"],
                "days_to_competition": bundle["days_to_competition"],
                "source_scope": bundle["source_scope_competition"],
                "source_event_id": str(bundle["source_event_id"]),
                "discipline": bundle["discipline"],
                "pool": bundle["pool_competition"],
                "category": bundle["category"],
                "round_stage": round_record["round_stage"],
                "round_name": round_record["round_name"],
            })
    result = pd.DataFrame(rows)
    result["event_date"] = pd.to_datetime(result["event_date"], errors="raise")
    result["observed_on"] = pd.to_datetime(result["observed_on"], errors="raise")
    result["event_cluster"] = (
        result["source_scope"].astype(str) + "|" + result["source_event_id"].astype(str)
        + "|" + result["discipline"].astype(str) + "|" + result["pool"].astype(str)
        + "|" + result["category"].astype(str)
    )
    return result


def build_estimability(
    physical: pd.DataFrame,
    board: pd.DataFrame,
    bundles: pd.DataFrame,
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    catalog = observation_catalog(physical, board)
    rounds = explode_bundle_rounds(bundles)
    candidate_contexts = candidate_round_contexts(candidates)
    rounds = rounds.merge(
        candidate_contexts.assign(pending_item_candidate=True),
        on=["source_scope", "source_event_id", "category", "round_stage"],
        how="left",
        validate="many_to_one",
    )
    rounds["pending_item_candidate"] = rounds["pending_item_candidate"].eq(True)
    groups = [
        "observation_family", "metric_id", "capacity_dimension",
        "measurement_protocol",
    ]
    profiles: list[dict[str, object]] = []
    for key, observed in catalog.groupby(groups, sort=True, dropna=False):
        linked = rounds.loc[
            rounds["observation_id"].astype(str).isin(set(observed["observation_id"].astype(str)))
        ].copy()
        boulder = linked.loc[linked["discipline"].astype(str).eq("Boulder")].copy()
        repeats = observed.groupby("athlete_id").size()
        linked_observations = linked["observation_id"].nunique()
        record = dict(zip(groups, key, strict=True))
        record.update({
            "input_observations": int(len(observed)),
            "input_athletes": int(observed["athlete_id"].nunique()),
            "athletes_with_repeats": int(repeats.ge(2).sum()),
            "athlete_repeat_fraction": float(repeats.ge(2).mean()),
            "linked_observations": int(linked_observations),
            "linked_observation_fraction": float(linked_observations / len(observed)),
            "linked_athletes": int(linked["athlete_id"].nunique()),
            "linked_round_rows": int(len(linked)),
            "linked_event_contexts": int(linked["event_cluster"].nunique()),
            "linked_source_events": int(linked[
                ["source_scope", "source_event_id"]
            ].drop_duplicates().shape[0]),
            "linked_sources": int(linked["source_scope"].nunique()),
            "event_effective_clusters": effective_clusters(linked["event_cluster"]),
            "max_event_cluster_share": max_cluster_share(linked["event_cluster"]),
            "athlete_effective_clusters": effective_clusters(linked["athlete_id"]),
            "max_athlete_share": max_cluster_share(linked["athlete_id"]),
            "first_observed_on": observed["observed_on"].min().date().isoformat(),
            "last_observed_on": observed["observed_on"].max().date().isoformat(),
            "first_future_event_date": (
                linked["event_date"].min().date().isoformat() if len(linked) else ""
            ),
            "last_future_event_date": (
                linked["event_date"].max().date().isoformat() if len(linked) else ""
            ),
            "future_event_years": int(linked["event_date"].dt.year.nunique()),
            "median_days_to_competition": (
                float(linked["days_to_competition"].median()) if len(linked) else None
            ),
            "boulder_round_rows": int(len(boulder)),
            "boulder_round_rows_with_pending_item_candidate": int(
                boulder["pending_item_candidate"].sum()
            ),
            "pending_item_candidate_coverage": (
                float(boulder["pending_item_candidate"].mean()) if len(boulder) else None
            ),
            "completed_human_demand_tag_coverage": 0.0,
            "evidence_state": "continuous_profile_no_eligibility_cliff",
            "model_input_authorized": False,
        })
        profiles.append(record)
    output = pd.DataFrame(profiles).sort_values(
        ["observation_family", "capacity_dimension", "metric_id", "measurement_protocol"],
        kind="stable",
    ).reset_index(drop=True)
    report = {
        "schema": SCHEMA,
        "status": "CONTINUOUS_ESTIMABILITY_PROFILE_READY_NO_MODEL_FIT",
        "coverage": {
            "measurement_profiles": int(len(output)),
            "physical_profiles": int(output["observation_family"].eq("physical").sum()),
            "board_profiles": int(output["observation_family"].eq("board").sum()),
            "input_observations": int(len(catalog)),
            "linked_observations": int(rounds["observation_id"].nunique()),
            "linked_event_contexts": int(rounds["event_cluster"].nunique()),
            "future_event_years": int(rounds["event_date"].dt.year.nunique()),
        },
        "semantics": {
            "hard_sample_cutoff_used": False,
            "mcmc_ess_treated_as_data_support": False,
            "effective_clusters_definition": "inverse Herfindahl index over linked row shares",
            "pending_item_candidate_is_completed_tag": False,
            "missing_metric_imputed_low": False,
            "current_outcomes_used_to_select_model": False,
        },
        "method_ladder": {
            "current": "counts/concentration profile and frozen chronological data contract",
            "next_after_independent_tags": "strongly shrunk monotone additive Zone/Top-given-Zone support challenger",
            "withheld": [
                "physical-by-demand interactions before independent tags",
                "CES/soft-min before stable compensation evidence",
                "one-sided access/frontier before repeated matched observations",
                "athlete-specific training prescriptions",
            ],
        },
        "next_gate": (
            "Collect independent demand tags and prospective outcomes; compare model "
            "sensitivity to prior, single-athlete, single-event and missingness assumptions "
            "without turning any count into a pass/fail threshold."
        ),
    }
    return output, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical", type=Path, required=True)
    parser.add_argument("--board", type=Path, required=True)
    parser.add_argument("--bundles", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output, report = build_estimability(
        pd.read_csv(args.physical),
        pd.read_csv(args.board),
        pd.read_csv(args.bundles, low_memory=False),
        pd.read_csv(args.candidates, low_memory=False),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "measurement_profiles.csv"
    output.to_csv(output_path, index=False, lineterminator="\n")
    report["bindings"] = {
        "physical_sha256": sha256(args.physical),
        "board_sha256": sha256(args.board),
        "athlete_event_bundles_sha256": sha256(args.bundles),
        "item_candidates_sha256": sha256(args.candidates),
    }
    report["output"] = {
        "rows": len(output), "sha256": sha256(output_path),
    }
    (args.output_dir / "receipt.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["coverage"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
