"""Measure coaching evidence unlocked by each pending Boulder demand tag."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path

import pandas as pd


SCHEMA = "physical-item-tag-unlock-v1"


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


def write_deterministic(frame: pd.DataFrame, path: Path) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(
            filename="", mode="wb", compresslevel=9, mtime=0, fileobj=raw
        ) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                frame.to_csv(text, index=False, lineterminator="\n")


def explode_boulder_rounds(bundles: pd.DataFrame) -> pd.DataFrame:
    boulder = bundles.loc[bundles["discipline"].astype(str).eq("Boulder")].copy()
    rows: list[dict[str, object]] = []
    for record in boulder.to_dict(orient="records"):
        vector = json.loads(str(record["round_vector_json"]))
        for round_record in vector:
            rows.append({
                "observation_id": record["observation_id"],
                "athlete_id": record["athlete_id"],
                "observation_family": record["observation_family"],
                "source_scope": record["source_scope_competition"],
                "source_event_id": str(record["source_event_id"]),
                "event_name": record["event_name"],
                "category": record["category"],
                "pool": record["pool_competition"],
                "round_stage": round_record["round_stage"],
            })
    return pd.DataFrame(rows)


def build_unlock(
    bundles: pd.DataFrame,
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if set(candidates["tag_status"].astype(str)) != {"human_demand_tags_needed"}:
        raise ValueError("candidate table contains a non-pending tag state")
    rounds = explode_boulder_rounds(bundles)
    candidates = candidates.copy()
    candidates["source_event_id"] = candidates["source_event_id"].astype(str)
    candidates["round_stage"] = candidates["round_name"].map(round_stage)
    candidate_keys = ["source_scope", "source_event_id", "category", "round_stage"]
    item_columns = candidate_keys + [
        "priority_rank", "problem_id", "boulder_number", "leaderboard_route_id",
        "event_date", "event_name", "round_name", "tag_status",
    ]
    joined = rounds.merge(
        candidates[item_columns],
        on=candidate_keys,
        how="left",
        suffixes=("_result", "_candidate"),
        validate="many_to_many",
        indicator=True,
    )
    joined["item_coverage_status"] = joined["_merge"].map({
        "both": "pending_human_demand_tag",
        "left_only": "no_item_candidate_for_round_context",
        "right_only": "unreachable",
    }).astype(str)
    joined = joined.drop(columns="_merge")
    covered = joined.loc[joined["problem_id"].notna()].copy()
    unlock = covered.groupby(
        ["priority_rank", "problem_id", "source_scope", "source_event_id",
         "event_date", "event_name_candidate", "category", "round_name",
         "round_stage", "boulder_number", "leaderboard_route_id", "tag_status"],
        dropna=False,
        sort=True,
    ).agg(
        coaching_athletes_unlocked=("athlete_id", "nunique"),
        coaching_observations_unlocked=("observation_id", "nunique"),
        physical_observations_unlocked=(
            "observation_family", lambda values: int((values == "physical").sum())
        ),
        board_observations_unlocked=(
            "observation_family", lambda values: int((values == "board").sum())
        ),
        observation_round_links=("observation_id", "size"),
    ).reset_index()
    unlock = unlock.sort_values(
        ["coaching_athletes_unlocked", "coaching_observations_unlocked", "priority_rank"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    unlock.insert(0, "coaching_unlock_rank", range(1, len(unlock) + 1))

    round_contexts = rounds[
        ["source_scope", "source_event_id", "event_name", "category", "pool", "round_stage"]
    ].drop_duplicates()
    available_contexts = candidates[candidate_keys].drop_duplicates()
    gaps = round_contexts.merge(
        available_contexts,
        on=candidate_keys,
        how="left",
        indicator=True,
        validate="one_to_one",
    )
    gaps = gaps.loc[gaps["_merge"].eq("left_only")].drop(columns="_merge").sort_values(
        ["source_scope", "source_event_id", "category", "round_stage"], kind="stable"
    ).reset_index(drop=True)
    gaps["coverage_status"] = "no_item_candidate_for_round_context"

    report = {
        "schema": SCHEMA,
        "status": "HUMAN_TAG_UNLOCK_PRIORITY_READY_NO_TAGS_COMPLETED",
        "coverage": {
            "boulder_bundles": int(bundles["discipline"].astype(str).eq("Boulder").sum()),
            "boulder_round_rows": int(len(rounds)),
            "candidate_items": int(candidates["problem_id"].nunique()),
            "candidate_items_linked_to_coaching_timeline": int(unlock["problem_id"].nunique()),
            "covered_round_contexts": int(round_contexts.merge(
                available_contexts, on=candidate_keys, how="inner"
            ).shape[0]),
            "uncovered_round_contexts": int(len(gaps)),
            "athletes_with_any_candidate": int(covered["athlete_id"].nunique()),
            "observations_with_any_candidate": int(covered["observation_id"].nunique()),
        },
        "claims": {
            "completed_human_demand_tags": 0,
            "candidate_status_is_not_a_tag": True,
            "shared_terrain_evidence_used_as_demand_tag": False,
            "style_imputed_for_uncovered_events": False,
            "model_fit": False,
            "coaching_prescription_authorized": False,
        },
        "next_gate": (
            "Complete independent human demand tags beginning with high-unlock items; "
            "then measure inter-rater agreement before any ceiling/transfer model fit."
        ),
    }
    return unlock, gaps, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundles", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    bundles = pd.read_csv(args.bundles, low_memory=False)
    candidates = pd.read_csv(args.candidates, low_memory=False)
    unlock, gaps, report = build_unlock(bundles, candidates)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    unlock_path = args.output_dir / "item_unlock_priority.csv.gz"
    gaps_path = args.output_dir / "uncovered_round_contexts.csv.gz"
    write_deterministic(unlock, unlock_path)
    write_deterministic(gaps, gaps_path)
    report["bindings"] = {
        "athlete_event_bundles_sha256": sha256(args.bundles),
        "item_candidates_sha256": sha256(args.candidates),
    }
    report["outputs"] = {
        "unlock": {"rows": len(unlock), "sha256": sha256(unlock_path)},
        "gaps": {"rows": len(gaps), "sha256": sha256(gaps_path)},
    }
    (args.output_dir / "receipt.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["coverage"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
