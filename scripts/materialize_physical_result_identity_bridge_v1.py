"""Materialize governed physical-test to competition-result identity links.

Exact or reviewed links are usable for research joins. Name-only candidates are
isolated for human review and never promoted by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


SCHEMA = "physical-result-identity-bridge-v1"
GOVERNED_METHODS = {
    "IFSC source identity",
    "Exact normalized name + exact birth date",
    "Reviewed source-ID override",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_bridge(
    profiles_path: Path,
    links_path: Path,
    results_path: Path,
    physical_overrides_path: Path,
    reviewed_overrides_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    profiles = pd.read_csv(profiles_path)
    links = pd.read_csv(links_path)
    results = pd.read_csv(results_path, low_memory=False)
    physical_overrides = pd.read_csv(physical_overrides_path)
    reviewed_overrides = pd.read_csv(reviewed_overrides_path)
    required_profiles = {
        "athlete_id", "athlete_name", "pool", "identity_status",
        "identity_match_method", "identity_evidence_score",
    }
    required_links = {
        "source_scope", "athlete_source_id", "global_id",
        "identity_match_method", "identity_confidence", "athlete_name",
        "birthday", "rated_rounds",
    }
    if missing := required_profiles.difference(profiles.columns):
        raise ValueError(f"profiles missing {sorted(missing)}")
    if missing := required_links.difference(links.columns):
        raise ValueError(f"identity links missing {sorted(missing)}")
    if profiles["athlete_id"].duplicated().any():
        raise ValueError("physical profile athlete IDs must be unique")

    profiles = profiles.copy()
    profiles["global_id"] = "IFSC:" + profiles["athlete_id"].astype(int).astype(str)
    scoped = links.loc[links["global_id"].astype(str).isin(profiles["global_id"])].copy()
    scoped["identity_confidence"] = pd.to_numeric(
        scoped["identity_confidence"], errors="raise"
    )
    scoped["rated_rounds"] = pd.to_numeric(scoped["rated_rounds"], errors="coerce")
    governed_mask = (
        scoped["identity_match_method"].isin(GOVERNED_METHODS)
        & scoped["identity_confidence"].ge(0.95)
    )
    governed = scoped.loc[governed_mask].copy()
    review = scoped.loc[~governed_mask].copy()

    profile_context = profiles[
        ["global_id", "athlete_name", "pool", "identity_status",
         "identity_match_method", "identity_evidence_score"]
    ].rename(columns={
        "athlete_name": "physical_athlete_name",
        "identity_match_method": "physical_identity_match_method",
    })
    governed = governed.merge(profile_context, on="global_id", validate="many_to_one")
    governed["bridge_status"] = "governed_research_join_allowed"
    governed["model_input_authorized"] = False
    governed["canonical_identity_mutated"] = False
    governed = governed.sort_values(
        ["global_id", "source_scope", "athlete_source_id"], kind="stable"
    ).reset_index(drop=True)

    review = review.merge(profile_context, on="global_id", validate="many_to_one")
    evidence_columns = [
        "source_scope", "athlete_source_id", "source_event_id", "event_date",
        "event_name", "round_name", "category", "rank_numeric", "n_athletes",
    ]
    evidence = results[evidence_columns].copy()
    evidence["athlete_source_id"] = evidence["athlete_source_id"].astype(str)
    review["athlete_source_id"] = review["athlete_source_id"].astype(str)
    evidence = evidence.merge(
        review[["source_scope", "athlete_source_id"]].drop_duplicates(),
        on=["source_scope", "athlete_source_id"],
        how="inner",
        validate="many_to_many",
    )
    evidence = evidence.sort_values(
        ["source_scope", "athlete_source_id", "event_date", "source_event_id"],
        ascending=[True, True, False, False],
        kind="stable",
    )
    evidence = evidence.drop_duplicates(
        ["source_scope", "athlete_source_id", "source_event_id", "round_name", "category"],
        keep="first",
    )
    evidence_map: dict[tuple[str, str], str] = {}
    for key, rows in evidence.groupby(["source_scope", "athlete_source_id"], sort=True):
        records = rows.head(12).copy()
        records["event_date"] = records["event_date"].astype(str)
        evidence_map[(str(key[0]), str(key[1]))] = json.dumps(
            records[
                ["event_date", "source_event_id", "event_name", "round_name",
                 "category", "rank_numeric", "n_athletes"]
            ].to_dict("records"),
            ensure_ascii=False,
            separators=(",", ":"),
        )
    review["competition_evidence_json"] = [
        evidence_map.get((str(source), str(source_id)), "[]")
        for source, source_id in zip(review["source_scope"], review["athlete_source_id"])
    ]
    review["competition_evidence_rows"] = review["competition_evidence_json"].map(
        lambda value: len(json.loads(value))
    )
    review["decision"] = "DEFER"
    review["review_reason"] = (
        "Name-only identity evidence is insufficient; compare date of birth, "
        "country, gender and attached competitions before ACCEPT_SAME or KEEP_SEPARATE."
    )
    review["model_input_authorized"] = False
    review["canonical_identity_mutated"] = False
    review = review.sort_values(
        ["global_id", "source_scope", "athlete_source_id"], kind="stable"
    ).reset_index(drop=True)

    if governed.empty or review.empty:
        raise ValueError("expected both governed links and unresolved review candidates")
    if not governed["identity_match_method"].isin(GOVERNED_METHODS).all():
        raise ValueError("ungoverned match method leaked into bridge")
    if governed["identity_confidence"].lt(0.95).any():
        raise ValueError("low-confidence identity leaked into bridge")
    if review["model_input_authorized"].any() or governed["model_input_authorized"].any():
        raise ValueError("identity bridge cannot authorize a model input")
    if review["competition_evidence_rows"].le(0).any():
        raise ValueError("every manual review candidate requires competition evidence")

    covered = set(governed["global_id"])
    profile_ids = set(profiles["global_id"])
    source_counts = governed["source_scope"].value_counts().sort_index().to_dict()
    report = {
        "schema": SCHEMA,
        "status": "RESEARCH_JOIN_READY_MANUAL_REVIEW_REQUIRED_NO_MODEL_INPUT",
        "bindings": {
            "physical_profiles_sha256": sha256(profiles_path),
            "identity_links_sha256": sha256(links_path),
            "source_results_sha256": sha256(results_path),
            "physical_overrides_sha256": sha256(physical_overrides_path),
            "reviewed_overrides_sha256": sha256(reviewed_overrides_path),
        },
        "coverage": {
            "physical_profiles": int(len(profiles)),
            "governed_source_links": int(len(governed)),
            "profiles_with_governed_non_ifsc_link": int(
                governed.loc[governed["source_scope"].ne("IFSC"), "global_id"].nunique()
            ),
            "profiles_without_governed_non_ifsc_link": int(
                len(profile_ids - set(
                    governed.loc[governed["source_scope"].ne("IFSC"), "global_id"]
                ))
            ),
            "governed_source_counts": {str(k): int(v) for k, v in source_counts.items()},
            "manual_review_candidates": int(len(review)),
            "digitalrock_governed_links": int(
                governed["source_scope"].eq("DIGITALROCK").sum()
            ),
            "profiles_with_any_governed_link": int(len(covered)),
        },
        "review_policy": {
            "auto_merge_by_name": False,
            "allowed_governed_methods": sorted(GOVERNED_METHODS),
            "decision_vocabulary": ["ACCEPT_SAME", "KEEP_SEPARATE", "DEFER"],
            "unresolved_candidates_are_model_inputs": False,
        },
        "authority": {
            "research_join": True,
            "canonical_identity_mutation": False,
            "rating_input": False,
            "physical_ceiling_fit": False,
            "app_change": False,
        },
        "evidence": {
            "physical_override_rows": int(len(physical_overrides)),
            "reviewed_cross_source_override_rows": int(len(reviewed_overrides)),
        },
    }
    return governed, review, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--links", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--physical-overrides", type=Path, required=True)
    parser.add_argument("--reviewed-overrides", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    governed, review, report = build_bridge(
        args.profiles, args.links, args.results,
        args.physical_overrides, args.reviewed_overrides
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    governed_path = args.output_dir / "governed_links.csv"
    review_path = args.output_dir / "manual_review_queue.csv"
    governed.to_csv(governed_path, index=False, lineterminator="\n")
    review.to_csv(review_path, index=False, lineterminator="\n")
    report["outputs"] = {
        "governed_links": {"rows": len(governed), "sha256": sha256(governed_path)},
        "manual_review_queue": {"rows": len(review), "sha256": sha256(review_path)},
    }
    (args.output_dir / "receipt.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["coverage"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
