"""Build a release-schema successor from the reviewed canonical identity rebuild.

This is a local staging producer.  It never overwrites ``data/`` and it grants
no deployment authority.  The input research artifact is accepted only when
its published identity-rebuild receipt, exact file hashes, and reviewed
IFSC:18545 -> IFSC:14843 decision all match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "data" / "release_identity_senior_wc_baseline_v1.json"
EXPECTED_DECISION = {
    "source_system": "IFSC",
    "alias_source_athlete_id": "18545",
    "canonical_global_id": "IFSC:14843",
    "automatic_new_identity_merges": False,
}
ATHLETE_FILE = "boulder_overview_athletes.parquet"
HISTORY_FILE = "boulder_overview_history.parquet"
INTERNAL_HISTORY_FILE = "boulder_global_history.parquet"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_candidate(candidate_root: Path) -> dict[str, object]:
    receipt_path = candidate_root / "data" / "identity_rebuild_validation_v1.json"
    receipt = load_json(receipt_path)
    if receipt.get("schema") != "identity-rebuild-validation-v1":
        raise ValueError("candidate identity receipt schema differs")
    if receipt.get("status") != "PASS_RESEARCH_IDENTITY_REBUILD_VALIDATED":
        raise ValueError("candidate identity receipt is not accepted research evidence")
    if receipt.get("reviewed_decision") != EXPECTED_DECISION:
        raise ValueError("candidate reviewed identity decision differs")
    authority = receipt.get("authority")
    if authority != {
        "additional_identity_merge": False,
        "deployment": False,
        "production_rating_promotion": False,
    }:
        raise ValueError("candidate authority boundary differs")
    bindings = receipt.get("artifact_bindings")
    if not isinstance(bindings, dict):
        raise ValueError("candidate artifact bindings are missing")
    for binding in bindings.values():
        if not isinstance(binding, dict):
            raise ValueError("candidate artifact binding is malformed")
        path = candidate_root / str(binding["relative_path"])
        if not path.is_file():
            raise ValueError(f"candidate file is missing: {path.name}")
        if path.stat().st_size != int(binding["candidate_bytes"]):
            raise ValueError(f"candidate byte count differs: {path.name}")
        if sha256(path) != binding["candidate_sha256"]:
            raise ValueError(f"candidate digest differs: {path.name}")
    return receipt


def _elementary_symmetric(values: np.ndarray, order: int) -> np.ndarray:
    output = np.zeros(order + 1, dtype=float)
    output[0] = 1.0
    reached = 0
    for value in np.asarray(values, dtype=float):
        reached = min(order, reached + 1)
        for degree in range(reached, 0, -1):
            output[degree] += value * output[degree - 1]
    return output


def joint_performance(
    ratings_before: np.ndarray,
    ranks: np.ndarray,
    prior_means: np.ndarray,
    prior_variances: np.ndarray,
    likelihood_strength: float,
    form_sd: float = 250.0,
    grid_step: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the frozen generalized Plackett-Luce contest posterior."""

    opponents = np.asarray(ratings_before, dtype=float)
    observed_ranks = np.asarray(ranks, dtype=float)
    priors = np.asarray(prior_means, dtype=float)
    variances = np.asarray(prior_variances, dtype=float)
    n_athletes = len(opponents)
    if n_athletes == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    low = float(min(np.nanmin(opponents), np.nanmin(priors)) - 1200.0)
    high = float(max(np.nanmax(opponents), np.nanmax(priors)) + 1200.0)
    grid = np.arange(low, high + grid_step, grid_step, dtype=float)
    centre = float(np.nanmedian(opponents))
    elo_log_scale = np.log(10.0) / 400.0
    opponent_log_worth = np.clip((opponents - centre) * elo_log_scale, -40, 40)
    candidate_log_worth = np.clip((grid - centre) * elo_log_scale, -40, 40)

    rank_values = np.sort(np.unique(observed_ranks[np.isfinite(observed_ranks)]))
    groups = [np.flatnonzero(observed_ranks == value) for value in rank_values]
    group_of = np.full(n_athletes, -1, dtype=int)
    remaining = np.arange(n_athletes, dtype=int)
    group_states: list[tuple[np.ndarray, int, np.ndarray]] = []
    for group_number, selected in enumerate(groups):
        group_size = len(selected)
        transformed = np.exp(opponent_log_worth[remaining] / group_size)
        group_states.append(
            (remaining.copy(), group_size, _elementary_symmetric(transformed, group_size))
        )
        group_of[selected] = group_number
        remaining = remaining[~np.isin(remaining, selected)]

    means = np.full(n_athletes, np.nan, dtype=float)
    sds = np.full(n_athletes, np.nan, dtype=float)
    strength = float(np.clip(likelihood_strength, 0.02, 1.0))
    for athlete_index in range(n_athletes):
        final_group = int(group_of[athlete_index])
        if final_group < 0:
            continue
        log_likelihood = np.zeros(len(grid), dtype=float)
        for group_number in range(final_group + 1):
            remaining_indices, group_size, all_symmetric = group_states[group_number]
            if athlete_index not in remaining_indices:
                continue
            original_beta = float(np.exp(opponent_log_worth[athlete_index] / group_size))
            without_focal = np.zeros(group_size + 1, dtype=float)
            without_focal[0] = 1.0
            for degree in range(1, group_size + 1):
                without_focal[degree] = max(
                    0.0,
                    all_symmetric[degree]
                    - original_beta * without_focal[degree - 1],
                )
            candidate_beta = np.exp(candidate_log_worth / group_size)
            denominator = without_focal[group_size] + (
                candidate_beta * without_focal[group_size - 1]
            )
            log_likelihood -= np.log(np.maximum(denominator, 1e-300))
            if group_number == final_group:
                log_likelihood += candidate_log_worth / group_size
        prior_sd = float(
            np.clip(
                np.sqrt(max(0.0, variances[athlete_index]) + form_sd**2),
                form_sd,
                520.0,
            )
        )
        log_posterior = (
            -0.5 * ((grid - priors[athlete_index]) / prior_sd) ** 2
            + strength * log_likelihood
        )
        weights = np.exp(log_posterior - np.nanmax(log_posterior))
        weight_sum = float(np.sum(weights))
        mean = float(np.sum(grid * weights) / weight_sum)
        variance = float(np.sum((grid - mean) ** 2 * weights) / weight_sum)
        means[athlete_index] = mean
        sds[athlete_index] = math.sqrt(max(0.0, variance))
    return means, sds


def add_joint_performance(history: pd.DataFrame) -> pd.DataFrame:
    output = history.copy()
    output["joint_ranking_performance_elo"] = np.nan
    output["joint_ranking_performance_elo_uncertainty"] = np.nan
    direct_levels = {
        "Youth World Championship",
        "World Cup",
        "Olympic qualifier",
        "World Championship",
        "Olympics",
    }
    for contest_number, (_, contest) in enumerate(
        output.groupby("contest_id", sort=False), start=1
    ):
        direct = str(contest["competition_level"].iloc[0]) in direct_levels
        strength = 1.0 if direct else (
            float(contest["shared_transfer"].iloc[0])
            * float(contest["information_quality"].iloc[0])
        )
        means, sds = joint_performance(
            pd.to_numeric(contest["event_start_global_rating"], errors="coerce").to_numpy(),
            pd.to_numeric(contest["model_rank"], errors="coerce").to_numpy(),
            pd.to_numeric(contest["event_start_global_rating"], errors="coerce").to_numpy(),
            pd.to_numeric(contest["event_start_rating_uncertainty"], errors="coerce")
            .pow(2)
            .to_numpy(),
            strength,
        )
        output.loc[contest.index, "joint_ranking_performance_elo"] = means
        output.loc[contest.index, "joint_ranking_performance_elo_uncertainty"] = sds
        if contest_number % 1000 == 0:
            print(f"joint posterior contests: {contest_number:,}", flush=True)
    return output


def build_athletes(candidate: pd.DataFrame, baseline_columns: list[str]) -> pd.DataFrame:
    output = candidate.copy()
    if "Canada projection — all evidence" in output:
        output = output.rename(
            columns={"Canada projection — all evidence": "canada_projection_all_evidence"}
        )
    output = output.drop(
        columns=[column for column in ("birthday", "birth_date_analysis_value") if column in output],
    )
    for column in ("WC+-ELO-Qualies-Flash", "WC+-ELO-Qualies-Flash evidence"):
        if column in output and output[column].notna().any():
            raise ValueError("canonical candidate contains unsupported WC+ Flash evidence")
        output[column] = np.nan
    age = pd.to_numeric(output["age"], errors="coerce")
    uncertainty_days = pd.to_numeric(
        output["birth_date_uncertainty_days"], errors="coerce"
    ).clip(lower=0)
    uncertainty_years = uncertainty_days / 365.2425
    output["age"] = np.floor(age * 10) / 10
    output["age_lower_years"] = np.floor((age - uncertainty_years.fillna(0)) * 10) / 10
    output["age_upper_years"] = np.ceil((age + uncertainty_years.fillna(0)) * 10) / 10
    output.loc[age.notna() & uncertainty_days.isna(), "age_upper_years"] = (
        output.loc[age.notna() & uncertainty_days.isna(), "age"] + 0.1
    )
    output.loc[age.notna() & uncertainty_days.eq(0), "age_upper_years"] = (
        output.loc[age.notna() & uncertainty_days.eq(0), "age"] + 0.1
    )
    output["age_precision_status"] = np.select(
        [age.isna(), uncertainty_days.fillna(np.inf).gt(0), uncertainty_days.eq(0)],
        [
            "unavailable",
            "source_interval_plus_public_tenth",
            "public_tenth_from_day_precision_source",
        ],
        default="public_tenth_source_uncertainty_unknown",
    )
    if set(output.columns) != set(baseline_columns):
        missing = sorted(set(baseline_columns) - set(output.columns))
        extra = sorted(set(output.columns) - set(baseline_columns))
        raise ValueError(f"athlete schema differs; missing={missing}; extra={extra}")
    output = output.loc[:, baseline_columns]
    if output["global_id"].eq("IFSC:18545").any():
        raise ValueError("alias identity remains in athlete output")
    if output.loc[output["global_id"].eq("IFSC:14843")].shape[0] != 1:
        raise ValueError("canonical identity is not unique in athlete output")
    return output


def build_history(internal: pd.DataFrame, baseline_columns: list[str]) -> pd.DataFrame:
    enriched = add_joint_performance(internal)
    missing = sorted(set(baseline_columns) - set(enriched.columns))
    if missing:
        raise ValueError(f"history source lacks release fields: {missing}")
    output = enriched.loc[:, baseline_columns].copy()
    if output["global_id"].eq("IFSC:18545").any():
        raise ValueError("alias identity remains in history output")
    canonical = output.loc[output["global_id"].eq("IFSC:14843")]
    if len(canonical) != 25:
        raise ValueError("canonical history does not contain 25 reviewed rows")
    if output[["joint_ranking_performance_elo", "joint_ranking_performance_elo_uncertainty"]].isna().any().any():
        raise ValueError("joint ranking posterior is incomplete")
    return output


def build(candidate_root: Path, output_root: Path) -> Path:
    candidate_root = candidate_root.resolve()
    receipt = validate_candidate(candidate_root)
    baseline = load_json(BASELINE)
    baseline_athletes = pd.read_parquet(ROOT / "data" / ATHLETE_FILE)
    baseline_history = pd.read_parquet(ROOT / "data" / HISTORY_FILE)
    if sha256(ROOT / "data" / ATHLETE_FILE) != baseline["files"]["athletes"]:
        raise ValueError("release athlete baseline differs")
    if sha256(ROOT / "data" / HISTORY_FILE) != baseline["files"]["history"]:
        raise ValueError("release history baseline differs")

    candidate_athletes = pd.read_parquet(candidate_root / "data" / ATHLETE_FILE)
    internal_history = pd.read_parquet(candidate_root / "data" / INTERNAL_HISTORY_FILE)
    athletes = build_athletes(candidate_athletes, list(baseline_athletes.columns))
    history = build_history(internal_history, list(baseline_history.columns))

    receipt_sha = sha256(candidate_root / "data" / "identity_rebuild_validation_v1.json")
    staging_parent = output_root.resolve()
    staging_parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".release-identity-v2-", dir=staging_parent))
    try:
        athletes.to_parquet(temporary / ATHLETE_FILE, index=False)
        history.to_parquet(temporary / HISTORY_FILE, index=False)
        manifest = {
            "schema": "release-identity-successor-v2",
            "status": "STAGED_LOCAL_NOT_AUTHORIZED_FOR_DEPLOYMENT",
            "source_identity_receipt_sha256": receipt_sha,
            "source_artifact_bindings": receipt["artifact_bindings"],
            "reviewed_decision": EXPECTED_DECISION,
            "outputs": {
                ATHLETE_FILE: {
                    "bytes": (temporary / ATHLETE_FILE).stat().st_size,
                    "sha256": sha256(temporary / ATHLETE_FILE),
                    "rows": len(athletes),
                    "columns": len(athletes.columns),
                },
                HISTORY_FILE: {
                    "bytes": (temporary / HISTORY_FILE).stat().st_size,
                    "sha256": sha256(temporary / HISTORY_FILE),
                    "rows": len(history),
                    "columns": len(history.columns),
                },
            },
            "wc_plus_flash_placeholder": "all_missing_no_direct_senior_flash_evidence",
            "authority": {
                "publication": False,
                "deployment": False,
                "production_rating_promotion": False,
                "additional_identity_merge": False,
            },
            "next_gate": "two-root byte reproduction, locked adult/youth/pathway calibration, app acceptance",
        }
        (temporary / "manifest.json").write_bytes(canonical_json_bytes(manifest))
        content_hash = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
        destination = staging_parent / f"staged-v2-{content_hash}"
        if destination.exists():
            shutil.rmtree(temporary)
            return destination
        temporary.replace(destination)
        return destination
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    destination = build(args.candidate_root, args.output_root)
    print(destination)


if __name__ == "__main__":
    main()
