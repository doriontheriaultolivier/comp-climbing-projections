"""Audit support for pooled and current-category YW-IFSC readiness.

This audit deliberately keeps three questions separate:

* does an athlete-specific pooled Youth Worlds context component improve
  chronological entry-round forecasts over the shared graph state;
* which literal Youth Worlds category fields and athlete transitions exist;
* can a literal category be interpreted as a governed age-category target.

The third question fails closed without a dated IFSC rule and an event-scoped
alias certificate.  Labels such as ``Youth A`` and ``U19`` are never equated
by this module.  The output is research evidence, not an app rating writer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SCHEMA = "yw-ifsc-reference-field-support-v1"
DISPLAY_SCALE = 400.0
EVENT_PERFORMANCE_SD = 155.0
YOUTH_DOMAIN = "ifsc_youth_world"
BASE_COMPONENTS = ("skill", "form")
FULL_COMPONENTS = ("skill", "form", "offset:ifsc_youth_world")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def component_projection(
    labels_json: str,
    means_json: str,
    covariance_json: str,
    components: Iterable[str],
) -> tuple[float, float]:
    """Return mean and SD of a component sum from the exact pre-event state."""

    labels = list(json.loads(labels_json))
    means = np.asarray(json.loads(means_json), dtype=float)
    covariance = np.asarray(json.loads(covariance_json), dtype=float)
    if covariance.shape != (len(labels), len(labels)) or means.shape != (len(labels),):
        raise ValueError("component state shape does not match its labels")
    requested = tuple(components)
    if any(component not in labels for component in requested):
        raise ValueError("required component is absent from the V4 state")
    index = [labels.index(component) for component in requested]
    mean = float(means[index].sum())
    variance = float(covariance[np.ix_(index, index)].sum())
    if variance < -1e-6:
        raise ValueError("component sum has negative variance")
    return mean, math.sqrt(max(0.0, variance))


def integrated_probability(
    left_mean: float,
    right_mean: float,
    left_sd: float,
    right_sd: float,
    *,
    display_scale: float = DISPLAY_SCALE,
    event_performance_sd: float = EVENT_PERFORMANCE_SD,
) -> float:
    """Exact V4 logistic-normal attenuation approximation."""

    q = math.log(10.0) / float(display_scale)
    variance = left_sd**2 + right_sd**2 + 2.0 * event_performance_sd**2
    q_effective = q / math.sqrt(1.0 + math.pi * q**2 * variance / 8.0)
    logit = float(np.clip(q_effective * (left_mean - right_mean), -60.0, 60.0))
    return 1.0 / (1.0 + math.exp(-logit))


def youth_entry_rows(prepared: pd.DataFrame, replay: pd.DataFrame) -> pd.DataFrame:
    keys = replay.loc[
        replay["target_domain"].astype(str).eq(YOUTH_DOMAIN),
        ["event_id", "event_date", "event_name", "pool"],
    ].drop_duplicates()
    if keys["event_id"].duplicated().any():
        raise ValueError("Youth Worlds replay event identifiers are not unique")
    working = prepared.copy()
    working["event_date"] = pd.to_datetime(working["event_date"], errors="coerce")
    keys = keys.assign(event_date=pd.to_datetime(keys["event_date"], errors="coerce"))
    entry = working.merge(
        keys,
        on=["event_date", "event_name", "pool"],
        how="inner",
        validate="many_to_one",
    )
    entry = entry.loc[entry["round_group"].astype(str).eq("Qualification")].copy()
    required = {"event_id", "contest_id", "global_id", "rank_numeric", "category"}
    if required - set(entry.columns):
        raise ValueError("prepared entry rows lack required fields")
    if entry.empty:
        raise ValueError("no Youth Worlds qualification rows were found")
    duplicates = entry.duplicated(["event_id", "contest_id", "global_id"], keep=False)
    if duplicates.any():
        raise ValueError("Youth Worlds entry rows contain duplicate athlete-contest keys")
    return entry


def pre_event_states(trace: pd.DataFrame) -> pd.DataFrame:
    youth = trace.loc[trace["target_domain"].astype(str).eq(YOUTH_DOMAIN)].copy()
    rows: list[dict[str, object]] = []
    for row in youth.itertuples(index=False):
        base_mean, base_sd = component_projection(
            row.component_labels_json,
            row.pre_component_means_json,
            row.pre_component_covariance_json,
            BASE_COMPONENTS,
        )
        full_mean, full_sd = component_projection(
            row.component_labels_json,
            row.pre_component_means_json,
            row.pre_component_covariance_json,
            FULL_COMPONENTS,
        )
        rows.append(
            {
                "event_id": str(row.event_id),
                "global_id": str(row.athlete_id),
                "base_mean": base_mean,
                "base_sd": base_sd,
                "pooled_yw_mean": full_mean,
                "pooled_yw_sd": full_sd,
                "pre_yw_offset": full_mean - base_mean,
            }
        )
    output = pd.DataFrame(rows)
    if output.duplicated(["event_id", "global_id"]).any():
        raise ValueError("pre-event trace contains duplicate Youth Worlds states")
    return output


def build_pair_forecasts(entry: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    joined = entry.merge(
        states,
        on=["event_id", "global_id"],
        how="left",
        validate="many_to_one",
    )
    if joined[["base_mean", "base_sd", "pooled_yw_mean", "pooled_yw_sd"]].isna().any().any():
        raise ValueError("a Youth Worlds entry has no pre-event V4 state")
    records: list[dict[str, object]] = []
    for (event_id, contest_id), field in joined.groupby(
        ["event_id", "contest_id"], sort=True
    ):
        field = field.sort_values("global_id", kind="stable").reset_index(drop=True)
        if len(field) < 2:
            continue
        for left_index in range(len(field) - 1):
            left = field.iloc[left_index]
            for right_index in range(left_index + 1, len(field)):
                right = field.iloc[right_index]
                left_rank = float(left["rank_numeric"])
                right_rank = float(right["rank_numeric"])
                outcome = 1.0 if left_rank < right_rank else 0.0 if left_rank > right_rank else 0.5
                record = {
                    "event_id": str(event_id),
                    "competition_id": "|".join(str(event_id).split("|")[:2]),
                    "contest_id": str(contest_id),
                    "event_date": pd.Timestamp(left["event_date"]),
                    "year": int(pd.Timestamp(left["event_date"]).year),
                    "pool": str(left["pool"]),
                    "category": str(left["category"]),
                    "left_id": str(left["global_id"]),
                    "right_id": str(right["global_id"]),
                    "outcome": outcome,
                }
                for model, mean_column, sd_column in (
                    ("shared_graph", "base_mean", "base_sd"),
                    ("shared_graph_plus_pooled_yw", "pooled_yw_mean", "pooled_yw_sd"),
                ):
                    record[f"p_{model}"] = integrated_probability(
                        float(left[mean_column]),
                        float(right[mean_column]),
                        float(left[sd_column]),
                        float(right[sd_column]),
                    )
                records.append(record)
    return pd.DataFrame(records)


def _loss(probability: pd.Series, outcome: pd.Series) -> pd.Series:
    p = probability.clip(1e-12, 1.0 - 1e-12)
    return -(outcome * np.log(p) + (1.0 - outcome) * np.log1p(-p))


def pair_metrics(pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groupings = [("overall", ["year"]), ("pool", ["year", "pool"]), ("literal_category", ["year", "pool", "category"])]
    for scope, columns in groupings:
        for keys, group in pairs.groupby(columns, sort=True):
            keys = keys if isinstance(keys, tuple) else (keys,)
            identity = dict(zip(columns, keys))
            for model in ("shared_graph", "shared_graph_plus_pooled_yw"):
                probability = group[f"p_{model}"]
                outcome = group["outcome"]
                per_contest = pd.DataFrame(
                    {
                        "contest_id": group["contest_id"],
                        "log_loss": _loss(probability, outcome),
                        "brier": (probability - outcome) ** 2,
                    }
                ).groupby("contest_id", as_index=False)[["log_loss", "brier"]].mean()
                rows.append(
                    {
                        "scope": scope,
                        **identity,
                        "model": model,
                        "competition_clusters": int(group["competition_id"].nunique()),
                        "entry_fields": int(group["contest_id"].nunique()),
                        "pairs": int(len(group)),
                        "field_balanced_log_loss": float(per_contest["log_loss"].mean()),
                        "field_balanced_brier": float(per_contest["brier"].mean()),
                    }
                )
    output = pd.DataFrame(rows)
    for column in ("year", "pool", "category"):
        if column not in output:
            output[column] = pd.NA
    return output[[
        "scope", "year", "pool", "category", "model", "competition_clusters",
        "entry_fields", "pairs", "field_balanced_log_loss", "field_balanced_brier",
    ]]


def support_fields(entry: pd.DataFrame) -> pd.DataFrame:
    working = entry.copy()
    working["year"] = pd.to_datetime(working["event_date"]).dt.year
    working["age_known"] = pd.to_numeric(working["age_at_event"], errors="coerce").notna()
    rows: list[dict[str, object]] = []
    for keys, group in working.groupby(
        ["year", "event_id", "pool", "category"], sort=True
    ):
        year, event_id, pool, category = keys
        ages = pd.to_numeric(group["age_at_event"], errors="coerce")
        rows.append(
            {
                "year": int(year),
                "event_id": str(event_id),
                "pool": str(pool),
                "literal_category": str(category),
                "athletes": int(group["global_id"].nunique()),
                "ranked_rows": int(len(group)),
                "age_known_rows": int(ages.notna().sum()),
                "observed_age_min": float(ages.min()) if ages.notna().any() else np.nan,
                "observed_age_max": float(ages.max()) if ages.notna().any() else np.nan,
                "category_semantics": "literal IFSC result label only; no timeless alias",
                "production_category_mapping": False,
            }
        )
    return pd.DataFrame(rows)


def transition_summaries(entry: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    appearances = entry[["event_date", "pool", "category", "global_id"]].drop_duplicates()
    appearances = appearances.assign(year=pd.to_datetime(appearances["event_date"]).dt.year)
    appearances = appearances.sort_values(["global_id", "pool", "year"], kind="stable")
    detail: list[dict[str, object]] = []
    for (athlete, pool), group in appearances.groupby(["global_id", "pool"], sort=False):
        records = group.to_dict("records")
        for prior, later in zip(records, records[1:]):
            detail.append(
                {
                    "global_id": str(athlete),
                    "pool": str(pool),
                    "from_year": int(prior["year"]),
                    "to_year": int(later["year"]),
                    "from_literal_category": str(prior["category"]),
                    "to_literal_category": str(later["category"]),
                    "year_gap": int(later["year"] - prior["year"]),
                    "interpretation": "observed athlete transition; not a rule alias",
                }
            )
    detail_frame = pd.DataFrame(detail)
    if detail_frame.empty:
        return detail_frame, detail_frame
    summary = (
        detail_frame.groupby(
            ["pool", "from_year", "to_year", "from_literal_category", "to_literal_category"],
            as_index=False,
        )["global_id"]
        .nunique()
        .rename(columns={"global_id": "athletes"})
    )
    summary["semantic_status"] = "empirical continuity only; no category equivalence claim"
    return detail_frame, summary


def reference_fields(entry: pd.DataFrame) -> pd.DataFrame:
    fields = support_fields(entry)
    rows: list[dict[str, object]] = []
    for row in fields.itertuples(index=False):
        if row.year == 2025 and str(row.literal_category).startswith(("U17", "U19")):
            status = "candidate prior-season literal reference for 2026 descriptive audit"
            target_year = 2026
        elif row.year == 2024:
            status = "legacy literal field; not auto-mapped to 2025 U17/U19"
            target_year = 2025
        else:
            status = "observed field; no fresh future validation opened"
            target_year = row.year + 1
        rows.append(
            {
                "origin_year": int(row.year),
                "target_year": int(target_year),
                "pool": row.pool,
                "literal_category": row.literal_category,
                "reference_athletes": int(row.athletes),
                "status": status,
                "current_readiness_publication_allowed": False,
            }
        )
    return pd.DataFrame(rows)


def rule_registry_summary(registry: pd.DataFrame) -> dict[str, object]:
    required = {"registry_schema", "source_scope", "raw_category_label"}
    if required - set(registry.columns):
        raise ValueError("rule registry does not match the governed schema")
    if set(registry["registry_schema"].dropna().astype(str)) != {
        "official-youth-category-rule-registry-v1.1"
    }:
        raise ValueError("unsupported youth-category registry revision")
    ifsc = registry.loc[registry["source_scope"].astype(str).eq("IFSC")]
    return {
        "registry_rows": int(len(registry)),
        "ifsc_rule_rows": int(len(ifsc)),
        "ifsc_event_alias_certificates": 0,
        "activation_status": "WITHHELD_NO_IFSC_EVENT_SCOPED_ALIAS_CERTIFICATES",
    }


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--rule-registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    prepared = pd.read_parquet(args.prepared)
    replay = pd.read_parquet(args.replay)
    trace = pd.read_parquet(args.trace)
    registry = pd.read_csv(args.rule_registry, low_memory=False)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    plan_events = [event for event in plan["events"] if event["target_domain"] == YOUTH_DOMAIN]
    entry = youth_entry_rows(prepared, replay)
    states = pre_event_states(trace)
    pairs = build_pair_forecasts(entry, states)
    metrics = pair_metrics(pairs)
    fields = support_fields(entry)
    transition_detail, transitions = transition_summaries(entry)
    references = reference_fields(entry)
    registry_evidence = rule_registry_summary(registry)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "support_fields": (fields, args.output_dir / "support_fields.csv"),
        "pair_metrics": (metrics, args.output_dir / "pair_metrics.csv"),
        "transition_summary": (transitions, args.output_dir / "transition_summary.csv"),
        "reference_fields": (references, args.output_dir / "reference_fields.csv"),
    }
    for frame, path in outputs.values():
        write_csv(frame, path)
    transition_detail_path = args.output_dir / "restricted_transition_detail.parquet"
    transition_detail.to_parquet(transition_detail_path, index=False)

    overall = metrics.loc[metrics["scope"].eq("overall")].pivot(
        index="year", columns="model", values=["field_balanced_log_loss", "field_balanced_brier"]
    )
    year_deltas: dict[str, dict[str, float]] = {}
    for year in sorted(overall.index):
        year_deltas[str(int(year))] = {
            "pooled_minus_shared_log_loss": float(
                overall.loc[year, ("field_balanced_log_loss", "shared_graph_plus_pooled_yw")]
                - overall.loc[year, ("field_balanced_log_loss", "shared_graph")]
            ),
            "pooled_minus_shared_brier": float(
                overall.loc[year, ("field_balanced_brier", "shared_graph_plus_pooled_yw")]
                - overall.loc[year, ("field_balanced_brier", "shared_graph")]
            ),
        }
    competition_clusters = entry.assign(
        competition_id=entry["event_id"].astype(str).str.split("|").str[:2].str.join("|")
    )["competition_id"].nunique()
    receipt = {
        "schema": SCHEMA,
        "status": "RESEARCH_ONLY_CURRENT_CATEGORY_READY_NOT_IDENTIFIED_FOR_PUBLICATION",
        "inputs": {
            "prepared": sha256(args.prepared),
            "replay": sha256(args.replay),
            "trace": sha256(args.trace),
            "plan": sha256(args.plan),
            "rule_registry": sha256(args.rule_registry),
        },
        "contract": {
            "display_scale": DISPLAY_SCALE,
            "event_performance_sd": EVENT_PERFORMANCE_SD,
            "baseline_components": list(BASE_COMPONENTS),
            "pooled_yw_components": list(FULL_COMPONENTS),
            "entry_round_only": True,
            "literal_categories_never_normalized": True,
            "zero_direct_yw_starts_is_not_a_withhold_reason": True,
        },
        "coverage": {
            "plan_event_pools": len(plan_events),
            "independent_competitions": int(competition_clusters),
            "entry_fields": int(entry["contest_id"].nunique()),
            "entry_athletes": int(entry["global_id"].nunique()),
            "canonical_pairs": int(len(pairs)),
            "transition_athletes": int(transition_detail["global_id"].nunique()) if not transition_detail.empty else 0,
        },
        "rule_evidence": registry_evidence,
        "year_deltas": year_deltas,
        "decisions": {
            "pooled_yw_context": "DESCRIPTIVE_CHRONOLOGICAL_SIGNAL_ONLY_THREE_COMPETITIONS",
            "literal_u17_u19_reference": "2025_TO_2026_DESCRIPTIVE_ONLY_2026_ALREADY_INSPECTED",
            "cross_era_category_mapping": "REJECTED_NO_TIMELESS_YOUTH_A_B_JUNIOR_MAPPING",
            "current_category_yw_ifsc_ready": "WITHHELD_PENDING_IFSC_RULE_ALIAS_AND_FRESH_2027_VALIDATION",
            "next_model": "shared YW context only; category-specific reference fields and joint simulation, no category-specific response parameters",
        },
        "outputs": {
            name: {"rows": int(len(frame)), "sha256": sha256(path)}
            for name, (frame, path) in outputs.items()
        } | {
            "restricted_transition_detail": {
                "rows": int(len(transition_detail)),
                "sha256": sha256(transition_detail_path),
            }
        },
        "authority": {"production": False, "app": False, "deployment": False},
    }
    receipt_path = args.output_dir / "receipt.json"
    receipt_path.write_text(canonical_json(receipt) + "\n", encoding="ascii")
    print(canonical_json(receipt))


if __name__ == "__main__":
    main()
