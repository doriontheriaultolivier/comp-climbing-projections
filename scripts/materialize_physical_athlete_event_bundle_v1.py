"""Bundle physical observations with later athlete-event round vectors."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path

import pandas as pd

SCHEMA = "physical-athlete-event-round-vector-v1"
GROUP_KEYS = [
    "observation_id", "athlete_id", "event_date", "source_scope_competition",
    "source_event_id", "discipline", "pool_competition", "category",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_deterministic_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(
            filename="", mode="wb", compresslevel=9, mtime=0, fileobj=raw
        ) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                frame.to_csv(text, index=False, lineterminator="\n")


def round_stage(name: object) -> str:
    label = str(name).strip().casefold()
    if "qualif" in label:
        return "qualification"
    if "semi" in label:
        return "semifinal"
    if "final" in label:
        return "final"
    return "unresolved"


def json_value(value: object) -> object:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def stable_round_vector(rows: pd.DataFrame) -> str:
    order = {"qualification": 0, "semifinal": 1, "final": 2, "unresolved": 3}
    records: list[dict[str, object]] = []
    for _, row in rows.assign(
        round_stage=rows["round_name"].map(round_stage),
    ).sort_values(
        ["round_name"], key=lambda series: series.map(
            lambda value: (order[round_stage(value)], str(value).casefold())
        ), kind="stable",
    ).iterrows():
        record = {
            key: json_value(row.get(key))
            for key in [
                "round_name", "round_stage", "rank_numeric", "n_athletes",
                "rank_pct", "tops", "zones", "top_attempts", "zone_attempts",
                "format_identifier",
            ]
        }
        records.append(record)
    return json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False)


def build_bundles(timeline: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    required = set(GROUP_KEYS) | {
        "event_name", "round_name", "observed_on", "days_to_competition",
        "observation_family", "metric_id", "value", "unit",
        "model_input_authorized",
    }
    missing = required - set(timeline.columns)
    if missing:
        raise ValueError(f"timeline missing columns: {sorted(missing)}")
    if timeline["model_input_authorized"].astype(bool).any():
        raise ValueError("timeline cannot authorize model input")
    if timeline.duplicated(GROUP_KEYS + ["round_name"]).any():
        raise ValueError("duplicate athlete-event round rows")

    passthrough = [
        "observation_family", "metric_id", "capacity_dimension", "protocol_id",
        "observed_on", "value", "unit", "reporting_window_days",
        "source_revision", "days_to_competition", "event_name",
    ]
    output_rows: list[dict[str, object]] = []
    for key, rows in timeline.groupby(GROUP_KEYS, dropna=False, sort=True):
        first = rows.iloc[0]
        if any(rows[column].nunique(dropna=False) != 1 for column in passthrough):
            raise ValueError(f"non-round fields drift within bundle {key}")
        stages = rows["round_name"].map(round_stage)
        record = dict(zip(GROUP_KEYS, key, strict=True))
        record.update({column: json_value(first[column]) for column in passthrough})
        record.update({
            "round_count": int(len(rows)),
            "qualification_round_count": int(stages.eq("qualification").sum()),
            "semifinal_round_count": int(stages.eq("semifinal").sum()),
            "final_round_count": int(stages.eq("final").sum()),
            "unresolved_round_count": int(stages.eq("unresolved").sum()),
            "round_vector_json": stable_round_vector(rows),
            "outcome_semantics": "lossless_round_vector_no_synthetic_event_score",
            "descriptive_only": True,
            "model_input_authorized": False,
        })
        output_rows.append(record)
    bundles = pd.DataFrame(output_rows).sort_values(
        ["athlete_id", "observed_on", "event_date", "source_scope_competition",
         "source_event_id", "discipline", "pool_competition", "category"],
        kind="stable",
    ).reset_index(drop=True)
    if bundles.duplicated(GROUP_KEYS).any():
        raise ValueError("bundle key is not unique")
    report = {
        "schema": SCHEMA,
        "status": "DESCRIPTIVE_ATHLETE_EVENT_ROUND_VECTORS_READY_NO_MODEL_FIT",
        "coverage": {
            "bundles": int(len(bundles)),
            "athletes": int(bundles["athlete_id"].nunique()),
            "observations": int(bundles["observation_id"].nunique()),
            "event_contexts": int(bundles[
                ["source_scope_competition", "source_event_id", "discipline",
                 "pool_competition", "category"]
            ].drop_duplicates().shape[0]),
            "round_rows_preserved": int(bundles["round_count"].sum()),
            "multi_round_bundles": int(bundles["round_count"].gt(1).sum()),
            "discipline_counts": {
                str(key): int(value)
                for key, value in bundles["discipline"].value_counts().items()
            },
        },
        "claims": {
            "distinct_disciplines_pools_categories_preserved": True,
            "round_results_preserved_as_vector": True,
            "synthetic_event_score_created": False,
            "best_round_selected": False,
            "model_fit": False,
            "training_prescription_authorized": False,
            "competition_prediction_authorized": False,
        },
        "next_gate": (
            "Join reviewed per-boulder demand tags to Boulder event contexts, retain "
            "untagged events explicitly, and define athlete/event-grouped chronological "
            "support-transfer candidate evaluation."
        ),
    }
    return bundles, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    timeline = pd.read_csv(args.timeline, low_memory=False)
    bundles, report = build_bundles(timeline)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "athlete_event_round_vectors.csv.gz"
    write_deterministic_gzip_csv(bundles, output)
    report["bindings"] = {"timeline_sha256": sha256(args.timeline)}
    report["output"] = {"rows": len(bundles), "sha256": sha256(output)}
    (args.output_dir / "receipt.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["coverage"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
