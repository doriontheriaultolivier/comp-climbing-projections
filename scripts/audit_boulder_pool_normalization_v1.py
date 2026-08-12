"""Audit source-level Boulder pool normalization across sources and eras."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EXPECTED = {"Boulder_Men", "Boulder_Women"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    required = {
        "discipline", "pool", "age_class", "category", "source_scope",
        "event_date", "event_name",
    }
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"source rows missing columns: {sorted(missing)}")
    frame = rows.copy()
    frame["event_date"] = pd.to_datetime(frame["event_date"], errors="coerce")
    frame["year"] = frame["event_date"].dt.year.astype("Int64")
    discipline = frame["discipline"].astype(str).str.strip().str.casefold()
    boulder = frame.loc[discipline.eq("boulder")].copy()
    mismatched = boulder.loc[~boulder["pool"].astype(str).isin(EXPECTED)].copy()
    cross_discipline = frame.loc[
        ~discipline.eq("boulder") & frame["pool"].astype(str).isin(EXPECTED)
    ].copy()
    unresolved = (
        pd.concat([mismatched, cross_discipline], ignore_index=True)
        if not mismatched.empty or not cross_discipline.empty
        else frame.iloc[0:0].copy()
    )
    summary = (
        boulder.assign(
            youth=boulder["age_class"].astype(str).str.strip().str.casefold().eq("youth")
        )
        .groupby(["source_scope", "year", "pool", "youth"], dropna=False)
        .agg(
            rows=("event_name", "size"),
            events=("event_name", "nunique"),
            categories=("category", "nunique"),
        )
        .reset_index()
        .sort_values(["year", "source_scope", "pool", "youth"], kind="stable")
    )
    receipt = {
        "schema": "boulder-pool-normalization-audit-v1",
        "status": "PASS" if unresolved.empty else "UNRESOLVED_HOLD",
        "source_rows": int(len(frame)),
        "boulder_rows": int(len(boulder)),
        "youth_boulder_rows": int(
            boulder["age_class"].astype(str).str.strip().str.casefold().eq("youth").sum()
        ),
        "unexpected_boulder_pool_rows": int(len(mismatched)),
        "non_boulder_rows_in_boulder_pools": int(len(cross_discipline)),
        "contract": (
            "discipline is authoritative; pool is a normalized sex pool; youth/adult "
            "status remains separate and original category is retained"
        ),
        "unknown_labels_are_not_guessed": True,
        "authority": {"model_fit": False, "promotion": False},
    }
    return summary, unresolved, receipt


def run(input_path: Path, output_dir: Path) -> dict:
    rows = pd.read_csv(
        input_path,
        compression="infer",
        low_memory=False,
        usecols=[
            "discipline", "pool", "age_class", "category", "source_scope",
            "event_date", "event_name",
        ],
    )
    summary, unresolved, receipt = audit(rows)
    receipt["input_sha256"] = sha256_file(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(
        output_dir / "source_era_summary.csv", index=False, lineterminator="\n"
    )
    unresolved.to_csv(
        output_dir / "unresolved_rows.csv", index=False, lineterminator="\n"
    )
    with (output_dir / "receipt.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATA / "source_results.csv.gz")
    parser.add_argument(
        "--output-dir", type=Path, default=DATA / "boulder_pool_normalization_v1"
    )
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
