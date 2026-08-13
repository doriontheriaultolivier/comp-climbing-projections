"""Stage an identity-correct compact Boulder bundle in the release schema.

The rating replay and compact producers live in the research repository while
the deployed Streamlit app has a narrower, newer data contract.  This adapter
binds both sides by hash, performs only declared schema transformations, and
writes a content-addressed local candidate.  It never overwrites ``data/`` and
never grants publication or deployment authority.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SCHEMA = "compact-boulder-identity-successor-v1"
ATHLETES = "boulder_overview_athletes.parquet"
HISTORY = "boulder_overview_history.parquet"
CORRELATIONS = "boulder_rating_correlations.csv"
CALIBRATION = "boulder_elo_calibration.csv"
INTERNAL_HISTORY = "boulder_global_history.parquet"
RELEASE_JOINT_PRODUCER = "scripts/build_release_identity_successor_v2.py"
RUNTIME_FILES = (ATHLETES, HISTORY, CORRELATIONS, CALIBRATION)
RAW_DOB_FIELDS = ("birthday", "birth_date_analysis_value")
CANADA_SOURCE_FIELD = "Canada projection — all evidence"
CANADA_RELEASE_FIELD = "canada_projection_all_evidence"
WC_FLASH_FIELDS = (
    "WC+-ELO-Qualies-Flash",
    "WC+-ELO-Qualies-Flash evidence",
)


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


def file_binding(path: Path, *, rows: int | None = None, columns: int | None = None) -> dict[str, object]:
    binding: dict[str, object] = {
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
    if rows is not None:
        binding["rows"] = rows
    if columns is not None:
        binding["columns"] = columns
    return binding


def adapt_athletes(candidate: pd.DataFrame, release_columns: list[str]) -> pd.DataFrame:
    """Map research output to the exact compact release schema."""

    output = candidate.copy()
    if CANADA_SOURCE_FIELD in output.columns:
        output = output.rename(columns={CANADA_SOURCE_FIELD: CANADA_RELEASE_FIELD})
    output = output.drop(
        columns=[field for field in RAW_DOB_FIELDS if field in output.columns]
    )
    for field in WC_FLASH_FIELDS:
        if field in output.columns and output[field].notna().any():
            raise ValueError(f"unsupported direct senior WC+ Flash evidence: {field}")
        output[field] = np.nan

    required_age = {"age", "birth_date_uncertainty_days"}
    if missing := sorted(required_age - set(output.columns)):
        raise ValueError("candidate age inputs are missing: " + ", ".join(missing))
    age = pd.to_numeric(output["age"], errors="coerce")
    uncertainty_days = pd.to_numeric(
        output["birth_date_uncertainty_days"], errors="coerce"
    ).clip(lower=0)
    uncertainty_years = uncertainty_days / 365.2425
    output["age"] = np.floor(age * 10.0) / 10.0
    output["age_lower_years"] = (
        np.floor((age - uncertainty_years.fillna(0.0)) * 10.0) / 10.0
    )
    output["age_upper_years"] = (
        np.ceil((age + uncertainty_years.fillna(0.0)) * 10.0) / 10.0
    )
    unknown_uncertainty = age.notna() & uncertainty_days.isna()
    exact_source_day = age.notna() & uncertainty_days.eq(0)
    output.loc[unknown_uncertainty | exact_source_day, "age_upper_years"] = (
        output.loc[unknown_uncertainty | exact_source_day, "age"] + 0.1
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

    missing = sorted(set(release_columns) - set(output.columns))
    extra = sorted(set(output.columns) - set(release_columns))
    if missing or extra:
        raise ValueError(f"athlete schema differs; missing={missing}; extra={extra}")
    output = output.loc[:, release_columns]
    if output.duplicated(["pool", "global_id"]).any():
        raise ValueError("candidate contains duplicate pool/global identities")
    if output["global_id"].eq("IFSC:18545").any():
        raise ValueError("reviewed Amari alias remains in athlete output")
    canonical = output.loc[output["global_id"].eq("IFSC:14843")]
    if len(canonical) != 1:
        raise ValueError("canonical Amari identity is not unique")
    if canonical["WC+-ELO"].notna().any():
        raise ValueError("Amari has numeric direct WC+ evidence without a senior WC+ start")
    return output


def adapt_history(candidate: pd.DataFrame, release_columns: list[str]) -> pd.DataFrame:
    missing = sorted(set(release_columns) - set(candidate.columns))
    if missing:
        raise ValueError("history source lacks release fields: " + ", ".join(missing))
    output = candidate.loc[:, release_columns].copy()
    if output["global_id"].eq("IFSC:18545").any():
        raise ValueError("reviewed Amari alias remains in history output")
    if len(output.loc[output["global_id"].eq("IFSC:14843")]) != 25:
        raise ValueError("canonical Amari history does not contain 25 reviewed rows")
    return output


def load_release_joint_producer(release_root: Path):
    """Load the exact release joint-posterior implementation as a bound input."""

    path = release_root / RELEASE_JOINT_PRODUCER
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("bound_release_joint_producer", path)
    if spec is None or spec.loader is None:
        raise ValueError("could not load release joint-posterior producer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "add_joint_performance", None)):
        raise ValueError("release joint-posterior producer lacks add_joint_performance")
    return module


def build(*, release_root: Path, output_root: Path) -> Path:
    release_root = release_root.resolve()
    output_root = output_root.resolve()
    release_data = release_root / "data"
    for filename in RUNTIME_FILES:
        if not (DATA / filename).is_file():
            raise FileNotFoundError(DATA / filename)
        if not (release_data / filename).is_file():
            raise FileNotFoundError(release_data / filename)

    source_athletes = pd.read_parquet(DATA / ATHLETES)
    source_history = pd.read_parquet(DATA / INTERNAL_HISTORY)
    release_athletes = pd.read_parquet(release_data / ATHLETES)
    release_history = pd.read_parquet(release_data / HISTORY)
    athletes = adapt_athletes(source_athletes, list(release_athletes.columns))
    joint_producer = load_release_joint_producer(release_root)
    source_history = joint_producer.add_joint_performance(source_history)
    history = adapt_history(source_history, list(release_history.columns))

    input_bindings = {
        "research": {
            **{name: file_binding(DATA / name) for name in RUNTIME_FILES},
            INTERNAL_HISTORY: file_binding(DATA / INTERNAL_HISTORY),
        },
        "release_schema": {
            **{name: file_binding(release_data / name) for name in RUNTIME_FILES},
            RELEASE_JOINT_PRODUCER: file_binding(
                release_root / RELEASE_JOINT_PRODUCER
            ),
        },
    }
    contract = {
        "schema": SCHEMA,
        "input_bindings": input_bindings,
        "reviewed_identity": {
            "alias": "IFSC:18545",
            "canonical": "IFSC:14843",
            "canonical_history_rows": 25,
            "direct_wc_plus_rating_withheld": True,
        },
        "transformations": {
            "release_schema_preserved": True,
            "raw_birth_dates_removed": list(RAW_DOB_FIELDS),
            "age_intervals_materialized": True,
            "canada_projection_field_renamed": True,
            "unsupported_wc_flash_fields_added_missing": list(WC_FLASH_FIELDS),
        },
    }
    address = hashlib.sha256(canonical_json_bytes(contract)).hexdigest()
    target = output_root / f"candidate-{address}"
    if target.exists():
        manifest = json.loads((target / "manifest.json").read_text(encoding="ascii"))
        for filename, binding in manifest["outputs"].items():
            if sha256(target / filename) != binding["sha256"]:
                raise ValueError("existing candidate output hash differs")
        return target

    output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".compact-boulder-", dir=output_root))
    try:
        athletes.to_parquet(temporary / ATHLETES, index=False)
        history.to_parquet(temporary / HISTORY, index=False)
        shutil.copy2(DATA / CORRELATIONS, temporary / CORRELATIONS)
        shutil.copy2(DATA / CALIBRATION, temporary / CALIBRATION)
        outputs = {
            ATHLETES: file_binding(
                temporary / ATHLETES, rows=len(athletes), columns=len(athletes.columns)
            ),
            HISTORY: file_binding(
                temporary / HISTORY, rows=len(history), columns=len(history.columns)
            ),
            CORRELATIONS: file_binding(temporary / CORRELATIONS),
            CALIBRATION: file_binding(temporary / CALIBRATION),
        }
        manifest = {
            **contract,
            "status": "STAGED_RESEARCH_NOT_AUTHORIZED_FOR_DEPLOYMENT",
            "content_address": address,
            "outputs": outputs,
            "authority": {
                "publication": False,
                "deployment": False,
                "production_rating_promotion": False,
            },
        }
        (temporary / "manifest.json").write_bytes(canonical_json_bytes(manifest))
        temporary.replace(target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / ".artifacts" / "compact-boulder-identity-successor-v1",
    )
    args = parser.parse_args()
    target = build(release_root=args.release_root, output_root=args.output_root)
    print(target)


if __name__ == "__main__":
    main()
