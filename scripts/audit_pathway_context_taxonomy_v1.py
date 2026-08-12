"""Predeclare direct-evidence labels for pathway rating heads.

Every valid result remains eligible for the shared athlete state.  This taxonomy
only says which target-context head receives *direct* evidence; it never filters
other rows out of the shared model and never infers identity or federation from
an athlete name.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SCHEMA = "pathway-context-taxonomy-audit-v1"
FEDERATION_SOURCES = {"CEC", "USAC", "FASI", "SAC-CAS", "FEDME"}
FEDERATION_TIERS = {
    "Regional / local",
    "Regional / local youth",
    "Regional / local masters",
    "National series",
    "National series youth",
    "National championship",
    "National championship youth",
}
IFSC_REGIONAL_TIERS = {
    "Continental series",
    "Continental series youth",
    "Continental championship",
    "Continental championship youth",
}


class TaxonomyError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regional_family(event_name: str) -> str | None:
    name = event_name.casefold()
    patterns = (
        ("PAN_AMERICA", ("pan america", "pan am")),
        ("EUROPE", ("europe", "european")),
        ("ASIA", ("asia", "asian")),
        ("OCEANIA", ("oceania",)),
        ("AFRICA", ("africa", "african")),
    )
    matched = [family for family, tokens in patterns if any(token in name for token in tokens)]
    return matched[0] if len(matched) == 1 else None


def classify_direct_context(row: pd.Series) -> tuple[str, str]:
    source = str(row["source_scope"]).strip()
    tier = str(row["event_tier"]).strip()
    name = str(row["event_name"]).strip()
    lower = name.casefold()
    youth_suffix = ":YOUTH" if str(row["rating_context"]) == "Youth" else ""

    if re.search(r"(?<![a-z])test(?![a-z])", lower):
        return "QUARANTINE", "event name is explicitly labelled TEST"
    if source == "IFSC" and (
        "olympic games" in lower or "olympic qualifier series" in lower
    ):
        return "OLYM_SCENARIO_INPUT", "Olympic/OQS evidence informs an exact scenario, not a fitted head"
    youth_world_name = bool(
        re.search(
            r"\byouth\s+world\s*champ|\bworld\s+climbing\s+youth\s+champ",
            lower,
        )
    )
    if source == "IFSC" and (tier == "World major youth" or youth_world_name):
        return (
            "IFSC_WORLD_YOUTH",
            "direct Youth World evidence; transition context, not adult WC evidence",
        )
    if source == "IFSC" and tier in {"World series", "World major"}:
        return "WC", "direct adult World series/World championship evidence"
    if source == "IFSC" and tier in IFSC_REGIONAL_TIERS:
        family = regional_family(name)
        if family is None:
            return "CONT:UNRESOLVED", "continental tier lacks one unambiguous region token"
        return f"CONT:{family}{youth_suffix}", "direct continental Series/Championship evidence"
    if source in {"CEC", "USAC"} and tier == "Continental / cross-border" and "nacs" in lower:
        return (
            "INTERFED:NORTH_AMERICA" + youth_suffix,
            "direct North American inter-federation circuit evidence (NACS)",
        )
    if source in FEDERATION_SOURCES and tier in FEDERATION_TIERS:
        return f"FED:{source}{youth_suffix}", "direct federation-level evidence"
    return "SHARED_BRIDGE_ONLY", "valid shared-skill evidence without a governed direct target-head label"


def build_event_taxonomy(rows: pd.DataFrame) -> pd.DataFrame:
    required = (
        "event_date", "event_name", "source_scope", "event_tier", "rating_context",
        "pool",
    )
    missing = set(required) - set(rows.columns)
    if missing:
        raise TaxonomyError(f"history missing columns: {sorted(missing)}")
    events = rows.loc[
        rows["pool"].astype(str).isin({"Boulder_Men", "Boulder_Women"}),
        list(required),
    ].drop_duplicates().copy()
    events["event_date"] = pd.to_datetime(events["event_date"], errors="raise").dt.date.astype(str)
    labels = [classify_direct_context(row) for _, row in events.iterrows()]
    events["direct_context_head"] = [label for label, _ in labels]
    events["classification_reason"] = [reason for _, reason in labels]
    events["all_results_update_shared_skill"] = True
    events["direct_label_controls_input_eligibility"] = False
    return events.sort_values(
        ["event_date", "source_scope", "event_name", "rating_context"], kind="stable"
    ).reset_index(drop=True)


def run(input_path: Path, event_output: Path, report_output: Path) -> dict:
    rows = pd.read_csv(
        input_path,
        compression="infer",
        low_memory=False,
        usecols=[
            "event_date", "event_name", "source_scope", "event_tier",
            "rating_context", "pool",
        ],
    )
    events = build_event_taxonomy(rows)
    counts = events.groupby("direct_context_head").size().sort_values(ascending=False)
    report = {
        "schema": SCHEMA,
        "status": "RESEARCH_TAXONOMY_ONLY",
        "input": {
            "path": str(input_path.relative_to(ROOT)),
            "sha256": sha256_file(input_path),
            "result_rows": int(len(rows)),
            "unique_event_contexts": int(len(events)),
        },
        "direct_event_context_counts": {key: int(value) for key, value in counts.items()},
        "invariants": {
            "all_results_update_shared_skill": bool(events["all_results_update_shared_skill"].all()),
            "direct_label_is_not_an_input_filter": bool((~events["direct_label_controls_input_eligibility"]).all()),
            "standalone_olympic_head_fitted": False,
            "ambiguous_region_auto_assigned": False,
        },
        "next_gate": (
            "Review unresolved context events, then compare fully pooled, isolated, "
            "fixed-offset, and partially pooled context-head models chronologically."
        ),
        "authority": {"model_fit": False, "rating_promotion": False, "deployment": False},
    }
    event_output.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(event_output, index=False, lineterminator="\n")
    with report_output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATA / "athlete_age_progression.csv.gz")
    parser.add_argument("--events", type=Path, default=DATA / "pathway_context_event_taxonomy_v1.csv")
    parser.add_argument("--report", type=Path, default=DATA / "pathway_context_taxonomy_audit_v1.json")
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.events, args.report), sort_keys=True))


if __name__ == "__main__":
    main()
