"""Audit governed physical identities against frozen exact-item evidence.

The governed identity bridge may resolve a source node that is absent from the
older identity-safe rating snapshot.  Such a resolution is useful for evidence
inventory, but it must not manufacture a pre-event rating state.  This audit
therefore overlays source-node identities only; exact frozen snapshots and the
strict ``item_calibration_eligible`` contract remain byte-for-byte unchanged.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.boulder_terrain_problem_adapter import (
    IdentityLookups,
    is_hidden_or_test_event,
    load_identity_lookups,
    normalize_round_row,
)


SCHEMA = "physical-item-identity-overlay-audit-v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_governed_overlay(
    links_path: Path, base: IdentityLookups
) -> tuple[IdentityLookups, set[str], dict[str, int]]:
    links = pd.read_csv(links_path, dtype=str).fillna("")
    required = {
        "source_scope",
        "athlete_source_id",
        "global_id",
        "bridge_status",
        "model_input_authorized",
        "canonical_identity_mutated",
    }
    if missing := required.difference(links.columns):
        raise ValueError(f"governed links missing {sorted(missing)}")
    if not links["bridge_status"].eq("governed_research_join_allowed").all():
        raise ValueError("non-governed identity leaked into overlay")
    if links["model_input_authorized"].str.casefold().isin({"true", "1"}).any():
        raise ValueError("identity bridge cannot authorize model input")
    if links["canonical_identity_mutated"].str.casefold().isin({"true", "1"}).any():
        raise ValueError("identity overlay cannot mutate canonical identities")

    mappings: dict[tuple[str, str], str] = {}
    for row in links.itertuples(index=False):
        key = (str(row.source_scope).upper(), str(row.athlete_source_id))
        global_id = str(row.global_id)
        if key in mappings and mappings[key] != global_id:
            raise ValueError(f"conflicting governed source-node mapping: {key}")
        mappings[key] = global_id

    conflicts = {
        key
        for key, global_id in mappings.items()
        if key in base.source_node_ids and base.source_node_ids[key] != global_id
    }
    merged = dict(base.source_node_ids)
    merged.update(mappings)
    overlay = IdentityLookups(
        exact_snapshots=base.exact_snapshots,
        source_node_ids=merged,
        ambiguous_source_nodes=frozenset(
            key for key in base.ambiguous_source_nodes if key not in mappings
        ),
    )
    diagnostics = {
        "governed_links": int(len(links)),
        "governed_source_nodes": int(len(mappings)),
        "nodes_already_in_frozen_lookup": int(
            sum(key in base.source_node_ids for key in mappings)
        ),
        "governed_overrides_of_old_node_assignment": int(len(conflicts)),
    }
    return overlay, set(links["global_id"].astype(str)), diagnostics


def _rows(path: Path) -> Iterable[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def audit_overlay(
    staged_paths: list[Path], identity_path: Path, governed_links_path: Path
) -> tuple[pd.DataFrame, dict[str, object]]:
    base = load_identity_lookups(identity_path)
    overlay, target_ids, overlay_counts = load_governed_overlay(
        governed_links_path, base
    )
    records: list[dict[str, object]] = []
    counts = {
        "target_boulder_problem_rows_base": 0,
        "target_boulder_problem_rows_overlay": 0,
        "marker_identity_eligible_base": 0,
        "marker_identity_eligible_overlay": 0,
        "item_calibration_eligible_base": 0,
        "item_calibration_eligible_overlay": 0,
        "newly_identity_resolved_marker_rows": 0,
        "newly_calibration_eligible_rows": 0,
        "identity_assignment_changed_rows": 0,
        "marker_identity_assignment_changed_rows": 0,
    }
    staged_boulder_nodes: set[tuple[str, str]] = set()
    staged_boulder_target_ids: set[str] = set()
    for path in staged_paths:
        for row in _rows(path):
            if str(row.get("discipline", "")).casefold() != "boulder":
                continue
            if str(row.get("event_date", "")) < "2021-01-01":
                continue
            if (
                str(row.get("source_scope", "")).upper() == "CEC"
                and is_hidden_or_test_event(row.get("event_name"))
            ):
                continue
            staged_boulder_nodes.add(
                (
                    str(row.get("source_scope", "")).upper(),
                    str(row.get("athlete_source_id", "")),
                )
            )
            base_evidence = normalize_round_row(row, base)
            overlay_evidence = normalize_round_row(row, overlay)
            if len(base_evidence) != len(overlay_evidence):
                raise ValueError("identity overlay changed problem-row cardinality")
            for before, after in zip(base_evidence, overlay_evidence, strict=True):
                before_target = before.athlete_id in target_ids
                after_target = after.athlete_id in target_ids
                counts["target_boulder_problem_rows_base"] += int(before_target)
                counts["target_boulder_problem_rows_overlay"] += int(after_target)
                counts["marker_identity_eligible_base"] += int(
                    before_target and before.marker_identity_eligible
                )
                counts["marker_identity_eligible_overlay"] += int(
                    after_target and after.marker_identity_eligible
                )
                counts["item_calibration_eligible_base"] += int(
                    before_target and before.item_calibration_eligible
                )
                counts["item_calibration_eligible_overlay"] += int(
                    after_target and after.item_calibration_eligible
                )
                newly_resolved = (
                    after_target
                    and after.marker_identity_eligible
                    and not (before_target and before.marker_identity_eligible)
                )
                newly_calibration = (
                    after_target
                    and after.item_calibration_eligible
                    and not (before_target and before.item_calibration_eligible)
                )
                counts["newly_identity_resolved_marker_rows"] += int(newly_resolved)
                counts["newly_calibration_eligible_rows"] += int(newly_calibration)
                identity_changed = before.athlete_id != after.athlete_id
                counts["identity_assignment_changed_rows"] += int(identity_changed)
                counts["marker_identity_assignment_changed_rows"] += int(
                    identity_changed and after.marker_identity_eligible
                )
                if newly_resolved or newly_calibration or identity_changed:
                    staged_boulder_target_ids.add(after.athlete_id)
                    records.append(
                        {
                            "prior_athlete_id": before.athlete_id,
                            "governed_athlete_id": after.athlete_id,
                            "source_scope": after.source_scope,
                            "source_event_id": after.source_event_id,
                            "competition_id": after.competition_id,
                            "problem_id": after.problem_id,
                            "event_date": after.event_date,
                            "identity_provenance": after.identity_provenance,
                            "marker_identity_eligible": after.marker_identity_eligible,
                            "item_calibration_eligible": after.item_calibration_eligible,
                            "identity_assignment_changed": identity_changed,
                            "withhold_reason": (
                                "missing_exact_frozen_pre_event_round_snapshot"
                                if not after.item_calibration_eligible
                                else ""
                            ),
                        }
                    )
    delta = pd.DataFrame(records)
    if not delta.empty:
        delta = delta.sort_values(
            ["event_date", "competition_id", "governed_athlete_id", "problem_id"],
            kind="stable",
        ).reset_index(drop=True)
    governed_nodes = {
        (str(row.source_scope).upper(), str(row.athlete_source_id))
        for row in pd.read_csv(governed_links_path, dtype=str).itertuples(index=False)
    }
    report: dict[str, object] = {
        "schema": SCHEMA,
        "status": "RESEARCH_GOVERNED_NODE_CONFLICTS_MONITORED_NO_EFFECTIVE_ITEM_CHANGE",
        "bindings": {
            "identity_safe_sha256": sha256(identity_path),
            "governed_links_sha256": sha256(governed_links_path),
            "staged_problem_evidence": {
                path.name: sha256(path) for path in staged_paths
            },
        },
        "coverage": {
            **overlay_counts,
            **counts,
            "governed_nodes_present_in_staged_boulder": int(
                len(governed_nodes.intersection(staged_boulder_nodes))
            ),
            "athletes_with_newly_resolved_marker_rows": int(
                len(staged_boulder_target_ids)
            ),
        },
        "claims": {
            "exact_frozen_snapshots_modified": False,
            "calibration_eligibility_weakened": False,
            "new_rating_state_imputed": False,
            "newly_resolved_rows_are_model_inputs": False,
            "identity_inventory_use_allowed": True,
            "model_fit_allowed": False,
        },
    }
    return delta, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity-safe", type=Path, required=True)
    parser.add_argument("--governed-links", type=Path, required=True)
    parser.add_argument("--staged", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--public-report", type=Path, required=True)
    args = parser.parse_args()
    delta, report = audit_overlay(
        list(args.staged), args.identity_safe, args.governed_links
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    delta_path = args.output_dir / "newly_identity_resolved_item_rows.csv"
    delta.to_csv(delta_path, index=False, lineterminator="\n")
    report["restricted_output"] = {
        "rows": int(len(delta)),
        "sha256": sha256(delta_path),
        "bytes": delta_path.stat().st_size,
    }
    args.public_report.parent.mkdir(parents=True, exist_ok=True)
    args.public_report.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
