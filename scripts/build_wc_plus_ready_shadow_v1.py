"""Build an all-athlete WC+ readiness shadow from the verified V4 graph.

Direct WC+ demonstrated level and projected WC+ readiness are intentionally
separate.  Zero direct starts never causes a projection to be blank: only a
structurally unanchored graph state is not estimable.  Absolute probability
intervals remain withheld because the predecessor's covariance calibration did
not pass its release gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


SCHEMA = "wc-plus-ready-shadow-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def support_band(row: pd.Series) -> str:
    direct = int(row["direct_wc_starts"])
    weight = float(row["anchored_effective_weight"])
    opponents = int(row["unique_anchored_opponents"])
    if direct >= 2:
        return "direct WC+ evidence"
    if direct == 1:
        return "one direct WC+ start plus graph evidence"
    if weight >= 10.0 and opponents >= 20:
        return "substantial connected indirect evidence"
    if weight >= 2.0 and opponents >= 8:
        return "moderate connected indirect evidence"
    if weight > 0.0 and opponents > 0:
        return "limited connected indirect evidence"
    return "not estimable — no anchored graph connection"


def build_shadow(
    snapshot: pd.DataFrame,
    replay: pd.DataFrame,
    direct: pd.DataFrame,
    initializer: pd.DataFrame,
) -> pd.DataFrame:
    latest = (
        replay.sort_values(["event_index", "event_id"], kind="stable")
        .groupby(["pool", "athlete_id"], as_index=False)
        .tail(1)
        .rename(columns={"athlete_id": "global_id"})
    )
    latest = latest[
        [
            "pool",
            "global_id",
            "anchored_events",
            "anchored_comparisons",
            "unique_anchored_opponents",
            "anchored_effective_weight",
        ]
    ]
    initial = initializer.rename(columns={"athlete_id": "global_id"})[
        [
            "pool",
            "global_id",
            "component_anchored",
            "effective_competitions",
            "unique_opponents",
        ]
    ]
    direct_counts = (
        direct.groupby(["pool", "athlete_id"], as_index=False)["event_id"]
        .nunique()
        .rename(columns={"athlete_id": "global_id", "event_id": "direct_wc_starts"})
    )
    youth_direct_counts = (
        replay.loc[replay["target_domain"].astype(str).eq("ifsc_youth_world")]
        .groupby(["pool", "athlete_id"], as_index=False)["event_id"]
        .nunique()
        .rename(
            columns={
                "athlete_id": "global_id",
                "event_id": "direct_youth_ifsc_events",
            }
        )
    )
    output = snapshot.merge(latest, on=["pool", "global_id"], how="left")
    output = output.merge(initial, on=["pool", "global_id"], how="left")
    output = output.merge(direct_counts, on=["pool", "global_id"], how="left")
    output = output.merge(
        youth_direct_counts, on=["pool", "global_id"], how="left"
    )
    for column in (
        "anchored_events",
        "anchored_comparisons",
        "unique_anchored_opponents",
        "anchored_effective_weight",
        "direct_wc_starts",
        "direct_youth_ifsc_events",
    ):
        output[column] = pd.to_numeric(output[column], errors="coerce").fillna(0)

    # Athletes without a post-2024 replay can still have a valid anchored
    # initializer state.  Carry its continuous support rather than inventing a
    # starts threshold.
    no_replay = output["anchored_effective_weight"].eq(0)
    anchored_initial = output["component_anchored"].fillna(False).astype(bool)
    output.loc[no_replay & anchored_initial, "anchored_effective_weight"] = pd.to_numeric(
        output.loc[no_replay & anchored_initial, "effective_competitions"],
        errors="coerce",
    ).fillna(0)
    output.loc[no_replay & anchored_initial, "unique_anchored_opponents"] = pd.to_numeric(
        output.loc[no_replay & anchored_initial, "unique_opponents"], errors="coerce"
    ).fillna(0)

    connected = (
        output["anchored_effective_weight"].gt(0)
        & output["unique_anchored_opponents"].gt(0)
    )
    output["wc_plus_ready"] = pd.to_numeric(
        output["internal_graph_wc_central"], errors="coerce"
    ).where(connected)
    # The predecessor pooled every Youth World age category within sex. Keep
    # this coordinate as an internal sensitivity, but do not present it as the
    # athlete's current-category YW-IFSC Ready value.
    output["yw_ifsc_pooled_sensitivity"] = (
        pd.to_numeric(output["stable_skill_mean"], errors="coerce")
        + pd.to_numeric(output["form_at_snapshot"], errors="coerce")
        + pd.to_numeric(output["youth_world_offset_at_snapshot"], errors="coerce")
    ).where(connected)
    output["yw_ifsc_ready"] = np.nan
    output["reg_ifsc_ready"] = np.nan
    output["wc_plus_ready_status"] = np.select(
        [
            connected & output["direct_wc_starts"].gt(0),
            connected & output["direct_wc_starts"].eq(0),
        ],
        [
            "projected readiness with direct WC+ evidence",
            "projected readiness from connected evidence; no direct WC+ start",
        ],
        default="not estimable — no anchored graph connection",
    )
    output["wc_plus_ready_evidence"] = output.apply(support_band, axis=1)
    output["yw_ifsc_ready_status"] = np.where(
        connected,
        "withheld - predecessor pools Youth World age categories; category-aware replay required",
        "not estimable — no anchored graph connection",
    )
    output["reg_ifsc_ready_status"] = (
        "withheld - predecessor mixes youth and senior/open regional IFSC offsets"
    )
    output["yw_ifsc_demonstrated_eligible"] = output[
        "direct_youth_ifsc_events"
    ].gt(0)
    output["wc_plus_demonstrated_eligible"] = output["direct_wc_starts"].gt(0)
    output["numeric_interval_status"] = (
        "withheld — absolute covariance calibration gate failed"
    )
    output["production_status"] = "research shadow; not authorized for app"
    columns = [
        "pool",
        "global_id",
        "snapshot_at_utc",
        "wc_plus_ready",
        "wc_plus_ready_status",
        "wc_plus_ready_evidence",
        "yw_ifsc_ready",
        "yw_ifsc_ready_status",
        "yw_ifsc_pooled_sensitivity",
        "reg_ifsc_ready",
        "reg_ifsc_ready_status",
        "direct_youth_ifsc_events",
        "yw_ifsc_demonstrated_eligible",
        "direct_wc_starts",
        "wc_plus_demonstrated_eligible",
        "anchored_events",
        "anchored_comparisons",
        "unique_anchored_opponents",
        "anchored_effective_weight",
        "internal_graph_wc_state_sd",
        "internal_graph_wc_predictive_sd",
        "numeric_interval_status",
        "production_status",
    ]
    return output[columns].sort_values(["pool", "global_id"], kind="stable").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v4-root", type=Path, required=True)
    parser.add_argument("--initializer-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    v4 = args.v4_root.resolve()
    initializer_root = args.initializer_root.resolve()
    manifest_path = v4 / "manifest.json"
    verification_path = v4 / "verification_receipt.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "RESEARCH_ONLY_NOT_FOR_APP_INTEGRATION":
        raise ValueError("unexpected V4 status")
    if not verification.get("independent_full_recomputation"):
        raise ValueError("V4 independent recomputation did not pass")
    paths = {
        "snapshot": v4 / "snapshot_full_states.v4.parquet",
        "replay": v4 / "replay_history.v4.parquet",
        "direct": v4 / "direct_wc_history.v4.parquet",
        "initializer": initializer_root / "states.parquet",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    shadow = build_shadow(
        pd.read_parquet(paths["snapshot"]),
        pd.read_parquet(paths["replay"]),
        pd.read_parquet(paths["direct"]),
        pd.read_parquet(paths["initializer"]),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "wc_plus_ready_shadow.parquet"
    shadow.to_parquet(output_path, index=False)
    receipt = {
        "schema": SCHEMA,
        "status": "RESEARCH_SHADOW_NOT_AUTHORIZED_FOR_APP",
        "labels": {
            "overall": "Overall",
            "youth_world_demonstrated": "YW-IFSC",
            "senior_regional_ifsc_demonstrated": "REG-IFSC",
            "youth_world_projected": "YW-IFSC Ready",
            "senior_regional_ifsc_projected": "REG-IFSC Ready",
            "wc_plus_demonstrated": "WC+",
            "wc_plus_projected": "WC+ Ready",
        },
        "semantics": {
            "yw_ifsc": "demonstrated Youth World level in an event-date governed youth category",
            "reg_ifsc": "demonstrated senior/open regional or continental IFSC level; youth evidence enters indirectly through the shared graph",
            "yw_ifsc_ready": "projected Youth World reference-field level for the athlete's event-date category; currently withheld pending category-aware replay",
            "reg_ifsc_ready": "projected senior/open regional IFSC level; currently withheld because the predecessor mixed youth and senior/open regional offsets",
            "wc_plus": "demonstrated level from direct senior WC+ evidence",
            "wc_plus_ready": "projected senior WC+ level from all connected evidence",
            "zero_direct_starts_is_not_a_withhold_reason": True,
            "not_estimable_requires_no_anchored_graph_connection": True,
            "absolute_numeric_intervals_withheld": True,
        },
        "inputs": {
            "v4_manifest": sha256(manifest_path),
            "v4_verification": sha256(verification_path),
            **{name: sha256(path) for name, path in paths.items()},
        },
        "output": {
            "rows": len(shadow),
            "estimated_rows": int(shadow["wc_plus_ready"].notna().sum()),
            "no_direct_start_estimated_rows": int(
                (shadow["wc_plus_ready"].notna() & shadow["direct_wc_starts"].eq(0)).sum()
            ),
            "yw_ifsc_pooled_sensitivity_rows": int(
                shadow["yw_ifsc_pooled_sensitivity"].notna().sum()
            ),
            "sha256": sha256(output_path),
        },
        "authority": {"production": False, "app": False, "deployment": False},
    }
    (args.output_dir / "receipt.json").write_text(
        canonical_json(receipt) + "\n", encoding="ascii"
    )
    print(canonical_json(receipt))


if __name__ == "__main__":
    main()
