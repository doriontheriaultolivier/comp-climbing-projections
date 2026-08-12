"""Deterministic Boulder-first analysis targets for verified 2026 broadcasts.

The ledger produced here is a *relative competition schedule*, not a claim
about broadcast timestamps.  It combines exact official start order with an
explicit round-format model so later clock/graphics/pose detection has stable
athlete-boulder targets to align to the video.
"""

from __future__ import annotations

import hashlib
import gzip
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


PIPELINE_VERSION = "ifsc-boulder-athlete-window-plan-v1"
TARGET_EVENT_IDS = (1479, 1480, 1482)
TARGET_ROUNDS = ("Semi-final", "Final")
BOULDER_COUNT = 4
ANALYSIS_WINDOW_SECONDS = 300
SEMIFINAL_TURN_SECONDS = 300
SEMIFINAL_REST_SECONDS = 300
FINAL_TURN_SECONDS = 240
FORMAT_RULE_SOURCE_URL = (
    "https://images.ifsc-climbing.org/ifsc/image/private/t_q_good/prd/"
    "brsjsyh2ea4fxinl8nsa.pdf"
)
DEFAULT_CONFIRMED_TECHNICAL_DELAY_SECONDS = 300
CONFIRMED_TECHNICAL_STATUS = "confirmed_official_or_broadcast_evidence"
TECHNICAL_PIPELINE_VERSION = "ifsc-boulder-confirmed-technicals-v1"
TECHNICAL_EVIDENCE_SOURCE_TYPES = {
    "official_result_note",
    "official_communication",
    "broadcast_graphic",
    "broadcast_commentary",
    "broadcast_graphic_and_commentary",
}
TECHNICAL_INCIDENT_COLUMNS = (
    "pipeline_version",
    "technical_incident_id",
    "category_round_id",
    "video_id",
    "incident_segment_id",
    "incident_simultaneous_group_id",
    "athlete_source_id",
    "boulder_number",
    "evidence_start_seconds",
    "evidence_end_seconds",
    "evidence_status",
    "evidence_source_type",
    "evidence_source_url",
    "evidence_sha256",
    "evidence_note",
    "scheduled_delay_seconds",
    "delay_applies_from_logical_slot",
    "observed_at_utc",
)


def sha256_canonical_text_file(path: Path) -> str:
    """Hash UTF-8 CSV/JSON text independently of LF/CRLF and gzip headers."""

    if path.suffix.lower() == ".gz":
        with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
            text = handle.read()
    else:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            text = handle.read()
    canonical = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Backward-compatible name for canonical text-ledger hashing."""

    return sha256_canonical_text_file(path)


def _required_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing: {', '.join(sorted(missing))}")


def _positive_int(value: object, label: str) -> int:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number) or int(number) <= 0 or float(number) != int(number):
        raise ValueError(f"{label} must be a positive integer")
    return int(number)


def _target_sources(manifest: pd.DataFrame) -> pd.DataFrame:
    _required_columns(
        manifest,
        {
            "event_id",
            "category_round_id",
            "event",
            "event_date",
            "discipline",
            "gender",
            "round",
            "video_id",
            "official_title",
            "official_youtube_url",
            "duration_seconds",
            "duration_basis",
            "duration_video_details_sha256",
            "official_round_results_api_url",
        },
        "video manifest",
    )
    selected = manifest.loc[
        pd.to_numeric(manifest["event_id"], errors="coerce").isin(TARGET_EVENT_IDS)
        & manifest["discipline"].astype(str).eq("Boulder")
        & manifest["round"].astype(str).isin(TARGET_ROUNDS)
    ].copy()
    if len(selected) != 12:
        raise ValueError(
            "Boulder-first plan requires exactly 12 Madrid, Prague and Innsbruck "
            "semi-final/final broadcasts"
        )
    key = ["event_id", "gender", "round"]
    if selected.duplicated(key).any():
        raise ValueError("target video scopes must be unique by event, gender and round")
    expected = {
        (event_id, gender, round_name)
        for event_id in TARGET_EVENT_IDS
        for gender in ("Men", "Women")
        for round_name in TARGET_ROUNDS
    }
    observed = {
        (int(row.event_id), str(row.gender), str(row.round))
        for row in selected.itertuples(index=False)
    }
    if observed != expected:
        raise ValueError("target video manifest does not cover all required Boulder scopes")
    return selected


def _round_schedule(
    round_name: str,
    athlete_count: int,
    start_order: int,
    boulder_number: int,
) -> tuple[int, int, int, int, str, str, str, str]:
    """Return slot, turn start/end, analysis end and declared format labels."""

    if round_name == "Semi-final":
        # One athlete enters Boulder 1 every five minutes. Athletes rest for
        # one five-minute rotation before their next boulder. At steady state,
        # as many as four athletes are climbing concurrently.
        slot = start_order + 2 * (boulder_number - 1)
        turn_seconds = SEMIFINAL_TURN_SECONDS
        format_model = "semi_final_circuit_5min_climb_5min_rest"
        timing_basis = "declared_2026_boulder_circuit_rule"
        relative_timing_status = "deterministic_fixed_rotation_schedule"
        turn_end_trigger = "five_minute_buzzer"
    elif round_name == "Final":
        # Finals are boulder-major: the field attempts one boulder before the
        # next boulder begins. The competitive turn is four minutes. The
        # analysis target deliberately keeps five minutes to retain entrances,
        # exits and broadcast cuts around that turn.
        slot = (boulder_number - 1) * athlete_count + start_order
        turn_seconds = FINAL_TURN_SECONDS
        format_model = "final_boulder_major_4min_turn"
        timing_basis = "declared_2026_boulder_final_rule_plus_5min_analysis_context"
        relative_timing_status = "nominal_maximum_turn_schedule_requires_clock_alignment"
        turn_end_trigger = "top_or_four_minute_buzzer"
    else:
        raise ValueError(f"unsupported Boulder round: {round_name}")
    turn_start = (slot - 1) * turn_seconds
    turn_end = turn_start + turn_seconds
    analysis_end = turn_start + ANALYSIS_WINDOW_SECONDS
    return (
        slot,
        turn_start,
        turn_end,
        analysis_end,
        format_model,
        timing_basis,
        relative_timing_status,
        turn_end_trigger,
    )


def build_boulder_window_plan(
    manifest: pd.DataFrame,
    start_orders: pd.DataFrame,
    *,
    manifest_sha256: str,
    start_order_sha256: str,
) -> pd.DataFrame:
    """Build one stable target per athlete and boulder for the 12 priority scopes."""

    sources = _target_sources(manifest)
    _required_columns(
        start_orders,
        {
            "event_id",
            "category_round_id",
            "event",
            "discipline",
            "gender",
            "round",
            "athlete_source_id",
            "athlete_name",
            "country",
            "start_order_status",
            "exact_start_order",
            "official_endpoint_url",
            "response_sha256_canonical_json",
            "observed_at_utc",
        },
        "start-order enrichment",
    )
    target_round_ids = set(
        pd.to_numeric(sources["category_round_id"], errors="raise").astype(int)
    )
    starters = start_orders.loc[
        pd.to_numeric(start_orders["category_round_id"], errors="coerce").isin(
            target_round_ids
        )
    ].copy()
    if not starters["start_order_status"].astype(str).eq(
        "official_exact_start_order"
    ).all():
        raise ValueError("every Boulder target requires exact official scalar start order")

    rows: list[dict[str, object]] = []
    source_by_round = {
        int(row.category_round_id): row for row in sources.itertuples(index=False)
    }
    for category_round_id in sorted(target_round_ids):
        source = source_by_round[category_round_id]
        scope = starters.loc[
            pd.to_numeric(starters["category_round_id"], errors="coerce").eq(
                category_round_id
            )
        ].copy()
        athlete_count = len(scope)
        expected_count = 24 if str(source.round) == "Semi-final" else 8
        if athlete_count != expected_count:
            raise ValueError(
                f"{category_round_id}: expected {expected_count} exact starters, "
                f"found {athlete_count}"
            )
        scope["exact_order_value"] = pd.to_numeric(
            scope["exact_start_order"], errors="raise"
        ).astype(int)
        if set(scope["exact_order_value"]) != set(range(1, athlete_count + 1)):
            raise ValueError(f"{category_round_id}: exact start order is not contiguous")
        scope = scope.sort_values("exact_order_value", kind="stable")
        for athlete in scope.itertuples(index=False):
            athlete_id = _positive_int(
                athlete.athlete_source_id,
                f"{category_round_id} athlete_source_id",
            )
            start_order = int(athlete.exact_order_value)
            for boulder_number in range(1, BOULDER_COUNT + 1):
                (
                    slot,
                    turn_start,
                    turn_end,
                    analysis_end,
                    format_model,
                    timing_basis,
                    relative_timing_status,
                    turn_end_trigger,
                ) = _round_schedule(
                    str(source.round), athlete_count, start_order, boulder_number
                )
                segment_id = (
                    f"B26-CR{category_round_id}-A{athlete_id}-B{boulder_number:02d}"
                )
                group_id = f"B26-CR{category_round_id}-T{slot:03d}"
                rows.append(
                    {
                        "pipeline_version": PIPELINE_VERSION,
                        "segment_id": segment_id,
                        "simultaneous_group_id": group_id,
                        "event_id": int(source.event_id),
                        "category_round_id": category_round_id,
                        "event": str(source.event),
                        "event_date": str(source.event_date),
                        "discipline": "Boulder",
                        "gender": str(source.gender),
                        "round": str(source.round),
                        "video_id": str(source.video_id),
                        "official_title": str(source.official_title),
                        "official_youtube_url": str(source.official_youtube_url),
                        "broadcast_duration_seconds": int(source.duration_seconds),
                        "athlete_source_id": athlete_id,
                        "athlete_name": str(athlete.athlete_name),
                        "country": str(athlete.country),
                        "official_exact_start_order": start_order,
                        "boulder_number": boulder_number,
                        "logical_climbing_slot": slot,
                        "relative_modeled_turn_start_seconds": turn_start,
                        "relative_modeled_turn_end_seconds": turn_end,
                        "modeled_competition_turn_seconds": turn_end - turn_start,
                        "analysis_target_start_seconds": turn_start,
                        "analysis_target_end_seconds": analysis_end,
                        "analysis_target_seconds": ANALYSIS_WINDOW_SECONDS,
                        "format_model": format_model,
                        "timing_rule_basis": timing_basis,
                        "relative_timing_status": relative_timing_status,
                        "turn_end_trigger": turn_end_trigger,
                        "format_rule_source_url": FORMAT_RULE_SOURCE_URL,
                        "schedule_identity_status": "official_exact_start_order",
                        "schedule_identity_confidence": 1.0,
                        "broadcast_alignment_status": (
                            "unresolved_requires_clock_graphics_or_pose_detection"
                        ),
                        "broadcast_start_seconds": "",
                        "broadcast_end_seconds": "",
                        "broadcast_alignment_confidence": "",
                        "timing_claim_scope": (
                            "relative_format_plan_only_not_observed_competition_or_broadcast_timestamp"
                        ),
                        "multiple_athlete_tracking_required": (
                            str(source.round) == "Semi-final"
                        ),
                        "pose_tracking_scope": (
                            "multi_person_across_four_simultaneous_boulders"
                            if str(source.round) == "Semi-final"
                            else "single_active_climber_expected_but_detect_all_people"
                        ),
                        "official_round_results_api_url": str(
                            source.official_round_results_api_url
                        ),
                        "official_start_order_endpoint_url": str(
                            athlete.official_endpoint_url
                        ),
                        "official_start_order_response_sha256": str(
                            athlete.response_sha256_canonical_json
                        ),
                        "official_start_order_observed_at_utc": str(
                            athlete.observed_at_utc
                        ),
                        "video_duration_basis": str(source.duration_basis),
                        "video_duration_metadata_sha256": str(
                            source.duration_video_details_sha256
                        ),
                        "source_manifest_sha256": manifest_sha256,
                        "start_order_ledger_sha256": start_order_sha256,
                    }
                )
    frame = pd.DataFrame(rows)
    group_members = (
        frame.groupby("simultaneous_group_id", sort=False)["segment_id"]
        .agg(list)
        .to_dict()
    )
    frame["simultaneous_group_size"] = frame["simultaneous_group_id"].map(
        lambda value: len(group_members[value])
    )
    frame["simultaneous_segment_ids"] = frame["simultaneous_group_id"].map(
        lambda value: ";".join(group_members[value])
    )
    return frame.sort_values(
        [
            "event_date",
            "event_id",
            "gender",
            "round",
            "logical_climbing_slot",
            "boulder_number",
            "official_exact_start_order",
        ],
        kind="stable",
    ).reset_index(drop=True)


def empty_technical_incidents() -> pd.DataFrame:
    """Return the governed empty schema; absence of evidence means no technical."""

    return pd.DataFrame(columns=list(TECHNICAL_INCIDENT_COLUMNS))


def normalize_confirmed_technical_incidents(
    incidents: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    """Validate explicit technical evidence and resolve its schedule position.

    A broadcast gap, clock discontinuity or long pause is not sufficient. The
    caller must supply an affirmative official/broadcast observation with a
    bounded evidence interval and content hash.
    """

    _required_columns(incidents, set(TECHNICAL_INCIDENT_COLUMNS), "technical ledger")
    if incidents.empty:
        normalized = incidents.copy()
        normalized["resolved_incident_logical_slot"] = pd.Series(dtype="int64")
        normalized["resolved_scheduled_delay_seconds"] = pd.Series(dtype="int64")
        normalized["resolved_delay_applies_from_logical_slot"] = pd.Series(dtype="int64")
        return normalized
    if incidents["technical_incident_id"].astype(str).eq("").any():
        raise ValueError("technical incident ID cannot be blank")
    if incidents["technical_incident_id"].astype(str).duplicated().any():
        raise ValueError("technical incident IDs must be unique")
    if not incidents["pipeline_version"].astype(str).eq(TECHNICAL_PIPELINE_VERSION).all():
        raise ValueError("technical incident pipeline version is unsupported")
    if not incidents["evidence_status"].astype(str).eq(CONFIRMED_TECHNICAL_STATUS).all():
        raise ValueError("a technical requires affirmative official or broadcast evidence")
    forbidden_sources = incidents["evidence_source_type"].astype(str).str.lower().isin(
        {"gap", "gap_only", "timing_gap_inference", "schedule_gap"}
    )
    if forbidden_sources.any():
        raise ValueError("a timing gap alone cannot confirm a technical")
    if not incidents["evidence_source_type"].astype(str).isin(
        TECHNICAL_EVIDENCE_SOURCE_TYPES
    ).all():
        raise ValueError("technical evidence source must be affirmative and governed")

    target_by_id = targets.set_index("segment_id", drop=False)
    normalized_rows: list[dict[str, object]] = []
    for raw in incidents.to_dict("records"):
        incident_id = str(raw["technical_incident_id"])
        segment_id = str(raw["incident_segment_id"])
        if segment_id not in target_by_id.index:
            raise ValueError(f"{incident_id}: incident segment is not in the target ledger")
        target = target_by_id.loc[segment_id]
        if isinstance(target, pd.DataFrame):
            raise ValueError(f"{incident_id}: incident segment is not unique")
        identity_checks = {
            "category_round_id": int(target["category_round_id"]),
            "athlete_source_id": int(target["athlete_source_id"]),
            "boulder_number": int(target["boulder_number"]),
        }
        for field, expected in identity_checks.items():
            if _positive_int(raw[field], f"{incident_id} {field}") != expected:
                raise ValueError(f"{incident_id}: {field} does not match incident segment")
        if str(raw["video_id"]) != str(target["video_id"]):
            raise ValueError(f"{incident_id}: video ID does not match incident segment")
        if str(raw["incident_simultaneous_group_id"]) != str(
            target["simultaneous_group_id"]
        ):
            raise ValueError(f"{incident_id}: simultaneous group does not match segment")

        evidence_start = pd.to_numeric(
            pd.Series([raw["evidence_start_seconds"]]), errors="coerce"
        ).iloc[0]
        evidence_end = pd.to_numeric(
            pd.Series([raw["evidence_end_seconds"]]), errors="coerce"
        ).iloc[0]
        if (
            pd.isna(evidence_start)
            or pd.isna(evidence_end)
            or float(evidence_start) < 0
            or float(evidence_end) <= float(evidence_start)
            or float(evidence_end) > float(target["broadcast_duration_seconds"])
        ):
            raise ValueError(f"{incident_id}: evidence interval is invalid")
        if not str(raw["evidence_source_url"]).strip():
            raise ValueError(f"{incident_id}: evidence source URL is required")
        if not str(raw["evidence_note"]).strip():
            raise ValueError(f"{incident_id}: evidence note is required")
        if not pd.Series([str(raw["evidence_sha256"])]).str.fullmatch(
            r"[0-9a-f]{64}"
        ).iloc[0]:
            raise ValueError(f"{incident_id}: evidence SHA256 is invalid")
        if not str(raw["observed_at_utc"]).strip():
            raise ValueError(f"{incident_id}: observation timestamp is required")

        delay_raw = str(raw["scheduled_delay_seconds"]).strip()
        delay = (
            DEFAULT_CONFIRMED_TECHNICAL_DELAY_SECONDS
            if delay_raw == ""
            else _positive_int(delay_raw, f"{incident_id} scheduled delay")
        )
        incident_slot = int(target["logical_climbing_slot"])
        applies_raw = str(raw["delay_applies_from_logical_slot"]).strip()
        applies_from = (
            incident_slot + 1
            if applies_raw == ""
            else _positive_int(applies_raw, f"{incident_id} delay start slot")
        )
        if applies_from <= incident_slot:
            raise ValueError(
                f"{incident_id}: downstream delay must begin after the incident slot"
            )
        normalized_rows.append(
            {
                **raw,
                "evidence_start_seconds": float(evidence_start),
                "evidence_end_seconds": float(evidence_end),
                "resolved_incident_logical_slot": incident_slot,
                "resolved_scheduled_delay_seconds": delay,
                "resolved_delay_applies_from_logical_slot": applies_from,
            }
        )
    return pd.DataFrame(normalized_rows)


def apply_confirmed_technical_delays(
    targets: pd.DataFrame,
    incidents: pd.DataFrame,
    *,
    technical_ledger_sha256: str = "",
) -> pd.DataFrame:
    """Attach confirmed technical flags and cumulative downstream offsets."""

    normalized = normalize_confirmed_technical_incidents(incidents, targets)
    if technical_ledger_sha256 and not pd.Series([technical_ledger_sha256]).str.fullmatch(
        r"[0-9a-f]{64}"
    ).iloc[0]:
        raise ValueError("technical incident ledger SHA256 is invalid")
    output = targets.copy()
    output["technical_incident_ledger_sha256"] = technical_ledger_sha256
    output["technical_flag"] = False
    output["technical_incident_ids"] = ""
    output["technical_evidence_intervals_json"] = "[]"
    output["technical_evidence_provenance_json"] = "[]"
    output["technical_scheduled_delay_seconds"] = 0
    output["cumulative_confirmed_technical_delay_seconds"] = 0

    for incident in normalized.to_dict("records"):
        segment_mask = output["segment_id"].astype(str).eq(
            str(incident["incident_segment_id"])
        )
        segment_index = output.index[segment_mask]
        if len(segment_index) != 1:
            raise ValueError("confirmed technical segment must resolve exactly once")
        index = segment_index[0]
        output.at[index, "technical_flag"] = True
        existing_ids = str(output.at[index, "technical_incident_ids"])
        output.at[index, "technical_incident_ids"] = ";".join(
            value
            for value in (existing_ids, str(incident["technical_incident_id"]))
            if value
        )
        intervals = json.loads(str(output.at[index, "technical_evidence_intervals_json"]))
        intervals.append(
            {
                "start_seconds": incident["evidence_start_seconds"],
                "end_seconds": incident["evidence_end_seconds"],
            }
        )
        output.at[index, "technical_evidence_intervals_json"] = json.dumps(
            intervals,
            sort_keys=True,
            separators=(",", ":"),
        )
        provenance = json.loads(
            str(output.at[index, "technical_evidence_provenance_json"])
        )
        provenance.append(
            {
                "evidence_source_type": str(incident["evidence_source_type"]),
                "evidence_source_url": str(incident["evidence_source_url"]),
                "evidence_sha256": str(incident["evidence_sha256"]),
                "evidence_note": str(incident["evidence_note"]),
                "observed_at_utc": str(incident["observed_at_utc"]),
            }
        )
        output.at[index, "technical_evidence_provenance_json"] = json.dumps(
            provenance,
            sort_keys=True,
            separators=(",", ":"),
        )
        output.at[index, "technical_scheduled_delay_seconds"] = int(
            output.at[index, "technical_scheduled_delay_seconds"]
        ) + int(
            incident["resolved_scheduled_delay_seconds"]
        )
        downstream = (
            pd.to_numeric(output["category_round_id"], errors="coerce").eq(
                int(incident["category_round_id"])
            )
            & pd.to_numeric(output["logical_climbing_slot"], errors="coerce").ge(
                int(incident["resolved_delay_applies_from_logical_slot"])
            )
        )
        output.loc[
            downstream, "cumulative_confirmed_technical_delay_seconds"
        ] += int(incident["resolved_scheduled_delay_seconds"])

    cumulative = pd.to_numeric(
        output["cumulative_confirmed_technical_delay_seconds"], errors="raise"
    ).astype(int)
    output["adjusted_relative_modeled_turn_start_seconds"] = pd.to_numeric(
        output["relative_modeled_turn_start_seconds"], errors="raise"
    ).astype(int) + cumulative
    output["adjusted_relative_modeled_turn_end_seconds"] = pd.to_numeric(
        output["relative_modeled_turn_end_seconds"], errors="raise"
    ).astype(int) + cumulative
    output["adjusted_analysis_target_start_seconds"] = pd.to_numeric(
        output["analysis_target_start_seconds"], errors="raise"
    ).astype(int) + cumulative
    output["adjusted_analysis_target_end_seconds"] = pd.to_numeric(
        output["analysis_target_end_seconds"], errors="raise"
    ).astype(int) + cumulative
    output["technical_adjustment_status"] = "no_confirmed_technical_evidence"
    delayed = cumulative.gt(0)
    output.loc[
        delayed, "technical_adjustment_status"
    ] = "downstream_confirmed_technical_delay_applied"
    output.loc[
        output["technical_flag"].astype(bool), "technical_adjustment_status"
    ] = "confirmed_technical_target"
    output.loc[
        output["technical_flag"].astype(bool) & delayed,
        "technical_adjustment_status",
    ] = "confirmed_technical_target_with_prior_delay"
    return output


def build_simultaneous_group_plan(targets: pd.DataFrame) -> pd.DataFrame:
    """Collapse athlete-boulder targets into one row per relative time slot."""

    rows: list[dict[str, object]] = []
    for group_id, group in targets.groupby("simultaneous_group_id", sort=False):
        ordered = group.sort_values(
            ["boulder_number", "official_exact_start_order"], kind="stable"
        )
        first = ordered.iloc[0]
        cumulative_values = set(
            pd.to_numeric(
                ordered.get(
                    "cumulative_confirmed_technical_delay_seconds",
                    pd.Series([0] * len(ordered), index=ordered.index),
                ),
                errors="raise",
            ).astype(int)
        )
        if len(cumulative_values) != 1:
            raise ValueError(f"{group_id}: one time group cannot have different offsets")
        technical_mask = ordered.get(
            "technical_flag", pd.Series(False, index=ordered.index)
        ).astype(bool)
        technical_ids = [
            incident_id
            for value in ordered.loc[technical_mask].get(
                "technical_incident_ids", pd.Series(dtype=str)
            )
            for incident_id in str(value).split(";")
            if incident_id
        ]
        technical_intervals: list[dict[str, object]] = []
        technical_provenance: list[dict[str, object]] = []
        for value in ordered.loc[technical_mask].get(
            "technical_evidence_intervals_json", pd.Series(dtype=str)
        ):
            technical_intervals.extend(json.loads(str(value)))
        for value in ordered.loc[technical_mask].get(
            "technical_evidence_provenance_json", pd.Series(dtype=str)
        ):
            technical_provenance.extend(json.loads(str(value)))
        rows.append(
            {
                "pipeline_version": PIPELINE_VERSION,
                "simultaneous_group_id": group_id,
                "event_id": int(first["event_id"]),
                "category_round_id": int(first["category_round_id"]),
                "event": first["event"],
                "gender": first["gender"],
                "round": first["round"],
                "video_id": first["video_id"],
                "logical_climbing_slot": int(first["logical_climbing_slot"]),
                "relative_modeled_turn_start_seconds": int(
                    first["relative_modeled_turn_start_seconds"]
                ),
                "relative_modeled_turn_end_seconds": int(
                    first["relative_modeled_turn_end_seconds"]
                ),
                "cumulative_confirmed_technical_delay_seconds": int(
                    first.get("cumulative_confirmed_technical_delay_seconds", 0)
                ),
                "adjusted_relative_modeled_turn_start_seconds": int(
                    first.get(
                        "adjusted_relative_modeled_turn_start_seconds",
                        first["relative_modeled_turn_start_seconds"],
                    )
                ),
                "adjusted_relative_modeled_turn_end_seconds": int(
                    first.get(
                        "adjusted_relative_modeled_turn_end_seconds",
                        first["relative_modeled_turn_end_seconds"],
                    )
                ),
                "simultaneous_group_size": len(ordered),
                "technical_flag": bool(technical_mask.any()),
                "technical_incident_ids": ";".join(technical_ids),
                "technical_evidence_intervals_json": json.dumps(
                    technical_intervals, sort_keys=True, separators=(",", ":")
                ),
                "technical_evidence_provenance_json": json.dumps(
                    technical_provenance, sort_keys=True, separators=(",", ":")
                ),
                "technical_scheduled_delay_seconds": int(
                    pd.to_numeric(
                        ordered.get(
                            "technical_scheduled_delay_seconds",
                            pd.Series([0] * len(ordered), index=ordered.index),
                        ),
                        errors="raise",
                    ).sum()
                ),
                "active_boulder_numbers": ";".join(
                    str(int(value)) for value in ordered["boulder_number"]
                ),
                "athlete_source_ids": ";".join(
                    str(int(value)) for value in ordered["athlete_source_id"]
                ),
                "athlete_names": ";".join(ordered["athlete_name"].astype(str)),
                "segment_ids": ";".join(ordered["segment_id"].astype(str)),
                "broadcast_alignment_status": first["broadcast_alignment_status"],
                "timing_claim_scope": first["timing_claim_scope"],
                "multiple_athlete_tracking_required": bool(
                    first["multiple_athlete_tracking_required"]
                ),
                "source_manifest_sha256": first["source_manifest_sha256"],
                "start_order_ledger_sha256": first["start_order_ledger_sha256"],
                "technical_incident_ledger_sha256": first.get(
                    "technical_incident_ledger_sha256", ""
                ),
            }
        )
    return pd.DataFrame(rows)


def validate_boulder_window_plan(
    targets: pd.DataFrame, groups: pd.DataFrame
) -> list[str]:
    errors: list[str] = []
    if len(targets) != 768:
        errors.append(f"expected 768 athlete-boulder targets, found {len(targets)}")
    if targets.get("segment_id", pd.Series(dtype=str)).astype(str).duplicated().any():
        errors.append("segment IDs must be unique")
    if not targets.get("analysis_target_seconds", pd.Series(dtype=float)).eq(300).all():
        errors.append("every analysis target must be exactly five minutes")
    if not targets.get("broadcast_alignment_status", pd.Series(dtype=str)).astype(str).eq(
        "unresolved_requires_clock_graphics_or_pose_detection"
    ).all():
        errors.append("unobserved broadcast alignment must remain explicitly unresolved")
    finals = targets.get("round", pd.Series(dtype=str)).astype(str).eq("Final")
    if not targets.loc[finals, "relative_timing_status"].astype(str).eq(
        "nominal_maximum_turn_schedule_requires_clock_alignment"
    ).all():
        errors.append("final timing must remain nominal until early tops and clock timing are observed")
    if targets.get("broadcast_start_seconds", pd.Series(dtype=str)).astype(str).ne("").any():
        errors.append("relative plans cannot invent broadcast start timestamps")
    if targets.get("broadcast_end_seconds", pd.Series(dtype=str)).astype(str).ne("").any():
        errors.append("relative plans cannot invent broadcast end timestamps")
    if not targets.get("schedule_identity_status", pd.Series(dtype=str)).astype(str).eq(
        "official_exact_start_order"
    ).all():
        errors.append("every target must use exact official start order")
    if "technical_flag" not in targets:
        errors.append("technical evidence fields are required")
    else:
        no_flag = ~targets["technical_flag"].astype(bool)
        if targets.loc[no_flag, "technical_incident_ids"].astype(str).ne("").any():
            errors.append("targets without confirmed evidence cannot carry technical IDs")
        if targets.loc[no_flag, "technical_evidence_intervals_json"].astype(str).ne(
            "[]"
        ).any():
            errors.append("targets without confirmed evidence cannot carry intervals")
    technical_ledger_hash = targets.get(
        "technical_incident_ledger_sha256", pd.Series(dtype=str)
    ).astype(str)
    if technical_ledger_hash.eq("").any() or technical_ledger_hash.str.fullmatch(
        r"[0-9a-f]{64}"
    ).ne(True).any():
        errors.append("every target must retain the technical incident ledger SHA256")
    cumulative = pd.to_numeric(
        targets.get(
            "cumulative_confirmed_technical_delay_seconds", pd.Series(dtype=float)
        ),
        errors="coerce",
    )
    if cumulative.isna().any() or cumulative.lt(0).any():
        errors.append("confirmed technical delay must be a non-negative integer offset")
    else:
        expected_adjusted = pd.to_numeric(
            targets["relative_modeled_turn_start_seconds"], errors="coerce"
        ) + cumulative
        observed_adjusted = pd.to_numeric(
            targets.get(
                "adjusted_relative_modeled_turn_start_seconds", pd.Series(dtype=float)
            ),
            errors="coerce",
        )
        if not expected_adjusted.equals(observed_adjusted):
            errors.append("adjusted relative start must equal base start plus confirmed delay")
    if targets.get("official_start_order_response_sha256", pd.Series(dtype=str)).astype(
        str
    ).str.fullmatch(r"[0-9a-f]{64}").ne(True).any():
        errors.append("every target needs exact start-order response provenance")
    for round_id, scope in targets.groupby("category_round_id"):
        expected_athletes = 24 if str(scope.iloc[0]["round"]) == "Semi-final" else 8
        if scope["athlete_source_id"].nunique() != expected_athletes:
            errors.append(f"{round_id}: wrong athlete coverage")
        if len(scope) != expected_athletes * BOULDER_COUNT:
            errors.append(f"{round_id}: each athlete needs four boulder targets")
        if set(scope["boulder_number"].astype(int)) != {1, 2, 3, 4}:
            errors.append(f"{round_id}: boulder coverage must be 1 through 4")
        if str(scope.iloc[0]["round"]) == "Semi-final":
            if scope["simultaneous_group_size"].max() != 4:
                errors.append(f"{round_id}: circuit plan must reach four simultaneous climbers")
        elif scope["simultaneous_group_size"].max() != 1:
            errors.append(f"{round_id}: final plan cannot invent simultaneous active climbers")
    if len(groups) != targets["simultaneous_group_id"].nunique():
        errors.append("group ledger must have one row per simultaneous group")
    if groups.get("simultaneous_group_id", pd.Series(dtype=str)).astype(str).duplicated().any():
        errors.append("simultaneous group IDs must be unique")
    if "technical_flag" not in groups:
        errors.append("simultaneous groups require explicit technical flags")
    else:
        no_group_flag = ~groups["technical_flag"].astype(bool)
        if groups.loc[no_group_flag, "technical_incident_ids"].astype(str).ne("").any():
            errors.append("unflagged time groups cannot carry technical incident IDs")
    return errors


def records(frame: pd.DataFrame) -> Iterable[dict[str, object]]:
    """Return a typing-friendly record iterator for downstream tools."""

    return frame.to_dict("records")
