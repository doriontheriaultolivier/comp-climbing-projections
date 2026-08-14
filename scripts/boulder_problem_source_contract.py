"""Versioned source contract for athlete-by-boulder federation payloads.

The live federation endpoints expose raw ascent dictionaries.  This isolated
research module decodes one dictionary without modifying the shared archive.
Explicit marker flags are authoritative.  Numeric points are a fallback only
when the caller has already bound the result to the modern IFSC 25/10 schema;
an arbitrary historical score is never guessed into a zone or top.
Combined Boulder & Lead 5/10/25 evidence has its own explicit-flags-only
schema: points are retained, while Low Zone, Zone and Top are decoded only
from their source flags until a separate numeric contract is established.

The raw flags, points and attempt fields remain present in the returned record
and contradictions are explicit.  This makes a later staged acquisition
auditable and prevents parser convenience from becoming model truth.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


IFSC_25_10_V1 = "ifsc_boulder_25_10_v1"
IFSC_COMBINED_5_10_25_FLAGS_V1 = "ifsc_combined_5_10_25_explicit_flags_v1"
LEGACY_EXPLICIT_FLAGS = "legacy_boulder_explicit_flags"
SUPPORTED_SCHEMAS = {
    IFSC_25_10_V1,
    IFSC_COMBINED_5_10_25_FLAGS_V1,
    LEGACY_EXPLICIT_FLAGS,
}
CEC_CANADIAN_A_JR_SHARED_V1 = "cec_canadian_a_jr_shared_v1"
DIRECT_BOULDER_ASCENTS = "direct_ranking_ascents"
COMBINED_BOULDER_ASCENTS = "combined_stage_boulder_ascents"
SUPPORTED_SOURCE_LANES = {DIRECT_BOULDER_ASCENTS, COMBINED_BOULDER_ASCENTS}


class BoulderSourceContractError(ValueError):
    """Raised when source/scoring semantics are not explicit enough."""


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        number = float(value)
        return None if not math.isfinite(number) else number != 0.0
    text = str(value).strip().casefold()
    if not text:
        return None
    if text in {"true", "t", "yes", "y", "1"}:
        return True
    if text in {"false", "f", "no", "n", "0"}:
        return False
    return None


def _optional_number(value: object) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _optional_positive_int(value: object) -> int | None:
    number = _optional_number(value)
    if number is None or number < 1.0 or not float(number).is_integer():
        return None
    return int(number)


def _stage(top: bool, zone: bool) -> str:
    return "top" if top else "zone_only" if zone else "no_zone"


def _ordered_stage(top: bool, zone: bool, low_zone: bool) -> str:
    if top:
        return "top"
    if zone:
        return "zone_only"
    if low_zone:
        return "low_zone_only"
    return "no_marker"


def _points_stage(points: float | None) -> str | None:
    if points is None or points < 0.0 or points > 25.0 + 1.0e-9:
        return None
    if points > 10.0 + 1.0e-9:
        return "top"
    if points > 0.0:
        return "zone_only"
    return "no_zone"


@dataclass(frozen=True)
class DecodedProblemOutcome:
    problem_index: int
    route_id: str | None
    route_name: str | None
    reached_low_zone: bool | None
    reached_zone: bool | None
    reached_top: bool | None
    stage: str
    ordered_stage: str
    marker_evidence: str
    scoring_schema: str
    raw_top_flag: object
    raw_zone_flag: object
    raw_low_zone_flag: object
    raw_points: object
    points: float | None
    raw_top_tries: object
    raw_zone_tries: object
    raw_low_zone_tries: object
    total_attempts: int | None
    attempts_to_low_zone: int | None
    attempts_to_zone: int | None
    attempts_to_top: int | None
    contradictions: tuple[str, ...]


@dataclass(frozen=True)
class ProblemIdentity:
    marker_key: str
    identity_quality: str
    leaderboard_route_id: str | None
    terrain_set_alias: str | None = None


@dataclass(frozen=True)
class RawBoulderAscentGroup:
    """One raw Boulder stage extracted without interpreting its markers.

    Ordinary Boulder rounds expose ``ranking[*].ascents`` directly. Combined
    Boulder & Lead rounds instead nest the same evidence below
    ``ranking[*].combined_stages``. Keeping the source lane and stage metadata
    prevents the combined overall score from being mistaken for a problem
    outcome or counted in addition to the nested Boulder stage.
    """

    source_lane: str
    stage_name: str
    stage_rank: object
    stage_score: object
    ascents: tuple[Mapping[str, Any], ...]
    contradictions: tuple[str, ...] = ()


def _is_boulder_stage(stage: Mapping[str, Any]) -> bool:
    labels = (stage.get("stage_name"), stage.get("kind"), stage.get("name"))
    normalized = {
        " ".join(str(value or "").strip().casefold().replace("_", " ").split())
        for value in labels
    }
    return "boulder" in normalized


def extract_boulder_ascent_groups(
    ranking_entry: Mapping[str, Any],
    *,
    discipline: str,
) -> tuple[RawBoulderAscentGroup, ...]:
    """Extract raw direct or combined-stage Boulder ascents, never both.

    Direct ascents remain authoritative if a payload unexpectedly publishes
    both representations; a contradiction flag requires later QA against the
    nested copy. Multiple explicit Boulder combined stages remain separate
    groups rather than being concatenated into a fictitious single round.
    """

    normalized_discipline = " ".join(
        str(discipline or "").strip().casefold().replace("_", " ").split()
    )
    if "boulder" not in normalized_discipline:
        return ()
    direct_raw = ranking_entry.get("ascents")
    direct = (
        tuple(value for value in direct_raw if isinstance(value, Mapping))
        if isinstance(direct_raw, list)
        else ()
    )
    combined_raw = ranking_entry.get("combined_stages")
    combined: list[RawBoulderAscentGroup] = []
    if isinstance(combined_raw, list):
        for stage in combined_raw:
            if not isinstance(stage, Mapping) or not _is_boulder_stage(stage):
                continue
            raw_ascents = stage.get("ascents")
            ascents = (
                tuple(value for value in raw_ascents if isinstance(value, Mapping))
                if isinstance(raw_ascents, list)
                else ()
            )
            if not ascents:
                continue
            combined.append(
                RawBoulderAscentGroup(
                    source_lane=COMBINED_BOULDER_ASCENTS,
                    stage_name=str(stage.get("stage_name") or "Boulder"),
                    stage_rank=stage.get("stage_rank"),
                    stage_score=stage.get("stage_score"),
                    ascents=ascents,
                )
            )
    # Direct ascents are Boulder evidence only in a pure Boulder payload.
    # Speed and Lead also use an ``ascents`` field with incompatible meaning;
    # combined Boulder & Lead must use its explicit nested Boulder stage.
    direct_is_boulder = normalized_discipline == "boulder"
    if direct and direct_is_boulder:
        contradictions = (
            ("direct_and_combined_boulder_ascents_both_present",)
            if combined
            else ()
        )
        return (
            RawBoulderAscentGroup(
                source_lane=DIRECT_BOULDER_ASCENTS,
                stage_name="Boulder",
                stage_rank=ranking_entry.get("rank"),
                stage_score=ranking_entry.get("score"),
                ascents=direct,
                contradictions=contradictions,
            ),
        )
    return tuple(combined)


def decode_problem_outcome(
    ascent: Mapping[str, Any],
    *,
    problem_index: int,
    scoring_schema: str,
    source_lane: str,
) -> DecodedProblemOutcome:
    """Decode one ascent with explicit precedence and contradiction flags."""

    if scoring_schema not in SUPPORTED_SCHEMAS:
        raise BoulderSourceContractError("unsupported or undeclared scoring schema")
    if source_lane not in SUPPORTED_SOURCE_LANES:
        raise BoulderSourceContractError("unsupported or undeclared Boulder source lane")
    combined_schema = scoring_schema == IFSC_COMBINED_5_10_25_FLAGS_V1
    combined_lane = source_lane == COMBINED_BOULDER_ASCENTS
    if combined_schema != combined_lane:
        raise BoulderSourceContractError(
            "combined Boulder source lane and 5/10/25 explicit schema must be bound together"
        )
    if int(problem_index) < 1:
        raise BoulderSourceContractError("problem_index must be positive")

    raw_top = ascent.get("top")
    raw_zone = ascent.get("zone")
    raw_low_zone = ascent.get("low_zone")
    # Do not use ``a or b``: numeric zero is real modern points evidence.
    raw_points = ascent["points"] if "points" in ascent else ascent.get("score")
    points = _optional_number(raw_points)
    explicit_top = _optional_bool(raw_top)
    explicit_zone = _optional_bool(raw_zone)
    explicit_low_zone = _optional_bool(raw_low_zone)
    points_stage = _points_stage(points) if scoring_schema == IFSC_25_10_V1 else None
    contradictions: list[str] = []

    if any(
        value is not None
        for value in (explicit_top, explicit_zone, explicit_low_zone)
    ):
        evidence = "explicit_api_flags"
        # Preserve tri-state source evidence.  A partially populated payload
        # must not turn every omitted flag into False.  Only logical marker
        # implications are resolved here (Top => Zone => Low Zone); otherwise
        # the exact stage remains unknown and is withheld downstream.
        reached_top = explicit_top
        reached_zone = explicit_zone
        reached_low_zone = explicit_low_zone
        if explicit_top is True and explicit_zone is False:
            contradictions.append("explicit_top_true_zone_false_normalized_top_implies_zone")
        if explicit_zone is True and explicit_low_zone is False:
            contradictions.append(
                "explicit_zone_true_low_zone_false_normalized_zone_implies_low_zone"
            )

        if reached_top is True:
            reached_zone = True
            reached_low_zone = True
        if reached_zone is True:
            reached_low_zone = True
        # The inverse implications are safe as well: failure to reach a lower
        # marker rules out every higher marker.
        if reached_low_zone is False:
            reached_zone = False
            reached_top = False
        if reached_zone is False:
            reached_top = False

        if combined_schema:
            if reached_top is True:
                stage = "top"
                ordered_stage = "top"
            elif reached_top is False and reached_zone is True:
                stage = "zone_only"
                ordered_stage = "zone_only"
            elif (
                reached_top is False
                and reached_zone is False
                and reached_low_zone is True
            ):
                stage = "no_zone"
                ordered_stage = "low_zone_only"
            elif (
                reached_top is False
                and reached_zone is False
                and reached_low_zone is False
            ):
                stage = "no_zone"
                ordered_stage = "no_marker"
            else:
                stage = "unknown"
                ordered_stage = "unknown"
        else:
            if reached_top is True:
                stage = "top"
            elif reached_top is False and reached_zone is True:
                stage = "zone_only"
            elif reached_top is False and reached_zone is False:
                stage = "no_zone"
            else:
                stage = "unknown"
            ordered_stage = stage
            # Ordinary low-zone is auxiliary evidence.  It does not identify
            # the scored Z/T stage, but a reached Zone logically includes it.
            if reached_zone is True:
                reached_low_zone = True

        if points_stage is not None and stage != "unknown" and points_stage != stage:
            contradictions.append(
                f"explicit_stage_{stage}_conflicts_with_points_stage_{points_stage}"
            )
    elif scoring_schema == IFSC_25_10_V1 and points_stage is not None:
        stage = points_stage
        reached_top = stage == "top"
        reached_zone = stage in {"zone_only", "top"}
        reached_low_zone = reached_zone
        ordered_stage = _ordered_stage(
            reached_top, reached_zone, reached_low_zone
        )
        evidence = "derived_ifsc_25_10_points"
    else:
        stage = "unknown"
        ordered_stage = "unknown"
        reached_top = None
        reached_zone = None
        reached_low_zone = None
        if (
            scoring_schema == IFSC_COMBINED_5_10_25_FLAGS_V1
            and points is not None
        ):
            evidence = "combined_5_10_25_points_retained_explicit_flags_required"
        elif points is not None:
            evidence = "legacy_points_not_decoded_without_explicit_flags"
        else:
            evidence = "missing_marker_evidence"
        if scoring_schema == IFSC_25_10_V1 and points is not None:
            contradictions.append("modern_points_outside_0_to_25")

    raw_top_tries = ascent.get("top_tries")
    raw_zone_tries = ascent.get("zone_tries")
    raw_low_zone_tries = ascent.get("low_zone_tries")
    top_tries = _optional_positive_int(raw_top_tries)
    zone_tries = _optional_positive_int(raw_zone_tries)
    low_zone_tries = _optional_positive_int(raw_low_zone_tries)
    total_attempts = (
        None
        if scoring_schema == IFSC_COMBINED_5_10_25_FLAGS_V1
        else top_tries
    )
    attempts_to_low_zone = (
        low_zone_tries if reached_low_zone is True else None
    )
    attempts_to_zone = zone_tries if reached_zone is True else None
    attempts_to_top = top_tries if reached_top is True else None

    if reached_top is True and top_tries is None:
        contradictions.append("top_without_valid_top_tries")
    if reached_zone is True and zone_tries is None:
        contradictions.append("zone_without_valid_zone_tries")
    if (
        reached_low_zone is True
        and low_zone_tries is None
        and (
            scoring_schema == IFSC_COMBINED_5_10_25_FLAGS_V1
            or explicit_low_zone is True
        )
    ):
        contradictions.append("low_zone_without_valid_low_zone_tries")
    if top_tries is not None and zone_tries is not None and top_tries < zone_tries:
        contradictions.append("top_tries_below_zone_tries")
    if (
        zone_tries is not None
        and low_zone_tries is not None
        and zone_tries < low_zone_tries
    ):
        contradictions.append("zone_tries_below_low_zone_tries")

    if scoring_schema == IFSC_25_10_V1 and points is not None and stage != "unknown":
        expected = None
        if stage == "top" and top_tries is not None:
            expected = 25.0 - 0.1 * (top_tries - 1)
        elif stage == "zone_only" and zone_tries is not None:
            expected = 10.0 - 0.1 * (zone_tries - 1)
        elif stage == "no_zone":
            expected = 0.0
        if expected is not None and abs(points - expected) > 0.051:
            contradictions.append("points_do_not_match_25_10_attempt_deduction")

    route_id_value = ascent.get("route_id")
    if route_id_value is None:
        route_id_value = ascent.get("id")
    route_name_value = ascent.get("route_name")
    if route_name_value is None:
        route_name_value = ascent.get("name")
    return DecodedProblemOutcome(
        problem_index=int(problem_index),
        route_id=None if route_id_value is None else str(route_id_value),
        route_name=None if route_name_value is None else str(route_name_value),
        reached_low_zone=reached_low_zone,
        reached_zone=reached_zone,
        reached_top=reached_top,
        stage=stage,
        ordered_stage=ordered_stage,
        marker_evidence=evidence,
        scoring_schema=scoring_schema,
        raw_top_flag=raw_top,
        raw_zone_flag=raw_zone,
        raw_low_zone_flag=raw_low_zone,
        raw_points=raw_points,
        points=points,
        raw_top_tries=raw_top_tries,
        raw_zone_tries=raw_zone_tries,
        raw_low_zone_tries=raw_low_zone_tries,
        total_attempts=total_attempts,
        attempts_to_low_zone=attempts_to_low_zone,
        attempts_to_zone=attempts_to_zone,
        attempts_to_top=attempts_to_top,
        contradictions=tuple(contradictions),
    )


def problem_identity(
    *,
    source_scope: str,
    source_event_id: int | str,
    result_url: str,
    outcome: DecodedProblemOutcome,
    reviewed_terrain_set_alias: str | None = None,
) -> ProblemIdentity:
    """Stable leaderboard identity plus an optional reviewed physical alias.

    Federation route IDs are retained even when two category leaderboards are
    later confirmed to share one physical boulder.  A terrain alias is never
    inferred here; it must come from the reviewed Canadian/YWCH sharing rules.
    """

    source = str(source_scope).strip().upper()
    event = str(source_event_id).strip()
    url = str(result_url).strip()
    if not source or not event or not url:
        raise BoulderSourceContractError("source, event and result_url are required")
    if outcome.route_id is not None:
        key = f"{source}|event:{event}|route:{outcome.route_id}"
        quality = "federation_route_id"
    else:
        key = f"{source}|event:{event}|result:{url}|problem:{outcome.problem_index}"
        quality = "result_url_problem_index_fallback"
    alias = None
    if reviewed_terrain_set_alias is not None:
        alias = str(reviewed_terrain_set_alias).strip() or None
    return ProblemIdentity(
        marker_key=key,
        identity_quality=quality,
        leaderboard_route_id=outcome.route_id,
        terrain_set_alias=alias,
    )


def reviewed_cec_terrain_alias(
    *,
    sharing_rule: str,
    source_event_id: int | str,
    round_name: str,
    gender: str,
    category: str,
    problem_index: int,
) -> str:
    """Return the explicitly reviewed Canadian A/Jr physical-terrain alias.

    The federation currently assigns different leaderboard route IDs to U19
    and U21 even when the Canadian event used the same physical boulders.  The
    caller must opt into the versioned rule; it is never inferred from similar
    names.  U17/B, U15/C, men/women and Youth Worlds remain separate.
    """

    if sharing_rule != CEC_CANADIAN_A_JR_SHARED_V1:
        raise BoulderSourceContractError("unsupported or undeclared terrain-sharing rule")
    normalized_category = " ".join(str(category).strip().upper().split())
    if not (normalized_category.startswith("U19") or normalized_category.startswith("U21")):
        raise BoulderSourceContractError("CEC A/Jr alias is valid only for U19/U21")
    normalized_gender = " ".join(str(gender).strip().casefold().split())
    if normalized_gender not in {"men", "women"}:
        raise BoulderSourceContractError("gender-specific terrain is required")
    normalized_round = "_".join(str(round_name).strip().casefold().replace("-", " ").split())
    if not normalized_round or int(problem_index) < 1:
        raise BoulderSourceContractError("round and positive problem_index are required")
    return (
        f"CEC|event:{str(source_event_id).strip()}|terrain:A_JR|"
        f"{normalized_gender}|{normalized_round}|problem:{int(problem_index)}"
    )


__all__ = [
    "BoulderSourceContractError",
    "CEC_CANADIAN_A_JR_SHARED_V1",
    "COMBINED_BOULDER_ASCENTS",
    "DecodedProblemOutcome",
    "DIRECT_BOULDER_ASCENTS",
    "IFSC_25_10_V1",
    "IFSC_COMBINED_5_10_25_FLAGS_V1",
    "LEGACY_EXPLICIT_FLAGS",
    "ProblemIdentity",
    "RawBoulderAscentGroup",
    "SUPPORTED_SCHEMAS",
    "SUPPORTED_SOURCE_LANES",
    "decode_problem_outcome",
    "extract_boulder_ascent_groups",
    "problem_identity",
    "reviewed_cec_terrain_alias",
]
