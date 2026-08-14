"""Isolated staged-evidence adapter for the terrain-response challenger.

This module reads the bounded federation problem-evidence artifacts and the
identity-safe Stage-A artifact.  It does not merge either source into the
shared archive and it never fits a rating model.

The adapter preserves three identities separately:

* ``problem_id`` is the immutable leaderboard route identity;
* ``terrain_set_alias`` is an optional reviewed physical-terrain alias and
  never replaces ``problem_id``;
* ``athlete_id`` is taken from the identity-safe research artifact only when
  the source node has one unambiguous research identity.  Otherwise a stable
  source-local research ID is used rather than guessing a person bridge.

Combined Boulder & Lead 5/10/25 evidence is a three-hurdle chain
(``low-zone -> zone -> top``).  The current terrain prototype is a two-hurdle
model, so combined rows are retained and counted but fail closed for Item-Elo
or response fitting.  In particular, low-zone-only is never coerced to Zone.

The source API supplies ordinal marker tries, not a predeclared opportunity
horizon.  They remain provenance in :class:`TerrainProblemEvidence`; the
``ProblemOutcome`` conversion deliberately withholds them so an aggregate
counter cannot be mistaken for conditional post-zone exposure.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence

import pandas as pd

try:
    from .boulder_problem_source_contract import (
        CEC_CANADIAN_A_JR_SHARED_V1,
        COMBINED_BOULDER_ASCENTS,
        DIRECT_BOULDER_ASCENTS,
        IFSC_COMBINED_5_10_25_FLAGS_V1,
        LEGACY_EXPLICIT_FLAGS,
        decode_problem_outcome,
        problem_identity,
        reviewed_cec_terrain_alias,
    )
    from .boulder_terrain_response import (
        CONFIRMED_PROCEDURES,
        DEFAULT_RATING_POOL,
        ProblemOutcome,
    )
except ImportError:  # pragma: no cover - direct-script fallback
    from boulder_problem_source_contract import (  # type: ignore
        CEC_CANADIAN_A_JR_SHARED_V1,
        COMBINED_BOULDER_ASCENTS,
        DIRECT_BOULDER_ASCENTS,
        IFSC_COMBINED_5_10_25_FLAGS_V1,
        LEGACY_EXPLICIT_FLAGS,
        decode_problem_outcome,
        problem_identity,
        reviewed_cec_terrain_alias,
    )
    from boulder_terrain_response import (  # type: ignore
        CONFIRMED_PROCEDURES,
        DEFAULT_RATING_POOL,
        ProblemOutcome,
    )


ADAPTER_SCHEMA = "boulder-terrain-problem-adapter-v1"
START_DATE = "2021-01-01"
REVIEWED_CEC_SHARED_TERRAIN_EVENTS = frozenset({"224"})
_HIDDEN_TEST_PATTERN = re.compile(
    r"(?i)(?:\btest\b|\bhidden\b|pratique\s+pour\s+nouveau\s+hj)"
)
_COMBINED_FORMAT_PREFIX = "bl_"
_UTC = timezone.utc


class TerrainProblemAdapterError(ValueError):
    """Raised when staged evidence cannot be normalized without guessing."""


def _key_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        number = float(value)
        return None if not math.isfinite(number) else bool(number)
    text = str(value).strip().casefold()
    if text in {"true", "t", "yes", "y", "1"}:
        return True
    if text in {"false", "f", "no", "n", "0"}:
        return False
    return None


def _finite_or_none(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _positive_int_or_none(value: object) -> int | None:
    number = _finite_or_none(value)
    if number is None or number < 1.0 or not float(number).is_integer():
        return None
    return int(number)


def _procedure(value: object) -> str:
    text = " ".join(str(value or "").strip().casefold().replace("_", " ").split())
    if "onsight" in text or "on sight" in text:
        return "onsight"
    if "flash" in text:
        return "flash"
    if "scramble" in text:
        return "scramble"
    return "unknown"


def _age_band(category: object, age_class: object) -> str:
    category_text = " ".join(str(category or "").strip().upper().split())
    mappings = (
        (("U15", "YOUTH C"), "U15"),
        (("U17", "YOUTH B"), "U17"),
        (("U19", "YOUTH A"), "U19"),
        (("U21", "JUNIOR"), "U21"),
    )
    for labels, band in mappings:
        if any(label in category_text for label in labels):
            return band
    age_text = " ".join(str(age_class or "").strip().split())
    return "Senior/Open" if "senior" in age_text.casefold() else age_text or "Unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_hidden_or_test_event(event_name: object) -> bool:
    """Fail-closed exclusion for staged CEC test/hidden fixtures."""

    return bool(_HIDDEN_TEST_PATTERN.search(str(event_name or "")))


@dataclass(frozen=True)
class StableSnapshot:
    athlete_id: str
    stable_mean: float
    stable_sd: float
    confirmed_procedure: str
    identity_status: str
    identity_provenance: str


@dataclass(frozen=True)
class IdentityLookups:
    exact_snapshots: Mapping[tuple[str, str, str, str], StableSnapshot]
    source_node_ids: Mapping[tuple[str, str], str]
    ambiguous_source_nodes: frozenset[tuple[str, str]]


@dataclass(frozen=True)
class TerrainProblemEvidence:
    source_scope: str
    source_event_id: str
    event_date: str
    event_name: str
    event_tier: str
    category: str
    age_band: str
    round_name: str
    round_rank_numeric: float | None
    round_rank_percentile: float | None
    source_url: str
    competition_id: str
    athlete_source_id: str
    athlete_id: str
    identity_provenance: str
    identity_status: str
    problem_index: int
    problem_id: str
    leaderboard_route_id: str | None
    problem_identity_quality: str
    terrain_set_alias: str | None
    marker_chain: str
    reached_low_zone: bool | None
    reached_zone: bool | None
    reached_top: bool | None
    source_marker_schema: str
    marker_evidence: str
    source_attempts_to_low_zone: int | None
    source_attempts_to_zone: int | None
    source_attempts_to_top: int | None
    attempt_semantics: str
    stable_mean: float | None
    stable_sd: float | None
    confirmed_procedure: str
    chronology_precision: str
    contradictions: tuple[str, ...]

    @property
    def marker_identity_eligible(self) -> bool:
        return (
            self.problem_identity_quality == "federation_route_id"
            and self.reached_zone is not None
            and self.reached_top is not None
            and not self.contradictions
        )

    @property
    def ordinary_two_hurdle(self) -> bool:
        return self.marker_chain == "ordinary_zone_top_2h"

    @property
    def item_calibration_eligible(self) -> bool:
        return (
            self.marker_identity_eligible
            and self.ordinary_two_hurdle
            and self.stable_mean is not None
            and self.stable_sd is not None
            and self.stable_sd > 0.0
            and self.confirmed_procedure in CONFIRMED_PROCEDURES
            and self.identity_provenance == "exact_identity_safe_round_snapshot"
        )


def load_identity_lookups(identity_safe_path: Path) -> IdentityLookups:
    """Load exact frozen snapshots and fail-closed source-node identities."""

    columns = [
        "source_scope",
        "source_event_id",
        "source_url",
        "athlete_source_id",
        "global_id",
        "identity_integrity_status",
        "confirmed_procedure",
        "event_start_global_rating",
        "event_start_rating_uncertainty",
    ]
    frame = pd.read_parquet(identity_safe_path, columns=columns)
    for column in (
        "source_scope",
        "source_event_id",
        "source_url",
        "athlete_source_id",
        "global_id",
    ):
        frame[column] = frame[column].map(_key_text)
    exact_key = ["source_scope", "source_event_id", "source_url", "athlete_source_id"]
    if frame.duplicated(exact_key, keep=False).any():
        raise TerrainProblemAdapterError(
            "identity-safe artifact has duplicate exact round/athlete keys"
        )

    exact: dict[tuple[str, str, str, str], StableSnapshot] = {}
    for row in frame.itertuples(index=False):
        mean = _finite_or_none(row.event_start_global_rating)
        sd = _finite_or_none(row.event_start_rating_uncertainty)
        if mean is None or sd is None or sd <= 0.0:
            continue
        key = (
            _key_text(row.source_scope).upper(),
            _key_text(row.source_event_id),
            _key_text(row.source_url),
            _key_text(row.athlete_source_id),
        )
        exact[key] = StableSnapshot(
            athlete_id=_key_text(row.global_id),
            stable_mean=mean,
            stable_sd=sd,
            confirmed_procedure=_procedure(row.confirmed_procedure),
            identity_status=_key_text(row.identity_integrity_status),
            identity_provenance="exact_identity_safe_round_snapshot",
        )

    node_groups = frame.groupby(["source_scope", "athlete_source_id"], sort=False)[
        "global_id"
    ].agg(lambda values: tuple(sorted(set(_key_text(value) for value in values if _key_text(value)))))
    source_nodes: dict[tuple[str, str], str] = {}
    ambiguous: set[tuple[str, str]] = set()
    for raw_key, global_ids in node_groups.items():
        key = (_key_text(raw_key[0]).upper(), _key_text(raw_key[1]))
        if len(global_ids) == 1:
            source_nodes[key] = global_ids[0]
        else:
            ambiguous.add(key)
    return IdentityLookups(exact, source_nodes, frozenset(ambiguous))


def _identity_for_round(
    row: Mapping[str, object], lookups: IdentityLookups
) -> tuple[str, str, str, StableSnapshot | None]:
    scope = _key_text(row.get("source_scope")).upper()
    athlete_source_id = _key_text(row.get("athlete_source_id"))
    exact_key = (
        scope,
        _key_text(row.get("source_event_id")),
        _key_text(row.get("source_url")),
        athlete_source_id,
    )
    snapshot = lookups.exact_snapshots.get(exact_key)
    if snapshot is not None:
        return (
            snapshot.athlete_id,
            snapshot.identity_provenance,
            snapshot.identity_status,
            snapshot,
        )
    node_key = (scope, athlete_source_id)
    if node_key in lookups.source_node_ids:
        return (
            lookups.source_node_ids[node_key],
            "unique_identity_safe_source_node_without_round_snapshot",
            "stable_identity_missing_frozen_round_snapshot",
            None,
        )
    fallback = f"RESEARCH-SOURCE:{scope}:{athlete_source_id}"
    provenance = (
        "ambiguous_identity_safe_source_node_split_locally"
        if node_key in lookups.ambiguous_source_nodes
        else "source_local_identity_not_in_snapshot"
    )
    return fallback, provenance, "source_local_only", None


def normalize_round_row(
    row: Mapping[str, object], lookups: IdentityLookups
) -> tuple[TerrainProblemEvidence, ...]:
    """Normalize one staged Boulder leaderboard row without model fitting."""

    scope = _key_text(row.get("source_scope")).upper()
    event_id = _key_text(row.get("source_event_id"))
    result_url = _key_text(row.get("source_url"))
    event_name = _key_text(row.get("event_name"))
    athlete_source_id = _key_text(row.get("athlete_source_id"))
    if not scope or not event_id or not result_url or not athlete_source_id:
        raise TerrainProblemAdapterError("source/event/result/athlete identity is required")
    raw_outcomes = row.get("boulder_outcomes_json")
    if isinstance(raw_outcomes, str):
        try:
            outcomes = json.loads(raw_outcomes or "[]")
        except json.JSONDecodeError as exc:
            raise TerrainProblemAdapterError("invalid boulder_outcomes_json") from exc
    elif isinstance(raw_outcomes, Sequence):
        outcomes = list(raw_outcomes)
    else:
        outcomes = []
    if not isinstance(outcomes, list):
        raise TerrainProblemAdapterError("boulder_outcomes_json must decode to a list")

    athlete_id, identity_provenance, identity_status, snapshot = _identity_for_round(
        row, lookups
    )
    combined_format = _key_text(row.get("format_identifier")).casefold().startswith(
        _COMBINED_FORMAT_PREFIX
    )
    normalized: list[TerrainProblemEvidence] = []
    for ordinal, raw_ascent in enumerate(outcomes, start=1):
        if not isinstance(raw_ascent, Mapping):
            continue
        problem_index_value = raw_ascent.get("problem_index", ordinal)
        try:
            problem_index = int(problem_index_value)
        except (TypeError, ValueError) as exc:
            raise TerrainProblemAdapterError("problem_index must be an integer") from exc
        raw_low_zone = _optional_bool(raw_ascent.get("low_zone"))
        raw_zone = _optional_bool(raw_ascent.get("zone"))
        # Format/lane, not the field name alone, defines the hurdle chain.
        # On ordinary federation leaderboards ``low_zone`` is auxiliary
        # evidence below the scored Zone. It must not be promoted to the
        # primary Zone: IFSC event 1394, for example, has four scored Zones
        # plus one low-zone-only boulder and an official `0T4z`, not `0T5z`.
        # In B&L combined lanes it is the declared 5-point marker below the
        # 10-point Zone. Schema remains round-bound either way.
        three_hurdle = combined_format
        decode_payload = dict(raw_ascent)
        if not three_hurdle:
            # The auxiliary marker does not enter the primary Z/T decoder.
            # It remains on TerrainProblemEvidence with its own raw success
            # ordinal for future auxiliary-marker research.
            decode_payload.pop("low_zone", None)
            decode_payload.pop("low_zone_tries", None)
        # The staged file already preserves explicit marker flags.  Still bind
        # the correct schema to the correct source lane so 5/10/25 combined
        # evidence can never fall through an ordinary positive-points rule.
        decoded = decode_problem_outcome(
            decode_payload,
            problem_index=problem_index,
            scoring_schema=(
                IFSC_COMBINED_5_10_25_FLAGS_V1
                if three_hurdle
                else LEGACY_EXPLICIT_FLAGS
            ),
            source_lane=(
                COMBINED_BOULDER_ASCENTS
                if three_hurdle
                else DIRECT_BOULDER_ASCENTS
            ),
        )
        low_zone = decoded.reached_low_zone if three_hurdle else raw_low_zone
        terrain_alias = None
        category = _key_text(row.get("category"))
        if (
            scope == "CEC"
            and event_id in REVIEWED_CEC_SHARED_TERRAIN_EVENTS
            and category.upper().startswith(("U19", "U21"))
        ):
            terrain_alias = reviewed_cec_terrain_alias(
                sharing_rule=CEC_CANADIAN_A_JR_SHARED_V1,
                source_event_id=event_id,
                round_name=_key_text(row.get("round_name")),
                gender=_key_text(row.get("gender")),
                category=category,
                problem_index=problem_index,
            )
        identity = problem_identity(
            source_scope=scope,
            source_event_id=event_id,
            result_url=result_url,
            outcome=decoded,
            reviewed_terrain_set_alias=terrain_alias,
        )
        normalized.append(
            TerrainProblemEvidence(
                source_scope=scope,
                source_event_id=event_id,
                event_date=_key_text(row.get("event_date")),
                event_name=event_name,
                event_tier=_key_text(row.get("event_tier")) or "Unknown",
                category=category,
                age_band=_age_band(category, row.get("age_class")),
                round_name=_key_text(row.get("round_name")),
                round_rank_numeric=_finite_or_none(row.get("rank_numeric")),
                round_rank_percentile=_finite_or_none(row.get("rank_pct")),
                source_url=result_url,
                competition_id=f"{scope}|event:{event_id}",
                athlete_source_id=athlete_source_id,
                athlete_id=athlete_id,
                identity_provenance=identity_provenance,
                identity_status=identity_status,
                problem_index=problem_index,
                problem_id=identity.marker_key,
                leaderboard_route_id=identity.leaderboard_route_id,
                problem_identity_quality=identity.identity_quality,
                terrain_set_alias=identity.terrain_set_alias,
                marker_chain=(
                    "combined_low_zone_zone_top_3h"
                    if three_hurdle
                    else "ordinary_zone_top_2h"
                ),
                reached_low_zone=low_zone,
                reached_zone=decoded.reached_zone,
                reached_top=decoded.reached_top,
                source_marker_schema=(
                    "combined_explicit_low_zone_zone_top_5_10_25"
                    if three_hurdle
                    else "ordinary_explicit_zone_top_auxiliary_low_zone_retained"
                ),
                marker_evidence=decoded.marker_evidence,
                source_attempts_to_low_zone=_positive_int_or_none(
                    raw_ascent.get("low_zone_tries")
                ),
                source_attempts_to_zone=decoded.attempts_to_zone,
                source_attempts_to_top=decoded.attempts_to_top,
                attempt_semantics="ordinal_marker_success_attempt_no_predeclared_horizon",
                stable_mean=(snapshot.stable_mean if snapshot else None),
                stable_sd=(snapshot.stable_sd if snapshot else None),
                confirmed_procedure=(
                    snapshot.confirmed_procedure if snapshot else "unknown"
                ),
                chronology_precision="event_date_only_whole_competition_boundary",
                contradictions=decoded.contradictions,
            )
        )
    return tuple(normalized)


def to_problem_outcome(evidence: TerrainProblemEvidence) -> ProblemOutcome:
    """Convert one contract-eligible row; attempts remain deliberately withheld."""

    if not evidence.item_calibration_eligible:
        raise TerrainProblemAdapterError("evidence is not two-hurdle ProblemOutcome-ready")
    try:
        start = datetime.fromisoformat(evidence.event_date).replace(tzinfo=_UTC)
    except ValueError as exc:
        raise TerrainProblemAdapterError("event_date must be ISO YYYY-MM-DD") from exc
    return ProblemOutcome(
        athlete_id=evidence.athlete_id,
        competition_id=evidence.competition_id,
        problem_id=evidence.problem_id,
        competition_start=start,
        frozen_at=start - timedelta(microseconds=1),
        result_available_at=start + timedelta(days=1),
        stable_mean=float(evidence.stable_mean),
        stable_sd=float(evidence.stable_sd),
        procedure=evidence.confirmed_procedure,
        reached_zone=bool(evidence.reached_zone),
        reached_top=bool(evidence.reached_top),
        rating_pool_id=DEFAULT_RATING_POOL,
        attempts_to_zone=None,
        attempts_to_top=None,
        attempt_cap=1,
    )


def _stream_csv(path: Path) -> Iterable[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def audit_problem_evidence(
    staged_paths: Sequence[Path],
    identity_safe_path: Path,
    *,
    start_date: str = START_DATE,
    minimum_athlete_items: int = 6,
    minimum_athlete_competitions: int = 2,
) -> tuple[dict[str, object], list[TerrainProblemEvidence]]:
    """Return exact eligibility counts and a small deterministic QA sample."""

    lookups = load_identity_lookups(identity_safe_path)
    counts: Counter[str] = Counter()
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    excluded_events: dict[tuple[str, str], tuple[str, str]] = {}
    eligible_rows: list[TerrainProblemEvidence] = []
    sample_by_class: dict[str, list[TerrainProblemEvidence]] = defaultdict(list)
    context_counts: dict[str, Counter[str]] = defaultdict(Counter)
    context_athletes: dict[str, set[str]] = defaultdict(set)
    context_item_athletes: dict[str, set[str]] = defaultdict(set)
    context_bottom_half_athletes: dict[str, set[str]] = defaultdict(set)
    context_item_outcomes: dict[
        str, dict[tuple[str, str], list[tuple[bool, bool]]]
    ] = defaultdict(lambda: defaultdict(list))
    eligible_appearances: set[
        tuple[str, str, str, str, str, str, float | None]
    ] = set()

    def context_key(evidence: TerrainProblemEvidence) -> str:
        return "|".join(
            (evidence.source_scope, evidence.event_tier, evidence.age_band)
        )

    def count(label: str, scope: str, amount: int = 1) -> None:
        counts[label] += amount
        by_source[scope][label] += amount

    for staged_path in staged_paths:
        for row in _stream_csv(staged_path):
            scope = _key_text(row.get("source_scope")).upper()
            if _key_text(row.get("discipline")).casefold() != "boulder":
                continue
            if _key_text(row.get("event_date")) < start_date:
                continue
            count("staged_boulder_round_rows", scope)
            try:
                raw_problem_count = len(json.loads(row.get("boulder_outcomes_json") or "[]"))
            except json.JSONDecodeError:
                raw_problem_count = 0
                count("invalid_problem_json_round_rows", scope)
            count("staged_boulder_problem_rows", scope, raw_problem_count)
            if raw_problem_count == 0:
                count("empty_problem_evidence_round_rows", scope)
                if _key_text(row.get("format_identifier")).casefold().startswith(
                    _COMBINED_FORMAT_PREFIX
                ):
                    count("combined_round_rows_awaiting_nested_evidence", scope)
            if scope == "CEC" and is_hidden_or_test_event(row.get("event_name")):
                count("excluded_hidden_test_round_rows", scope)
                count("excluded_hidden_test_problem_rows", scope, raw_problem_count)
                excluded_events[(scope, _key_text(row.get("source_event_id")))] = (
                    _key_text(row.get("event_name")),
                    _key_text(row.get("event_date")),
                )
                continue
            evidence_rows = normalize_round_row(row, lookups)
            count("normalized_problem_rows", scope, len(evidence_rows))
            for evidence in evidence_rows:
                context = context_key(evidence)
                context_counts[context]["normalized_rows"] += 1
                context_athletes[context].add(evidence.athlete_id)
                if (
                    evidence.round_rank_percentile is not None
                    and evidence.round_rank_percentile <= 0.5
                ):
                    context_counts[context]["bottom_half_normalized_rows"] += 1
                    context_bottom_half_athletes[context].add(evidence.athlete_id)
                label = evidence.marker_chain
                count(label + "_rows", scope)
                if evidence.reached_low_zone and evidence.reached_zone is False:
                    count(
                        "combined_low_zone_only_rows"
                        if not evidence.ordinary_two_hurdle
                        else "ordinary_auxiliary_low_zone_below_primary_zone_rows",
                        scope,
                    )
                if evidence.ordinary_two_hurdle and evidence.reached_low_zone is not None:
                    count("ordinary_auxiliary_low_zone_observed_rows", scope)
                if evidence.terrain_set_alias:
                    count("reviewed_cec_u19_u21_alias_rows", scope)
                if evidence.identity_provenance == "exact_identity_safe_round_snapshot":
                    count("exact_frozen_snapshot_rows", scope)
                else:
                    count("missing_exact_frozen_snapshot_rows", scope)
                if evidence.marker_identity_eligible:
                    count("marker_identity_eligible_rows", scope)
                if evidence.item_calibration_eligible:
                    count("item_calibration_eligible_rows", scope)
                    eligible_rows.append(evidence)
                    context_counts[context]["item_calibration_eligible_rows"] += 1
                    context_item_athletes[context].add(evidence.athlete_id)
                    eligible_appearances.add(
                        (
                            evidence.event_date,
                            evidence.athlete_id,
                            context,
                            evidence.competition_id,
                            evidence.category,
                            evidence.round_name,
                            evidence.round_rank_numeric,
                        )
                    )
                    context_item_outcomes[context][
                        (evidence.competition_id, evidence.problem_id)
                    ].append((bool(evidence.reached_zone), bool(evidence.reached_top)))
                else:
                    count("item_calibration_withheld_rows", scope)
                    if not evidence.ordinary_two_hurdle:
                        count("withheld_combined_three_hurdle_rows", scope)
                    if evidence.problem_identity_quality != "federation_route_id":
                        count("withheld_missing_federation_route_id_rows", scope)
                    if evidence.reached_zone is None or evidence.reached_top is None:
                        count("withheld_unknown_marker_rows", scope)
                    if evidence.contradictions:
                        count("withheld_marker_contradiction_rows", scope)
                    if evidence.identity_provenance != "exact_identity_safe_round_snapshot":
                        count("withheld_missing_exact_snapshot_rows", scope)
                    if evidence.confirmed_procedure not in CONFIRMED_PROCEDURES:
                        count("withheld_unconfirmed_procedure_rows", scope)
                if len(sample_by_class[label]) < 8:
                    sample_by_class[label].append(evidence)

    item_field_sizes = Counter((row.competition_id, row.problem_id) for row in eligible_rows)
    peer_eligible = [
        row
        for row in eligible_rows
        if item_field_sizes[(row.competition_id, row.problem_id)] >= 2
    ]
    athlete_competitions: dict[str, set[str]] = defaultdict(set)
    athlete_rows = Counter()
    for row in peer_eligible:
        athlete_competitions[row.athlete_id].add(row.competition_id)
        athlete_rows[row.athlete_id] += 1
    repeated_athletes = {
        athlete_id
        for athlete_id, row_count in athlete_rows.items()
        if row_count >= minimum_athlete_items
        and len(athlete_competitions[athlete_id]) >= minimum_athlete_competitions
    }
    chronological = [row for row in peer_eligible if row.athlete_id in repeated_athletes]
    for row in chronological:
        context_counts[context_key(row)]["chronological_response_candidate_rows"] += 1

    counts["loo_peer_eligible_item_rows"] = len(peer_eligible)
    counts["chronological_response_candidate_rows"] = len(chronological)
    counts["chronological_response_candidate_athletes"] = len(repeated_athletes)
    for scope in {row.source_scope for row in peer_eligible}:
        scoped = [row for row in peer_eligible if row.source_scope == scope]
        by_source[scope]["loo_peer_eligible_item_rows"] = len(scoped)
        by_source[scope]["chronological_response_candidate_rows"] = sum(
            row.athlete_id in repeated_athletes for row in scoped
        )
        by_source[scope]["chronological_response_candidate_athletes"] = len(
            {row.athlete_id for row in scoped if row.athlete_id in repeated_athletes}
        )

    # Validate every eligible row through the actual research dataclass without
    # fitting a model.  Also audit the whole-competition frozen-state invariant.
    state_by_event_athlete: dict[tuple[str, str], tuple[float, float]] = {}
    state_conflicts = 0
    for evidence in eligible_rows:
        outcome = to_problem_outcome(evidence)
        state_key = (outcome.competition_id, outcome.athlete_id)
        state = (outcome.stable_mean, outcome.stable_sd)
        previous = state_by_event_athlete.setdefault(state_key, state)
        state_conflicts += int(previous != state)
    counts["problem_outcome_contract_valid_rows"] = len(eligible_rows)
    counts["whole_competition_frozen_state_conflict_rows"] = state_conflicts

    # Establish context bridges only from strictly earlier competitions.  An
    # edge appears on the date the later context is first observed; rolling
    # origins must filter these edges by target date.
    appearances_by_athlete: dict[
        str, list[tuple[str, str, str, str, str, float | None]]
    ] = defaultdict(list)
    rounds_by_competition_category: dict[tuple[str, str], set[str]] = defaultdict(set)
    for (
        date,
        athlete_id,
        context,
        competition_id,
        category,
        round_name,
        rank_numeric,
    ) in eligible_appearances:
        appearances_by_athlete[athlete_id].append(
            (date, context, competition_id, category, round_name, rank_numeric)
        )
        rounds_by_competition_category[
            (competition_id, category.casefold())
        ].add(round_name.casefold())
    prior_bridge_athletes: dict[str, set[str]] = defaultdict(set)
    prior_ifsc_bridge_athletes: dict[str, set[str]] = defaultdict(set)
    prior_world_bridge_athletes: dict[str, set[str]] = defaultdict(set)
    bridge_edges: dict[
        tuple[str, str], dict[str, object]
    ] = {}
    youth_winner_events: dict[str, set[str]] = defaultdict(set)
    for athlete_id, appearances in appearances_by_athlete.items():
        ordered = sorted(set(appearances), key=lambda value: (value[0], value[2], value[1]))
        prior: list[tuple[str, str, str]] = []
        for date, context, competition_id, category, round_name, rank_numeric in ordered:
            deciding_round = (
                "final" in round_name.casefold()
                or len(
                    rounds_by_competition_category[
                        (competition_id, category.casefold())
                    ]
                )
                == 1
            )
            if (
                context.rsplit("|", 1)[-1] != "Senior/Open"
                and rank_numeric == 1.0
                and deciding_round
            ):
                youth_winner_events[athlete_id].add(competition_id)
            earlier = [value for value in prior if value[0] < date]
            earlier_contexts = {value[1] for value in earlier if value[1] != context}
            if earlier_contexts:
                prior_bridge_athletes[context].add(athlete_id)
            if any(value.split("|", 1)[0] == "IFSC" for value in earlier_contexts):
                prior_ifsc_bridge_athletes[context].add(athlete_id)
            if any("|World " in value for value in earlier_contexts):
                prior_world_bridge_athletes[context].add(athlete_id)
            for origin in earlier_contexts:
                edge = bridge_edges.setdefault(
                    (origin, context),
                    {
                        "origin_context": origin,
                        "destination_context": context,
                        "first_established_date": date,
                        "athlete_ids": set(),
                        "later_competition_ids": set(),
                    },
                )
                edge["first_established_date"] = min(
                    str(edge["first_established_date"]), date
                )
                edge["athlete_ids"].add(athlete_id)  # type: ignore[union-attr]
                edge["later_competition_ids"].add(competition_id)  # type: ignore[union-attr]
            prior.append((date, context, competition_id))

    repeated_youth_winners = {
        athlete_id
        for athlete_id, competitions in youth_winner_events.items()
        if len(competitions) >= 2
    }

    # Retrospective component map at the artifact's latest event date. It is a
    # QA diagnostic only; target replay must rebuild it using edges established
    # strictly before that target.
    adjacency: dict[str, dict[str, int]] = defaultdict(dict)
    serialized_edges: list[dict[str, object]] = []
    for edge in bridge_edges.values():
        origin = str(edge["origin_context"])
        destination = str(edge["destination_context"])
        athletes = set(edge["athlete_ids"])  # type: ignore[arg-type]
        strength = len(athletes)
        adjacency[origin][destination] = max(adjacency[origin].get(destination, 0), strength)
        adjacency[destination][origin] = max(adjacency[destination].get(origin, 0), strength)
        serialized_edges.append(
            {
                "origin_context": origin,
                "destination_context": destination,
                "first_established_date": edge["first_established_date"],
                "bridge_athletes": strength,
                "later_competitions": len(edge["later_competition_ids"]),  # type: ignore[arg-type]
                "example_athlete_ids": sorted(athletes)[:5],
            }
        )

    all_contexts = sorted(context_counts)
    component_by_context: dict[str, int] = {}
    components: list[list[str]] = []
    for node in all_contexts:
        if node in component_by_context:
            continue
        component_id = len(components)
        stack = [node]
        members: list[str] = []
        component_by_context[node] = component_id
        while stack:
            current = stack.pop()
            members.append(current)
            for neighbor in adjacency.get(current, {}):
                if neighbor not in component_by_context:
                    component_by_context[neighbor] = component_id
                    stack.append(neighbor)
        components.append(sorted(members))

    world_contexts = {context for context in all_contexts if "|World " in context}

    def world_path(start: str) -> tuple[int | None, int | None]:
        if start in world_contexts:
            return 0, None
        queue: list[tuple[str, int, int | None]] = [(start, 0, None)]
        seen = {start}
        while queue:
            node, distance, bottleneck = queue.pop(0)
            for neighbor, strength in adjacency.get(node, {}).items():
                if neighbor in seen:
                    continue
                next_bottleneck = strength if bottleneck is None else min(bottleneck, strength)
                if neighbor in world_contexts:
                    return distance + 1, next_bottleneck
                seen.add(neighbor)
                queue.append((neighbor, distance + 1, next_bottleneck))
        return None, None

    context_coverage: dict[str, dict[str, int]] = {}
    for context in sorted(context_counts):
        summary = dict(context_counts[context])
        athletes = context_athletes[context]
        summary["normalized_athletes"] = len(athletes)
        summary["bottom_half_normalized_athletes"] = len(
            context_bottom_half_athletes[context]
        )
        summary["item_calibration_eligible_athletes"] = len(
            context_item_athletes[context]
        )
        summary["athletes_with_strictly_prior_other_context"] = len(
            prior_bridge_athletes[context]
        )
        summary["athletes_with_strictly_prior_ifsc_context"] = len(
            prior_ifsc_bridge_athletes[context]
        )
        summary["athletes_with_strictly_prior_world_context"] = len(
            prior_world_bridge_athletes[context]
        )
        summary["repeated_youth_winner_athletes"] = len(
            context_item_athletes[context] & repeated_youth_winners
        )
        distance, bottleneck = world_path(context)
        summary["retrospective_component_id"] = component_by_context[context]
        summary["retrospective_graph_distance_to_world_tier"] = (
            distance if distance is not None else -1
        )
        summary["weakest_bridge_athlete_count_on_shortest_world_path"] = (
            bottleneck if bottleneck is not None else 0
        )
        summary["component_location_uncertainty_eligible"] = int(
            distance == 0 or (distance is not None and (bottleneck or 0) >= 3)
        )
        item_groups = context_item_outcomes[context]
        summary["item_count"] = len(item_groups)
        for outcomes in item_groups.values():
            n_rows = len(outcomes)
            zones = sum(zone for zone, _ in outcomes)
            tops = sum(top for _, top in outcomes)
            if zones == 0:
                summary["items_all_no_zone_censored"] = (
                    summary.get("items_all_no_zone_censored", 0) + 1
                )
                summary["items_no_post_zone_opportunity"] = (
                    summary.get("items_no_post_zone_opportunity", 0) + 1
                )
            elif zones == n_rows:
                summary["items_all_zone_censored"] = (
                    summary.get("items_all_zone_censored", 0) + 1
                )
            else:
                summary["items_mixed_zone"] = summary.get("items_mixed_zone", 0) + 1
            if zones > 0:
                if tops == 0:
                    summary["items_all_no_top_given_zone_censored"] = (
                        summary.get("items_all_no_top_given_zone_censored", 0) + 1
                    )
                elif tops == zones:
                    summary["items_all_top_given_zone_censored"] = (
                        summary.get("items_all_top_given_zone_censored", 0) + 1
                    )
                else:
                    summary["items_mixed_top_given_zone"] = (
                        summary.get("items_mixed_top_given_zone", 0) + 1
                    )
        context_coverage[context] = dict(sorted(summary.items()))

    input_files = {
        str(path): {"sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in [*staged_paths, identity_safe_path]
    }
    manifest: dict[str, object] = {
        "schema": ADAPTER_SCHEMA,
        "research_only": True,
        "production_merged": False,
        "full_history_model_fit_run": False,
        "start_date": start_date,
        "rating_pool_id": DEFAULT_RATING_POOL,
        "attempt_policy": (
            "per-problem marker success-attempt ordinals preserved as provenance; "
            "round aggregate top_attempts/zone_attempts are ignored for official tie "
            "score because they can include raw failed-hurdle tries; all attempt "
            "evidence is withheld from ProblemOutcome until a predeclared opportunity "
            "horizon exists"
        ),
        "combined_policy": (
            "retain low-zone/zone/top evidence; withhold every 3-hurdle row from "
            "the current 2-hurdle terrain model"
        ),
        "censoring_policy": (
            "all-no-Zone, all-Zone, all-no-Top-given-Zone and all-Top-given-Zone "
            "items remain eligible; the Item-Elo posterior reports one-sided or "
            "prior-only status and broad uncertainty rather than dropping easy/hard items"
        ),
        "identity_policy": (
            "exact identity-safe round snapshot for model-ready rows; unique source-node "
            "identity for marker-only rows; ambiguous or absent nodes split source-locally"
        ),
        "reviewed_cec_shared_terrain_events": sorted(
            REVIEWED_CEC_SHARED_TERRAIN_EVENTS
        ),
        "eligibility": {
            "item_calibration": (
                "ordinary 2-hurdle explicit marker + federation route ID + exact "
                "identity-safe frozen Global-Elo snapshot + confirmed procedure"
            ),
            "chronological_response_candidate": (
                f"item eligible, LOO field >=2, athlete >= {minimum_athlete_items} "
                f"problems across >= {minimum_athlete_competitions} whole competitions; "
                "still requires rolling-origin Item-Elo calibration before fitting"
            ),
        },
        "counts": dict(sorted(counts.items())),
        "context_coverage": context_coverage,
        "connectivity_contract": {
            "edge_rule": (
                "same stable athlete observed in origin context at a strictly earlier "
                "competition date than destination; same-event/later evidence forbidden"
            ),
            "artifact_as_of": max((value[0] for value in eligible_appearances), default=None),
            "rolling_origin_requirement": (
                "rebuild edges/components using first_established_date < target start; "
                "retrospective component IDs are QA only"
            ),
            "repeated_youth_winner_definition": (
                "rank 1 in the Final (or sole published round) of at least two "
                "distinct youth competition IDs"
            ),
            "repeated_youth_winner_athletes": len(repeated_youth_winners),
            "component_location_uncertainty_candidate_rule": (
                "same world context or a retrospective path whose weakest edge has >=3 "
                "distinct bridge athletes; research eligibility only, not validation"
            ),
            "component_count": len(components),
            "components": [
                {"component_id": index, "contexts": members}
                for index, members in enumerate(components)
            ],
            "directed_bridge_edges": sorted(
                serialized_edges,
                key=lambda value: (
                    str(value["first_established_date"]),
                    str(value["origin_context"]),
                    str(value["destination_context"]),
                ),
            ),
        },
        "focus_cec_regional_local_youth_u15": context_coverage.get(
            "CEC|Regional / local youth|U15", {}
        ),
        "by_source": {
            scope: dict(sorted(source_counts.items()))
            for scope, source_counts in sorted(by_source.items())
        },
        "excluded_hidden_test_events": [
            {
                "source_scope": scope,
                "source_event_id": event_id,
                "event_name": value[0],
                "event_date": value[1],
            }
            for (scope, event_id), value in sorted(excluded_events.items())
        ],
        "input_files": input_files,
        "remaining_gates": [
            (
                f"{counts.get('combined_round_rows_awaiting_nested_evidence', 0):,} "
                "combined round rows still require the nested Boulder-stage refresh; "
                "they are not silently treated as zero outcomes"
            ),
            (
                f"{counts.get('combined_low_zone_zone_top_3h_rows', 0):,} staged "
                "three-hurdle rows are retained but require a generic "
                "low-zone/zone/top response model before use"
            ),
            "source attempt ordinals need a defensible predeclared opportunity horizon",
            "future Item-Elo and athlete-response fitting must be competition-cross-fitted inside rolling origins",
            (
                "context bridge components must be rebuilt with edge first_established_date "
                "strictly before every target; retrospective connectivity cannot initialize "
                "a past athlete"
            ),
            "style tags and frozen target-course distributions are not supplied by this adapter",
        ],
    }
    sample = [
        evidence
        for label in sorted(sample_by_class)
        for evidence in sample_by_class[label]
    ]
    return manifest, sample


def write_smoke_outputs(
    manifest: Mapping[str, object],
    sample: Sequence[TerrainProblemEvidence],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / "normalized_sample.csv"
    rows = [asdict(row) for row in sample]
    if rows:
        with sample_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                row["contradictions"] = " | ".join(row["contradictions"])
                writer.writerow(row)
    else:
        sample_path.write_text("", encoding="utf-8")
    bound_manifest = dict(manifest)
    bound_manifest["output_files"] = {
        sample_path.name: {
            "bytes": sample_path.stat().st_size,
            "sha256": _sha256(sample_path),
        }
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(bound_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "ADAPTER_SCHEMA",
    "IdentityLookups",
    "REVIEWED_CEC_SHARED_TERRAIN_EVENTS",
    "StableSnapshot",
    "TerrainProblemAdapterError",
    "TerrainProblemEvidence",
    "audit_problem_evidence",
    "is_hidden_or_test_event",
    "load_identity_lookups",
    "normalize_round_row",
    "to_problem_outcome",
    "write_smoke_outputs",
]
