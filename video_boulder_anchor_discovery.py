"""Two-pass, shallow Boulder broadcast-anchor discovery contracts.

The module is Gemini-compatible but imports no Gemini or vision runtime. It
plans direct official-YouTube windows and validates observable evidence only.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Iterable

import pandas as pd


PIPELINE_VERSION = "ifsc-boulder-anchor-discovery-v1"
DISCOVERY_SECONDS = 1500
DISCOVERY_FPS = 0.25
VERIFICATION_CONTEXT_SECONDS = 60
VERIFICATION_FPS = 1.0
TARGET_EVENTS = {1479, 1480, 1482}
TARGET_ROUNDS = {"Semi-final", "Final"}
ANCHOR_TYPES = {
    "competition_clock_graphic",
    "athlete_nameplate_graphic",
    "boulder_number_graphic",
    "round_transition_graphic",
    "technical_incident_graphic",
}
SCENE_TYPES = {
    "hard_scene_cut",
    "camera_view_switch",
    "replay_start",
    "replay_end",
}
CUE_TYPES = {"visible_name", "bib", "country_code", "commentary_name", "graphic_name"}
FORBIDDEN_KEYS = {
    "tactics",
    "tactical_observations",
    "affect",
    "emotion",
    "psychology",
    "elo",
    "training",
    "route_tags",
    "beta",
    "movement",
    "coaching_advice",
}
BOULDER_NUMBERS = {"M1", "M2", "M3", "M4", "unknown"}
REFERENCE_HINTS = {"turn_start", "turn_end", "unknown"}
VERIFICATION_STATUSES = {"supported", "conflicts", "not_visible"}


def _obvious_source_semantic_conflict(source_candidate: dict[str, object]) -> str:
    """Return a deterministic reason when an anchor label contradicts its literal evidence."""

    anchor_type = str(source_candidate.get("anchor_type", "")).strip()
    clock_text = str(source_candidate.get("clock_text", "")).strip()
    graphic_text = str(source_candidate.get("graphic_text", "")).strip()
    literal = f"{clock_text} {graphic_text}".casefold()
    cues = source_candidate.get("identity_cues", [])
    cue_types = {
        str(cue.get("cue_type", "")).strip()
        for cue in cues
        if isinstance(cue, dict) and str(cue.get("cue_text", "")).strip()
    } if isinstance(cues, list) else set()

    weather_literal = bool(
        re.search(r"\b(?:temperature|humidity|weather)\b|\d{1,2}\s*°\s*[cf]\b", literal)
    )
    ranking_literal = bool(re.search(r"\b(?:ranking|standings?)\b", literal))
    calendar_literal = bool(re.search(r"\b(?:calendar|schedule)\b", literal))
    round_literal = bool(
        re.search(r"\b(?:qualification|qualifying|semi(?:-final)?|final|round)\b", literal)
    )
    explicit_identity = bool(cue_types & {"visible_name", "graphic_name", "bib"})

    if anchor_type == "competition_clock_graphic" and weather_literal:
        return "temperature, humidity or weather text is not a competition clock"
    if anchor_type == "athlete_nameplate_graphic" and (
        ranking_literal or (not explicit_identity and "world climbing" in literal)
    ):
        return "event or ranking graphics without an explicit athlete identity are not nameplates"
    if anchor_type == "round_transition_graphic" and not round_literal and (
        ranking_literal or calendar_literal
    ):
        return "calendar or ranking graphics are not round transitions"
    return ""


@dataclass(frozen=True)
class AnchorWindow:
    window_id: str
    pass_name: str
    event_id: int
    category_round_id: int
    event: str
    gender: str
    round: str
    video_id: str
    youtube_url: str
    start_seconds: int
    end_seconds: int
    fps: float
    source_candidate_id: str = ""
    source_discovery_window_id: str = ""
    source_candidate_json: str = ""


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-")


def _target_manifest(manifest: pd.DataFrame, event_id: int | None = None, *, target_events: set[int] | None = None) -> pd.DataFrame:
    target_events = TARGET_EVENTS if target_events is None else set(target_events)
    if not target_events:
        raise ValueError("anchor discovery requires target events")
    required = {
        "event_id", "category_round_id", "event", "discipline", "gender",
        "round", "video_id", "official_youtube_url", "duration_seconds",
        "official_channel", "metadata_status",
    }
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError("video manifest is missing: " + ", ".join(sorted(missing)))
    selected = manifest.loc[
        pd.to_numeric(manifest["event_id"], errors="coerce").isin(target_events)
        & manifest["discipline"].astype(str).eq("Boulder")
        & manifest["round"].astype(str).isin(TARGET_ROUNDS)
    ].copy()
    if event_id is not None:
        if event_id not in target_events:
            raise ValueError("event_id must be in the declared target events")
        selected = selected.loc[
            pd.to_numeric(selected["event_id"], errors="coerce").eq(event_id)
        ].copy()
    expected_sources = 4 if event_id is not None else 4 * len(target_events)
    if len(selected) != expected_sources:
        raise ValueError(
            f"anchor discovery requires exactly {expected_sources} priority Boulder broadcasts"
        )
    if not selected["official_channel"].astype(str).eq("World Climbing").all():
        raise ValueError("anchor discovery sources must be official World Climbing uploads")
    if not selected["metadata_status"].astype(str).eq(
        "Verified by public YouTube oEmbed"
    ).all():
        raise ValueError("anchor discovery sources require verified public metadata")
    return selected


def build_discovery_plan(
    manifest: pd.DataFrame,
    *,
    window_seconds: int = DISCOVERY_SECONDS,
    fps: float = DISCOVERY_FPS,
    event_id: int | None = None,
    target_events: set[int] | None = None,
) -> list[AnchorWindow]:
    """Plan 20–30 minute, non-overlapping, low-FPS discovery windows."""

    if not 1200 <= window_seconds <= 1800:
        raise ValueError("discovery windows must be 20 to 30 minutes")
    if not 0 < fps <= 0.5:
        raise ValueError("discovery FPS must be above 0 and no greater than 0.5")
    selected = _target_manifest(manifest, event_id=event_id, target_events=target_events)
    rows: list[AnchorWindow] = []
    ordered = selected.sort_values(
        ["event_id", "gender", "round", "category_round_id"], kind="stable"
    )
    for source in ordered.itertuples(index=False):
        duration = int(source.duration_seconds)
        part_count = max(1, math.ceil(duration / window_seconds))
        while part_count > 1 and duration / part_count < 1200:
            part_count -= 1
        if duration / part_count > 1800:
            raise ValueError(f"cannot partition {source.video_id} into 20-30 minute windows")
        boundaries = [round(duration * number / part_count) for number in range(part_count + 1)]
        for number, (start, end) in enumerate(zip(boundaries, boundaries[1:]), start=1):
            rows.append(
                AnchorWindow(
                    window_id=(
                        f"BAD26-CR{int(source.category_round_id)}-"
                        f"{str(source.video_id)}-D{number:03d}"
                    ),
                    pass_name="discovery",
                    event_id=int(source.event_id),
                    category_round_id=int(source.category_round_id),
                    event=str(source.event),
                    gender=str(source.gender),
                    round=str(source.round),
                    video_id=str(source.video_id),
                    youtube_url=str(source.official_youtube_url),
                    start_seconds=start,
                    end_seconds=end,
                    fps=float(fps),
                )
            )
    return rows


def _all_candidates(record: dict[str, object]) -> Iterable[tuple[str, dict[str, object]]]:
    for field in ("anchor_candidates", "scene_candidates"):
        values = record.get(field, [])
        if isinstance(values, list):
            for value in values:
                if isinstance(value, dict):
                    yield field, value


def build_verification_plan(
    discovery_records: list[dict[str, object]],
    *,
    context_seconds: int = VERIFICATION_CONTEXT_SECONDS,
    fps: float = VERIFICATION_FPS,
    candidate_fields: tuple[str, ...] = ("anchor_candidates", "scene_candidates"),
) -> list[AnchorWindow]:
    """Plan short second-pass windows around explicit first-pass candidates."""

    if not 20 <= context_seconds <= 120:
        raise ValueError("verification context must be 20 to 120 seconds")
    if not 0 < fps <= 2:
        raise ValueError("verification FPS must be above 0 and no greater than 2")
    allowed_fields = {"anchor_candidates", "scene_candidates"}
    if not candidate_fields or len(candidate_fields) != len(set(candidate_fields)) or not set(candidate_fields) <= allowed_fields:
        raise ValueError("verification candidate fields must be a unique non-empty subset of anchor_candidates and scene_candidates")
    rows: list[AnchorWindow] = []
    seen: set[str] = set()
    half = context_seconds / 2
    for record in discovery_records:
        source_window = str(record.get("window_id", ""))
        duration = int(record.get("video_duration_seconds", 0))
        for field, candidate in _all_candidates(record):
            if field not in candidate_fields:
                continue
            candidate_id = str(candidate.get("candidate_id", "")).strip()
            try:
                timestamp = float(candidate.get("broadcast_seconds"))
            except (TypeError, ValueError):
                continue
            if not candidate_id or duration <= 0 or not 0 <= timestamp <= duration:
                continue
            identity = f"{source_window}:{field}:{candidate_id}"
            digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
            window_id = f"BAD26-V-{digest}"
            if window_id in seen:
                continue
            seen.add(window_id)
            start = max(0, int(timestamp - half))
            end = min(duration, start + context_seconds)
            start = max(0, end - context_seconds)
            rows.append(
                AnchorWindow(
                    window_id=window_id,
                    pass_name="verification",
                    event_id=int(record["event_id"]),
                    category_round_id=int(record["category_round_id"]),
                    event=str(record["event"]),
                    gender=str(record["gender"]),
                    round=str(record["round"]),
                    video_id=str(record["video_id"]),
                    youtube_url=str(record["youtube_url"]),
                    start_seconds=start,
                    end_seconds=end,
                    fps=float(fps),
                    source_candidate_id=candidate_id,
                    source_discovery_window_id=source_window,
                    source_candidate_json=json.dumps(
                        candidate,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                )
            )
    return sorted(rows, key=lambda value: (value.event_id, value.video_id, value.start_seconds, value.window_id))


def plan_frame(windows: Iterable[AnchorWindow]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pipeline_version": PIPELINE_VERSION,
                **window.__dict__,
                "window_duration_seconds": window.end_seconds - window.start_seconds,
                "media_download_required": False,
                "shallow_evidence_only": True,
            }
            for window in windows
        ]
    )


def response_shape(pass_name: str) -> dict[str, object]:
    common = {
        "window_id": "must equal requested window ID",
        "window_notes": ["short observable note; no analysis or advice"],
        "anchor_candidates": [
            {
                "candidate_id": "unique in window",
                "broadcast_seconds": 0.0,
                "evidence_start_seconds": 0.0,
                "evidence_end_seconds": 0.0,
                "anchor_type": "one governed anchor type",
                "clock_text": "explicit visible clock text or empty",
                "graphic_text": "explicit visible graphic text or empty",
                "boulder_number": "M1 | M2 | M3 | M4 | unknown",
                "identity_cues": [
                    {
                        "cue_type": "visible_name | bib | country_code | commentary_name | graphic_name",
                        "cue_text": "literal observed cue",
                        "confidence_0_1": 0.0,
                    }
                ],
                "reference_point_hint": "turn_start | turn_end | unknown",
                "confidence_0_1": 0.0,
                "observable_evidence_note": "what is explicitly visible or audible",
                "uncertainty_note": "what remains uncertain",
            }
        ],
        "scene_candidates": [
            {
                "candidate_id": "unique in window",
                "broadcast_seconds": 0.0,
                "evidence_start_seconds": 0.0,
                "evidence_end_seconds": 0.0,
                "scene_type": "hard_scene_cut | camera_view_switch | replay_start | replay_end",
                "boulder_number": "M1 | M2 | M3 | M4 | unknown",
                "confidence_0_1": 0.0,
                "observable_evidence_note": "literal scene evidence",
                "uncertainty_note": "what remains uncertain",
            }
        ],
        "visible_intervals": [
            {
                "interval_id": "unique in window",
                "start_seconds": 0.0,
                "end_seconds": 0.0,
                "boulder_number": "M1 | M2 | M3 | M4 | unknown",
                "identity_cues": [],
                "confidence_0_1": 0.0,
                "uncertainty_note": "camera or identity limitation",
            }
        ],
    }
    if pass_name == "verification":
        common["verification"] = {
            "source_candidate_id": "must equal requested candidate ID",
            "status": "supported | conflicts | not_visible",
            "confidence_0_1": 0.0,
            "observable_evidence_note": "short evidence",
            "uncertainty_note": "short uncertainty",
        }
    return common


def build_prompt(window: AnchorWindow) -> str:
    pass_instruction = (
        "Scan broadly for explicit clocks, graphics, scene changes and visible athlete/boulder intervals."
        if window.pass_name == "discovery"
        else (
            "Verify only the requested candidate. Report supported, conflicts or not_visible; "
            "do not promote it to fact."
        )
    )
    candidate_context = ""
    if window.pass_name == "verification":
        candidate_context = (
            "The exact discovery candidate to verify is below. Judge this specific claim, "
            "including its candidate type, literal text, timing and identity cues. A nearby "
            "graphic is not enough when it is a different kind of graphic. For example, an "
            "event title or ranking table does not support athlete_nameplate_graphic; temperature "
            "or weather text is not a competition clock; a ranking table does not support "
            "round_transition_graphic; and an athlete nameplate needs an "
            "athlete name or equivalent explicit identity cue. Use conflicts when the visible "
            "content contradicts or materially misclassifies the candidate, and not_visible when "
            "the requested evidence cannot be seen.\n"
            f"SOURCE_CANDIDATE_JSON={window.source_candidate_json}\n\n"
        )
    return (
        "You are a shallow evidence extractor for an official World Climbing Boulder broadcast. "
        + pass_instruction
        + " Use absolute broadcast seconds. Record only literal clock/graphic text, scene events, "
        "candidate identity cues, boulder number, visible intervals and uncertainty. Do not return "
        "movement, beta, tactics, route demands, affect, emotion, psychology, coaching advice, "
        "training recommendations, Elo or performance projections. Multiple climbers may be active; "
        "do not collapse them into one identity. Empty arrays are correct when evidence is absent.\n\n"
        f"Window {window.window_id}: seconds {window.start_seconds}-{window.end_seconds}; "
        f"{window.event}; {window.gender} {window.round}; video {window.video_id}; "
        f"source candidate {window.source_candidate_id or 'none'}.\n\n"
        + candidate_context
        + "Return JSON only with this exact shallow structure:\n"
        + json.dumps(response_shape(window.pass_name), ensure_ascii=False, indent=2)
    )


def build_response_json_schema(pass_name: str) -> dict[str, object]:
    cue = {
        "type": "object",
        "additionalProperties": False,
        "required": ["cue_type", "cue_text", "confidence_0_1"],
        "properties": {
            "cue_type": {"type": "string", "enum": sorted(CUE_TYPES)},
            "cue_text": {"type": "string"},
            "confidence_0_1": {"type": "number", "minimum": 0, "maximum": 1},
        },
    }
    timed = {
        "broadcast_seconds": {"type": "number", "minimum": 0},
        "evidence_start_seconds": {"type": "number", "minimum": 0},
        "evidence_end_seconds": {"type": "number", "minimum": 0},
    }
    anchor = {
        "type": "object", "additionalProperties": False,
        "required": [
            "candidate_id", *timed, "anchor_type", "clock_text", "graphic_text",
            "boulder_number", "identity_cues", "reference_point_hint", "confidence_0_1",
            "observable_evidence_note", "uncertainty_note",
        ],
        "properties": {
            "candidate_id": {"type": "string"}, **timed,
            "anchor_type": {"type": "string", "enum": sorted(ANCHOR_TYPES)},
            "clock_text": {"type": "string"}, "graphic_text": {"type": "string"},
            "boulder_number": {"type": "string", "enum": ["M1", "M2", "M3", "M4", "unknown"]},
            "identity_cues": {"type": "array", "items": cue},
            "reference_point_hint": {"type": "string", "enum": ["turn_start", "turn_end", "unknown"]},
            "confidence_0_1": {"type": "number", "minimum": 0, "maximum": 1},
            "observable_evidence_note": {"type": "string"},
            "uncertainty_note": {"type": "string"},
        },
    }
    scene = {
        "type": "object", "additionalProperties": False,
        "required": [
            "candidate_id", *timed, "scene_type", "boulder_number", "confidence_0_1",
            "observable_evidence_note", "uncertainty_note",
        ],
        "properties": {
            "candidate_id": {"type": "string"}, **timed,
            "scene_type": {"type": "string", "enum": sorted(SCENE_TYPES)},
            "boulder_number": {"type": "string", "enum": ["M1", "M2", "M3", "M4", "unknown"]},
            "confidence_0_1": {"type": "number", "minimum": 0, "maximum": 1},
            "observable_evidence_note": {"type": "string"},
            "uncertainty_note": {"type": "string"},
        },
    }
    interval = {
        "type": "object", "additionalProperties": False,
        "required": ["interval_id", "start_seconds", "end_seconds", "boulder_number", "identity_cues", "confidence_0_1", "uncertainty_note"],
        "properties": {
            "interval_id": {"type": "string"},
            "start_seconds": {"type": "number", "minimum": 0},
            "end_seconds": {"type": "number", "minimum": 0},
            "boulder_number": {"type": "string", "enum": ["M1", "M2", "M3", "M4", "unknown"]},
            "identity_cues": {"type": "array", "items": cue},
            "confidence_0_1": {"type": "number", "minimum": 0, "maximum": 1},
            "uncertainty_note": {"type": "string"},
        },
    }
    properties: dict[str, object] = {
        "window_id": {"type": "string"},
        "window_notes": {"type": "array", "items": {"type": "string"}},
        "anchor_candidates": {"type": "array", "items": anchor},
        "scene_candidates": {"type": "array", "items": scene},
        "visible_intervals": {"type": "array", "items": interval},
    }
    required = list(properties)
    if pass_name == "verification":
        properties["verification"] = {
            "type": "object", "additionalProperties": False,
            "required": ["source_candidate_id", "status", "confidence_0_1", "observable_evidence_note", "uncertainty_note"],
            "properties": {
                "source_candidate_id": {"type": "string"},
                "status": {"type": "string", "enum": ["supported", "conflicts", "not_visible"]},
                "confidence_0_1": {"type": "number", "minimum": 0, "maximum": 1},
                "observable_evidence_note": {"type": "string"},
                "uncertainty_note": {"type": "string"},
            },
        }
        required.append("verification")
    return {
        "type": "object", "additionalProperties": False,
        "required": required, "properties": properties,
    }


def build_video_part(types_module: object, window: AnchorWindow) -> object:
    return types_module.Part(
        file_data=types_module.FileData(file_uri=window.youtube_url),
        video_metadata=types_module.VideoMetadata(
            start_offset=f"{window.start_seconds}s",
            end_offset=f"{window.end_seconds}s",
            fps=window.fps,
        ),
    )


def _forbidden_keys(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_KEYS:
                found.add(str(key))
            found.update(_forbidden_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_forbidden_keys(child))
    return found


def _has_exact_keys(value: dict[str, object], expected: set[str]) -> bool:
    return set(value) == expected


def _cue_is_valid(cue: object) -> bool:
    if not isinstance(cue, dict) or not _has_exact_keys(
        cue, {"cue_type", "cue_text", "confidence_0_1"}
    ):
        return False
    try:
        confidence = float(cue.get("confidence_0_1"))
    except (TypeError, ValueError):
        return False
    return (
        cue.get("cue_type") in CUE_TYPES
        and bool(str(cue.get("cue_text", "")).strip())
        and 0 <= confidence <= 1
    )


def validate_response(payload: dict[str, object], window: AnchorWindow) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict) or payload.get("window_id") != window.window_id:
        return ["response window_id must match the requested window"]
    if forbidden := _forbidden_keys(payload):
        errors.append("forbidden analytical fields: " + ", ".join(sorted(forbidden)))
    expected_keys = {
        "window_id", "window_notes", "anchor_candidates", "scene_candidates",
        "visible_intervals",
    }
    if window.pass_name == "verification":
        expected_keys.add("verification")
    if set(payload) != expected_keys:
        errors.append("response top-level fields must match the shallow schema exactly")
    if not isinstance(payload.get("window_notes"), list) or any(
        not isinstance(note, str) or not note.strip()
        for note in payload.get("window_notes", [])
    ):
        errors.append("window_notes must be a list")
    all_ids: list[str] = []
    for field, type_field, allowed in (
        ("anchor_candidates", "anchor_type", ANCHOR_TYPES),
        ("scene_candidates", "scene_type", SCENE_TYPES),
    ):
        values = payload.get(field)
        if not isinstance(values, list):
            errors.append(f"{field} must be a list")
            continue
        for candidate in values:
            if not isinstance(candidate, dict):
                errors.append(f"{field} entries must be objects")
                continue
            candidate_id = str(candidate.get("candidate_id", "")).strip()
            all_ids.append(candidate_id)
            common_keys = {
                "candidate_id", "broadcast_seconds", "evidence_start_seconds",
                "evidence_end_seconds", "boulder_number", "confidence_0_1",
                "observable_evidence_note", "uncertainty_note", type_field,
            }
            expected_candidate_keys = (
                common_keys
                | {"clock_text", "graphic_text", "identity_cues", "reference_point_hint"}
                if field == "anchor_candidates"
                else common_keys
            )
            if not _has_exact_keys(candidate, expected_candidate_keys):
                errors.append(f"{candidate_id}: fields must match the shallow schema exactly")
            try:
                timestamp = float(candidate.get("broadcast_seconds"))
                start = float(candidate.get("evidence_start_seconds"))
                end = float(candidate.get("evidence_end_seconds"))
                confidence = float(candidate.get("confidence_0_1"))
            except (TypeError, ValueError):
                errors.append(f"{candidate_id}: candidate timing/confidence is invalid")
                continue
            if not window.start_seconds <= start <= timestamp <= end <= window.end_seconds:
                errors.append(
                    f"{candidate_id}: evidence interval {start}-{timestamp}-{end} is outside "
                    f"source window {window.start_seconds}-{window.end_seconds}"
                )
            if not 0 <= confidence <= 1:
                errors.append(f"{candidate_id}: confidence must be 0-1")
            if candidate.get(type_field) not in allowed:
                errors.append(f"{candidate_id}: {type_field} is invalid")
            if candidate.get("boulder_number") not in BOULDER_NUMBERS:
                errors.append(f"{candidate_id}: boulder_number is invalid")
            for note in ("observable_evidence_note", "uncertainty_note"):
                if not str(candidate.get(note, "")).strip():
                    errors.append(f"{candidate_id}: {note} cannot be blank")
            if field == "anchor_candidates":
                cues = candidate.get("identity_cues")
                if not isinstance(cues, list):
                    errors.append(f"{candidate_id}: identity_cues must be a list")
                elif any(not _cue_is_valid(cue) for cue in cues):
                    errors.append(f"{candidate_id}: identity cue is invalid")
                if candidate.get("reference_point_hint") not in REFERENCE_HINTS:
                    errors.append(f"{candidate_id}: reference_point_hint is invalid")
                if not str(candidate.get("clock_text", "")).strip() and not str(
                    candidate.get("graphic_text", "")
                ).strip():
                    errors.append(f"{candidate_id}: anchor needs literal clock or graphic text")
    intervals = payload.get("visible_intervals")
    if not isinstance(intervals, list):
        errors.append("visible_intervals must be a list")
    else:
        for interval in intervals:
            if not isinstance(interval, dict):
                errors.append("visible interval must be an object")
                continue
            interval_id = str(interval.get("interval_id", "")).strip()
            all_ids.append(interval_id)
            if not _has_exact_keys(interval, {
                "interval_id", "start_seconds", "end_seconds", "boulder_number",
                "identity_cues", "confidence_0_1", "uncertainty_note",
            }):
                errors.append(f"{interval_id}: fields must match the shallow schema exactly")
            try:
                start = float(interval.get("start_seconds"))
                end = float(interval.get("end_seconds"))
                confidence = float(interval.get("confidence_0_1"))
            except (TypeError, ValueError):
                errors.append(f"{interval_id}: visible interval timing/confidence is invalid")
                continue
            if not window.start_seconds <= start < end <= window.end_seconds:
                errors.append(
                    f"{interval_id}: visible interval {start}-{end} is outside source window "
                    f"{window.start_seconds}-{window.end_seconds}"
                )
            if not 0 <= confidence <= 1:
                errors.append(f"{interval_id}: confidence must be 0-1")
            if interval.get("boulder_number") not in BOULDER_NUMBERS:
                errors.append(f"{interval_id}: boulder_number is invalid")
            cues = interval.get("identity_cues")
            if not isinstance(cues, list) or any(not _cue_is_valid(cue) for cue in cues):
                errors.append(f"{interval_id}: identity cue is invalid")
            if not str(interval.get("uncertainty_note", "")).strip():
                errors.append(f"{interval_id}: uncertainty_note cannot be blank")
    if "" in all_ids or len(all_ids) != len(set(all_ids)):
        errors.append("all candidate and interval IDs must be nonblank and unique")
    if window.pass_name == "verification":
        source_candidate: dict[str, object] | None = None
        try:
            decoded_candidate = json.loads(window.source_candidate_json)
            source_candidate = decoded_candidate if isinstance(decoded_candidate, dict) else None
            if source_candidate is None:
                errors.append("verification source candidate must be a JSON object")
            elif str(source_candidate.get("candidate_id", "")) != window.source_candidate_id:
                errors.append("verification source candidate JSON must match the plan ID")
        except (TypeError, ValueError, json.JSONDecodeError):
            errors.append("verification source candidate JSON is invalid")
        verification = payload.get("verification")
        if not isinstance(verification, dict):
            errors.append("verification object is required in pass two")
        else:
            if not _has_exact_keys(verification, {
                "source_candidate_id", "status", "confidence_0_1",
                "observable_evidence_note", "uncertainty_note",
            }):
                errors.append("verification fields must match the shallow schema exactly")
            if str(verification.get("source_candidate_id", "")) != window.source_candidate_id:
                errors.append("verification source candidate must match the plan")
            if verification.get("status") not in VERIFICATION_STATUSES:
                errors.append("verification status is invalid")
            if verification.get("status") == "supported" and source_candidate is not None:
                if conflict := _obvious_source_semantic_conflict(source_candidate):
                    errors.append(
                        "verification cannot support a semantically contradictory source claim: "
                        + conflict
                    )
            try:
                confidence = float(verification.get("confidence_0_1"))
                if not 0 <= confidence <= 1:
                    errors.append("verification confidence must be 0-1")
            except (TypeError, ValueError):
                errors.append("verification confidence is invalid")
            for note in ("observable_evidence_note", "uncertainty_note"):
                if not str(verification.get(note, "")).strip():
                    errors.append(f"verification {note} cannot be blank")
    elif "verification" in payload:
        errors.append("discovery response cannot carry verification output")
    return errors
