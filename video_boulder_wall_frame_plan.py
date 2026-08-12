"""Plan reviewable Boulder-wall frame candidates from verified anchor windows.

This is intentionally a planning layer, not a vision claim.  A Gemini anchor
review can identify a Boulder slot and visible-athlete intervals, but cannot
prove that a selected video frame has no climber or is the best view of the
holds.  The resulting frames therefore stay *candidates for human/tagger
review* until a later visual-review workflow accepts one.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable


PIPELINE_VERSION = "ifsc-boulder-wall-frame-plan-v1"
BOULDER_SLOTS = {"M1", "M2", "M3", "M4"}
SAFETY_GATES = (
    "production_use_allowed",
    "athlete_scoring_allowed",
    "athlete_comparison_allowed",
    "elo_update_allowed",
)


@dataclass(frozen=True)
class FrameCandidate:
    candidate_id: str
    event_id: int
    category_round_id: int
    event: str
    gender: str
    round: str
    video_id: str
    youtube_url: str
    boulder_slot: str
    source_window_id: str
    source_candidate_id: str
    source_anchor_seconds: float
    frame_seconds: float
    nearest_visible_interval_clearance_seconds: float
    source_review_sha256: str


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _number(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if result < 0:
        raise ValueError(f"{field} must be non-negative")
    return result


def _free_segments(start: float, end: float, intervals: Iterable[tuple[float, float]], margin: float) -> list[tuple[float, float]]:
    """Return parts of a review window not covered by reported visible people."""

    blocked = sorted(
        (max(start, left - margin), min(end, right + margin))
        for left, right in intervals
        if right > start and left < end
    )
    merged: list[list[float]] = []
    for left, right in blocked:
        if left >= right:
            continue
        if not merged or left > merged[-1][1]:
            merged.append([left, right])
        else:
            merged[-1][1] = max(merged[-1][1], right)
    free: list[tuple[float, float]] = []
    cursor = start
    for left, right in merged:
        if left > cursor:
            free.append((cursor, left))
        cursor = max(cursor, right)
    if cursor < end:
        free.append((cursor, end))
    return free


def _clearance(point: float, intervals: Iterable[tuple[float, float]]) -> float:
    values = [min(abs(point - left), abs(point - right)) if not left <= point <= right else 0.0 for left, right in intervals]
    return min(values) if values else float("inf")


def plan_frame_candidates(
    records: Iterable[dict[str, object]], *, minimum_clearance_seconds: float = 2.0
) -> list[FrameCandidate]:
    """Choose at most one maximally-clear timestamp per supported anchor window.

    Only an explicit supported verification record, whose source anchor has a
    governed slot, can emit a candidate.  We preserve the source interval and
    choose the midpoint of the largest reported-person-free segment.  This
    produces reproducible candidates without falsely calling them empty walls.
    """

    if not 0 <= minimum_clearance_seconds <= 10:
        raise ValueError("minimum_clearance_seconds must be between 0 and 10")
    output: list[FrameCandidate] = []
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or record.get("pass_name") != "verification":
            raise ValueError("frame planning requires verification records only")
        if any(record.get(gate) is not False for gate in SAFETY_GATES):
            raise ValueError("verification record has an open safety gate")
        response = record.get("response")
        if not isinstance(response, dict):
            raise ValueError("verification record has no response")
        verification = response.get("verification")
        if not isinstance(verification, dict) or verification.get("status") != "supported":
            continue
        raw_source = record.get("source_candidate_json")
        try:
            source = json.loads(str(raw_source))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("verification record has invalid source candidate JSON") from exc
        if not isinstance(source, dict):
            raise ValueError("verification source candidate must be an object")
        slot = str(source.get("boulder_number", ""))
        if slot not in BOULDER_SLOTS:
            continue
        source_id = str(record.get("source_candidate_id", ""))
        if source.get("candidate_id") != source_id or verification.get("source_candidate_id") != source_id:
            raise ValueError("verification source candidate binding mismatch")
        start = _number(record.get("window_start_seconds"), "window_start_seconds")
        end = _number(record.get("window_end_seconds"), "window_end_seconds")
        if not start < end:
            raise ValueError("verification window must be non-empty")
        intervals: list[tuple[float, float]] = []
        for item in response.get("visible_intervals", []):
            if not isinstance(item, dict):
                raise ValueError("visible interval must be an object")
            if item.get("boulder_number") not in {slot, "unknown"}:
                continue
            left = _number(item.get("start_seconds"), "visible interval start_seconds")
            right = _number(item.get("end_seconds"), "visible interval end_seconds")
            if not start <= left < right <= end:
                raise ValueError("visible interval escapes verification window")
            intervals.append((left, right))
        free = _free_segments(start, end, intervals, minimum_clearance_seconds)
        if not free:
            continue
        # Largest gap, then earliest time: deterministic and amenable to review.
        left, right = max(free, key=lambda segment: (segment[1] - segment[0], -segment[0]))
        frame_seconds = round((left + right) / 2, 3)
        clearance = _clearance(frame_seconds, intervals)
        if clearance < minimum_clearance_seconds:
            continue
        identity = f"{record.get('window_id')}:{source_id}:{slot}:{frame_seconds:.3f}"
        candidate_id = "BWF-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        if candidate_id in seen:
            raise ValueError("duplicate frame candidate")
        seen.add(candidate_id)
        output.append(FrameCandidate(
            candidate_id=candidate_id,
            event_id=int(record["event_id"]),
            category_round_id=int(record["category_round_id"]),
            event=str(record["event"]), gender=str(record["gender"]), round=str(record["round"]),
            video_id=str(record["video_id"]), youtube_url=str(record["youtube_url"]),
            boulder_slot=slot, source_window_id=str(record["window_id"]),
            source_candidate_id=source_id,
            source_anchor_seconds=_number(source.get("broadcast_seconds"), "source anchor broadcast_seconds"),
            frame_seconds=frame_seconds,
            nearest_visible_interval_clearance_seconds=round(clearance, 3),
            source_review_sha256=sha256_bytes(_canonical_json(record)),
        ))
    return sorted(output, key=lambda item: (item.event_id, item.category_round_id, item.boulder_slot, item.frame_seconds, item.candidate_id))


def plan_records(candidates: Iterable[FrameCandidate]) -> list[dict[str, object]]:
    return [
        {
            "pipeline_version": PIPELINE_VERSION,
            **candidate.__dict__,
            "candidate_status": "REQUIRES_VISUAL_EMPTY_WALL_REVIEW",
            "empty_wall_verified": False,
            "media_download_required": True,
            **{gate: False for gate in SAFETY_GATES},
        }
        for candidate in candidates
    ]
