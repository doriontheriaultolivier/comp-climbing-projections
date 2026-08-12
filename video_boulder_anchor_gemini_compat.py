"""Gemini structured-output compatibility helpers for anchor discovery.

The evidence contract stays owned by :mod:`video_boulder_anchor_discovery`.
This module keeps the live-accepted provider schema unchanged. It adds concise
prompt guidance and post-response budget validation without introducing schema
keywords that previously caused HTTP 400 responses.
"""

from __future__ import annotations

from video_boulder_anchor_discovery import AnchorWindow, build_response_json_schema


SCHEMA_PROFILE = "gemini-flash-lite-accepted-schema-v2"
OUTPUT_BUDGETS = {
    "window_notes": 6,
    "anchor_candidates": 32,
    "scene_candidates": 24,
    "visible_intervals": 32,
}
MAX_IDENTITY_CUES_PER_ITEM = 4


def build_gemini_prompt(window: AnchorWindow) -> str:
    """Return concise instructions without duplicating the response schema."""

    task = (
        "Scan broadly for explicit clocks, graphics, scene changes, and visible "
        "athlete/boulder intervals."
        if window.pass_name == "discovery"
        else (
            "Verify only the requested candidate. Classify it as supported, "
            "conflicts, or not_visible; do not promote it to fact."
        )
    )
    verification_rule = ""
    if window.pass_name == "verification":
        verification_rule = (
            " Verify the exact SOURCE_CANDIDATE_JSON claim, including its type, text, "
            "timing and identity cues. A nearby but different graphic conflicts: event titles "
            "and rankings are not athlete nameplates; weather/temperature text is not a "
            "competition clock; rankings are not round transitions; an athlete nameplate needs "
            "explicit athlete identity. Use not_visible when the claim "
            "cannot be seen."
        )
    return (
        "You extract shallow, directly visible evidence from an official World "
        "Climbing Boulder broadcast. "
        f"{task} Use absolute broadcast seconds inside the requested window. "
        "Record literal clock/graphic text, scene events, candidate identity cues, "
        "boulder number, visible intervals, confidence, and uncertainty only. "
        "Do not infer movement, beta, tactics, route demands, affect, emotion, "
        "psychology, coaching advice, training, Elo, or performance projections. "
        "Multiple climbers may be visible; keep their intervals separate. Do not "
        "emit every repeated clock frame: keep one useful candidate per distinct "
        "turn, graphic, or transition. Use M1, M2, M3, and M4 as neutral internal "
        "tokens for Boulder slots 1, 2, 3, and 4 in every gender; the M prefix does "
        "not mean men's. Preserve any literal W1/M1-style label in graphic_text. "
        "Use unknown when the slot is not visible. Return at most 6 window notes, "
        "32 anchor candidates, 24 scene candidates, 32 visible intervals, and 4 "
        "identity cues per candidate or interval. Empty arrays are valid when evidence is absent. "
        "The API response schema defines the exact JSON fields; return only that JSON."
        + verification_rule
        + "\n\n"
        f"window_id={window.window_id}\n"
        f"broadcast_seconds={window.start_seconds}..{window.end_seconds}\n"
        f"event={window.event}\n"
        f"category={window.gender} {window.round}\n"
        f"video_id={window.video_id}\n"
        f"source_candidate_id={window.source_candidate_id or 'none'}\n"
        f"SOURCE_CANDIDATE_JSON={window.source_candidate_json or 'none'}"
    )


def build_gemini_response_schema(window: AnchorWindow) -> dict[str, object]:
    """Return the exact schema shape already accepted by the live Gemini API.

    The first cloud run proved this canonical schema is accepted. Adding
    provider-side ``maxItems`` and request-specific enums subsequently caused
    HTTP 400 for every event. Bounds remain enforced in the concise prompt and
    by prompt guidance and a local fail-closed budget validator, where
    violations can be retried and audited instead of making the request invalid.
    """

    return build_response_json_schema(window.pass_name)


def validate_gemini_output_budget(payload: dict[str, object]) -> list[str]:
    """Reject oversized responses locally without changing the provider schema."""

    errors: list[str] = []
    for field, limit in OUTPUT_BUDGETS.items():
        value = payload.get(field)
        if isinstance(value, list) and len(value) > limit:
            errors.append(f"{field} has {len(value)} items; maximum is {limit}")

    for field in ("anchor_candidates", "visible_intervals"):
        values = payload.get(field)
        if not isinstance(values, list):
            continue
        for index, item in enumerate(values):
            if not isinstance(item, dict):
                continue
            cues = item.get("identity_cues")
            if isinstance(cues, list) and len(cues) > MAX_IDENTITY_CUES_PER_ITEM:
                item_id = item.get("candidate_id") or item.get("interval_id") or index
                errors.append(
                    f"{field} {item_id} has {len(cues)} identity cues; "
                    f"maximum is {MAX_IDENTITY_CUES_PER_ITEM}"
                )
    return errors
