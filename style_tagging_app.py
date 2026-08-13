"""Standalone public tagger for governed Boulder route and segment observations."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

import pandas as pd
import streamlit as st


# Keep the tagger independently deployable.  It only needs the governed
# inventory file, not the projection application's import graph.
DATA = Path(__file__).resolve().parent / "data"


BOULDER_LABEL = re.compile(r"^([MW])\s*[-:]?\s*([1-9][0-9]*)$", re.IGNORECASE)
CORE_TAG_LABELS = {
    "physical_0_3": "Physical demand",
    "technical_0_3": "Technical demand",
    "coordination_0_3": "Coordination demand",
}
ROUTE_FIELD_LABELS = {
    "wall_overhang_0_3": "Wall angle demand", "three_dimensionality_0_3": "Three-dimensional movement",
    "dual_texture_friction_0_3": "Dual-texture / friction demand", "hold_positivity_0_3": "Handhold positivity",
    "crimp_edge_0_3": "Crimp / edge use", "open_hand_sloper_0_3": "Open-hand / sloper use",
    "pinch_0_3": "Pinch use", "compression_opposition_0_3": "Compression / opposition",
    "slow_technical_0_3": "Slow technical movement", "upper_pull_0_3": "Upper-body pulling",
    "upper_push_0_3": "Upper-body pushing", "fast_upper_accuracy_0_3": "Fast hand accuracy",
    "fast_lower_accuracy_0_3": "Fast foot accuracy", "lower_explosivity_0_3": "Lower-body explosivity",
    "body_tension_core_0_3": "Body tension / core", "mobility_flexibility_0_3": "Mobility / flexibility",
    "reach_span_0_3": "Reach / span", "read_complexity_0_3": "Reading complexity",
    "precision_risk_0_3": "Precision risk", "attempt_cost_0_3": "Attempt cost",
}


def canonical_boulder_label(value: str) -> str:
    match = BOULDER_LABEL.fullmatch(value.strip())
    if match is None:
        raise ValueError("Use a label such as M1 or W3")
    return f"{match.group(1).upper()}{int(match.group(2))}"


def route_fields() -> dict[str, str]:
    return dict(ROUTE_FIELD_LABELS)


FRAME_PROVENANCE_FIELDS = (
    "frame_candidate_id", "frame_sha256", "source_media_sha256",
    "source_video_id", "source_frame_seconds",
)


def matching_frame_receipts(receipt: object, problem: dict[str, object]) -> list[dict[str, object]]:
    """Return only review-only media candidates bound to this exact round/Boulder."""
    if not isinstance(receipt, dict) or not isinstance(receipt.get("frames"), list):
        return []
    try:
        round_ids = {int(value) for value in str(problem["source_round_ids"]).split(",") if value.strip()}
        boulder_number = int(problem["boulder_number"])
    except (KeyError, TypeError, ValueError):
        return []
    matched: list[dict[str, object]] = []
    for frame in receipt["frames"]:
        if not isinstance(frame, dict):
            continue
        slot = str(frame.get("boulder_slot", ""))
        try:
            slot_number = int(slot[1:]) if len(slot) > 1 and slot[0] in {"M", "W"} else -1
            round_id = int(frame.get("category_round_id"))
        except (TypeError, ValueError):
            continue
        if (
            round_id in round_ids and slot_number == boulder_number
            and frame.get("candidate_status") == "REQUIRES_VISUAL_EMPTY_WALL_REVIEW"
            and frame.get("empty_wall_verified") is False
        ):
            matched.append(frame)
    return sorted(matched, key=lambda frame: (str(frame.get("frame_seconds")), str(frame.get("candidate_id"))))


def build_record(
    problem: dict[str, object], *, confidence: str, pre_zone_direction: str,
    post_zone_direction: str, core_values: dict[str, tuple[int, int]],
    detailed_values: dict[str, tuple[int, int]], optional_tags_completed: bool,
    reviewer_code: str = "", image_name: str = "", image_bytes: bytes = b"",
    frame: dict[str, object] | None = None,
) -> dict[str, object]:
    """Create a schema-v4 record bound to a governed boulder and its segments."""
    number = int(problem["boulder_number"])
    record: dict[str, object] = {
        "schema_version": "4.0", "tag_taxonomy_version": "2026-08-03.1", "record_type": "style",
        "submitted_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "competition": str(problem["event_name"]), "competition_date": str(problem["event_date"]),
        "source_scope": str(problem["source_scope"]), "source_event_ids": str(problem["source_event_ids"]),
        "source_round_ids": str(problem["source_round_ids"]), "round": str(problem["round_group"]),
        "round_uid": str(problem["round_uid"]), "gender_terrain": str(problem["gender"]),
        "terrain_group": str(problem["terrain_group"]), "boulder": f"B{number}",
        "boulder_uid": str(problem["boulder_uid"]),
        "pre_zone_segment_uid": str(problem["pre_zone_segment_uid"]),
        "post_zone_segment_uid": str(problem["post_zone_segment_uid"]),
        "expected_boulders": int(problem["boulder_count"]),
        "boulder_count_status": str(problem["boulder_count_status"]),
        "boulder_count_source": "data/boulder_problem_inventory.csv.gz",
        "pre_zone_direction": pre_zone_direction, "post_zone_direction": post_zone_direction,
        "optional_tags_completed": optional_tags_completed, "confidence": confidence,
        "contributor": reviewer_code,
        "image_name": image_name,
        "image_sha256": hashlib.sha256(image_bytes).hexdigest() if image_bytes else "",
        "image_in_bundle": "" if not image_bytes else "uploaded_to_shared_backend",
    }
    if frame is not None:
        provenance = {
            "frame_candidate_id": str(frame.get("candidate_id", "")),
            "frame_sha256": str(frame.get("frame_sha256", "")),
            "source_media_sha256": str(frame.get("source_media_sha256", "")),
            "source_video_id": str(frame.get("video_id", "")),
            "source_frame_seconds": float(frame.get("frame_seconds")),
        }
        if not all(provenance[key] for key in FRAME_PROVENANCE_FIELDS):
            raise ValueError("frame provenance is incomplete")
        record.update(provenance)
    for field, (pre_zone, post_zone) in core_values.items():
        record[f"pre_zone_{field}"] = pre_zone
        record[f"post_zone_{field}"] = post_zone
    if optional_tags_completed:
        for field, (pre_zone, post_zone) in detailed_values.items():
            record[f"pre_zone_{field}"] = pre_zone
            record[f"post_zone_{field}"] = post_zone
    return record


@st.cache_data(show_spinner=False, ttl=1800, max_entries=1)
def tagging_priority_queue() -> pd.DataFrame:
    priority_path = DATA / "physical_item_tagging_priority_v1_1.csv"
    if not priority_path.exists():
        priority_path = DATA / "physical_item_tagging_priority_v1.csv"
    queue = (
        pd.read_csv(priority_path, low_memory=False)
        if priority_path.exists()
        else pd.DataFrame()
    )
    coaching_path = DATA / "physical_item_tag_unlock_app_v1.csv"
    if (
        coaching_path.exists()
        and not queue.empty
        and "problem_id" in queue.columns
    ):
        coaching = pd.read_csv(coaching_path, low_memory=False)
        if "problem_id" in coaching.columns and coaching["problem_id"].is_unique:
            queue = queue.merge(coaching, on="problem_id", how="left", validate="one_to_one")
    return queue


@st.cache_data(show_spinner=False, ttl=1800, max_entries=1)
def problem_inventory() -> pd.DataFrame:
    path = DATA / "boulder_problem_inventory.csv.gz"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, low_memory=False)
    frame["event_name"] = frame["event_name"].fillna("").astype(str).str.strip()
    frame["event_date"] = pd.to_datetime(frame["event_date"], errors="coerce")
    priority = tagging_priority_queue()
    return apply_tagging_priority(frame.loc[frame["event_name"].ne("")], priority)


def apply_tagging_priority(inventory: pd.DataFrame, priority: pd.DataFrame) -> pd.DataFrame:
    """Attach identity-free item priorities through exact round/problem keys."""

    rows = inventory.copy().reset_index(drop=True)
    rows["_inventory_index"] = rows.index
    rows["priority_rank"] = pd.NA
    rows["priority_source_items"] = pd.NA
    rows["priority_linked_athletes"] = pd.NA
    rows["priority_linked_outcomes"] = pd.NA
    rows["priority_board_linked_outcomes"] = 0
    rows["priority_top_given_zone_pairs"] = 0
    rows["priority_zone_pairs"] = 0
    rows["coaching_unlock_rank"] = pd.NA
    rows["coaching_unlock_source_items"] = 0
    rows["coaching_unlock_athletes"] = pd.NA
    rows["coaching_unlock_observations"] = pd.NA
    rows["coaching_unlock_physical_observations"] = pd.NA
    rows["coaching_unlock_board_observations"] = pd.NA
    required = {
        "source_scope", "source_event_id", "source_round_id", "boulder_number",
        "priority_rank", "linked_athletes", "linked_outcomes",
    }
    if not priority.empty and required.issubset(priority.columns):
        expanded = rows[
            ["_inventory_index", "source_scope", "source_event_id", "source_round_ids", "boulder_number"]
        ].copy()
        expanded["source_round_id"] = expanded["source_round_ids"].astype(str).str.split("|")
        expanded = expanded.explode("source_round_id")
        keys = ["source_scope", "source_event_id", "source_round_id", "boulder_number"]
        for column in keys:
            expanded[column] = expanded[column].astype(str).str.strip()
        optional = {
            "board_linked_outcomes": "priority_board_linked_outcomes",
            "top_given_zone_discordant_pairs": "priority_top_given_zone_pairs",
            "zone_discordant_pairs": "priority_zone_pairs",
        }
        coaching_optional = {
            "coaching_unlock_rank": "coaching_unlock_rank",
            "coaching_athletes_unlocked": "coaching_unlock_athletes",
            "coaching_observations_unlocked": "coaching_unlock_observations",
            "physical_observations_unlocked": "coaching_unlock_physical_observations",
            "board_observations_unlocked": "coaching_unlock_board_observations",
        }
        queue_columns = list(required) + [
            column for column in optional if column in priority.columns
        ] + [column for column in coaching_optional if column in priority.columns]
        queue = priority[queue_columns].copy()
        for column in keys:
            queue[column] = queue[column].astype(str).str.strip()
        matched = expanded.merge(queue, on=keys, how="inner")
        if not matched.empty:
            aggregations = {
                "priority_rank": ("priority_rank", "min"),
                "priority_source_items": ("priority_rank", "count"),
                "priority_linked_athletes": ("linked_athletes", "max"),
                "priority_linked_outcomes": ("linked_outcomes", "sum"),
            }
            aggregations.update(
                {
                    output: (source, "sum")
                    for source, output in optional.items()
                    if source in matched.columns
                }
            )
            if "coaching_unlock_rank" in matched.columns:
                aggregations["coaching_unlock_source_items"] = (
                    "coaching_unlock_rank", "count"
                )
            aggregations.update({
                output: (source, "min" if source == "coaching_unlock_rank" else "max")
                for source, output in coaching_optional.items()
                if source in matched.columns
            })
            attached = matched.groupby("_inventory_index", as_index=False).agg(
                **aggregations,
            )
            rows = rows.drop(
                columns=[
                    "priority_rank", "priority_source_items",
                    "priority_linked_athletes", "priority_linked_outcomes",
                    "priority_board_linked_outcomes",
                    "priority_top_given_zone_pairs", "priority_zone_pairs",
                    "coaching_unlock_rank", "coaching_unlock_source_items",
                    "coaching_unlock_athletes", "coaching_unlock_observations",
                    "coaching_unlock_physical_observations",
                    "coaching_unlock_board_observations",
                ]
            ).merge(attached, on="_inventory_index", how="left")
    for column in (
        "priority_board_linked_outcomes",
        "priority_top_given_zone_pairs",
        "priority_zone_pairs",
    ):
        if column not in rows.columns:
            rows[column] = 0
        rows[column] = rows[column].fillna(0).astype(int)
    if "coaching_unlock_source_items" not in rows.columns:
        rows["coaching_unlock_source_items"] = 0
    rows["coaching_unlock_source_items"] = (
        rows["coaching_unlock_source_items"].fillna(0).astype(int)
    )
    for column in (
        "coaching_unlock_rank", "coaching_unlock_athletes",
        "coaching_unlock_observations", "coaching_unlock_physical_observations",
        "coaching_unlock_board_observations",
    ):
        if column not in rows.columns:
            rows[column] = pd.NA
    rows["priority_status"] = rows["priority_rank"].notna().map(
        {True: "Physical-transfer priority", False: "General governed inventory"}
    )
    return rows.sort_values(
        ["coaching_unlock_rank", "priority_rank", "event_date", "event_name", "round_group", "gender", "boulder_number"],
        ascending=[True, True, False, True, True, True, True], na_position="last", kind="stable",
    ).drop(columns="_inventory_index").reset_index(drop=True)


def problem_display(row: pd.Series, *, prefer_coaching: bool = True) -> str:
    prefix = {"Men": "M", "Women": "W"}.get(str(row.gender), "B")
    route = f"{prefix}{int(row.boulder_number)} · governed {row.boulder_uid}"
    if prefer_coaching and pd.notna(row.get("coaching_unlock_rank")):
        return (
            f"Coaching {int(row.coaching_unlock_rank)} · {route} · "
            f"{int(row.coaching_unlock_observations)} dated observations"
        )
    if pd.notna(row.get("priority_rank")):
        return (
            f"Priority {int(row.priority_rank)} · {route} · "
            f"{int(row.priority_linked_athletes)} linked athletes"
        )
    return route


def tagging_coverage_milestones(
    priority: pd.DataFrame,
    milestones: tuple[int, ...] = (10, 25, 50, 100),
) -> pd.DataFrame:
    """Summarize cumulative information unlocked by ranked human review.

    Linked-athlete counts are athlete-item links and may repeat one athlete
    across several items. Pair counts are within-item comparison opportunities,
    not independent observations or claims of model readiness.
    """
    required = {
        "priority_rank",
        "competition_id",
        "linked_athletes",
        "board_linked_outcomes",
        "top_given_zone_discordant_pairs",
        "zone_discordant_pairs",
    }
    if priority.empty or not required.issubset(priority.columns):
        return pd.DataFrame()
    rows = priority.copy()
    for column in required - {"competition_id"}:
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    rows = rows.loc[rows["priority_rank"].notna()].sort_values(
        "priority_rank", kind="stable"
    )
    result: list[dict[str, int]] = []
    for requested in milestones:
        count = min(int(requested), len(rows))
        if count <= 0:
            continue
        selected = rows.head(count)
        result.append({
            "Reviewed items": count,
            "Competitions touched": int(selected["competition_id"].nunique()),
            "Athlete-item links": int(selected["linked_athletes"].sum()),
            "Board-linked outcomes": int(selected["board_linked_outcomes"].sum()),
            "Top|Zone discordant pairs": int(
                selected["top_given_zone_discordant_pairs"].sum()
            ),
            "Zone discordant pairs": int(selected["zone_discordant_pairs"].sum()),
        })
    return pd.DataFrame(result)


def backend_url() -> str:
    try:
        return str(st.secrets.get("STYLE_TAG_WEBHOOK_URL", "")).strip()
    except (FileNotFoundError, KeyError):
        return ""


def save_remotely(url: str, record: dict[str, object], image_bytes: bytes = b"") -> tuple[bool, str]:
    payload: dict[str, object] = {"record": record}
    if image_bytes:
        payload["image_base64"] = base64.b64encode(image_bytes).decode("ascii")
    request = urlrequest.Request(url, data=json.dumps(payload, separators=(",", ":")).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlrequest.urlopen(request, timeout=20) as response:
            answer = json.loads(response.read().decode("utf-8"))
        return bool(answer.get("ok")), str(answer.get("message", "Saved"))
    except (urlerror.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return False, f"Shared save failed: {exc}"


def shared_records_url(url: str, limit: int = 100) -> str:
    return f"{url}{'&' if '?' in url else '?'}{urlparse.urlencode({'action': 'list', 'limit': limit})}"


@st.cache_data(show_spinner=False, ttl=120, max_entries=1)
def load_shared_records(url: str) -> tuple[list[dict[str, object]], str]:
    try:
        with urlrequest.urlopen(shared_records_url(url), timeout=15) as response:
            answer = json.loads(response.read().decode("utf-8"))
        records = answer.get("records", [])
        if not answer.get("ok") or not isinstance(records, list):
            return [], str(answer.get("message", "Shared tag list is unavailable"))
        return [item for item in records if isinstance(item, dict)], ""
    except (urlerror.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return [], f"Shared tag list is unavailable: {exc}"


def normalize_reviewer_code(value: object) -> str:
    code = str(value).strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{3,24}", code) is None:
        raise ValueError("Use 3-24 letters, numbers, underscores or hyphens")
    return code


def reviewed_boulder_uids(
    records: list[dict[str, object]], reviewer_code: str | None = None,
) -> set[str]:
    """Return exact Boulder identities reviewed by this reviewer or anyone."""

    return {
        str(record["boulder_uid"])
        for record in records
        if isinstance(record, dict) and str(record.get("boulder_uid", "")).strip()
        and (
            reviewer_code is None
            or str(record.get("contributor", "")).strip() == reviewer_code
        )
    }


def independent_review_coverage(records: list[dict[str, object]]) -> pd.DataFrame:
    """Count distinct declared reviewers without treating duplicates as independent."""

    rows = pd.DataFrame([
        {
            "boulder_uid": str(record.get("boulder_uid", "")).strip(),
            "reviewer_code": str(record.get("contributor", "")).strip(),
        }
        for record in records
        if isinstance(record, dict)
        and str(record.get("boulder_uid", "")).strip()
        and str(record.get("contributor", "")).strip()
    ])
    if rows.empty:
        return pd.DataFrame(columns=["boulder_uid", "independent_reviewers"])
    return rows.groupby("boulder_uid", as_index=False).agg(
        independent_reviewers=("reviewer_code", "nunique")
    )


def independent_core_tag_agreement(records: list[dict[str, object]]) -> pd.DataFrame:
    """Describe core-tag disagreement where at least two reviewers exist.

    If a reviewer saved multiple records for one Boulder, only their latest
    timestamped record is retained. This table is descriptive and reports its
    actual paired Boulder count; it never turns that count into a pass/fail gate.
    """

    fields = [
        f"{segment}_{field}"
        for segment in ("pre_zone", "post_zone")
        for field in CORE_TAG_LABELS
    ]
    rows = pd.DataFrame([
        record for record in records
        if isinstance(record, dict)
        and str(record.get("boulder_uid", "")).strip()
        and str(record.get("contributor", "")).strip()
        and all(field in record for field in fields)
    ])
    if rows.empty:
        return pd.DataFrame()
    rows["submitted_at_utc"] = pd.to_datetime(
        rows.get("submitted_at_utc"), errors="coerce", utc=True
    )
    rows = rows.sort_values("submitted_at_utc", kind="stable").drop_duplicates(
        ["boulder_uid", "contributor"], keep="last"
    )
    paired = rows.loc[
        rows.groupby("boulder_uid")["contributor"].transform("nunique").ge(2)
    ].copy()
    if paired.empty:
        return pd.DataFrame()
    output: list[dict[str, object]] = []
    for field in fields:
        values = pd.to_numeric(paired[field], errors="coerce")
        by_boulder = paired.assign(_value=values).groupby("boulder_uid")["_value"]
        ranges = by_boulder.max() - by_boulder.min()
        output.append({
            "Core tag": field,
            "Double-reviewed boulders": int(ranges.notna().sum()),
            "Exact agreement": float(ranges.eq(0).mean()),
            "Mean reviewer range (0-3)": float(ranges.mean()),
            "Maximum reviewer range (0-3)": float(ranges.max()),
        })
    return pd.DataFrame(output)


def pending_review_inventory(
    inventory: pd.DataFrame,
    records: list[dict[str, object]],
    *,
    reviewer_code: str | None = None,
    include_completed: bool = False,
) -> pd.DataFrame:
    """Keep governed ordering while hiding exact completed tasks by default."""

    if include_completed or inventory.empty or "boulder_uid" not in inventory:
        return inventory.copy()
    completed = reviewed_boulder_uids(records, reviewer_code=reviewer_code)
    return inventory.loc[~inventory["boulder_uid"].astype(str).isin(completed)].copy()


def main() -> None:
    st.set_page_config(page_title="Comp Climbing Boulder Tags", page_icon="B", layout="wide")
    st.title("Comp Climbing Boulder Tags")
    st.caption("Tag terrain demand, not athlete ability. Records bind to the governed boulder and its two segments.")
    st.info("0 absent · 1 secondary · 2 important · 3 defining. Score start-to-Zone and Zone-to-Top separately.")
    inventory = problem_inventory()
    if inventory.empty:
        st.error("The governed boulder inventory is unavailable; tagging is disabled until it is restored.")
        return
    review_order = st.radio(
        "Review order",
        ("Coaching evidence unlocked", "Generic physical/Kilter evidence"),
        horizontal=True,
        help=(
            "Coaching order prioritizes pending tags that connect to the most dated "
            "physical and board observations. Generic order preserves the earlier "
            "outcome-discordance priority. Neither is a model-readiness threshold."
        ),
    )
    if review_order == "Generic physical/Kilter evidence":
        inventory = inventory.sort_values(
            ["priority_rank", "event_date", "event_name", "round_group", "gender", "boulder_number"],
            ascending=[True, False, True, True, True, True],
            na_position="last",
            kind="stable",
        ).reset_index(drop=True)
    url = backend_url()
    shared: list[dict[str, object]] = []
    shared_error = ""
    if url:
        shared, shared_error = load_shared_records(url)
    session_records = st.session_state.get("style_tag_records", [])
    all_records = [*shared, *session_records]
    completed_uids = reviewed_boulder_uids(all_records)
    reviewer_code_raw = st.text_input(
        "Reviewer code",
        value=str(st.session_state.get("style_tag_reviewer_code", "")),
        max_chars=24,
        help=(
            "Use a stable pseudonym, not your name or email. A boulder is hidden only "
            "after this reviewer has tagged it, allowing independent second reviews."
        ),
    )
    try:
        reviewer_code = normalize_reviewer_code(reviewer_code_raw)
        st.session_state["style_tag_reviewer_code"] = reviewer_code
    except ValueError:
        reviewer_code = ""
        st.warning("Enter a pseudonymous reviewer code to start or save a review.")
    agreement = independent_review_coverage(all_records)
    independently_double_reviewed = int(
        agreement["independent_reviewers"].ge(2).sum()
        if not agreement.empty else 0
    )
    prioritized = inventory.loc[inventory["priority_rank"].notna()]
    reviewed_prioritized = int(
        prioritized["boulder_uid"].astype(str).isin(completed_uids).sum()
    )
    source_priority_items = int(prioritized["priority_source_items"].fillna(0).sum())
    top_pair_tasks = int(prioritized["priority_top_given_zone_pairs"].gt(0).sum())
    top_pairs = int(prioritized["priority_top_given_zone_pairs"].sum())
    zone_pairs = int(prioritized["priority_zone_pairs"].sum())
    st.caption(
        f"{len(prioritized):,} governed Boulder tagging tasks cover all "
        f"{source_priority_items:,} physical/Kilter priority source items. Shared terrain "
        "lets one reviewed tag serve multiple source records; priority is continuous, not "
        "an inclusion cutoff."
    )
    milestones = tagging_coverage_milestones(tagging_priority_queue())
    if not milestones.empty:
        st.markdown("**What a short ranked review session unlocks**")
        st.dataframe(milestones, hide_index=True, width="stretch")
        st.caption(
            "Athlete-item links repeat athletes across items. Discordant-pair counts are "
            "within-item opportunities for comparing outcomes after a shared demand tag; "
            "they are not independent competitions or a model-release threshold."
        )
    st.caption(
        f"The evidence-rich first layer is {top_pair_tasks:,} tasks covering "
        f"{top_pairs:,} exact both-board Top-given-Zone comparisons; the full queue "
        f"also covers {zone_pairs:,} Zone comparisons. This orders human review by "
        "expected evidence reuse, not by athlete importance."
    )
    include_completed = st.checkbox(
        "Include already reviewed boulders",
        value=False,
        help="Completed tasks remain available for inspection but are hidden during rapid review.",
    )
    review_inventory = pending_review_inventory(
        inventory,
        all_records,
        reviewer_code=reviewer_code or None,
        include_completed=include_completed,
    )
    st.caption(
        f"Progress: {reviewed_prioritized:,}/{len(prioritized):,} prioritized governed "
        "tasks have a saved review. The form starts at the highest-priority unreviewed task."
    )
    st.caption(
        f"Independent-review coverage: {independently_double_reviewed:,} boulders have "
        "reviews from at least two distinct declared reviewer codes. Duplicate submissions "
        "from one code do not count. Agreement is not estimated until this count is nonzero."
    )
    agreement_table = independent_core_tag_agreement(all_records)
    if not agreement_table.empty:
        st.dataframe(agreement_table, hide_index=True, width="stretch")
        st.caption(
            "Agreement uses each reviewer's latest record per Boulder and reports the "
            "actual double-reviewed count. It is descriptive, not a model-release gate."
        )
    flash = st.session_state.pop("style_tag_flash", None)
    if isinstance(flash, tuple) and len(flash) == 2:
        (st.success if flash[0] == "success" else st.warning)(str(flash[1]))
    if review_inventory.empty:
        st.success("Every governed Boulder in the current view has a saved review.")
        return
    events = review_inventory[["event_name", "event_date"]].drop_duplicates()
    event_labels = [f"{row.event_date.date().isoformat() if pd.notna(row.event_date) else 'Date unknown'} — {row.event_name}" for row in events.itertuples(index=False)]
    with st.form("style_tag_form", clear_on_submit=True):
        event_label = st.selectbox("Competition", event_labels)
        event_name = event_label.split(" — ", 1)[-1]
        event_rows = review_inventory.loc[review_inventory.event_name.eq(event_name)]
        round_name = st.selectbox(
            "Round",
            event_rows["round_group"].dropna().astype(str).drop_duplicates().tolist(),
        )
        round_rows = event_rows.loc[event_rows.round_group.eq(round_name)]
        terrain = st.selectbox(
            "Terrain",
            round_rows["gender"].dropna().astype(str).drop_duplicates().tolist(),
        )
        terrain_rows = round_rows.loc[round_rows.gender.eq(terrain)]
        prefer_coaching = review_order == "Coaching evidence unlocked"
        choices = terrain_rows.apply(
            lambda row: problem_display(row, prefer_coaching=prefer_coaching), axis=1
        ).tolist()
        chosen_display = st.selectbox("Boulder", choices)
        selected_problem = terrain_rows.loc[terrain_rows.apply(
            lambda row: problem_display(row, prefer_coaching=prefer_coaching), axis=1
        ).eq(chosen_display)].iloc[0].to_dict()
        st.caption(f"{selected_problem['boulder_count_status']} count: {int(selected_problem['boulder_count'])}; terrain: {selected_problem['terrain_group']}")
        if pd.notna(selected_problem.get("priority_rank")):
            source_items = int(selected_problem.get("priority_source_items", 1))
            st.info(
                f"Physical-transfer priority {int(selected_problem['priority_rank'])}: "
                f"this tagging task covers {source_items} exact source item"
                f"{'s' if source_items != 1 else ''}; each source item links up to "
                f"{int(selected_problem['priority_linked_athletes'])} athletes. This is a "
                "review-efficiency heuristic, not model evidence."
            )
            top_pairs_for_task = int(selected_problem.get("priority_top_given_zone_pairs", 0))
            zone_pairs_for_task = int(selected_problem.get("priority_zone_pairs", 0))
            if top_pairs_for_task or zone_pairs_for_task:
                st.caption(
                    f"Expected exact-item reuse: {top_pairs_for_task} "
                    f"Top-given-Zone pair{'s' if top_pairs_for_task != 1 else ''} and "
                    f"{zone_pairs_for_task} Zone pair{'s' if zone_pairs_for_task != 1 else ''}."
                )
        if pd.notna(selected_problem.get("coaching_unlock_rank")):
            st.success(
                f"Coaching evidence unlock {int(selected_problem['coaching_unlock_rank'])}: "
                f"the highest-linked exact source item covers "
                f"{int(selected_problem['coaching_unlock_athletes'])} tested athletes and "
                f"{int(selected_problem['coaching_unlock_observations'])} dated observations "
                f"({int(selected_problem['coaching_unlock_physical_observations'])} physical, "
                f"{int(selected_problem['coaching_unlock_board_observations'])} board). "
                "Counts prioritize human review only; they contain no athlete identity or test value."
            )
        confidence = st.select_slider("Confidence", options=("Low", "Moderate", "High"), value="Moderate")
        directions = st.columns(2)
        pre_direction = directions[0].selectbox("Start to Zone direction", ("Up", "Diagonal", "Sideways", "Mixed / unclear"))
        post_direction = directions[1].selectbox("Zone to Top direction", ("Up", "Diagonal", "Sideways", "Mixed / unclear"))
        st.subheader("Core demand")
        core_values: dict[str, tuple[int, int]] = {}
        for column, (field, label) in zip(st.columns(3), CORE_TAG_LABELS.items()):
            with column:
                st.markdown(f"**{label}**")
                pre = st.slider("Start to Zone", 0, 3, 0, key=f"pre_core_{field}")
                post = st.slider("Zone to Top", 0, 3, 0, key=f"post_core_{field}")
                core_values[field] = (pre, post)
        optional_tags_completed = st.checkbox("Add detailed route-demand tags", value=False)
        detailed_values: dict[str, tuple[int, int]] = {}
        if optional_tags_completed:
            st.subheader("Detailed route demand")
            columns = st.columns(4)
            for index, (field, label) in enumerate(route_fields().items()):
                with columns[index % 4]:
                    st.markdown(f"**{label}**")
                    pre = st.slider("Start to Zone", 0, 3, 0, key=f"pre_detail_{field}")
                    post = st.slider("Zone to Top", 0, 3, 0, key=f"post_detail_{field}")
                    detailed_values[field] = (pre, post)
        image = st.file_uploader("Boulder image (optional)", type=("jpg", "jpeg", "png"))
        receipt_file = st.file_uploader("Frame extraction receipt (optional)", type=("json",))
        matched_frames: list[dict[str, object]] = []
        if receipt_file is not None:
            try:
                matched_frames = matching_frame_receipts(json.loads(receipt_file.getvalue()), selected_problem)
            except (UnicodeDecodeError, json.JSONDecodeError):
                st.warning("The frame receipt is not valid JSON.")
            if matched_frames:
                st.caption(f"{len(matched_frames)} review-only frame candidate(s) match this Boulder. Select the image file separately; the receipt does not assert an empty wall.")
            else:
                st.caption("No review-only frame candidate in this receipt matches the selected governed Boulder.")
        selected_frame = st.selectbox(
            "Frame provenance (optional)",
            [None, *matched_frames],
            format_func=lambda frame: "No receipt provenance" if frame is None else f"{frame['candidate_id']} · {frame['frame_seconds']}s",
        )
        submitted = st.form_submit_button(
            "Save style-tag proposal", type="primary", disabled=not reviewer_code
        )
    if submitted:
        image_bytes = image.getvalue() if image else b""
        record = build_record(selected_problem, confidence=confidence, pre_zone_direction=pre_direction, post_zone_direction=post_direction, core_values=core_values, detailed_values=detailed_values, optional_tags_completed=optional_tags_completed, reviewer_code=reviewer_code, image_name=image.name if image else "", image_bytes=image_bytes, frame=selected_frame)
        st.session_state.setdefault("style_tag_records", []).append(record)
        url = backend_url()
        if url:
            saved, message = save_remotely(url, record, image_bytes)
            st.session_state["style_tag_flash"] = (
                "success" if saved else "warning",
                message,
            )
            if saved:
                load_shared_records.clear()
        else:
            st.session_state["style_tag_flash"] = (
                "success",
                "Saved in this session. Download the review file before closing the tab.",
            )
        st.rerun()
    records = st.session_state.get("style_tag_records", [])
    if records:
        st.subheader("Current review records")
        st.dataframe(pd.DataFrame(records), hide_index=True, width="stretch")
        st.download_button("Download style-tag review JSON", json.dumps(records, indent=2), "comp_climbing_style_tags.json", "application/json")
    if url:
        st.divider(); st.subheader("Recent shared tags")
        if st.button("Refresh shared tags"):
            load_shared_records.clear()
        if shared_error:
            st.caption(shared_error)
        elif shared:
            visible = pd.DataFrame(shared)
            columns = [key for key in ("submitted_at_utc", "competition_date", "competition", "round", "gender_terrain", "boulder", "confidence", "image_public_url") if key in visible]
            st.dataframe(visible[columns], hide_index=True, width="stretch")
        else:
            st.caption("No shared tags yet.")


if __name__ == "__main__":
    main()
