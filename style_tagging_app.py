"""Standalone schema-v2 public tagger for Comp Climbing Boulder Tags."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import json
import re
from urllib import error as urlerror
from urllib import request as urlrequest

import pandas as pd
import streamlit as st

from comp_climbing_app import DATA


BOULDER_LABEL = re.compile(r"^([MW])\s*[-:]?\s*([1-4])$", re.IGNORECASE)
CORE_TAG_LABELS = {
    "physical_0_3": "Physical demand",
    "technical_0_3": "Technical demand",
    "coordination_0_3": "Coordination demand",
    "verticality_0_3": "Verticality demand",
}
ROUTE_FIELD_LABELS = {
    "wall_overhang_0_3": "Wall angle demand",
    "three_dimensionality_0_3": "Three-dimensional movement",
    "dual_texture_friction_0_3": "Dual-texture / friction demand",
    "hold_positivity_0_3": "Handhold positivity",
    "crimp_edge_0_3": "Crimp / edge use",
    "open_hand_sloper_0_3": "Open-hand / sloper use",
    "pinch_0_3": "Pinch use",
    "compression_opposition_0_3": "Compression / opposition",
    "slow_technical_0_3": "Slow technical movement",
    "upper_pull_0_3": "Upper-body pulling",
    "upper_push_0_3": "Upper-body pushing",
    "fast_upper_accuracy_0_3": "Fast hand accuracy",
    "fast_lower_accuracy_0_3": "Fast foot accuracy",
    "lower_explosivity_0_3": "Lower-body explosivity",
    "body_tension_core_0_3": "Body tension / core",
    "mobility_flexibility_0_3": "Mobility / flexibility",
    "reach_span_0_3": "Reach / span",
    "read_complexity_0_3": "Reading complexity",
    "precision_risk_0_3": "Precision risk",
    "attempt_cost_0_3": "Attempt cost",
}


def canonical_boulder_label(value: str) -> str:
    """Canonicalise the public route IDs used by IFSC Boulder broadcasts."""
    match = BOULDER_LABEL.fullmatch(value.strip())
    if match is None:
        raise ValueError("Use M1-M4 or W1-W4")
    return f"{match.group(1).upper()}{match.group(2)}"


def route_fields() -> dict[str, str]:
    """Return optional detailed route-demand fields for the v2 tag record."""
    return dict(ROUTE_FIELD_LABELS)


def build_record(
    *, competition: str, competition_date: str, round_name: str, category: str,
    boulder: str, confidence: str, top_direction: str, zone_direction: str,
    core_values: dict[str, tuple[int, int]], detailed_values: dict[str, tuple[int, int]],
    optional_tags_completed: bool, image_name: str = "", image_bytes: bytes = b"",
) -> dict[str, object]:
    """Build a schema-v2 record accepted by the shared style-tag backend."""
    record: dict[str, object] = {
        "schema_version": "2.0",
        "submitted_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "competition": competition,
        "competition_date": competition_date,
        "round": round_name,
        "gender_terrain": category,
        "boulder": boulder,
        "top_direction": top_direction,
        "zone_direction": zone_direction,
        "optional_tags_completed": optional_tags_completed,
        "confidence": confidence,
        "image_name": image_name,
        "image_sha256": hashlib.sha256(image_bytes).hexdigest() if image_bytes else "",
        "image_in_bundle": "" if not image_bytes else "uploaded_to_shared_backend",
    }
    for field, (top_value, zone_value) in core_values.items():
        record[f"top_{field}"] = top_value
        record[f"zone_{field}"] = zone_value
    if optional_tags_completed:
        for field, (top_value, zone_value) in detailed_values.items():
            record[f"top_{field}"] = top_value
            record[f"zone_{field}"] = zone_value
    return record


@st.cache_data(show_spinner=False, ttl=1800, max_entries=1)
def round_inventory() -> pd.DataFrame:
    """Load source-reported round and boulder counts for the tag selector."""
    path = DATA / "boulder_round_inventory.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path, usecols=[
            "event_name", "event_date", "round_group", "gender", "category",
            "boulder_count", "boulder_count_status",
        ])
    except ValueError:
        return pd.DataFrame()
    if frame.empty:
        return frame
    frame["event_name"] = frame["event_name"].fillna("").astype(str).str.strip()
    frame["event_date"] = pd.to_datetime(frame["event_date"], errors="coerce")
    frame["boulder_count"] = pd.to_numeric(frame["boulder_count"], errors="coerce")
    return frame.loc[frame["event_name"].ne("")].drop_duplicates().sort_values(
        ["event_date", "event_name", "round_group", "gender"], ascending=[False, True, True, True]
    )


def boulder_options(gender: str, count: object) -> list[str]:
    """Return only source-supported M/W boulder labels when a count is known."""
    try:
        integer_count = int(float(count))
    except (TypeError, ValueError):
        return []
    if integer_count < 1 or integer_count > 12 or gender not in {"Men", "Women"}:
        return []
    prefix = "M" if gender == "Men" else "W"
    return [f"{prefix}{number}" for number in range(1, integer_count + 1)]


def backend_url() -> str:
    try:
        return str(st.secrets.get("STYLE_TAG_WEBHOOK_URL", "")).strip()
    except (FileNotFoundError, KeyError):
        return ""


def save_remotely(url: str, record: dict[str, object], image_bytes: bytes = b"") -> tuple[bool, str]:
    payload: dict[str, object] = {"record": record}
    if image_bytes:
        payload["image_base64"] = base64.b64encode(image_bytes).decode("ascii")
    request = urlrequest.Request(
        url, data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urlrequest.urlopen(request, timeout=20) as response:
            answer = json.loads(response.read().decode("utf-8"))
        return bool(answer.get("ok")), str(answer.get("message", "Saved"))
    except (urlerror.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return False, f"Shared save failed: {exc}"


def main() -> None:
    st.set_page_config(page_title="Comp Climbing Boulder Tags", page_icon="B", layout="wide")
    st.title("Comp Climbing Boulder Tags")
    st.caption("Tag terrain demand, not athlete ability. Every saved proposal uses the shared v2 style-tag schema.")
    st.info("0 absent · 1 secondary · 2 important · 3 defining. Score Zone and Top separately.")
    inventory = round_inventory()
    catalog = inventory[["event_name", "event_date"]].drop_duplicates() if not inventory.empty else pd.DataFrame()
    labels = ["Custom competition"] + [
        f"{row.event_date.date().isoformat() if pd.notna(row.event_date) else 'Date unknown'} — {row.event_name}"
        for row in catalog.itertuples(index=False)
    ]
    with st.form("style_tag_form", clear_on_submit=True):
        event_label = st.selectbox("Competition", labels)
        custom_event = st.text_input("Custom competition", disabled=event_label != "Custom competition")
        selected_inventory = pd.DataFrame()
        if event_label != "Custom competition" and not inventory.empty:
            selected_name = event_label.split(" — ", 1)[-1]
            selected_inventory = inventory.loc[inventory["event_name"].eq(selected_name)]
        available_rounds = sorted(selected_inventory["round_group"].dropna().astype(str).unique()) or ["Qualification", "Semi-final", "Final", "Other"]
        round_name = st.selectbox("Round", available_rounds)
        selected_round = selected_inventory.loc[selected_inventory["round_group"].eq(round_name)]
        available_terrain = sorted(selected_round["gender"].dropna().astype(str).unique()) or ["Men", "Women", "Mixed / unknown"]
        category = st.selectbox("Terrain", available_terrain)
        selected_terrain = selected_round.loc[selected_round["gender"].eq(category)]
        count = selected_terrain["boulder_count"].dropna().iloc[0] if selected_terrain["boulder_count"].notna().any() else None
        supported_boulders = boulder_options(category, count)
        route_context = st.columns(2)
        if supported_boulders:
            boulder = route_context[0].selectbox("Boulder", supported_boulders)
            route_context[1].caption(f"Source reports {len(supported_boulders)} boulders for this terrain and round.")
        else:
            boulder = route_context[0].text_input("Boulder", placeholder="M1 or W3")
            route_context[1].caption("No source count is available; enter a verified M/W route label.")
        confidence = st.select_slider("Confidence", options=("Low", "Moderate", "High"), value="Moderate")
        directions = st.columns(2)
        zone_direction = directions[0].selectbox("Start to Zone direction", ("Up", "Diagonal", "Sideways", "Mixed / unclear"))
        top_direction = directions[1].selectbox("Zone to Top direction", ("Up", "Diagonal", "Sideways", "Mixed / unclear"))
        st.subheader("Core demand")
        core_values: dict[str, tuple[int, int]] = {}
        for column, (field, label) in zip(st.columns(4), CORE_TAG_LABELS.items()):
            with column:
                st.markdown(f"**{label}**")
                zone = st.slider("Zone", 0, 3, 0, key=f"zone_core_{field}")
                top = st.slider("Top", 0, 3, 0, key=f"top_core_{field}")
                core_values[field] = (top, zone)
        optional_tags_completed = st.checkbox("Add detailed route-demand tags", value=False)
        detailed_values: dict[str, tuple[int, int]] = {}
        if optional_tags_completed:
            st.subheader("Detailed route demand")
            columns = st.columns(4)
            for index, (field, label) in enumerate(route_fields().items()):
                with columns[index % 4]:
                    st.markdown(f"**{label}**")
                    zone = st.slider("Zone", 0, 3, 0, key=f"zone_detail_{field}")
                    top = st.slider("Top", 0, 3, 0, key=f"top_detail_{field}")
                    detailed_values[field] = (top, zone)
        image = st.file_uploader("Boulder image (optional)", type=("jpg", "jpeg", "png"))
        submitted = st.form_submit_button("Save style-tag proposal", type="primary")
    if submitted:
        try:
            route_id = canonical_boulder_label(boulder)
        except ValueError as exc:
            st.error(str(exc))
            return
        competition = custom_event.strip() if event_label == "Custom competition" else event_label.split(" — ", 1)[-1]
        if not competition:
            st.error("Choose or enter a competition.")
            return
        competition_date = ""
        if event_label != "Custom competition":
            selected = catalog.iloc[labels.index(event_label) - 1]
            if pd.notna(selected.event_date):
                competition_date = selected.event_date.date().isoformat()
        image_bytes = image.getvalue() if image else b""
        record = build_record(
            competition=competition, competition_date=competition_date, round_name=round_name,
            category=category, boulder=route_id, confidence=confidence, top_direction=top_direction,
            zone_direction=zone_direction, core_values=core_values, detailed_values=detailed_values,
            optional_tags_completed=optional_tags_completed, image_name=image.name if image else "",
            image_bytes=image_bytes,
        )
        st.session_state.setdefault("style_tag_records", []).append(record)
        url = backend_url()
        if url:
            saved, message = save_remotely(url, record, image_bytes)
            (st.success if saved else st.warning)(message)
        else:
            st.success("Saved in this session. Download the review file before closing the tab.")
    records = st.session_state.get("style_tag_records", [])
    if records:
        st.subheader("Current review records")
        st.dataframe(pd.DataFrame(records), hide_index=True, width="stretch")
        st.download_button("Download style-tag review JSON", json.dumps(records, indent=2), "comp_climbing_style_tags.json", "application/json")


if __name__ == "__main__":
    main()
