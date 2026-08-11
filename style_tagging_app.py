"""Standalone public route-demand tagger for Comp Climbing Boulder Tags.

It intentionally owns its UI rather than importing an internal renderer from
the projections application.  Route tags are review records, never athlete
ratings or training prescriptions.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import re
from urllib import error as urlerror
from urllib import request as urlrequest

import pandas as pd
import streamlit as st

from comp_climbing_app import DATA


BOULDER_LABEL = re.compile(r"^([MW])\s*[-:]?\s*([1-4])$", re.IGNORECASE)
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
    """Return the v10.7 public route-demand fields bundled with this app."""
    return dict(ROUTE_FIELD_LABELS)


@st.cache_data(show_spinner=False, ttl=1800, max_entries=1)
def event_catalog() -> pd.DataFrame:
    path = DATA / "boulder_overview_history.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["event_name", "event_date"])
    try:
        frame = pd.read_parquet(path, columns=["event_name", "event_date"])
    except (KeyError, ValueError):
        frame = pd.read_parquet(path)
        frame = frame[[column for column in ("event_name", "event_date") if column in frame]]
    if "event_name" not in frame:
        return pd.DataFrame(columns=["event_name", "event_date"])
    frame["event_name"] = frame["event_name"].fillna("").astype(str).str.strip()
    frame["event_date"] = pd.to_datetime(frame.get("event_date"), errors="coerce")
    return frame.loc[frame["event_name"].ne("")].drop_duplicates().sort_values(
        ["event_date", "event_name"], ascending=[False, True]
    )


def backend_url() -> str:
    try:
        return str(st.secrets.get("STYLE_TAG_WEBHOOK_URL", "")).strip()
    except (FileNotFoundError, KeyError):
        return ""


def save_remotely(url: str, record: dict[str, object]) -> tuple[bool, str]:
    body = json.dumps({"record": record}, separators=(",", ":")).encode("utf-8")
    request = urlrequest.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlrequest.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return bool(payload.get("ok")), str(payload.get("message", "Saved"))
    except (urlerror.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return False, f"Shared save failed: {exc}"


def main() -> None:
    st.set_page_config(page_title="Comp Climbing Boulder Tags", page_icon="🧗", layout="wide")
    st.title("Comp Climbing Boulder Tags")
    st.caption("Tag what the route demands. These are review records, not athlete labels or ratings.")
    st.info("0 absent · 1 secondary · 2 important · 3 defining. Tag only what is visible in the route evidence.")

    catalog = event_catalog()
    labels = ["Custom competition"] + [
        f"{row.event_date.date().isoformat() if pd.notna(row.event_date) else 'Date unknown'} — {row.event_name}"
        for row in catalog.itertuples(index=False)
    ]
    with st.form("route_demand_tag_form", clear_on_submit=True):
        event_label = st.selectbox("Competition", labels)
        custom_event = st.text_input("Custom competition", disabled=event_label != "Custom competition")
        context = st.columns(4)
        round_name = context[0].selectbox("Round", ("Qualification", "Semi-final", "Final"))
        category = context[1].selectbox("Category", ("Men", "Women", "Mixed / unknown"))
        boulder = context[2].text_input("Boulder", placeholder="M1 or W3")
        confidence = context[3].select_slider("Confidence", options=("Low", "Moderate", "High"), value="Moderate")
        st.subheader("Route demand")
        fields = route_fields()
        values: dict[str, int] = {}
        columns = st.columns(4)
        for index, (field, label) in enumerate(fields.items()):
            values[field] = columns[index % 4].slider(label.title(), 0, 3, 0, key=f"tag_{field}")
        source_url = st.text_input("Evidence URL (optional)", placeholder="https://…")
        submitted = st.form_submit_button("Save route-demand proposal", type="primary")
    if submitted:
        try:
            route_id = canonical_boulder_label(boulder)
        except ValueError as exc:
            st.error(str(exc))
            return
        if event_label == "Custom competition":
            competition = custom_event.strip()
        else:
            competition = event_label.split(" — ", 1)[-1]
        if not competition:
            st.error("Choose or enter a competition.")
            return
        record: dict[str, object] = {
            "schema_version": "v10.7_route_demand",
            "status": "human_or_ai_proposal_pending_review",
            "submitted_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "event_name": competition,
            "category": category,
            "round": round_name,
            "boulder_number": route_id,
            "confidence": confidence,
            "evidence_url": source_url.strip(),
            "route_tags": values,
        }
        st.session_state.setdefault("route_tag_records", []).append(record)
        url = backend_url()
        if url:
            saved, message = save_remotely(url, record)
            (st.success if saved else st.warning)(message)
        else:
            st.success("Saved in this session. Download the review file before closing the tab.")
    records = st.session_state.get("route_tag_records", [])
    if records:
        st.subheader("Current review records")
        st.dataframe(pd.DataFrame(records), hide_index=True, width="stretch")
        st.download_button(
            "Download route-demand review JSON",
            json.dumps(records, indent=2, ensure_ascii=False),
            "comp_climbing_route_demand_tags.json",
            "application/json",
        )


if __name__ == "__main__":
    main()
