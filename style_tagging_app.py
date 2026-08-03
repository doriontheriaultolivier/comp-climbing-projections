"""Lightweight public boulder-style annotation app."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from comp_climbing_app import DATA, render_style_tagging_v2, style_tag_backend_url


@st.cache_data(show_spinner=False, ttl=1800, max_entries=1)
def event_history() -> pd.DataFrame:
    columns = [
        "event_name", "event_date", "source_scope", "source_event_id",
        "round_group", "confirmed_procedure", "pool",
    ]
    compact_path = DATA / "style_event_catalog.csv"
    inventory_path = DATA / "boulder_round_inventory.csv"
    if compact_path.exists():
        compact = pd.read_csv(compact_path, low_memory=False)
        if inventory_path.exists():
            inventory = pd.read_csv(inventory_path, low_memory=False)
            procedure_keys = [
                "source_scope", "source_event_id", "event_name", "event_date",
                "pool", "round_group",
            ]
            available_keys = [
                key for key in procedure_keys
                if key in compact.columns and key in inventory.columns
            ]
            procedure = compact[
                [*available_keys, "confirmed_procedure"]
            ].drop_duplicates(available_keys)
            return inventory.merge(procedure, on=available_keys, how="left")
        return compact
    path = DATA / "boulder_overview_history.parquet"
    if not path.exists():
        return pd.DataFrame(columns=columns)
    try:
        return pd.read_parquet(path, columns=columns)
    except (KeyError, ValueError):
        frame = pd.read_parquet(path)
        available = [column for column in columns if column in frame]
        return frame[available]


def main() -> None:
    st.set_page_config(
        page_title="Comp Climbing - Boulder Style Tags",
        page_icon="🧗",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(
        """
        <style>
        .block-container{max-width:1320px;padding-top:1.2rem;padding-bottom:4rem}
        h1,h2,h3{color:#102F2B;letter-spacing:-.025em}
        .stCaption{color:#627571}
        @media(max-width:640px){
          .block-container{padding:.7rem .75rem 3rem}
          h1{font-size:2rem!important}
          [data-testid="stHorizontalBlock"]{flex-wrap:wrap}
          [data-testid="column"]{min-width:100%!important}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Comp Climbing Boulder Tags")
    st.caption(
        "A public annotation tool feeding style-specific performance and physical-demand analysis."
    )
    if style_tag_backend_url():
        st.success("Shared database connected", icon="✅")
    else:
        st.warning(
            "Shared database connection pending. Entries remain exportable and recoverable "
            "from this session.",
            icon="⚠️",
        )
    render_style_tagging_v2(event_history(), standalone=True)
    st.divider()
    st.link_button(
        "Open Comp Climbing Projections",
        "https://comp-climbing-projections.streamlit.app/",
    )


if __name__ == "__main__":
    main()
