"""Lightweight public boulder-style annotation app."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from comp_climbing_app import DATA, render_style_tagging_v2, style_tag_backend_url


@st.cache_data(show_spinner=False, ttl=1800, max_entries=1)
def event_history() -> pd.DataFrame:
    path = DATA / "boulder_overview_history.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["event_name", "event_date"])
    try:
        return pd.read_parquet(path, columns=["event_name", "event_date"])
    except (KeyError, ValueError):
        frame = pd.read_parquet(path)
        available = [column for column in ["event_name", "event_date"] if column in frame]
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
