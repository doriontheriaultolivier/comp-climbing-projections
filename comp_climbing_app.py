"""Boulder-first interface for Comp Climbing Projections.

The legacy product stays on its own release branch and URL.  This module loads
only the compact artifacts needed by the Overview so Streamlit Community Cloud
does not retain the full research warehouse in memory.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RATING_ORDER = ["Global-ELO", "IFSC-ELO", "WR-ELO"]
FORMAT_OPTIONS = ["All formats", "Onsight", "Flash", "Scramble"]
ALL_RATINGS = [
    "Global-ELO", "Global-ELO-Onsight", "Global-ELO-Scramble",
    "Global-ELO-Flash", "WR-ELO", "WR-ELO-Qualies", "WR-ELO-Semies",
    "WR-ELO-Finals", "IFSC-ELO", "IFSC-ELO-Qualies",
    "IFSC-ELO-Semies", "IFSC-ELO-Finals",
]
DEFAULT_ATHLETES = ["Oscar Baudrand", "Matthew Rodriguez", "Colin Duffy"]
DISPLAY_OVERRIDES = {
    "baudrandoscar": "Oscar Baudrand",
    "matthewrodriguez": "Matthew Rodriguez",
    "colinduffy": "Colin Duffy",
    "madisonrichardson": "Madison Richardson",
    "babetteroy": "Babette Roy",
    "anniesanders": "Annie Sanders",
}
PALETTE = {
    "ink": "#102F2B",
    "teal": "#0B7A75",
    "coral": "#F26B5B",
    "gold": "#E6A23C",
    "blue": "#4285A9",
    "muted": "#71817E",
}


def transparent(color: str, alpha: float = 0.12) -> str:
    value = color.lstrip("#")
    red, green, blue = (int(value[index:index + 2], 16) for index in (0, 2, 4))
    return f"rgba({red},{green},{blue},{alpha})"


def plain_key(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    plain = "".join(
        char.lower() if char.isalnum() else " "
        for char in text
        if not unicodedata.combining(char)
    )
    # IFSC commonly stores SURNAME Given while CEC/CNR stores Given SURNAME.
    # A token-sorted key reconciles that presentation difference; pool remains
    # part of every rating join so men/women and disciplines cannot cross-match.
    return "".join(sorted(plain.split()))


def friendly_name(value: object) -> str:
    text = str(value or "").strip()
    return DISPLAY_OVERRIDES.get(plain_key(text), text)


@st.cache_data(show_spinner=False, ttl=900, max_entries=2)
def read_data() -> dict[str, pd.DataFrame]:
    files = {
        "athletes": ("boulder_overview_athletes.parquet", "parquet"),
        "history": ("boulder_overview_history.parquet", "parquet"),
        "correlations": ("boulder_rating_correlations.csv", "csv"),
        "calibration": ("boulder_elo_calibration.csv", "csv"),
        "rosters": ("program_rosters.csv", "csv"),
    }
    output: dict[str, pd.DataFrame] = {}
    for key, (filename, kind) in files.items():
        path = DATA / filename
        if not path.exists():
            output[key] = pd.DataFrame()
            continue
        output[key] = (
            pd.read_parquet(path)
            if kind == "parquet"
            else pd.read_csv(path, low_memory=False)
        )
    return output


def wide_athletes(ratings: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    if ratings.empty:
        return pd.DataFrame()
    index = ["pool", "global_id"]
    values = ratings.pivot_table(
        index=index,
        columns="rating_family",
        values="rating",
        aggfunc="first",
    ).reset_index()
    evidence = ratings.pivot_table(
        index=index,
        columns="rating_family",
        values="contests_seen",
        aggfunc="first",
    ).add_suffix(" evidence").reset_index()
    metadata_columns = [
        "pool", "global_id", "athlete_name", "nationality", "gender",
        "birthday", "birth_date_analysis_value", "birth_date_confidence",
        "birth_date_uncertainty_days", "days_since_last_result",
        "ifsc_athlete_id",
    ]
    available = [column for column in metadata_columns if column in ratings]
    metadata = (
        ratings.sort_values("last_result_date" if "last_result_date" in ratings else "rating_family")
        .drop_duplicates(index, keep="last")[available]
    )
    athletes = values.merge(evidence, on=index, how="left").merge(
        metadata, on=index, how="left"
    )
    birth = pd.to_datetime(
        athletes.get("birth_date_analysis_value", athletes.get("birthday")),
        errors="coerce",
    )
    as_of = (
        pd.to_datetime(history["event_date"], errors="coerce").max()
        if not history.empty else pd.Timestamp(date.today())
    )
    athletes["age"] = (as_of - birth).dt.days / 365.2425
    athletes["name_key"] = athletes["athlete_name"].map(plain_key)
    athletes["country"] = athletes.get("nationality", "").fillna("")

    athletes["momentum"] = np.nan
    if not history.empty:
        history = history.copy()
        history["event_date"] = pd.to_datetime(history["event_date"], errors="coerce")
        recent = history.loc[history["event_date"].ge(as_of - pd.Timedelta(days=365))]
        change = (
            recent.sort_values("event_date")
            .groupby(index)["rating_after"]
            .agg(["first", "last"])
        )
        change["momentum"] = change["last"] - change["first"]
        athletes = athletes.merge(
            change[["momentum"]].reset_index(), on=index, how="left",
            suffixes=("", "_recent"),
        )
        athletes["momentum"] = athletes["momentum_recent"].combine_first(
            athletes["momentum"]
        )
        athletes = athletes.drop(columns=["momentum_recent"])
    athletes["momentum"] = athletes["momentum"].fillna(0.0)
    athletes["gender"] = athletes.get("gender", athletes["pool"].str.rsplit("_", n=1).str[-1])
    return athletes


def attach_cnr(athletes: pd.DataFrame, cnr: pd.DataFrame) -> pd.DataFrame:
    if athletes.empty:
        return pd.DataFrame()
    if cnr.empty:
        athletes["cnr_rank"] = np.nan
        athletes["cnr_value"] = np.nan
        return athletes
    boulder = cnr.loc[cnr["discipline"].eq("Boulder")].copy()
    boulder["name_key"] = boulder["athlete_name"].map(plain_key)
    boulder = boulder.rename(columns={"athlete_name": "cnr_athlete_name"})
    columns = [
        "pool", "name_key", "cnr_athlete_name", "cnr_rank", "cnr_value",
        "cnr_as_of",
    ]
    boulder = boulder[columns].drop_duplicates(["pool", "name_key"])
    output = athletes.merge(boulder, on=["pool", "name_key"], how="outer")
    output["athlete_name"] = output["athlete_name"].combine_first(
        output["cnr_athlete_name"]
    )
    output["gender"] = output["gender"].combine_first(
        output["pool"].str.rsplit("_", n=1).str[-1]
    )
    output.loc[
        output["country"].isna() & output["cnr_rank"].notna(), "country"
    ] = "CAN"
    output["country"] = output["country"].fillna("")
    output["momentum"] = output["momentum"].fillna(0.0)
    return output.drop(columns=["cnr_athlete_name"])


def attach_world_ranking(athletes: pd.DataFrame, wr: pd.DataFrame) -> pd.DataFrame:
    if athletes.empty:
        return athletes
    if wr.empty:
        for column in ("world_event_rank", "starts_365", "top_40"):
            athletes[column] = np.nan
        return athletes
    boulder = wr.loc[wr["discipline"].eq("Boulder")].copy()
    boulder["name_key"] = boulder["athlete_name"].map(plain_key)
    keep = [
        "pool", "name_key", "world_event_rank", "starts_365", "top_40",
        "world_event_points_365", "points_per_start", "country",
    ]
    boulder = boulder[keep].sort_values("world_event_rank").drop_duplicates(
        ["pool", "name_key"]
    )
    output = athletes.merge(
        boulder, on=["pool", "name_key"], how="left", suffixes=("", "_wr")
    )
    output["country"] = output["country_wr"].combine_first(output["country"])
    return output.drop(columns=["country_wr"])


def enrich_public_ages(
    athletes: pd.DataFrame,
    profiles: pd.DataFrame,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """Fill only missing ages from the strongest public profile match."""

    if athletes.empty or profiles.empty:
        return athletes
    profiles = profiles.copy()
    profiles["name_key"] = profiles["athlete_name"].map(plain_key)
    profiles["birth_date_confidence_score"] = pd.to_numeric(
        profiles.get("birth_date_confidence_score"), errors="coerce"
    ).fillna(0)
    profiles = (
        profiles.sort_values(
            ["birth_date_confidence_score", "fetched_at_utc"],
            ascending=[False, False],
        )
        .drop_duplicates("name_key")
    )
    profile_dates = profiles[[
        "name_key", "birth_date_analysis_value", "birth_date_confidence",
        "birth_date_uncertainty_days",
    ]].rename(columns={
        "birth_date_analysis_value": "profile_birth_date",
        "birth_date_confidence": "profile_birth_confidence",
        "birth_date_uncertainty_days": "profile_birth_uncertainty_days",
    })
    output = athletes.merge(profile_dates, on="name_key", how="left")
    birth = pd.to_datetime(output["profile_birth_date"], errors="coerce")
    missing_age = output["age"].isna() & birth.notna()
    output.loc[missing_age, "age"] = (as_of - birth.loc[missing_age]).dt.days / 365.2425
    output["birth_date_confidence"] = output.get(
        "birth_date_confidence", pd.Series(index=output.index, dtype=object)
    ).combine_first(output["profile_birth_confidence"])
    output["birth_date_uncertainty_days"] = pd.to_numeric(
        output.get("birth_date_uncertainty_days"), errors="coerce"
    ).combine_first(pd.to_numeric(output["profile_birth_uncertainty_days"], errors="coerce"))
    return output.drop(columns=[
        "profile_birth_date", "profile_birth_confidence",
        "profile_birth_uncertainty_days",
    ])


def roster_names(
    mode: str,
    athletes: pd.DataFrame,
    history: pd.DataFrame,
    rosters: pd.DataFrame,
) -> list[str]:
    if mode == "EEQ" and not rosters.empty:
        return sorted(rosters.loc[rosters["organization"].eq("EEQ"), "athlete_name"].dropna().unique())
    if mode == "YNT Tier 1" and not history.empty:
        event = history.loc[
            history["event_name"].astype(str).str.contains(
                "Youth Championship Arco 2026", case=False, na=False
            )
            & history["source_scope"].eq("IFSC")
            & history["nationality"].astype(str).str.upper().eq("CAN")
        ]
        return sorted(event["athlete_name"].dropna().unique())
    if mode == "Canadian National Team proxy":
        return sorted(
            athletes.loc[athletes["cnr_rank"].le(15), "athlete_name"]
            .dropna().unique()
        )
    return []


def selected_rows(athletes: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    keys = {plain_key(name) for name in names}
    return athletes.loc[athletes["name_key"].isin(keys)].copy()


def rating_help() -> str:
    return (
        "Every displayed family uses the same anchor: 2000 means a fitted 50% "
        "chance of reaching a semifinal at a randomly sampled 2025 IFSC Open "
        "World Cup, within the athlete's gender pool. This shifts the scale for "
        "interpretation without changing athlete order or model updates. "
        "Global-ELO uses every de-duplicated Boulder result on one Open World-Cup "
        "readiness scale. IFSC-ELO uses IFSC results only. WR-ELO uses only events "
        "that award IFSC World Ranking points. Specialist ratings are shown only "
        "with at least two eligible rounds and shrink toward Global-ELO while "
        "evidence is limited. Performance-ELO is one round's isolated level, not "
        "the athlete's stable rating."
    )


def rating_transform_controls(key: str, default: str) -> str:
    columns = st.columns([1.55, 1])
    family = columns[0].segmented_control(
        "Rating evidence", RATING_ORDER, default=default,
        help=rating_help(), key=f"{key}_family",
    )
    format_name = columns[1].selectbox(
        "Round format", FORMAT_OPTIONS, index=0,
        disabled=family != "Global-ELO",
        help=(
            "Format-specific ratings are available for the Global evidence pool. "
            "IFSC and WR transformations already describe a narrower event pool."
        ),
        key=f"{key}_format",
    )
    if family == "Global-ELO" and format_name != "All formats":
        return f"Global-ELO-{format_name}"
    return family


def correlation_note(
    correlations: pd.DataFrame, family: str, pool: str | None = None
) -> str:
    if correlations.empty or family == "WR-ELO":
        return "WR-ELO is the reference scale in this view."
    rows = correlations.loc[correlations["rating_family"].eq(family)]
    if pool:
        rows = rows.loc[rows["pool"].eq(pool)]
    rows = rows.dropna(subset=["spearman_correlation"])
    if rows.empty:
        return "Not enough paired evidence to report a stable relationship with WR-ELO."
    value = float(np.average(rows["spearman_correlation"], weights=rows["athletes"]))
    n = int(rows["athletes"].sum())
    return (
        f"Rank correlation with WR-ELO: {value:.2f} across {n} paired athlete-pools. "
        "This shows association, not why the ratings differ; setting specificity, "
        "training environment, attendance, travel and selection remain mixed together."
    )


def add_selected_highlight(
    figure: go.Figure,
    frame: pd.DataFrame,
    selected: list[str],
    x: str,
    y: str,
) -> None:
    focus = selected_rows(frame, selected).dropna(subset=[x, y])
    for _, row in focus.iterrows():
        figure.add_trace(
            go.Scatter(
                x=[row[x]], y=[row[y]], mode="markers+text",
                text=[friendly_name(row["athlete_name"])], textposition="top left",
                marker={
                    "size": 14, "color": "rgba(0,0,0,0)",
                    "line": {"width": 2, "color": "#36524E"},
                    "symbol": "diamond" if row.get("gender") == "Women" else "circle",
                },
                hoverinfo="skip",
                showlegend=False,
            )
        )


def displacement_lines(
    figure: go.Figure,
    frame: pd.DataFrame,
    stage: str,
    x: str,
) -> None:
    transitions = []
    if stage.startswith("Global-ELO-"):
        transitions.append(("Global-ELO", stage, "#9AA7A4"))
    if stage in {"IFSC-ELO", "WR-ELO"}:
        transitions.append(("Global-ELO", "IFSC-ELO", "#9AA7A4"))
    if stage == "WR-ELO":
        transitions.append(("IFSC-ELO", "WR-ELO", PALETTE["gold"]))
    for start, end, color in transitions:
        subset = frame.dropna(subset=[x, start, end])
        for _, row in subset.iterrows():
            figure.add_trace(
                go.Scatter(
                    x=[row[x], row[x]], y=[row[start], row[end]],
                    mode="lines", line={"width": 1, "color": color},
                    opacity=0.22, hoverinfo="skip", showlegend=False,
                )
            )


def pool_scatter(
    frame: pd.DataFrame,
    x: str,
    rating: str,
    selected: list[str],
    title: str,
    canadian_outline: bool = False,
) -> go.Figure:
    plot = frame.dropna(subset=[x, rating]).copy()
    if plot.empty:
        return go.Figure().update_layout(title="No matched ratings for this view")
    if "display_name" not in plot:
        plot["display_name"] = plot["athlete_name"].map(friendly_name)
    country = (
        plot.get("country", pd.Series("", index=plot.index))
        .fillna("").astype(str).str.upper()
    )
    country_codes = {
        "CANADA": "CAN", "UNITED STATES": "USA", "FRANCE": "FRA",
        "AUSTRALIA": "AUS", "JAPAN": "JPN", "GREAT BRITAIN": "GBR",
        "UNITED KINGDOM": "GBR", "GERMANY": "GER", "ITALY": "ITA",
        "SLOVENIA": "SLO", "AUSTRIA": "AUT", "SWITZERLAND": "SUI",
    }
    plot["country_code"] = country.map(country_codes).fillna(country.str.slice(0, 3))
    age = pd.to_numeric(
        plot.get("age", pd.Series(np.nan, index=plot.index)), errors="coerce"
    )
    uncertainty_days = pd.to_numeric(
        plot.get(
            "birth_date_uncertainty_days", pd.Series(np.nan, index=plot.index)
        ),
        errors="coerce",
    )
    plot["age_hover"] = [
        f"{value:.1f} ± {uncertainty / 365.2425:.1f} years"
        if np.isfinite(value) and np.isfinite(uncertainty)
        else (f"{value:.1f} years" if np.isfinite(value) else "Unknown")
        for value, uncertainty in zip(age, uncertainty_days)
    ]
    days = pd.to_numeric(
        plot.get("days_since_last_result", pd.Series(np.nan, index=plot.index)),
        errors="coerce",
    )
    plot["days_hover"] = [
        (
            f"{'🔴' if value > 180 else '🟠' if value > 90 else '🟢'} "
            f"{int(value)} days"
        ) if np.isfinite(value) else "Unknown"
        for value in days
    ]
    evidence_column = f"{rating} evidence"
    evidence = pd.to_numeric(
        plot.get(evidence_column, pd.Series(np.nan, index=plot.index)), errors="coerce"
    )
    plot["rounds_hover"] = evidence.fillna(0).astype(int)
    plot["momentum_hover"] = plot["momentum"].map(
        lambda value: f"{value:+.1f} Global-ELO / 365d"
    )
    figure = px.scatter(
        plot,
        x=x,
        y=rating,
        color="momentum",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        symbol="gender",
        hover_name="display_name",
        custom_data=[
            "country_code", "momentum_hover", "age_hover", "days_hover",
            "rounds_hover", "display_name",
        ],
        title=title,
    )
    figure.update_traces(
        marker={"size": 9, "opacity": 0.72},
        hovertemplate=(
            "<b>%{customdata[5]} (%{customdata[0]})</b><br>"
            f"{rating}: %{{y:,.0f}}<br>"
            "Momentum: %{customdata[1]}<br>"
            "Age: %{customdata[2]}<br>"
            "Last result: %{customdata[3]}<br>"
            "Included rounds: %{customdata[4]}<extra></extra>"
        ),
    )
    if canadian_outline and "country" in plot:
        canadians = plot.loc[plot["country"].astype(str).str.upper().eq("CAN")]
        figure.add_trace(
            go.Scatter(
                x=canadians[x], y=canadians[rating], mode="markers",
                marker={"size": 13, "color": "rgba(0,0,0,0)", "line": {"width": 2, "color": "#111"}},
                hoverinfo="skip", name="Canada", showlegend=True,
            )
        )
    displacement_lines(figure, plot, rating, x)
    add_selected_highlight(figure, plot, selected, x, rating)
    figure.update_layout(
        height=610,
        margin={"l": 88, "r": 36, "t": 76, "b": 138},
        legend={
            "title": {"text": "Gender"}, "orientation": "h",
            "x": 0, "xanchor": "left", "y": -0.18, "yanchor": "top",
        },
        coloraxis_colorbar={
            "title": {"text": "Momentum · Global-ELO / 365d", "side": "top"},
            "orientation": "h", "x": 1, "xanchor": "right",
            "y": -0.25, "yanchor": "top", "len": 0.46, "thickness": 12,
            "nticks": 3, "tickangle": 0,
        },
        hovermode="closest",
    )
    x_titles = {
        "cnr_rank": "CNR rank", "age": "Age (years)",
        "ifsc_rank": "IFSC-ELO rank", "world_event_rank": "IFSC World Ranking",
    }
    figure.update_xaxes(
        title={"text": x_titles.get(x, x.replace("_", " ").title()), "standoff": 16},
        tickformat=",.0f", separatethousands=True, automargin=True,
    )
    figure.update_yaxes(
        title={"text": rating, "standoff": 12}, gridcolor="#E5ECEA",
        tickformat=",.0f", separatethousands=True, automargin=True,
    )
    return figure


def compare_text(
    frame: pd.DataFrame,
    selected: list[str],
    rating: str,
    context: str = "",
) -> str:
    focus = selected_rows(frame, selected).dropna(subset=[rating]).sort_values(rating, ascending=False)
    if focus.empty:
        return "The selected athletes do not yet have matched evidence in this pool."
    leader = focus.iloc[0]
    if len(focus) == 1:
        return f"{friendly_name(leader['athlete_name'])} is shown at {leader[rating]:.0f} {rating}."
    trailer = focus.iloc[-1]
    gap = leader[rating] - trailer[rating]
    base = (
        f"{friendly_name(leader['athlete_name'])} leads this comparison at {leader[rating]:.0f}. "
        f"The displayed gap to {friendly_name(trailer['athlete_name'])} is {gap:.0f} points. "
    )
    if context == "Canadian":
        rank = leader.get("cnr_rank", np.nan)
        detail = (
            f"The same athlete is CNR #{int(rank)}; disagreement between CNR and "
            "Global-ELO is the useful review signal. "
            if pd.notna(rank) else "Their CNR rank is not matched in this snapshot. "
        )
    elif context == "IFSC":
        transport = leader.get("IFSC-ELO", np.nan) - leader.get("Global-ELO", np.nan)
        detail = (
            f"Their IFSC-minus-Global gap is {transport:+.0f}; this describes how "
            "their IFSC evidence differs, without assigning the cause. "
            if np.isfinite(transport) else "Their IFSC-specific evidence is still incomplete. "
        )
    elif context == "WR":
        rank = leader.get("world_event_rank", np.nan)
        starts = leader.get("starts_365", np.nan)
        detail = (
            f"Current World Ranking: {int(rank) if pd.notna(rank) else 'not ranked'} "
            f"from {int(starts) if pd.notna(starts) else 0} eligible starts in 365 days. "
        )
    elif context == "Progression":
        detail = f"Their Global-ELO changed {leader.get('momentum', 0):+.0f} over the latest 365-day window. "
    else:
        detail = ""
    return base + detail + (
        "Read the gap with evidence count and last-result date; it is a projection "
        "difference, not a selection verdict."
    )


def render_canadian_pool(
    athletes: pd.DataFrame,
    selected: list[str],
    correlations: pd.DataFrame,
) -> None:
    st.subheader("Canadian Pool")
    st.caption("Where every current Canadian CNR athlete sits, and what changes when the evidence becomes more competition-specific.")
    canadian = athletes.loc[athletes["cnr_rank"].notna()].copy()
    x_choice = st.segmented_control(
        "Horizontal axis", ["CNR rank", "Age"], default="CNR rank",
        help="CNR rank compares current Canadian ranking position. Age compares pathway timing.",
        key="canadian_x",
    )
    stage = rating_transform_controls("canadian", "Global-ELO")
    x = "cnr_rank" if x_choice == "CNR rank" else "age"
    figure = pool_scatter(
        canadian, x, stage, selected,
        f"Canadian CNR athletes — {stage}",
    )
    if x == "cnr_rank":
        figure.update_xaxes(autorange="reversed")
    st.plotly_chart(figure, width="stretch", theme=None)
    st.info(compare_text(canadian, selected, stage, "Canadian"), icon="↗️")
    st.caption(correlation_note(correlations, stage))
    missing = int(canadian[stage].isna().sum()) if stage in canadian else len(canadian)
    if missing:
        st.caption(f"{missing} CNR athlete(s) are retained in the pool but cannot be plotted on {stage} yet because eligible evidence is missing.")


def render_ifsc_pool(
    athletes: pd.DataFrame,
    history: pd.DataFrame,
    selected: list[str],
    correlations: pd.DataFrame,
) -> None:
    st.subheader("IFSC Pool")
    st.caption("Canadians beside every 2025–2026 IFSC Boulder finalist; a black ring identifies Canada.")
    finalist_ids: set[str] = set()
    if not history.empty:
        dates = pd.to_datetime(history["event_date"], errors="coerce")
        finalist_ids = set(
            history.loc[
                history["source_scope"].eq("IFSC")
                & history["round_group"].eq("Final")
                & dates.dt.year.isin([2025, 2026]),
                "global_id",
            ].astype(str)
        )
    pool = athletes.loc[
        athletes["global_id"].astype(str).isin(finalist_ids)
        | athletes["cnr_rank"].notna()
    ].copy()
    pool["ifsc_rank"] = pool.groupby("pool")["IFSC-ELO"].rank(ascending=False, method="min")
    x_choice = st.segmented_control(
        "Horizontal axis", ["IFSC rank", "Age"], default="Age", key="ifsc_x"
    )
    stage = rating_transform_controls("ifsc", "IFSC-ELO")
    x = "ifsc_rank" if x_choice == "IFSC rank" else "age"
    figure = pool_scatter(
        pool, x, stage, selected,
        f"Canadian CNR athletes and recent IFSC finalists — {stage}",
        canadian_outline=True,
    )
    if x == "ifsc_rank":
        figure.update_xaxes(autorange="reversed")
    st.plotly_chart(figure, width="stretch", theme=None)
    st.info(compare_text(pool, selected, stage, "IFSC"), icon="↗️")
    st.caption(correlation_note(correlations, stage))


def render_wr_pool(
    athletes: pd.DataFrame,
    selected: list[str],
    correlations: pd.DataFrame,
) -> None:
    st.subheader("WR Pool")
    st.caption("The current World Ranking top 40 plus every Canadian with a World Ranking start in the last 365 days.")
    pool = athletes.loc[
        athletes["world_event_rank"].le(40)
        | (
            athletes["country"].astype(str).str.upper().eq("CAN")
            & athletes["starts_365"].fillna(0).gt(0)
        )
    ].copy()
    stage = rating_transform_controls("wr", "WR-ELO")
    figure = pool_scatter(
        pool, "world_event_rank", stage, selected,
        f"World Ranking pool — {stage}", canadian_outline=True,
    )
    figure.update_xaxes(autorange="reversed", title="Current IFSC World Ranking")
    st.plotly_chart(figure, width="stretch", theme=None)
    st.info(compare_text(pool, selected, stage, "WR"), icon="↗️")
    st.caption(correlation_note(correlations, stage))

    available = sorted(
        country for country in pool["country"].dropna().astype(str).unique()
        if country and country.upper() != "CAN"
    )
    defaults = [country for country in ["FRA", "USA", "AUS"] if country in available]
    countries = st.multiselect(
        "Compare Canada with three countries",
        available,
        default=defaults,
        max_selections=3,
    )
    country_pool = pool.loc[pool["country"].isin(["CAN", *countries])].dropna(subset=["WR-ELO"])
    if not country_pool.empty:
        comparison = px.strip(
            country_pool, x="country", y="WR-ELO", color="country",
            hover_name="display_name",
            hover_data={"world_event_rank": True, "starts_365": True},
            title="Actual WR-ELO distribution of current participants",
        )
        comparison.update_traces(marker={"size": 10, "opacity": 0.65})
        comparison.update_layout(height=430, showlegend=False)
        st.plotly_chart(comparison, width="stretch", theme=None)


def age_group(age: float) -> str:
    if pd.isna(age):
        return "Age unknown"
    if age < 15:
        return "U15"
    if age < 17:
        return "U17"
    if age < 19:
        return "U19"
    if age < 21:
        return "U21"
    return "Senior"


def render_progression(
    athletes: pd.DataFrame,
    history: pd.DataFrame,
    selected: list[str],
    correlations: pd.DataFrame,
) -> None:
    st.subheader("Global progression")
    st.caption("Compare athletes at the same age, then separate current level from the direction of travel.")
    cohort = athletes.loc[
        ((athletes["age"] < 21) & athletes["cnr_rank"].notna())
        | athletes["cnr_rank"].le(15)
    ].copy()
    if not history.empty:
        dates = pd.to_datetime(history["event_date"], errors="coerce")
        canadian_youth_event = (
            history["source_scope"].eq("CEC")
            & history["event_name"].astype(str).str.contains(
                "Youth National", case=False, na=False
            )
        )
        latest_canadian_youth = dates.loc[canadian_youth_event].max()
        finalists = history.loc[
            canadian_youth_event
            & history["round_group"].eq("Final")
            & dates.eq(latest_canadian_youth)
        ]
        cohort = pd.concat(
            [cohort, athletes.loc[athletes["global_id"].isin(finalists["global_id"])]],
            ignore_index=True,
        ).drop_duplicates(["pool", "global_id"])
    cohort["Age group"] = cohort["age"].map(age_group)
    plot = cohort.dropna(subset=["age", "Global-ELO"])
    figure = px.scatter(
        plot, x="age", y="Global-ELO", color="Age group", symbol="gender",
        hover_name="display_name",
        hover_data={"cnr_rank": True, "momentum": ":.1f", "country": True},
        title="Current Canadian pathway — one Open World-readiness scale",
    )
    figure.update_traces(marker={"size": 10, "opacity": 0.65})
    add_selected_highlight(figure, plot, selected, "age", "Global-ELO")

    show_lines = st.toggle(
        "Show pathway reference lines",
        value=True,
        help=(
            "Shows the mean Global-ELO of current CNR top-five athletes by age "
            "band and the current IFSC-ELO level near a 50% semifinal rate. "
            "These are descriptive references, not selection standards."
        ),
    )
    if show_lines:
        top5 = athletes.loc[athletes["cnr_rank"].le(5)].dropna(subset=["age", "Global-ELO"])
        if not top5.empty:
            grouped = (
                top5.assign(age_year=top5["age"].round())
                .groupby(["gender", "age_year"], as_index=False)["Global-ELO"]
                .mean()
            )
            for gender, gender_rows in grouped.groupby("gender"):
                figure.add_trace(go.Scatter(
                    x=gender_rows["age_year"], y=gender_rows["Global-ELO"], mode="lines",
                    name=f"Current CNR top-five mean — {gender}",
                    line={
                        "color": PALETTE["teal"] if gender == "Men" else PALETTE["blue"],
                        "width": 3, "dash": "dot",
                    },
                ))
        for gender, pool_name in (("Men", "Boulder_Men"), ("Women", "Boulder_Women")):
            semi_level = semi_probability_reference(history, athletes, pool_name)
            if np.isfinite(semi_level):
                figure.add_hline(
                    y=semi_level, line_dash="dash",
                    line_color=PALETTE["coral"] if gender == "Men" else PALETTE["gold"],
                    annotation_text=f"Current IFSC ~50% semifinal — {gender}",
                )
    figure.update_layout(height=570, margin={"l": 20, "r": 20, "t": 70, "b": 20})
    st.plotly_chart(figure, width="stretch", theme=None)
    st.info(compare_text(cohort, selected, "Global-ELO", "Progression"), icon="↗️")

    projection_figure = progression_projection(athletes, history, selected)
    st.plotly_chart(projection_figure, width="stretch", theme=None)
    st.caption(
        "Projection rate = 65% of the athlete's bounded recent Global-ELO "
        "change + 35% of the median IFSC Performance-ELO change observed at "
        "the same age and gender. It assumes the trend continues; it is not a "
        "training-effect claim."
    )
    render_focus_hypotheses(athletes, history, selected)
    st.caption(correlation_note(correlations, "Global-ELO"))


def semi_probability_reference(
    history: pd.DataFrame,
    athletes: pd.DataFrame,
    pool: str | None = None,
) -> float:
    if history.empty:
        return np.nan
    latest_date = pd.Timestamp(
        pd.to_datetime(history["event_date"], errors="coerce").max()
    )
    recent_cutoff = latest_date - pd.Timedelta(days=365)
    starts = history.loc[
        history["source_scope"].eq("IFSC")
        & history["round_group"].eq("Qualification")
        & pd.to_datetime(history["event_date"], errors="coerce").ge(recent_cutoff)
    ].copy()
    if pool:
        starts = starts.loc[starts["pool"].eq(pool)]
    if starts.empty:
        return np.nan
    semis = history.loc[
        history["source_scope"].eq("IFSC")
        & history["round_group"].eq("Semi-final")
        & pd.to_datetime(history["event_date"], errors="coerce").ge(recent_cutoff)
    ]
    advanced_keys = set(zip(
        semis["source_event_id"].astype(str),
        semis["pool"].astype(str),
        semis["global_id"].astype(str),
    ))
    starts["advanced"] = [
        (str(event), str(athlete_pool), str(athlete)) in advanced_keys
        for event, athlete_pool, athlete in zip(
            starts["source_event_id"], starts["pool"], starts["global_id"]
        )
    ]
    current = athletes[["pool", "global_id", "IFSC-ELO"]]
    starts = starts.merge(current, on=["pool", "global_id"], how="left").dropna(subset=["IFSC-ELO"])
    if starts.empty:
        return np.nan
    bands = starts.assign(band=(starts["IFSC-ELO"] / 50).round() * 50).groupby("band")["advanced"].agg(["mean", "count"])
    bands = bands.loc[bands["count"].ge(8)]
    return float((bands["mean"] - 0.5).abs().idxmin()) if not bands.empty else np.nan


def progression_projection(
    athletes: pd.DataFrame, history: pd.DataFrame, selected: list[str]
) -> go.Figure:
    figure = go.Figure()
    focus = selected_rows(athletes, selected)
    if history.empty or focus.empty:
        return figure.update_layout(title="Progression projection unavailable")
    as_of = pd.to_datetime(history["event_date"], errors="coerce").max()
    history = history.copy()
    history["event_date"] = pd.to_datetime(history["event_date"], errors="coerce")
    age_rates = typical_age_progression(history)
    colors = [PALETTE["teal"], PALETTE["coral"], PALETTE["blue"]]
    for color, (_, athlete) in zip(colors, focus.iterrows()):
        rows = history.loc[
            history["global_id"].eq(athlete["global_id"])
            & history["pool"].eq(athlete["pool"])
        ].sort_values("event_date")
        if rows.empty:
            continue
        figure.add_trace(go.Scatter(
            x=rows["event_date"], y=rows["rating_after"], mode="lines",
            name=f"{athlete['athlete_name']} — observed", line={"color": color, "width": 3},
        ))
        momentum = float(np.clip(athlete.get("momentum", 0.0), -150, 150))
        age_year = int(round(athlete.get("age", np.nan))) if pd.notna(athlete.get("age")) else -1
        typical = age_rates.get((athlete.get("pool"), age_year), 0.0)
        projected_rate = float(np.clip(0.65 * momentum + 0.35 * typical, -150, 150))
        future_dates = pd.date_range(as_of, periods=13, freq="MS")
        central = float(athlete["Global-ELO"]) + projected_rate * np.arange(13) / 12
        uncertainty = 35 + 4 * np.arange(13)
        figure.add_trace(go.Scatter(
            x=future_dates, y=central, mode="lines",
            name=(
                f"{athlete['athlete_name']} — hypothesis "
                f"({projected_rate:+.0f}/year)"
            ),
            line={"color": color, "width": 3, "dash": "dash"},
        ))
        figure.add_trace(go.Scatter(
            x=list(future_dates) + list(future_dates[::-1]),
            y=list(central + uncertainty) + list((central - uncertainty)[::-1]),
            fill="toself", fillcolor=transparent(color),
            line={"color": "rgba(0,0,0,0)"}, hoverinfo="skip", showlegend=False,
        ))
    figure.add_vline(x=as_of.timestamp() * 1000, line_dash="dash", line_color="#555", annotation_text="Now")
    figure.update_layout(
        title="Observed Global-ELO and a 12-month bounded-trend hypothesis",
        height=500, yaxis_title="Global-ELO", xaxis_title="Event date",
        hovermode="x unified",
    )
    return figure


def typical_age_progression(history: pd.DataFrame) -> dict[tuple[str, int], float]:
    """Median IFSC Performance-ELO change at each age; descriptive only."""

    required = {
        "pool", "global_id", "event_date", "performance_elo",
        "age_at_event", "source_scope",
    }
    if history.empty or not required.issubset(history.columns):
        return {}
    rows = history.loc[history["source_scope"].eq("IFSC")].dropna(
        subset=["age_at_event", "performance_elo", "event_date"]
    ).copy()
    rows["age_year"] = pd.to_numeric(rows["age_at_event"], errors="coerce").round()
    rows = rows.loc[rows["age_year"].between(12, 45)]
    rows = rows.sort_values("event_date")
    segments = (
        rows.groupby(["pool", "global_id", "age_year"], as_index=False)
        .agg(
            first_date=("event_date", "first"),
            last_date=("event_date", "last"),
            first_rating=("performance_elo", "first"),
            last_rating=("performance_elo", "last"),
            observations=("performance_elo", "size"),
        )
    )
    segments["years"] = (
        pd.to_datetime(segments["last_date"]) - pd.to_datetime(segments["first_date"])
    ).dt.days / 365.2425
    segments = segments.loc[segments["years"].ge(0.20) & segments["observations"].ge(2)]
    segments["annual_change"] = (
        (segments["last_rating"] - segments["first_rating"]) / segments["years"]
    ).clip(-300, 300)
    typical = segments.groupby(["pool", "age_year"])["annual_change"].median()
    return {
        (str(pool), int(age)): float(value)
        for (pool, age), value in typical.items()
        if pd.notna(age) and np.isfinite(value)
    }


def render_focus_hypotheses(
    athletes: pd.DataFrame, history: pd.DataFrame, selected: list[str]
) -> None:
    focus = selected_rows(athletes, selected)
    rows = []
    for _, athlete in focus.iterrows():
        global_elo = athlete.get("Global-ELO", np.nan)
        semi = semi_probability_reference(history, athletes, athlete.get("pool"))
        wr_starts = athlete.get("starts_365", np.nan)
        momentum = athlete.get("momentum", 0.0)
        if pd.isna(global_elo):
            hypothesis = "Build a reliable competition baseline before choosing a pathway emphasis."
        elif np.isfinite(semi) and global_elo >= semi - 100 and (pd.isna(wr_starts) or wr_starts < 3):
            hypothesis = "Test targeted WR competition exposure; readiness appears close enough for the experience to be informative."
        elif momentum > 35:
            hypothesis = "Protect the improving training process; add WR starts selectively rather than chasing participation volume."
        else:
            hypothesis = "Prioritize raising repeatable performance; choose competitions that answer a specific readiness question."
        rows.append({"Athlete": friendly_name(athlete["athlete_name"]), "Working hypothesis": hypothesis})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("These are transparent decision hypotheses from rating level, recent change and WR exposure—not causal training prescriptions.")


def render_olympics(
    athletes: pd.DataFrame,
    selected: list[str],
    correlations: pd.DataFrame,
) -> None:
    st.subheader("Towards the Olympics")
    st.caption("A single view of World readiness, World Ranking access and evidence gaps. It does not claim qualification before the LA28 rules and field are modelled.")
    focus = selected_rows(athletes, selected)
    columns = st.columns(max(1, min(3, len(focus))))
    for column, (_, athlete) in zip(columns, focus.iterrows()):
        column.markdown(f"#### {friendly_name(athlete['athlete_name'])}")
        column.metric("Global-ELO", f"{athlete.get('Global-ELO', np.nan):.0f}" if pd.notna(athlete.get("Global-ELO")) else "—")
        column.metric("IFSC-ELO", f"{athlete.get('IFSC-ELO', np.nan):.0f}" if pd.notna(athlete.get("IFSC-ELO")) else "—")
        column.metric("WR-ELO", f"{athlete.get('WR-ELO', np.nan):.0f}" if pd.notna(athlete.get("WR-ELO")) else "—")
        rank = athlete.get("world_event_rank", np.nan)
        column.caption(f"Current World Ranking: {int(rank) if pd.notna(rank) else 'not ranked'} · starts/365d: {int(athlete.get('starts_365', 0) or 0)}")
    if not focus.empty:
        table = focus[[
            "athlete_name", "Global-ELO", "IFSC-ELO", "WR-ELO",
            "world_event_rank", "starts_365", "momentum",
        ]].copy()
        table["athlete_name"] = table["athlete_name"].map(friendly_name)
        table = table.rename(columns={
            "athlete_name": "Athlete", "world_event_rank": "World Ranking",
            "starts_365": "WR starts / 365d", "momentum": "Global-ELO change / 365d",
        })
        st.dataframe(table, hide_index=True, width="stretch")
    st.warning(
        "Olympic pathway probabilities remain a governed next step. Until the qualification rules, eligible fields and quota constraints are versioned, this page supports readiness discussion—not selection odds.",
        icon="⚠️",
    )
    st.caption(correlation_note(correlations, "Global-ELO"))


def top_ribbon(
    athletes: pd.DataFrame,
    history: pd.DataFrame,
    rosters: pd.DataFrame,
) -> tuple[list[str], str]:
    with st.container(border=True):
        top = st.columns([1.2, 1])
        mode = top[0].segmented_control(
            "Athlete set",
            ["Compare 3", "EEQ", "YNT Tier 1", "Canadian National Team proxy"],
            default="Compare 3",
            help=(
                "YNT Tier 1 means Canadian 2026 Youth Worlds participants. "
                "The national-team option is a clearly labelled CNR top-15 proxy; "
                "it is not presented as an official roster."
            ),
        )
        discipline = top[1].segmented_control(
            "Discipline", ["Boulder"], default="Boulder",
            help="Boulder is the governed release focus. Lead and Speed stay out of this interface until Boulder is excellent.",
        )
        names = sorted(athletes["athlete_name"].dropna().unique(), key=str.casefold)
        selected: list[str]
        if mode == "Compare 3":
            columns = st.columns(3)
            names_by_key = {plain_key(name): name for name in names}
            defaults = [
                names_by_key[plain_key(name)]
                for name in DEFAULT_ATHLETES
                if plain_key(name) in names_by_key
            ]
            while len(defaults) < 3 and names:
                candidate = names[min(len(defaults), len(names) - 1)]
                if candidate not in defaults:
                    defaults.append(candidate)
                else:
                    break
            selected = []
            for index, column in enumerate(columns):
                default = defaults[index] if index < len(defaults) else names[0]
                selected.append(column.selectbox(
                    "Main athlete" if index == 0 else f"Comparison {index + 1}",
                    names,
                    index=names.index(default),
                    format_func=friendly_name,
                    key=f"athlete_{index}",
                ))
        else:
            preset = roster_names(mode, athletes, history, rosters)
            matched = athletes.loc[athletes["name_key"].isin({plain_key(name) for name in preset}), "athlete_name"].dropna().unique().tolist()
            selected = st.multiselect(
                f"{mode} athletes",
                names,
                default=matched[:12],
                help="Choose up to the athletes you need; the three first selections receive the strongest visual emphasis.",
            )
            if mode == "Canadian National Team proxy":
                st.caption("Proxy only: current CNR top 15 by gender. Replace with the official roster when supplied.")
        return selected[:3], discipline


def render_rating_detail(
    athletes: pd.DataFrame,
    history: pd.DataFrame,
    selected: list[str],
) -> None:
    with st.expander("Compared athletes · all rating evidence"):
        focus = selected_rows(athletes, selected)
        if focus.empty:
            st.caption("No matched rating evidence.")
            return
        tabs = st.tabs([friendly_name(row["athlete_name"]) for _, row in focus.iterrows()])
        for tab, (_, athlete) in zip(tabs, focus.iterrows()):
            with tab:
                metrics = st.columns(3)
                for column, family in zip(metrics, RATING_ORDER):
                    value = athlete.get(family, np.nan)
                    column.metric(family, f"{value:.0f}" if pd.notna(value) else "—")
                detail = pd.DataFrame([
                    {
                        "Rating family": family,
                        "Elo": athlete.get(family),
                        "Eligible contests": athlete.get(f"{family} evidence"),
                    }
                    for family in ALL_RATINGS
                ])
                st.dataframe(
                    detail.style.format(
                        {"Elo": "{:.0f}", "Eligible contests": "{:.0f}"},
                        na_rep="—",
                    ),
                    hide_index=True,
                    width="stretch",
                )
                st.caption(
                    "Missing Elo means the minimum evidence rule was not met. "
                    "Eligible contests describes evidence quantity, not athlete quality."
                )
                rounds = history.loc[
                    history["global_id"].eq(athlete["global_id"])
                    & history["pool"].eq(athlete["pool"])
                ].sort_values("event_date", ascending=False).head(5)
                if not rounds.empty:
                    latest = rounds[[
                        "event_date", "event_name", "round_group",
                        "confirmed_procedure", "performance_elo",
                    ]].rename(columns={
                        "event_date": "Event date", "event_name": "Competition",
                        "round_group": "Round",
                        "confirmed_procedure": "Procedure",
                        "performance_elo": "Performance-ELO",
                    })
                    st.markdown("**Latest isolated round performances**")
                    st.dataframe(
                        latest.style.format({"Performance-ELO": "{:.0f}"}, na_rep="—"),
                        hide_index=True,
                        width="stretch",
                    )


def startup_status(data: dict[str, pd.DataFrame]) -> None:
    missing = [
        key for key in ("athletes", "history")
        if data[key].empty
    ]
    with st.expander("Data health", expanded=bool(missing)):
        if missing:
            st.error(
                "The overview is in safe fallback mode. Missing artifact(s): "
                + ", ".join(missing)
                + ". Run `python scripts/build_boulder_rating_families.py` before deployment."
            )
        else:
            latest = pd.to_datetime(data["history"]["event_date"], errors="coerce").max()
            st.success(
                f"Ready · {len(data['athletes']):,} Boulder athletes · results through {latest:%Y-%m-%d}."
            )
            calibration = data.get("calibration", pd.DataFrame())
            if not calibration.empty:
                starts = int(calibration["qualification_starts"].sum())
                events = int(calibration["events"].sum())
                st.caption(
                    f"Display scale checked: 2000 is the fitted 50% semifinal level "
                    f"from {starts:,} pre-event athlete-starts across {events} "
                    "gender-pool World Cup samples in 2025."
                )
        st.caption("Duplicate controls run before the rating build. Missing specialist evidence is withheld rather than silently replaced.")


def main() -> None:
    st.set_page_config(
        page_title="Comp Climbing Projections",
        page_icon="🧗",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(
        """
        <style>
        .block-container{max-width:1500px;padding-top:1.2rem;padding-bottom:4rem}
        h1,h2,h3{color:#102F2B;letter-spacing:-.025em}
        [data-testid="stMetric"]{background:#F4F8F7;border:1px solid #DCE7E4;border-radius:14px;padding:14px}
        [data-testid="stHorizontalBlock"]{gap:.8rem}
        .stCaption{color:#627571}
        @media(max-width:640px){
          .block-container{padding:.7rem .75rem 3rem}
          h1{font-size:2rem!important}
          [data-testid="stHorizontalBlock"]{flex-wrap:wrap}
          [data-testid="column"]{min-width:100%!important}
          .js-plotly-plot{min-height:420px}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Comp Climbing Projections")
    st.markdown("### Strength, depth and progression of canadian climbers: from local comps to the Olympics")
    st.caption("Boulder release · model evidence supports coaching and governance judgment; it does not replace it.")

    data = read_data()
    startup_status(data)
    if data["athletes"].empty:
        st.stop()
    athletes = data["athletes"].copy()
    athletes["display_name"] = athletes["athlete_name"].map(friendly_name)
    selected, _ = top_ribbon(athletes, data["history"], data["rosters"])
    if not selected:
        st.info("Select at least one athlete to begin.")
        st.stop()
    render_rating_detail(athletes, data["history"], selected)

    st.header("Overview")
    section = st.segmented_control(
        "Overview section",
        ["Canadian Pool", "IFSC Pool", "WR Pool", "Global progression", "Towards Olympics"],
        default="Canadian Pool",
        label_visibility="collapsed",
    )
    renderers = {
        "Canadian Pool": lambda: render_canadian_pool(athletes, selected, data["correlations"]),
        "IFSC Pool": lambda: render_ifsc_pool(athletes, data["history"], selected, data["correlations"]),
        "WR Pool": lambda: render_wr_pool(athletes, selected, data["correlations"]),
        "Global progression": lambda: render_progression(athletes, data["history"], selected, data["correlations"]),
        "Towards Olympics": lambda: render_olympics(athletes, selected, data["correlations"]),
    }
    renderers[section]()

    with st.expander("Rating glossary and model contract"):
        st.write(rating_help())
        st.markdown(
            "- **Global-ELO-Onsight / Scramble / Flash:** only rounds with a confirmed procedure.\n"
            "- **WR-ELO-Qualies / Semies / Finals:** only the named round of World Ranking events.\n"
            "- **IFSC-ELO-Qualies / Semies / Finals:** only the named round of non-para IFSC events.\n"
            "- **Performance-ELO:** the isolated level shown in one round, calculated from ratings frozen before the event."
        )


if __name__ == "__main__":
    main()
