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
ROUND_OPTIONS = ["All rounds", "Qualification", "Semi-final", "Final"]
QUALIFICATION_FORMATS = ["Flash + Onsight", "Flash", "Onsight"]
ALL_RATINGS = [
    "Global-ELO", "Global-ELO-Qualies", "Global-ELO-Qualies-Flash",
    "Global-ELO-Qualies-Onsight", "Global-ELO-Semies", "Global-ELO-Finals",
    "Global-ELO-Onsight", "Global-ELO-Scramble", "Global-ELO-Flash",
    "WR-ELO", "WR-ELO-Qualies", "WR-ELO-Qualies-Flash",
    "WR-ELO-Qualies-Onsight", "WR-ELO-Semies", "WR-ELO-Finals",
    "IFSC-ELO", "IFSC-ELO-Qualies", "IFSC-ELO-Qualies-Flash",
    "IFSC-ELO-Qualies-Onsight", "IFSC-ELO-Semies", "IFSC-ELO-Finals",
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
ATHLETE_COLORS = [
    "#0B7A75", "#F26B5B", "#4285A9", "#E6A23C", "#8E5AA7",
    "#2E8B57", "#D05A8A", "#6B7C93", "#A66A3F", "#4C9F9A",
]


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
        "physical_profiles": ("physical_test_profiles.csv", "csv"),
        "physical_associations": ("physical_test_associations.csv", "csv"),
        "physical_models": ("physical_model_summary.csv", "csv"),
        "model_backtest": ("model_backtest_summary.csv", "csv"),
        "sensitivity_metrics": ("latent_volatility_challenger_metrics.csv", "csv"),
        "sensitivity_status": ("latent_volatility_challenger_status.csv", "csv"),
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
    for family in ALL_RATINGS:
        if family not in values:
            values[family] = np.nan
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
        "interpretation without changing athlete order or model updates. Dashed "
        "final, podium and win lines are fitted from the same frozen 2025 "
        "athlete-starts; they are historical references, not current 2026 odds. "
        "Global-ELO uses every de-duplicated Boulder result on one Open World-Cup "
        "readiness scale. IFSC-ELO uses IFSC results only. WR-ELO uses only events "
        "that award IFSC World Ranking points. Specialist ratings are shown only "
        "with at least two eligible rounds and enough athletes to calibrate the "
        "family; they shrink toward Global-ELO while evidence is limited. "
        "Performance-ELO is one round's isolated level, not "
        "the athlete's stable rating."
    )


def rating_transform_controls(key: str, default: str) -> str:
    columns = st.columns([1.3, 1, 1])
    family = columns[0].segmented_control(
        "Rating evidence", RATING_ORDER, default=default,
        help=rating_help(), key=f"{key}_family",
    )
    round_name = columns[1].selectbox(
        "Round evidence", ROUND_OPTIONS, index=0, key=f"{key}_round",
        help="Use all eligible rounds, or isolate qualification, semifinal or final evidence.",
    )
    procedure = columns[2].selectbox(
        "Qualification procedure", QUALIFICATION_FORMATS, index=0,
        disabled=round_name != "Qualification", key=f"{key}_procedure",
        help="Flash includes shared beta/video rounds; Onsight keeps athletes isolated from other attempts.",
    )
    suffix = {
        "All rounds": "", "Qualification": "-Qualies",
        "Semi-final": "-Semies", "Final": "-Finals",
    }[round_name]
    rating = f"{family}{suffix}"
    if round_name == "Qualification" and procedure != "Flash + Onsight":
        rating = f"{rating}-{procedure}"
    return rating


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
        f"Relationship with WR-ELO: {value:.2f} (rank correlation), using {n} athletes "
        "who have both ratings. A value near 1 means the two ratings order athletes "
        "similarly. It does not tell us why an athlete differs between them."
    )


def add_selected_highlight(
    figure: go.Figure,
    frame: pd.DataFrame,
    selected: list[str],
    x: str,
    y: str,
) -> None:
    focus = selected_rows(frame, selected).dropna(subset=[x, y])
    order = {plain_key(name): index for index, name in enumerate(selected)}
    focus["_selection_order"] = focus["athlete_name"].map(
        lambda name: order.get(plain_key(name), len(order))
    )
    focus = focus.sort_values("_selection_order")
    offsets = [(-58, -34), (-68, 18), (-54, 42)]
    for position, (_, row) in enumerate(focus.iterrows()):
        color = ATHLETE_COLORS[position % len(ATHLETE_COLORS)]
        figure.add_trace(
            go.Scatter(
                x=[row[x]], y=[row[y]], mode="markers",
                marker={
                    "size": 14, "color": "rgba(0,0,0,0)",
                    "line": {"width": 2, "color": color},
                    "symbol": "diamond" if row.get("gender") == "Women" else "circle",
                },
                hoverinfo="skip", name=friendly_name(row["athlete_name"]),
                legendgroup="Compared athletes", showlegend=True,
            )
        )
        if position >= 3:
            continue
        ax, ay = offsets[position % len(offsets)]
        figure.add_annotation(
            x=row[x], y=row[y], text=friendly_name(row["athlete_name"]),
            showarrow=True, arrowhead=0, arrowwidth=1,
            arrowcolor="rgba(54,82,78,0.52)", ax=ax, ay=ay,
            font={"size": 12, "color": PALETTE["ink"]},
            bgcolor="rgba(255,255,255,0.62)", borderpad=2,
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


def outcome_threshold(
    calibration: pd.DataFrame, outcome: str, pool: str = "Boulder_All"
) -> float:
    column = f"display_elo_at_50pct_{outcome}"
    if calibration.empty or column not in calibration:
        return np.nan
    matched = calibration.loc[calibration["pool"].eq(pool), column]
    if matched.notna().any():
        return float(matched.dropna().iloc[0])
    pool_rows = calibration.loc[
        calibration["pool"].isin(["Boulder_Men", "Boulder_Women"]), column
    ]
    return float(pool_rows.mean()) if pool_rows.notna().any() else np.nan


def outcome_probability(
    rating: float, calibration: pd.DataFrame, outcome: str,
    pool: str = "Boulder_All",
) -> tuple[float, float]:
    """Return a fitted 2025 outcome probability and McFadden pseudo-R²."""

    if not np.isfinite(rating) or calibration.empty:
        return np.nan, np.nan
    rows = calibration.loc[calibration["pool"].eq(pool)]
    if rows.empty:
        rows = calibration.loc[calibration["pool"].eq("Boulder_All")]
    if rows.empty:
        return np.nan, np.nan
    row = rows.iloc[0]
    threshold = pd.to_numeric(pd.Series([
        row.get(f"display_elo_at_50pct_{outcome}")
    ]), errors="coerce").iloc[0]
    slope = pd.to_numeric(pd.Series([
        row.get(f"{outcome}_logistic_slope_per_100_native_elo")
    ]), errors="coerce").iloc[0]
    fit = pd.to_numeric(pd.Series([
        row.get(f"{outcome}_mcfadden_pseudo_r2")
    ]), errors="coerce").iloc[0]
    if not np.isfinite(threshold) or not np.isfinite(slope):
        return np.nan, float(fit) if np.isfinite(fit) else np.nan
    linear = np.clip((rating - threshold) / 100.0 * slope, -30, 30)
    return (
        float(1 / (1 + np.exp(-linear))),
        float(fit) if np.isfinite(fit) else np.nan,
    )


def rating_target(family: str) -> str:
    if "-Finals" in family:
        return "podium"
    if "-Semies" in family:
        return "final"
    return "semifinal"


def add_outcome_thresholds(
    figure: go.Figure, calibration: pd.DataFrame
) -> None:
    styles = [
        ("semifinal", "50% semifinal", "rgba(11,122,117,0.42)"),
        ("final", "50% final", "rgba(66,133,169,0.42)"),
        ("podium", "50% podium", "rgba(230,162,60,0.48)"),
        ("win", "50% win", "rgba(242,107,91,0.46)"),
    ]
    for outcome, label, color in styles:
        level = outcome_threshold(calibration, outcome)
        if not np.isfinite(level):
            continue
        figure.add_hline(
            y=level, line_dash="dash", line_width=1, line_color=color,
            annotation_text=f"{label} · 2025",
            annotation_position="top left",
            annotation_font={"size": 10, "color": color.replace("0.4", "0.9")},
        )


def pool_scatter(
    frame: pd.DataFrame,
    x: str,
    rating: str,
    selected: list[str],
    title: str,
    canadian_outline: bool = False,
    calibration: pd.DataFrame | None = None,
) -> go.Figure:
    if rating not in frame.columns:
        return go.Figure().update_layout(
            title=f"{rating} is not yet available in this data build"
        )
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
    add_outcome_thresholds(
        figure, calibration if calibration is not None else pd.DataFrame()
    )
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
    if rating not in frame.columns:
        return (
            f"{rating} is not present in this data build yet. The athlete selection "
            "is preserved while the specialist family is rebuilt."
        )
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
    calibration: pd.DataFrame,
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
        calibration=calibration,
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
    calibration: pd.DataFrame,
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
        calibration=calibration,
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
    calibration: pd.DataFrame,
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
        calibration=calibration,
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
        add_outcome_thresholds(comparison, calibration)
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
    calibration: pd.DataFrame,
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
        "Show current World top-10 pathway",
        value=True,
        help=(
            "Shows how today's World Ranking top 10 were rated at each age. "
            "An age bin appears only when it contains several athletes."
        ),
    )
    if show_lines:
        top10 = athletes.loc[athletes["world_event_rank"].le(10), ["pool", "global_id"]]
        pathway = history.merge(top10, on=["pool", "global_id"], how="inner")
        pathway = pathway.dropna(subset=["age_at_event", "rating_after"]).copy()
        pathway["age_year"] = (
            pd.to_numeric(pathway["age_at_event"], errors="coerce") * 2
        ).round() / 2
        pathway = pathway.merge(
            athletes[["pool", "global_id", "gender"]],
            on=["pool", "global_id"], how="left",
        )
        grouped = pathway.groupby(["gender", "age_year"], as_index=False).agg(
            rating=("rating_after", "mean"), athletes=("global_id", "nunique")
        )
        grouped = grouped.loc[grouped["athletes"].ge(3)]
        if not grouped.empty:
            for gender, gender_rows in grouped.groupby("gender"):
                figure.add_trace(go.Scatter(
                    x=gender_rows["age_year"], y=gender_rows["rating"], mode="lines",
                    name=f"Current WR top-10 history — {gender}",
                    line={
                        "color": PALETTE["teal"] if gender == "Men" else PALETTE["blue"],
                        "width": 3, "dash": "dot",
                    },
                ))
    add_outcome_thresholds(figure, calibration)
    figure.update_layout(height=570, margin={"l": 20, "r": 20, "t": 70, "b": 20})
    st.plotly_chart(figure, width="stretch", theme=None)
    st.info(compare_text(cohort, selected, "Global-ELO", "Progression"), icon="↗️")

    projection_figure = progression_projection(
        athletes, history, selected, calibration
    )
    st.plotly_chart(projection_figure, width="stretch", theme=None)
    st.caption(
        "Projection rate = 65% of the athlete's bounded recent Global-ELO "
        "change + 35% of the median IFSC Performance-ELO change observed at "
        "the same age and gender. It assumes the trend continues; it is not a "
        "training-effect claim."
    )
    render_matchup_matrix(athletes, history, selected)
    render_focus_hypotheses(athletes, history, selected, calibration)
    st.caption(correlation_note(correlations, "Global-ELO"))


def progression_projection(
    athletes: pd.DataFrame,
    history: pd.DataFrame,
    selected: list[str],
    calibration: pd.DataFrame,
) -> go.Figure:
    figure = go.Figure()
    focus = selected_rows(athletes, selected)
    if history.empty or focus.empty:
        return figure.update_layout(title="Progression projection unavailable")
    as_of = pd.to_datetime(history["event_date"], errors="coerce").max()
    history = history.copy()
    history["event_date"] = pd.to_datetime(history["event_date"], errors="coerce")
    age_rates = typical_age_progression(history)
    colors = [ATHLETE_COLORS[index % len(ATHLETE_COLORS)] for index in range(len(focus))]
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
    add_outcome_thresholds(figure, calibration)
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


def render_matchup_matrix(
    athletes: pd.DataFrame, history: pd.DataFrame, selected: list[str]
) -> None:
    """Pairwise Elo ordering now and under the same bounded 12-month hypothesis."""

    focus = selected_rows(athletes, selected).dropna(subset=["Global-ELO"]).copy()
    if focus.empty or len(focus) < 2:
        return
    focus["selection_order"] = focus["name_key"].map({
        plain_key(name): index for index, name in enumerate(selected)
    })
    focus = focus.sort_values("selection_order").head(12)
    age_rates = typical_age_progression(history)
    projected = {}
    for _, athlete in focus.iterrows():
        age_year = int(round(athlete["age"])) if pd.notna(athlete.get("age")) else -1
        typical = age_rates.get((athlete.get("pool"), age_year), 0.0)
        rate = float(np.clip(
            0.65 * float(np.clip(athlete.get("momentum", 0.0), -150, 150))
            + 0.35 * typical, -150, 150,
        ))
        projected[(athlete["pool"], athlete["global_id"])] = float(athlete["Global-ELO"]) + rate
    labels = [friendly_name(name) for name in focus["athlete_name"]]
    matrix = []
    for _, row in focus.iterrows():
        cells = []
        for _, opponent in focus.iterrows():
            if row["global_id"] == opponent["global_id"] and row["pool"] == opponent["pool"]:
                cells.append("—")
            elif row["pool"] != opponent["pool"]:
                cells.append("Different pool")
            else:
                now = 1 / (1 + 10 ** ((opponent["Global-ELO"] - row["Global-ELO"]) / 400))
                future = 1 / (1 + 10 ** ((
                    projected[(opponent["pool"], opponent["global_id"])]
                    - projected[(row["pool"], row["global_id"])]
                ) / 400))
                cells.append(f"{now:.0%} → {future:.0%}")
        matrix.append(cells)
    table = pd.DataFrame(matrix, columns=labels, index=labels)
    st.markdown("#### Pairwise rating comparison")
    st.caption(
        "Each cell is the row athlete's Elo-expected chance of placing ahead: now → "
        "the 12-month bounded-trend hypothesis. This is not a full competition simulation."
    )
    st.dataframe(table, width="stretch")


def render_focus_hypotheses(
    athletes: pd.DataFrame,
    history: pd.DataFrame,
    selected: list[str],
    calibration: pd.DataFrame,
) -> None:
    focus = selected_rows(athletes, selected)
    cards = []
    for _, athlete in focus.iterrows():
        global_elo = athlete.get("Global-ELO", np.nan)
        wr_elo = athlete.get("WR-ELO", np.nan)
        semi = outcome_threshold(calibration, "semifinal", athlete.get("pool"))
        wr_starts = athlete.get("starts_365", np.nan)
        momentum = athlete.get("momentum", 0.0)
        if pd.isna(global_elo):
            hypothesis = "Build a reliable competition baseline before choosing a pathway emphasis."
        elif np.isfinite(wr_elo) and global_elo - wr_elo >= 100:
            hypothesis = (
                "General performance is ahead of World-Ranking-specific performance. "
                "Prioritize WR-style simulations and selective WR starts; review onsight "
                "decision quality, setting specificity, travel and pressure response."
            )
        elif np.isfinite(semi) and global_elo >= semi - 100 and (pd.isna(wr_starts) or wr_starts < 3):
            hypothesis = "Test targeted WR competition exposure; readiness appears close enough for the experience to be informative."
        elif momentum > 35:
            hypothesis = "Protect the improving training process; add WR starts selectively rather than chasing participation volume."
        else:
            hypothesis = "Prioritize raising repeatable performance; choose competitions that answer a specific readiness question."
        cards.append((friendly_name(athlete["athlete_name"]), hypothesis))
    if cards:
        st.markdown("#### Working hypotheses")
        for start in range(0, len(cards), 3):
            columns = st.columns(min(3, len(cards) - start))
            for column, (name, hypothesis) in zip(columns, cards[start:start + 3]):
                with column.container(border=True):
                    st.markdown(f"**{name}**")
                    st.write(hypothesis)
        st.caption(
            "These are decision hypotheses from rating level, recent change and WR "
            "exposure—not causal training prescriptions. More starts can reveal a "
            "rating more clearly, but the current data do not prove that starts cause improvement."
        )


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
                default=matched,
                help="All matched members start selected. Uncheck any athlete to simplify an individual graph.",
            )
            if mode == "Canadian National Team proxy":
                st.caption("Proxy only: current CNR top 15 by gender. Replace with the official roster when supplied.")
        return selected, discipline


def render_rating_detail(
    athletes: pd.DataFrame,
    history: pd.DataFrame,
    selected: list[str],
    calibration: pd.DataFrame,
) -> None:
    st.subheader("Compared athletes · all rating evidence")
    focus = selected_rows(athletes, selected)
    if focus.empty:
        st.caption("No matched rating evidence. Athletes with no competition yet remain in the roster.")
        return
    focus["selection_order"] = focus["name_key"].map({
        plain_key(name): index for index, name in enumerate(selected)
    })
    focus = focus.sort_values("selection_order")
    rows = []
    for athlete_index, (_, athlete) in enumerate(focus.iterrows()):
        for family in ALL_RATINGS:
            value = pd.to_numeric(pd.Series([athlete.get(family)]), errors="coerce").iloc[0]
            if not np.isfinite(value):
                continue
            target = rating_target(family)
            probability, fit = outcome_probability(
                value, calibration, target, athlete.get("pool", "Boulder_All")
            )
            target_label = {
                "semifinal": "Make semifinal", "final": "Make final",
                "podium": "Make podium",
            }[target]
            estimate = "Not enough validation data"
            if np.isfinite(probability):
                estimate = f"{target_label}: {probability:.0%}"
                if np.isfinite(fit):
                    estimate += f" (fit {fit:.0%})"
            rows.append({
                "Athlete": friendly_name(athlete["athlete_name"]),
                "Rating family": family,
                "Elo": round(float(value)),
                "Included rounds": athlete.get(f"{family} evidence"),
                "Historical outcome estimate": estimate,
            })
    detail = pd.DataFrame(rows)
    if detail.empty:
        st.info("These athletes are kept in the roster but do not yet have competition evidence.")
        return
    legend = " · ".join(
        f"<span style='color:{ATHLETE_COLORS[i % len(ATHLETE_COLORS)]}'>●</span> {friendly_name(name)}"
        for i, name in enumerate(selected)
        if plain_key(name) in set(focus["name_key"])
    )
    st.markdown(legend, unsafe_allow_html=True)
    st.dataframe(
        detail,
        hide_index=True,
        width="stretch",
        height=min(720, 38 * len(detail) + 40),
        column_config={
            "Elo": st.column_config.NumberColumn(format="%d"),
            "Included rounds": st.column_config.NumberColumn(format="%d"),
            "Historical outcome estimate": st.column_config.TextColumn(width="large"),
        },
    )
    st.caption(
        "Fit is McFadden pseudo-R² for the 2025 outcome curve: how much Elo improves "
        "that outcome model over using only the average success rate. It is not the "
        "athlete's certainty and it is not ordinary R²."
    )
    if not history.empty:
        ids = focus[["pool", "global_id", "athlete_name"]]
        recent = history.merge(ids, on=["pool", "global_id"], how="inner", suffixes=("", "_selected"))
        recent["event_date"] = pd.to_datetime(recent["event_date"], errors="coerce")
        recent = (
            recent.sort_values("event_date", ascending=False)
            .groupby(["pool", "global_id"], as_index=False, group_keys=False)
            .head(3)
        )
        if not recent.empty:
            recent["Athlete"] = recent["athlete_name_selected"].map(friendly_name)
            latest = recent[[
                "Athlete", "event_date", "event_name", "round_group",
                "confirmed_procedure", "performance_elo",
            ]].rename(columns={
                "event_date": "Event date", "event_name": "Competition",
                "round_group": "Round", "confirmed_procedure": "Procedure",
                "performance_elo": "Performance-ELO",
            })
            st.markdown("#### Latest isolated round performances")
            st.dataframe(
                latest, hide_index=True, width="stretch",
                column_config={
                    "Event date": st.column_config.DateColumn(format="YYYY-MM-DD"),
                    "Performance-ELO": st.column_config.NumberColumn(format="%d"),
                },
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
                combined = calibration.loc[calibration["pool"].eq("Boulder_All")]
                counted = combined if not combined.empty else calibration
                starts = int(counted["qualification_starts"].sum())
                events = int(counted["events"].sum())
                st.caption(
                    f"Display scale checked: 2000 is the fitted 50% semifinal level "
                    f"from {starts:,} pre-event athlete-starts across {events} "
                    "gender-pool World Cup samples in 2025."
                )
        st.caption("Specialist ratings appear only when the athlete has enough matching round evidence.")


def render_physical_strength(
    athletes: pd.DataFrame, selected: list[str], data: dict[str, pd.DataFrame]
) -> None:
    st.header("Physical Strength")
    st.caption(
        "What testing and self-reported climbing grades explain about current Boulder "
        "ratings. Associations describe shared patterns; they do not prove that changing "
        "one test causes Elo to change."
    )
    profiles = data.get("physical_profiles", pd.DataFrame())
    associations = data.get("physical_associations", pd.DataFrame())
    models = data.get("physical_models", pd.DataFrame())
    if profiles.empty:
        st.info(
            "The governed physical-testing artifact is being rebuilt. Athlete names remain "
            "available even when they have no test or competition evidence yet."
        )
        return
    profiles = profiles.copy()
    profiles["name_key"] = profiles["athlete_name"].map(plain_key)
    rating_columns = [
        "pool", "name_key", "Global-ELO", "IFSC-ELO", "WR-ELO",
        "gender", "age",
    ]
    athlete_ratings = athletes[[
        column for column in rating_columns + ["Global-ELO evidence"]
        if column in athletes
    ]].copy()
    athlete_ratings = athlete_ratings.sort_values(
        "Global-ELO evidence" if "Global-ELO evidence" in athlete_ratings else "Global-ELO",
        ascending=False,
    ).drop_duplicates(["pool", "name_key"])
    profiles = profiles.merge(
        athlete_ratings,
        on=["pool", "name_key"], how="left",
    )
    tests = {
        "Self-reported 50% flash grade": "boulder_grade_50pct_flash_v",
        "Self-reported max grade in ≤3 physical sends (last 3 months)":
            "boulder_grade_3x_physical_sends_last_3_months_v",
    }
    test = st.selectbox("Test or self-reported grade", list(tests))
    value_column = tests[test]
    rating_candidates = [family for family in RATING_ORDER if family in profiles]
    rating = st.selectbox("Rating to explain", rating_candidates or ["Global-ELO"])
    plot = profiles.copy()
    if rating in plot and value_column in plot:
        plot[value_column] = pd.to_numeric(plot[value_column], errors="coerce")
        plot[rating] = pd.to_numeric(plot[rating], errors="coerce")
        plot = plot.dropna(subset=[value_column, rating])
        figure = px.scatter(
            plot, x=value_column, y=rating, hover_name="athlete_name",
            color="gender" if "gender" in plot else None,
            trendline=None,
            title=f"{test} and {rating}",
        )
        figure.update_traces(marker={"size": 10, "opacity": 0.68})
        focus = plot.loc[plot["name_key"].isin({plain_key(name) for name in selected})]
        for index, (_, row) in enumerate(focus.iterrows()):
            figure.add_trace(go.Scatter(
                x=[row[value_column]], y=[row[rating]], mode="markers+text",
                text=[friendly_name(row["athlete_name"])], textposition="top center",
                marker={"size": 15, "color": ATHLETE_COLORS[index % len(ATHLETE_COLORS)],
                        "line": {"width": 2, "color": PALETTE["ink"]}},
                name=friendly_name(row["athlete_name"]),
            ))
        figure.update_layout(height=540)
        st.plotly_chart(figure, width="stretch", theme=None)
        st.caption(
            f"{len(plot)} linked athletes are visible for this exact grade × rating pair. "
            "Grade values are converted to the common V-scale used in the testing source."
        )
    if not associations.empty:
        shown = associations.copy()
        grade_label = (
            "Boulder Grade 50% Flash" if value_column == "boulder_grade_50pct_flash_v"
            else "Boulder Grade 3x physical sends last 3 months"
        )
        shown = shown.loc[shown["grade_metric"].eq(grade_label)]
        elo_label = {
            "Global-ELO": "Current Global quality Elo",
            "IFSC-ELO": "Current production IFSC Elo",
            "WR-ELO": "Current WR Elo",
        }.get(rating)
        if elo_label and "elo_metric" in shown:
            shown = shown.loc[shown["elo_metric"].eq(elo_label)]
        st.markdown("#### Tested relationships")
        st.dataframe(shown, hide_index=True, width="stretch")
    if not models.empty:
        st.markdown("#### Combined models")
        st.dataframe(models, hide_index=True, width="stretch")
        st.caption(
            "Cross-validated means the model is judged on athletes it did not fit. "
            "A negative validation R² means the simple average would have predicted better."
        )


def render_maths_behind(
    athletes: pd.DataFrame, correlations: pd.DataFrame, calibration: pd.DataFrame,
    data: dict[str, pd.DataFrame],
) -> None:
    st.header("Maths behind")
    st.caption("What each model is for, what it can predict, and where it can fail.")
    comparison = pd.DataFrame([
        {
            "Model": "Global-ELO", "Best use": "Overall Open World-Cup readiness",
            "Evidence": "Local, national, youth and international rounds",
            "Strength": "Fast learning for athletes with little IFSC exposure",
            "Main caveat": "Local terrain and field strength must transport correctly",
        },
        {
            "Model": "IFSC-ELO", "Best use": "IFSC-specific readiness",
            "Evidence": "Non-para IFSC rounds",
            "Strength": "Closer competition environment and setting",
            "Main caveat": "Can lag for new or transformed athletes",
        },
        {
            "Model": "WR-ELO", "Best use": "World Ranking event performance",
            "Evidence": "Only WR-point events",
            "Strength": "Most specific to ranking access",
            "Main caveat": "Sparse and participation-selected evidence",
        },
        {
            "Model": "Performance-ELO", "Best use": "Describe one round",
            "Evidence": "One frozen pre-event field and observed result",
            "Strength": "Makes surprise and momentum visible",
            "Main caveat": "One round also contains terrain fit and ordinary noise",
        },
    ])
    st.dataframe(comparison, hide_index=True, width="stretch")
    backtest = data.get("model_backtest", pd.DataFrame())
    if not backtest.empty:
        st.markdown("#### Frozen next-competition backtest")
        st.dataframe(backtest, hide_index=True, width="stretch")
        st.caption(
            "Frozen means every prediction uses only information available before that "
            "competition. Higher rank correlation is better; lower log-loss and Brier "
            "error are better. This is the promotion test, not post-event fit."
        )
    sensitivity = data.get("sensitivity_metrics", pd.DataFrame())
    if not sensitivity.empty:
        boulder = sensitivity.loc[
            sensitivity.get("pool", pd.Series("", index=sensitivity.index))
            .astype(str).str.startswith("Boulder")
        ]
        if not boulder.empty:
            st.markdown("#### Faster-response challenger")
            st.dataframe(boulder, hide_index=True, width="stretch")
            st.caption(
                "The latent-volatility challenger reacts more to persistent Performance-ELO "
                "surprises. It slightly improved some ordering metrics but failed the "
                "predeclared probability-error gate, so it is research evidence—not production Elo."
            )
    if not calibration.empty:
        combined = calibration.loc[calibration["pool"].eq("Boulder_All")]
        shown = combined if not combined.empty else calibration.head(1)
        columns = [column for column in shown if (
            column.startswith("display_elo_at_50pct_") or column.endswith("mcfadden_pseudo_r2")
        )]
        if columns:
            st.markdown("#### 2025 outcome calibration")
            st.dataframe(shown[["pool", *columns]], hide_index=True, width="stretch")
    st.markdown("#### Why a large Performance-ELO surprise does not get full weight immediately")
    st.write(
        "A repeated surprise across independent competitions is stronger evidence of real "
        "change than several rounds at the same event. The production Elo stays zero-sum "
        "inside each event and lets uncertainty fall with evidence. A faster current-shape "
        "layer should be adopted only if it improves frozen next-event probability forecasts, "
        "not just because it follows the latest result more closely."
    )
    st.caption(correlation_note(correlations, "Global-ELO"))


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
    st.markdown("### Level, depth, physicality and progression of Canadian climbers: from local comps to the Olympics")
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
    workspace = st.segmented_control(
        "Workspace", ["Overview", "Physical Strength", "Maths behind"],
        default="Overview", label_visibility="collapsed",
    )
    if workspace == "Overview":
        render_rating_detail(athletes, data["history"], selected, data["calibration"])
        st.header("Overview")
        section = st.segmented_control(
            "Overview section",
            ["Canadian Pool", "IFSC Pool", "WR Pool", "Global progression", "Towards Olympics"],
            default="Canadian Pool",
            label_visibility="collapsed",
        )
        renderers = {
            "Canadian Pool": lambda: render_canadian_pool(
                athletes, selected, data["correlations"], data["calibration"]
            ),
            "IFSC Pool": lambda: render_ifsc_pool(
                athletes, data["history"], selected, data["correlations"],
                data["calibration"],
            ),
            "WR Pool": lambda: render_wr_pool(
                athletes, selected, data["correlations"], data["calibration"]
            ),
            "Global progression": lambda: render_progression(
                athletes, data["history"], selected, data["correlations"],
                data["calibration"],
            ),
            "Towards Olympics": lambda: render_olympics(
                athletes, selected, data["correlations"]
            ),
        }
        renderers[section]()
    elif workspace == "Physical Strength":
        render_physical_strength(athletes, selected, data)
    else:
        render_maths_behind(
            athletes, data["correlations"], data["calibration"], data
        )

    with st.expander("Rating glossary and model contract"):
        st.write(rating_help())
        st.markdown(
            "- **Global-ELO-Onsight / Scramble / Flash:** only rounds with a confirmed procedure.\n"
            "- **WR-ELO-Qualies / Semies / Finals:** only the named round of World Ranking events.\n"
            "- **IFSC-ELO-Qualies / Semies / Finals:** only the named round of non-para IFSC events.\n"
            "- **Performance-ELO:** the isolated level shown in one round, calculated from ratings frozen before the event."
        )
        st.markdown(
            "**Why one exceptional Performance-ELO does not automatically accelerate Elo.** "
            "One result can reflect real improvement, but also terrain fit and ordinary event noise. "
            "The tested one-state dynamic challenger became slightly better at ordering fields but "
            "worse at forecasting advancement probabilities after multi-round and cross-discipline "
            "dependence were corrected. Production therefore keeps the uncertainty-sensitive, "
            "zero-sum Elo. A future current-shape layer must first show repeated surprise across "
            "independent competitions and improve frozen next-event forecasts."
        )


if __name__ == "__main__":
    main()
