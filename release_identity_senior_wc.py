"""Release-safe guards for a reviewed identity correction and WC eligibility.

This module intentionally does *not* rewrite published ratings.  Joining an
alias into a canonical athlete changes the chronological rating state and must
be rebuilt by the rating producer, not approximated in the Streamlit process.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


IDENTITY_OVERRIDE_COLUMNS = (
    "source_scope",
    "athlete_source_id",
    "ifsc_athlete_id",
    "identity_confidence",
    "reviewer_note",
    "evidence_url",
)
REVIEWED_ALIAS_GLOBAL_ID = "IFSC:18545"
REVIEWED_CANONICAL_GLOBAL_ID = "IFSC:14843"


def load_reviewed_identity_overrides(path: Path) -> pd.DataFrame:
    """Load the deliberately narrow, reviewed override ledger.

    The release lane permits exactly the reviewed IFSC 18545 -> 14843 mapping.
    Additional rows are a new identity decision and must not silently enter a
    public build.
    """

    overrides = pd.read_csv(path, dtype="string")
    if tuple(overrides.columns) != IDENTITY_OVERRIDE_COLUMNS:
        raise ValueError("unexpected reviewed identity override schema")
    if len(overrides) != 1:
        raise ValueError("release override ledger must contain exactly one mapping")
    row = overrides.iloc[0]
    if (
        str(row["source_scope"]).upper() != "IFSC"
        or str(row["athlete_source_id"]) != "18545"
        or str(row["ifsc_athlete_id"]) != "14843"
        or float(row["identity_confidence"]) != 1.0
    ):
        raise ValueError("release override ledger is not the reviewed IFSC mapping")
    return overrides


def senior_wc_direct_evidence_mask(history: pd.DataFrame) -> pd.Series:
    """Return direct Senior/Open World-level evidence, never youth or regional.

    This is a display/release guard.  The producer must use the same semantic
    rule when it rebuilds the WC+ rating family.
    """

    if history.empty:
        return pd.Series(False, index=history.index, dtype=bool)
    names = history.get(
        "event_name", pd.Series("", index=history.index, dtype="string")
    ).astype("string")
    senior_name = names.str.contains(
        r"(?i)\b(?:world\s+(?:climbing\s+)?(?:cup|series|championships?)|"
        r"olympic(?:\s+qualifier)?(?:\s+series)?|oqs)\b",
        regex=True,
        na=False,
    )
    youth = names.str.contains(r"(?i)\byouth\b", regex=True, na=False)
    source = history.get(
        "source_scope", pd.Series("", index=history.index, dtype="string")
    ).astype("string").str.upper()
    return source.eq("IFSC") & senior_name & ~youth


def identity_rebuild_status(
    athletes: pd.DataFrame,
    history: pd.DataFrame,
) -> dict[str, object]:
    """Describe whether release inputs still require the producer rebuild."""

    athlete_ids = athletes.get("global_id", pd.Series(dtype="string")).astype(str)
    history_ids = history.get("global_id", pd.Series(dtype="string")).astype(str)
    alias_rows = int(history_ids.eq(REVIEWED_ALIAS_GLOBAL_ID).sum())
    return {
        "reviewed_alias_global_id": REVIEWED_ALIAS_GLOBAL_ID,
        "reviewed_canonical_global_id": REVIEWED_CANONICAL_GLOBAL_ID,
        "canonical_profile_count": int(athlete_ids.eq(REVIEWED_CANONICAL_GLOBAL_ID).sum()),
        "alias_profile_count": int(athlete_ids.eq(REVIEWED_ALIAS_GLOBAL_ID).sum()),
        "alias_history_rows": alias_rows,
        "requires_producer_rebuild": bool(alias_rows),
        "runtime_merge_performed": False,
    }
