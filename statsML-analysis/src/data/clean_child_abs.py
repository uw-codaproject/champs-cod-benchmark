"""
Clinical Abstraction Cleaning Script
-------------------------------------
Encodes CH-coded binary abstraction columns.
Drops narrative columns (based on text length heuristic).
Keeps messy numeric columns (unknowns/blanks as np.nan).
Encodes `cpmh_091` CH-coded location column as one-hot (safely preserving NaNs).
Flags unknown column types.
Allows explicit columns to drop via cols_to_drop parameter.
"""

import pandas as pd
import numpy as np
import re
import warnings

from src.data.constants import (
    CHILD_ABS_V1_UNKNOWN_VALUES,
    CHILD_ABS_V1_COLUMNS_TO_DROP,
    CHILD_ABS_V1_CLINICAL_NARRATIVE_COLUMNS,
    CHILD_ABS_V1_TRAIN_ALL_NULL_COLUMNS_INFANTCHILD
)

V1_COLS_TO_DROP = (
    CHILD_ABS_V1_CLINICAL_NARRATIVE_COLUMNS +
    CHILD_ABS_V1_COLUMNS_TO_DROP +
    CHILD_ABS_V1_TRAIN_ALL_NULL_COLUMNS_INFANTCHILD
)

# Pre-compute string versions of unknown values for faster matching
UNKNOWN_STRINGS = {str(v).strip().lower() for v in CHILD_ABS_V1_UNKNOWN_VALUES if v is not None}

# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def is_ch_code(value):
    """Check if a value matches CHxxxxx pattern."""
    if pd.isna(value):
        return False
    val_str = str(value).strip().lower()
    if val_str in ['nan', 'null', 'none', '']:
        return False
    return bool(re.fullmatch(r"ch\d{5}", val_str))


def is_binary_abstraction_column(series):
    """Determine if a column is a CH-coded abstraction column."""
    s = series.dropna().astype(str).str.strip().str.lower()
    s = s[~s.isin(['nan', 'null', 'none', ''])]
    if len(s) == 0:
        return False
    return s.apply(is_ch_code).all()


def encode_binary_abstraction(series):
    """
    Encode CH-coded binary columns.
    CH00001 -> 1. Everything else -> 0.
    True missing values and explicit 'unknowns' -> np.nan
    """
    s_str = series.astype(str).str.strip().str.lower()

    # Identify what should be treated as a true NaN
    is_missing = series.isna() | s_str.isin(['nan', 'null', 'none', '']) | s_str.isin(UNKNOWN_STRINGS)

    # Encode CH00001 as 1, others as 0
    encoded = np.where(s_str == "ch00001", 1, 0)

    # Re-inject NaNs where data was missing or unknown
    encoded = np.where(is_missing, np.nan, encoded)

    return pd.Series(encoded, index=series.index, dtype=float)


def encode_numeric_column(series):
    """Convert messy numeric column to float. Unknowns/blanks -> NaN."""
    s = series.astype(str).str.strip().str.lower()
    s = s.replace(list(UNKNOWN_STRINGS), np.nan)
    s = s.replace(["null", "nan", "none", ""], np.nan)

    return pd.to_numeric(s, errors='coerce')


def is_numeric_column(series):
    """Determine if a column can be treated as numeric."""
    try:
        s = encode_numeric_column(series)
        return s.dropna().shape[0] > 0
    except Exception:
        return False


def is_narrative_column(series, length_threshold=10):
    """Narrative if at least one non-unknown, non-null value has length > threshold."""
    s = series.dropna().astype(str).str.strip().str.lower()
    s = s[~s.isin(UNKNOWN_STRINGS | {"null", "nan", "none", ""})]
    return (s.str.len() > length_threshold).any()


def encode_categorical_locdelivery(series):
    """
    Robustly encode cpmh_091 CH-coded location column and one-hot encode it safely.
    Missing/Unknowns -> np.nan (propagated across dummy columns).
    """
    mapping = {
        "ch00012": "hospital",
        "ch00013": "health center",
        "ch00014": "health post",
        "ch00015": "home",
        "ch00016": "on the way",
        "ch00017": "other location"
    }

    s_str = series.astype(str).str.strip().str.lower()

    # Map missingness
    is_missing = series.isna() | s_str.isin(['nan', 'null', 'none', '']) | s_str.isin(UNKNOWN_STRINGS)

    # Map CH codes, unmapped -> 'other'
    s_mapped = s_str.map(lambda x: mapping.get(x, "other"))
    s_mapped[is_missing] = np.nan

    # Safe one-hot encode
    one_hot = pd.get_dummies(s_mapped, prefix=series.name, dtype=float)

    # Inject NaNs back across the row for missing data
    if is_missing.any():
        one_hot.loc[is_missing, :] = np.nan

    return one_hot


# ------------------------------------------------------------
# Main Cleaning Function
# ------------------------------------------------------------

def clean_clinical_abstraction(
    df,
    cols_to_drop=V1_COLS_TO_DROP,
    id_column=None,
    verbose=True
):

    cleaned_df = df.copy()

    protected_cols = set()
    if id_column:
        if id_column not in cleaned_df.columns:
            raise ValueError(f"id_column '{id_column}' not found.")
        protected_cols.add(id_column)

    # ------------------------
    # Drop explicitly requested columns (exclude protected)
    # ------------------------
    if cols_to_drop:
        existing_cols = [
            c for c in cols_to_drop
            if c in cleaned_df.columns and c not in protected_cols
        ]
        cleaned_df = cleaned_df.drop(columns=existing_cols)

    # Note: "Drop all-NaN columns" logic removed.
    # This is safely handled by the pipeline's remove_dead_columns() method.

    binary_cols = []
    numeric_cols = []
    narrative_cols = []
    other_cols = []

    # ------------------------
    # ONLY classify NON-protected columns
    # ------------------------
    cols_to_process = [
        c for c in cleaned_df.columns if c not in protected_cols
    ]

    for col in cols_to_process:
        series = cleaned_df[col]

        # Intercept the specific location column
        if col == "cpmh_091":
            one_hot_df = encode_categorical_locdelivery(series)
            cleaned_df = pd.concat(
                [cleaned_df.drop(columns=[col]), one_hot_df],
                axis=1
            )
            continue

        if is_binary_abstraction_column(series):
            binary_cols.append(col)

        elif is_numeric_column(series):
            numeric_cols.append(col)

        elif is_narrative_column(series):
            narrative_cols.append(col)

        else:
            other_cols.append(col)

    # ------------------------
    # Encode
    # ------------------------
    for col in binary_cols:
        cleaned_df[col] = encode_binary_abstraction(cleaned_df[col])

    for col in numeric_cols:
        cleaned_df[col] = encode_numeric_column(cleaned_df[col])

    # ------------------------
    # Drop narrative
    # ------------------------
    if narrative_cols:
        if verbose:
            print(f"\n[DROPPING {len(narrative_cols)} NARRATIVE COLUMNS]")
            if len(narrative_cols) <= 10:
                for col in narrative_cols:
                    print(f"  Dropping: {col}")

        cleaned_df = cleaned_df.drop(columns=narrative_cols)

    return cleaned_df, {
        "binary_columns": binary_cols,
        "numeric_columns": numeric_cols,
        "narrative_columns": narrative_cols,
        "protected_columns": list(protected_cols)
    }