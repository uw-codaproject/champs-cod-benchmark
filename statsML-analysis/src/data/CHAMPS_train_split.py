"""Prepare CHAMPS data for train/holdout splits with multi-label stratification."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

ROOT = Path(__file__).resolve().parents[2]  # repo root (this file lives in src/data/)
DATA_DIR = ROOT / "data" / "feb2026"        # overridable with --data-dir
OUTPUT_DIR = DATA_DIR / "processed"         # overridable with --output-dir

KEY_COLUMN = "champs_deid"
AGE_GROUP_COLUMN = "age_group"
MODULE_COLUMN = "module"

SOURCE_FILES: List[str] = [
    "CHAMPS_deid_basic_demographics.csv",
    "CHAMPS_deid_decode_results.csv",
    "CHAMPS_deid_lab_results.csv",
    "CHAMPS_deid_tac_results.csv",
    "CHAMPS_deid_verbal_autopsy.csv",
]

AGE_GROUP_TO_MODULE = {
    "Infant (28 days to less than 12 months)": "InfantChild",
    "Early Neonate (1 to 6 days)": "Neonate",
    "Late Neonate (7 to 27 days)": "Neonate",
    "Death in the first 24 hours": "Neonate",
    "Child (12 months to less than 60 Months)": "InfantChild",
    "Stillbirth": "Stillbirth",
}

MODULE_FILE_STEMS = {
    "InfantChild": "champs_infant_child",
    "Neonate": "champs_neonate",
    "Stillbirth": "champs_stillbirth",
    "Unknown": "champs_unknown",
}

TRAIN_FRACTION = 0.8
RANDOM_SEED = 2024
MIN_COMBO_COUNT = 5
CAUSE_COLUMNS = ["UC_champs_group_desc", "IC_champs_group_desc"]
GROUPED_CAUSE_COLUMNS = ["UC_grouped_cause", "IC_grouped_cause"]
CSV_ENCODING = "utf-8"


def load_source_frames(data_dir: Path, filenames: List[str]) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for idx, filename in enumerate(filenames):
        path = data_dir / filename
        df = pd.read_csv(path, low_memory=False)
        # Some releases (e.g. feb2026 decode results) contain repeated rows for a case; keep the first
        # (same rule as CHAMPSPipeline.drop_duplicates) so the one-to-one merge below is valid.
        df = df.drop_duplicates(subset=KEY_COLUMN, keep="first")
        if idx != 0 and AGE_GROUP_COLUMN in df.columns:
            df = df.drop(columns=[AGE_GROUP_COLUMN])
        frames[filename] = df
    return frames


def merge_on_key(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    frame_iter = iter(frames.values())
    merged = next(frame_iter)
    for df in frame_iter:
        merged = merged.merge(df, on=KEY_COLUMN, how="outer", validate="one_to_one")
    return merged


def assign_modules(df: pd.DataFrame) -> pd.DataFrame:
    modules = df[AGE_GROUP_COLUMN].map(AGE_GROUP_TO_MODULE)
    return df.assign(module=modules.fillna("Unknown"))


def group_sparse_cause_combinations(df: pd.DataFrame) -> pd.DataFrame:
    if not set(CAUSE_COLUMNS).issubset(df.columns):
        missing = set(CAUSE_COLUMNS) - set(df.columns)
        raise KeyError(f"Required cause columns missing from dataframe: {sorted(missing)}")

    df = df.copy()
    df[GROUPED_CAUSE_COLUMNS] = df[CAUSE_COLUMNS]

    fill_map = {
        GROUPED_CAUSE_COLUMNS[0]: "Unknown UC",
        GROUPED_CAUSE_COLUMNS[1]: "Unknown IC",
    }
    grouped_cause_df = df[GROUPED_CAUSE_COLUMNS].copy()
    grouped_cause_df = grouped_cause_df.fillna(fill_map)
    combo_counts = (
        grouped_cause_df.groupby(GROUPED_CAUSE_COLUMNS)
        .size()
        .reset_index(name="combo_count")
    )
    grouped_cause_df = grouped_cause_df.merge(combo_counts, on=GROUPED_CAUSE_COLUMNS, how="left")
    sparse_mask = grouped_cause_df["combo_count"] < MIN_COMBO_COUNT
    grouped_cause_df.loc[sparse_mask, GROUPED_CAUSE_COLUMNS] = "Other Cause"

    df[GROUPED_CAUSE_COLUMNS] = grouped_cause_df[GROUPED_CAUSE_COLUMNS]
    return df


def split_by_module(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    module_frames: Dict[str, pd.DataFrame] = {}
    for module_value, group_df in df.groupby(MODULE_COLUMN, dropna=False):
        module_key = module_value if isinstance(module_value, str) else "Unknown"
        module_frames[module_key] = group_df.copy()
    return module_frames


def build_label_sets(df: pd.DataFrame) -> Sequence[Sequence[str]]:
    label_sets: List[List[str]] = []
    for uc_value, ic_value in zip(df[GROUPED_CAUSE_COLUMNS[0]], df[GROUPED_CAUSE_COLUMNS[1]]):
        labels: List[str] = []
        if isinstance(uc_value, str) and uc_value:
            labels.append(uc_value)
        if isinstance(ic_value, str) and ic_value and ic_value not in labels:
            labels.append(ic_value)
        if not labels:
            labels.append("Other Cause")
        label_sets.append(labels)
    return label_sets


def stratified_split(module_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit  # only needed for random splits
    label_sets = build_label_sets(module_df)
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(label_sets)

    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        train_size=TRAIN_FRACTION,
        test_size=1 - TRAIN_FRACTION,
        random_state=RANDOM_SEED,
    )

    train_indices: Iterable[int]
    holdout_indices: Iterable[int]
    train_indices, holdout_indices = next(splitter.split(module_df, y))
    train_df = module_df.iloc[train_indices].copy()
    holdout_df = module_df.iloc[holdout_indices].copy()
    return train_df, holdout_df


def nested_splits(module_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if len(module_df) <= 1:
        empty = module_df.iloc[0:0].copy()
        return module_df.copy(), empty, empty

    outer_train_df, holdout_df = stratified_split(module_df)

    if len(outer_train_df) <= 1:
        inner_empty = outer_train_df.iloc[0:0].copy()
        return outer_train_df.copy(), inner_empty, holdout_df

    inner_train_df, inner_test_df = stratified_split(outer_train_df)
    return inner_train_df, inner_test_df, holdout_df


def write_module_splits(train_df: pd.DataFrame, test_df: pd.DataFrame, holdout_df: pd.DataFrame, module_name: str) -> None:
    stem = MODULE_FILE_STEMS.get(module_name, MODULE_FILE_STEMS["Unknown"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_path = OUTPUT_DIR / f"{stem}_train.csv"
    test_path = OUTPUT_DIR / f"{stem}_test.csv"
    holdout_path = OUTPUT_DIR / f"{stem}_holdout.csv"

    train_df.to_csv(train_path, index=False, encoding=CSV_ENCODING)
    test_df.to_csv(test_path, index=False, encoding=CSV_ENCODING)
    holdout_df.to_csv(holdout_path, index=False, encoding=CSV_ENCODING)

    print(
        f"{module_name}: wrote {len(train_df)} train rows -> {train_path}, "
        f"{len(test_df)} test rows -> {test_path}, "
        f"{len(holdout_df)} holdout rows -> {holdout_path}"
    )


def split_from_json(module_df: pd.DataFrame, split_json: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fixed train/test membership and row order taken from a split file ({"train": [...], "test": [...]})."""
    split = json.load(open(split_json))
    train_ids, test_ids = split["train"], split["test"]
    assert not set(train_ids) & set(test_ids), "train/test overlap in split file"
    indexed = module_df.set_index(KEY_COLUMN, drop=False)
    missing = (set(train_ids) | set(test_ids)) - set(indexed.index)
    assert not missing, f"{len(missing)} split ids not in the merged data"
    extra = set(indexed.index) - set(train_ids) - set(test_ids)
    if extra:
        print(f"note: {len(extra)} rows of this module are in neither split and are not written")
    train_df = indexed.loc[train_ids].reset_index(drop=True)
    test_df = indexed.loc[test_ids].reset_index(drop=True)
    return train_df, test_df


def main() -> None:
    global DATA_DIR, OUTPUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    ap.add_argument("--split-json", type=Path, default=None,
                    help="Use this fixed split for the InfantChild module (e.g. data/infant_child_split.json). "
                         "Writes only champs_infant_child_{train,test}.csv with the plain merged schema + module.")
    args = ap.parse_args()
    DATA_DIR, OUTPUT_DIR = args.data_dir, args.output_dir

    frames = load_source_frames(DATA_DIR, SOURCE_FILES)
    merged = merge_on_key(frames)
    merged_with_modules = assign_modules(merged)
    if args.split_json is not None:
        ic = merged_with_modules[merged_with_modules[MODULE_COLUMN] == "InfantChild"]
        train_df, test_df = split_from_json(ic, args.split_json)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stem = MODULE_FILE_STEMS["InfantChild"]
        for part, df in [("train", train_df), ("test", test_df)]:
            path = OUTPUT_DIR / f"{stem}_{part}.csv"
            df.to_csv(path, index=False, encoding=CSV_ENCODING)
            print(f"InfantChild {part}: {len(df)} rows, {df.shape[1]} cols -> {path}")
        return
    merged_with_causes = group_sparse_cause_combinations(merged_with_modules)
    module_frames = split_by_module(merged_with_causes)

    print(f"Merged dataset rows: {len(merged_with_causes)}")
    for module_name, module_df in module_frames.items():
        if module_df.empty:
            print(f"{module_name}: 0 rows (skipping split)")
            continue
        train_df, test_df, holdout_df = nested_splits(module_df)
        write_module_splits(train_df, test_df, holdout_df, module_name)


if __name__ == "__main__":
    main()
