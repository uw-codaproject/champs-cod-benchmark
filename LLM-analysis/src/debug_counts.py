from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

from champs_pipeline import CHAMPSDatasetBuilder, PipelineConfig


def main() -> None:
    cfg = PipelineConfig.from_yaml("configs/pipeline.yaml")
    cfg_data = cfg.data

    def step_counts(df, label: str) -> None:
        print(f"{label}: {len(df)}")

    # Load base tables
    narr = pd.read_csv(
        cfg_data.narrative_path(),
        usecols=[cfg_data.id_column, cfg_data.narrative_column],
        dtype=str,
        low_memory=False,
    )
    labels = pd.read_csv(
        cfg_data.label_path(),
        usecols=[cfg_data.id_column, cfg_data.label_column],
        dtype=str,
        low_memory=False,
    )

    # Merge
    df = narr.merge(labels, on=cfg_data.id_column, how="inner").rename(
        columns={
            cfg_data.narrative_column: "narrative",
            cfg_data.label_column: "label",
            cfg_data.id_column: "case_id",
        }
    )
    df["narrative"] = df["narrative"].fillna("").astype(str)
    df["label"] = df["label"].fillna("").astype(str)
    step_counts(df, "After merge")

    # Drop empty labels
    df = df[df["label"].str.strip().str.len().gt(0)]
    step_counts(df, "After non-empty label")

    # Age group merge/filter
    demo_path = cfg_data.root_path() / cfg_data.demographics_file
    demo = pd.read_csv(
        demo_path,
        usecols=[cfg_data.id_column, cfg_data.age_group_column],
        dtype=str,
        low_memory=False,
    )
    df = df.merge(
        demo,
        left_on="case_id",
        right_on=cfg_data.id_column,
        how="left",
    )
    if cfg_data.id_column in df.columns and cfg_data.id_column != "case_id":
        df = df.drop(columns=[cfg_data.id_column])

    age_series = df[cfg_data.age_group_column].fillna("").astype(str)
    mask = pd.Series(True, index=df.index)
    if cfg_data.include_age_groups:
        allowed = {g.lower().strip() for g in cfg_data.include_age_groups}
        mask = age_series.str.strip().str.lower().isin(allowed)
    elif cfg_data.age_group_keywords:
        keywords = [kw.lower() for kw in cfg_data.age_group_keywords]
        mask = age_series.str.lower().apply(lambda x: any(kw in x for kw in keywords))
    if cfg_data.age_group_required:
        mask &= age_series.str.strip().ne("")
    df = df.loc[mask]

    step_counts(df, "After age group filter")

    # Drop duplicates
    df = df.drop_duplicates(subset=["case_id"])
    step_counts(df, "After dedup case_id")

    # Clinical abstraction merge (if present)
    if cfg_data.clinical_abstraction_dir:
        root = Path(cfg_data.clinical_abstraction_dir)
        if not root.is_absolute():
            root = cfg_data.root_path().parent / root
        pattern = root / cfg_data.clinical_abstraction_glob
        files = [
            f
            for f in glob.glob(str(pattern))
            if "datadictionary" not in Path(f).name.lower()
            and "data_dictionary" not in Path(f).name.lower()
        ]
        id_col = cfg_data.clinical_abstraction_case_id_col or cfg_data.id_column
        text_cols = cfg_data.clinical_abstraction_text_cols
        merged = {}
        for file in files:
            try:
                adf = pd.read_csv(file, dtype=str, low_memory=False)
            except Exception:
                continue
            if id_col not in adf.columns:
                continue
            use_cols = [c for c in adf.columns if c != id_col]
            if text_cols:
                use_cols = [c for c in text_cols if c in adf.columns]
            for _, row in adf.iterrows():
                cid = str(row[id_col]).strip()
                if not cid:
                    continue
                parts = [
                    str(row[c]).strip()
                    for c in use_cols
                    if isinstance(row[c], str) and str(row[c]).strip()
                ]
                if parts:
                    merged.setdefault(cid, []).extend(parts)
        if merged:
            df["clinical_abstraction"] = df["case_id"].map(
                lambda x: "\n".join(merged.get(x, []))
            ).fillna("")
        step_counts(df, "After clinical abstraction merge")

    # Final text length filter
    if cfg_data.text_min_chars and cfg_data.text_min_chars > 0:
        df = df[df["narrative"].fillna("").astype(str).str.len() >= cfg_data.text_min_chars]
    step_counts(df, f"After text_min_chars >= {cfg_data.text_min_chars}")

    # Show a few examples
    print("\nSample cases (case_id, len(text)):")
    for cid, text in df[["case_id", "narrative"]].head(5).itertuples(index=False):
        print(cid, len(text))


if __name__ == "__main__":
    main()
