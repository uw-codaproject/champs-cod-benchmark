#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


def _load_code_map(
    nodes_path: Path,
    code_field: str,
    name_field: str,
    class_field: str,
    id_field: str,
    system: str,
) -> Dict[str, Dict[str, str]]:
    """
    Load a mapping from ICD code -> metadata (name, class_kind, node_id, system).
    """
    mapping: Dict[str, Dict[str, str]] = {}
    with nodes_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            code = (row.get(code_field) or "").strip()
            if not code:
                continue
            normalized = code.upper()
            if normalized in mapping:
                # Keep the first occurrence (most specific entries usually appear later,
                # but we avoid overwriting in case of duplicates).
                continue
            mapping[normalized] = {
                "code": code,
                "name": (row.get(name_field) or "").strip(),
                "class_kind": (row.get(class_field) or "").strip(),
                "node_id": (row.get(id_field) or "").strip(),
                "system": system,
            }
    return mapping


def _iter_codes(path: Path) -> Iterable[str]:
    for line in path.read_text(encoding="utf-8").splitlines():
        code = line.strip()
        if code:
            yield code


def _normalize_name(value: Optional[str]) -> str:
    name = (value or "").strip()
    if not name or name.lower() == "nan":
        return "Others"
    return name


def _normalize_name_with_fallback(value: Optional[str], fallback: str) -> str:
    name = (value or "").strip()
    if not name or name.lower() == "nan":
        return fallback
    return name


def match_codes(
    codes: Iterable[str],
    icd11_map: Dict[str, Dict[str, str]],
    icd10_map: Dict[str, Dict[str, str]],
) -> Tuple[list[Dict[str, str]], list[str]]:
    results: list[Dict[str, str]] = []
    missing: list[str] = []
    for code in codes:
        key = code.upper()
        row: Optional[Dict[str, str]] = None
        if key in icd11_map:
            row = icd11_map[key].copy()
        elif key in icd10_map:
            row = icd10_map[key].copy()

        if row is None:
            missing.append(code)
            results.append(
                {
                    "code": code,
                    "system": "",
                    "name": "Others",
                    "class_kind": "",
                    "node_id": "",
                    "status": "missing",
                }
            )
        else:
            name = _normalize_name_with_fallback(row.get("name"), "Others")
            results.append(
                {
                    "code": row.get("code", code),
                    "system": row.get("system", ""),
                    "name": name,
                    "class_kind": row.get("class_kind", ""),
                    "node_id": row.get("node_id", ""),
                    "status": "matched",
                }
            )
    return results, missing


def write_matches(output_path: Path, rows: list[Dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["code", "system", "name", "class_kind", "node_id", "status"]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_unique_names(output_path: Path, rows: list[Dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    names = {_normalize_name_with_fallback(row.get("name"), "Others") for row in rows}
    with output_path.open("w", encoding="utf-8", newline="") as f:
        for name in sorted(names):
            f.write(f"{name}\n")


def resolve_code_name(
    code: str,
    icd11_map: Dict[str, Dict[str, str]],
    icd10_map: Dict[str, Dict[str, str]],
) -> str:
    key = (code or "").strip().upper()
    if not key:
        return ""
    if key in icd11_map:
        name = _normalize_name(icd11_map[key].get("name"))
        return "" if name == "Others" else name
    if key in icd10_map:
        name = _normalize_name(icd10_map[key].get("name"))
        return "" if name == "Others" else name
    return ""


def write_decode_with_immediate_names(
    input_path: Path,
    output_path: Path,
    icd11_map: Dict[str, Dict[str, str]],
    icd10_map: Dict[str, Dict[str, str]],
    immediate_col: str,
    name_col: str,
    morbid_cols: Iterable[str],
    morbid_name_suffix: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = list(reader.fieldnames or [])
        if name_col not in fieldnames:
            fieldnames.append(name_col)
        morbid_name_cols = []
        for col in morbid_cols:
            name_field = f"{col}{morbid_name_suffix}"
            morbid_name_cols.append(name_field)
            if name_field not in fieldnames:
                fieldnames.append(name_field)
        with output_path.open("w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                code = row.get(immediate_col, "")
                row[name_col] = resolve_code_name(code, icd11_map, icd10_map)
                for col, name_field in zip(morbid_cols, morbid_name_cols):
                    morbid_code = row.get(col, "")
                    row[name_field] = resolve_code_name(morbid_code, icd11_map, icd10_map)
                writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Match ICD codes to disease names using ICD10/ICD11 KG node TSVs."
        )
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path(
            "data/CHAMPS_de_identified_data_new/outputs/cod_labels_detailed_new.txt"
        ),
        help="Path to code list (one ICD code per line).",
    )
    parser.add_argument(
        "--icd10-nodes",
        type=Path,
        default=Path("data/kg/icd10_nodes.tsv"),
        help="Path to icd10_nodes.tsv.",
    )
    parser.add_argument(
        "--icd11-nodes",
        type=Path,
        default=Path("data/kg/icd11_nodes.tsv"),
        help="Path to icd11_nodes.tsv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/CHAMPS_de_identified_data_new/outputs/cod_labels_detailed_with_names.tsv"
        ),
        help="Where to write the matched results TSV.",
    )
    parser.add_argument(
        "--names-output",
        type=Path,
        default=Path(
            "data/CHAMPS_de_identified_data_new/outputs/cod_labels_detailed_unique_names.txt"
        ),
        help="Where to write the unique disease names (one per line).",
    )
    parser.add_argument(
        "--decode-input",
        type=Path,
        default=Path("data/CHAMPS_de_identified_data_new/CHAMPS_deid_decode_results.csv"),
        help="CHAMPS decode CSV to augment with Immediate_COD name.",
    )
    parser.add_argument(
        "--decode-output",
        type=Path,
        default=Path("data/CHAMPS_de_identified_data_new/CHAMPS_deid_decode_modified.csv"),
        help="Output CSV with Immediate_COD name column.",
    )
    parser.add_argument(
        "--immediate-col",
        type=str,
        default="Immediate_COD",
        help="Column name containing Immediate_COD ICD codes.",
    )
    parser.add_argument(
        "--immediate-name-col",
        type=str,
        default="Immediate_COD_name",
        help="Column name to store the resolved Immediate_COD name.",
    )
    parser.add_argument(
        "--morbid-cols",
        type=str,
        default="Morbid_Condition_01,Morbid_Condition_02,Morbid_Condition_03,Morbid_Condition_04,Morbid_Condition_05,Morbid_Condition_06,Morbid_Condition_07,Morbid_Condition_08",
        help="Comma-separated Morbid_Condition columns to resolve.",
    )
    parser.add_argument(
        "--morbid-name-suffix",
        type=str,
        default="_name",
        help="Suffix appended to morbid columns for the resolved name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    icd10_map = _load_code_map(
        args.icd10_nodes,
        code_field="code",
        name_field="name",
        class_field="class_kind",
        id_field="id:ID",
        system="icd10",
    )
    icd11_map = _load_code_map(
        args.icd11_nodes,
        code_field="code",
        name_field="name",
        class_field="class_kind",
        id_field="id:ID",
        system="icd11",
    )

    codes = list(_iter_codes(args.labels))
    rows, missing = match_codes(codes, icd11_map, icd10_map)
    write_matches(args.output, rows)
    write_unique_names(args.names_output, rows)
    morbid_cols = [c.strip() for c in args.morbid_cols.split(",") if c.strip()]
    write_decode_with_immediate_names(
        args.decode_input,
        args.decode_output,
        icd11_map,
        icd10_map,
        args.immediate_col,
        args.immediate_name_col,
        morbid_cols,
        args.morbid_name_suffix,
    )

    print(f"Wrote {len(rows)} rows to {args.output.resolve()}")
    print(f"Wrote {args.names_output.resolve()} with unique names")
    print(f"Wrote {args.decode_output.resolve()} with Immediate_COD names")
    if missing:
        print(f"Unmatched codes: {len(missing)}")


if __name__ == "__main__":
    main()

