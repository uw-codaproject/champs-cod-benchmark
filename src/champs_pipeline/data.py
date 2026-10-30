from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import json
import math
import glob
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DataConfig


# ---------------------------
# Helpers
# ---------------------------

def is_missing(x) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    if isinstance(x, str) and x.strip() == "":
        return True
    return False


def normalize_binary(x):
    if is_missing(x):
        return None
    if isinstance(x, (int, float)):
        if x == 1:
            return "yes"
        if x == 0:
            return "no"
        return x
    if isinstance(x, str):
        v = x.strip().lower()
        if v in ("1", "yes", "y", "true", "t", "positive", "pos"):
            return "yes"
        if v in ("0", "no", "n", "false", "f", "negative", "neg"):
            return "no"
    return x


def load_icd_name_map(data_root: Path) -> Dict[str, str]:
    """
    Load ICD code -> name mapping from the generated TSV.
    Returns an empty map if the file does not exist.
    """
    mapping_path = data_root / "outputs" / "cod_labels_detailed_with_names.tsv"
    if not mapping_path.exists():
        print(f"[WARN] ICD name map not found: {mapping_path}")
        return {}
    df = pd.read_csv(mapping_path, sep="\t", dtype=str, low_memory=False)
    if "code" not in df.columns or "name" not in df.columns:
        print(f"[WARN] ICD name map missing required columns: {mapping_path}")
        return {}
    mapping: Dict[str, str] = {}
    for _, row in df.iterrows():
        code = str(row.get("code") or "").strip().upper()
        if not code:
            continue
        name = str(row.get("name") or "").strip()
        if not name or name.lower() == "nan":
            name = "Others"
        mapping[code] = name
    return mapping


def _should_map_icd_column(column_name: Optional[str]) -> bool:
    if not column_name:
        return False
    return not column_name.strip().lower().endswith("_name")


def _map_icd_codes_to_names(
    series: pd.Series, icd_name_map: Dict[str, str], default: str
) -> pd.Series:
    names = (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
        .map(icd_name_map)
    )
    names = names.fillna(default)
    if default == "":
        names = names.replace("Others", "")
    return names


def _lookup_full_field(mapping: Dict[str, str], field: str) -> Optional[str]:
    if not field:
        return None
    if field in mapping:
        return mapping[field]
    lower = field.lower()
    if lower in mapping:
        return mapping[lower]
    upper = field.upper()
    if upper in mapping:
        return mapping[upper]
    return None


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()

def _clean_label_text(text: str) -> str:
    cleaned = _strip_html(text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.rstrip(":")

def _normalize_field_key(field: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (field or "").lower())


def _parse_choice_map(raw: Optional[str]) -> Dict[str, str]:
    if not raw or not isinstance(raw, str):
        return {}
    choices: Dict[str, str] = {}
    for part in raw.split("|"):
        entry = part.strip()
        if not entry:
            continue
        if "," in entry:
            code, label = entry.split(",", 1)
            code = code.strip()
            label = label.strip()
            if code:
                choices[code] = label
        else:
            # Fallback for unusual encodings (keep as-is).
            choices[entry] = entry
    return choices


def _load_variable_dictionary(path: Path) -> Tuple[List[str], Dict[str, str]]:
    if not path.exists():
        print(f"[WARN] Variable dictionary not found: {path}")
        return [], {}
    df = pd.read_csv(path, dtype=str, low_memory=False, encoding="utf-8", encoding_errors="replace")
    var_col = "variable_name"
    desc_col = "description"
    if var_col not in df.columns or desc_col not in df.columns:
        print(f"[WARN] Variable dictionary missing expected columns: {path}")
        return [], {}
    variables: List[str] = []
    mapping: Dict[str, str] = {}
    for _, row in df.iterrows():
        raw_var = str(row.get(var_col) or "").strip()
        if not raw_var:
            continue
        label = _clean_label_text(str(row.get(desc_col) or "")) or raw_var
        if raw_var not in variables:
            variables.append(raw_var)
        # Store multiple lookup keys for robustness.
        for key in {raw_var, raw_var.strip(), raw_var.lower(), raw_var.upper(), _normalize_field_key(raw_var)}:
            if key:
                mapping[key] = label
    return variables, mapping


def _load_abstraction_dictionary(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        print(f"[WARN] Clinical abstraction dictionary not found: {path}")
        return {}
    df = pd.read_csv(path, dtype=str, low_memory=False, encoding="utf-8", encoding_errors="replace")

    def _normalize_col(name: str) -> str:
        return str(name or "").strip().strip('"').lstrip("\ufeff").lower()

    col_lookup = {_normalize_col(c): c for c in df.columns}
    field_col = col_lookup.get(_normalize_col("Variable / Field Name"))
    label_col = col_lookup.get(_normalize_col("Field Label"))
    choices_col = col_lookup.get(_normalize_col("Choices, Calculations, OR Slider Labels"))
    if not field_col or not label_col:
        print(f"[WARN] Clinical abstraction dictionary missing expected columns: {path}")
        return {}
    mapping: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        field = str(row.get(field_col) or "").strip()
        if not field:
            continue
        label = _clean_label_text(str(row.get(label_col) or "")) or field
        choices = _parse_choice_map(row.get(choices_col)) if choices_col else {}
        entry = {
            "label": label,
            "choices": choices,
            "field_label": _clean_label_text(str(row.get(label_col) or "")) or field,
        }
        mapping[field] = entry
        mapping[field.lower()] = entry
        mapping[field.upper()] = entry
        mapping[_normalize_field_key(field)] = entry
    return mapping


def _load_clinical_variable_list(path: Path) -> List[str]:
    if not path.exists():
        print(f"[WARN] Clinical variable list not found: {path}")
        return []
    df = pd.read_csv(path, dtype=str, low_memory=False, encoding="utf-8", encoding_errors="replace")
    var_col = "variable_name"
    if var_col not in df.columns:
        print(f"[WARN] Clinical variable list missing column '{var_col}': {path}")
        return []
    variables = []
    for val in df[var_col].fillna("").astype(str):
        name = val.strip()
        if not name:
            continue
        variables.append(name)
    return variables


def _lookup_field_meta(mapping: Dict[str, Dict[str, Any]], field: str) -> Optional[Dict[str, Any]]:
    if not field:
        return None
    if field in mapping:
        return mapping[field]
    lower = field.lower()
    if lower in mapping:
        return mapping[lower]
    upper = field.upper()
    if upper in mapping:
        return mapping[upper]
    normalized = _normalize_field_key(field)
    if normalized in mapping:
        return mapping[normalized]
    return None


def _format_choice_value(value: Any, choices: Dict[str, str]) -> str:
    raw = str(value).strip()
    if not choices:
        return raw
    if raw in choices:
        return choices[raw]
    lower_map = {str(k).strip().lower(): v for k, v in choices.items()}
    raw_lower = raw.lower()
    if raw_lower in lower_map:
        return lower_map[raw_lower]
    parts = [p.strip() for p in re.split(r"[;|]", raw) if p.strip()]
    if len(parts) > 1:
        mapped = [choices.get(part, lower_map.get(part.lower(), part)) for part in parts]
        return "; ".join(mapped)
    return raw


def _is_unknown_value(value: str) -> bool:
    lowered = value.strip().lower()
    if not lowered:
        return True
    # Only treat short, explicit markers as unknown/unavailable.
    unknown_markers = {
        "unknown",
        "unavailable",
        "unknown or unavailable",
        "unknown/unavailable",
        "unknown to respondent",
        "not available",
        "not applicable",
        "na",
        "n/a",
        "not known",
        "not recorded",
    }
    if lowered in unknown_markers:
        return True
    if lowered.startswith("unknown") or lowered.startswith("unavailable"):
        return True
    return False


def _is_dk_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return False
    lowered = str(value).strip().lower()
    if not lowered:
        return False
    return lowered in {"dk", "d/k", "dont know", "don't know", "do not know"}


def _is_unmapped_code(value: str) -> bool:
    return bool(re.fullmatch(r"CH\\d+", value.strip()))


def _is_age_related_va(field: str, label: Optional[str]) -> bool:
    checks = []
    if field:
        checks.append(field)
    if label:
        checks.append(label)
    for text in checks:
        lowered = str(text).lower()
        if any(token in lowered for token in ("age", "year", "month", "day", "hour", "week")):
            return True
    return False


def describe_mits_field(field: str, lab_test_map: Dict[str, str], tac_assay_map: Dict[str, str]) -> str:
    for prefix, specimen in (
        ("lung_", "lung"),
        ("bld_sp_", "blood (spun)"),
        ("bld_", "blood"),
        ("CSF_", "CSF"),
        ("rect_swab_", "rectal swab"),
        ("NP_OP_", "NP/OP swab"),
    ):
        if field.startswith(prefix):
            assay_desc = _lookup_full_field(tac_assay_map, field)
            if not assay_desc:
                return field
            return assay_desc

    test_desc = _lookup_full_field(lab_test_map, field)
    if test_desc:
        return test_desc

    return field


def read_subset(csv_path: Path, cols: List[str], limit: Optional[int] = None) -> pd.DataFrame:
    if not csv_path.exists():
        print(f"[WARN] File not found: {csv_path}")
        return pd.DataFrame(columns=cols)

    df = pd.read_csv(csv_path, low_memory=False, nrows=limit)
    keep = [c for c in cols if c in df.columns]
    missing = sorted(set(cols) - set(keep))
    if missing:
        print(f"[WARN] {csv_path.name}: missing {len(missing)} cols (up to 15 shown): {missing[:15]}")
    return df[keep]


class PostProcessor:
    def __init__(self, allowed_labels: List[str]):
        normalized = [label.strip().lower() for label in allowed_labels if str(label).strip()]
        self.allowed_labels = normalized
        self.label_lookup = {label: label for label in normalized}

    def post_process_generations_and_references(
        self,
        records: List[Dict[str, Any]],
        generations: List[str],
    ) -> Tuple[List[str], List[str], List[str], List[List[str]], List[List[str]]]:
        """
        Normalize generations to strings and build normalized labels/top-k inputs for evaluation.
        """
        cleaned_generations = ["" if gen is None else str(gen) for gen in generations]
        normalized_refs: List[str] = []
        normalized_preds: List[str] = []
        normalized_topk: List[List[str]] = []
        decode_targets: List[List[str]] = []
        for record, text in zip(records, cleaned_generations):
            target_values: List[str] = []
            normalized_refs.append(self._normalize_label(record.get("label")))
            topk = self._extract_topk(text, max_items=3)
            normalized_topk.append(topk)
            normalized_preds.append(topk[0] if topk else "NA")
            if record.get("label"):
                target_values.append(self._normalize_label(record["label"]))
            if record.get("immediate_label"):
                target_values.append(self._normalize_label(record["immediate_label"]))
            for value in record.get("comorbidity_labels", []) or []:
                if value:
                    target_values.append(self._normalize_label(value))
            decode_targets.append(target_values)
        return cleaned_generations, normalized_refs, normalized_preds, normalized_topk, decode_targets

    def extract_cause_name(self, text: str) -> Tuple[Optional[str], Optional[dict]]:
        """
        Best-effort extraction of a cause name from a model generation.
        Returns (cause_name, parsed_json). cause_name may be None if not found.
        """
        parsed = self._parse_json_block(text or "")
        if parsed:
            for key in ("cause_name", "cause_of_death", "cause_code"):
                value = parsed.get(key)
                if value:
                    return str(value).strip(), parsed
        return None, parsed

    def extract_cause_with_fallback(
        self, text: str, allow_numeric_prefix: bool = True
    ) -> Optional[str]:
        """
        Extract a cause using JSON first, then by searching for an allowed label
        inside the text. Optionally strips numeric prefixes from the parsed value.
        """
        candidate, _ = self.extract_cause_name(text or "")
        if candidate and allow_numeric_prefix:
            candidate = self._strip_numeric_prefix(candidate)
        if candidate and self._matches_allowed(candidate):
            return candidate
        return None

    def extract_ranked_causes(self, text: str, max_items: int = 3) -> List[str]:
        """
        Extract a ranked list of causes from a model generation.
        Uses JSON parsing when possible and falls back to a single best cause.
        """
        parsed = self._parse_json_block(text or "")
        candidates: List[str] = []
        if parsed:
            list_candidates = None
            for key in ("top_causes", "top3", "causes", "predictions", "ranked_causes"):
                if isinstance(parsed.get(key), list):
                    list_candidates = parsed[key]
                    break
            if list_candidates is None and isinstance(parsed.get("cause_name"), list):
                list_candidates = parsed["cause_name"]
            if list_candidates:
                for item in list_candidates:
                    value = None
                    if isinstance(item, str):
                        value = item
                    elif isinstance(item, dict):
                        for key in ("cause_name", "cause", "label", "name"):
                            if item.get(key):
                                value = item[key]
                                break
                    if value is None:
                        continue
                    value = self._strip_numeric_prefix(str(value))
                    if self._matches_allowed(value):
                        candidates.append(value)

        if not candidates:
            fallback = self.extract_cause_with_fallback(text)
            if fallback:
                candidates = [fallback]

        seen = set()
        ordered: List[str] = []
        for candidate in candidates:
            key = candidate.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            ordered.append(candidate)
            if len(ordered) >= max_items:
                break
        return ordered

    def _extract_topk(self, text: str, max_items: int = 3) -> List[str]:
        candidates = self.extract_ranked_causes(text, max_items=max_items)
        return [self._normalize_label(candidate) for candidate in candidates]

    def _normalize_label(self, candidate: Optional[str]) -> str:
        normalized = str(candidate or "").strip().lower()
        if not normalized:
            return "NA"
        return self.label_lookup.get(normalized, "NA")

    def _matches_allowed(self, value: str) -> bool:
        norm = value.strip().lower()
        return norm in self.label_lookup

    @staticmethod
    def _parse_json_block(text: str) -> Optional[dict]:
        if not text:
            return None
        text = PostProcessor._extract_harmony_final(text)
        decoder = json.JSONDecoder()
        for idx, ch in enumerate(text):
            if ch != "{":
                continue
            try:
                parsed, end = decoder.raw_decode(text[idx:])
                return parsed
            except json.JSONDecodeError:
                continue
        return None

    @staticmethod
    def _extract_harmony_final(text: str) -> str:
        """
        Extract the assistant final channel content from GPT-OSS Harmony output.
        Falls back to the original text if no final marker is found.
        """
        if not text:
            return ""
        marker = "assistantfinal"
        if marker in text:
            return text.split(marker, 1)[1].lstrip()
        tag = "<|start|>assistant<|message|>"
        if tag in text:
            tail = text.split(tag, 1)[1]
            return tail.lstrip().removeprefix("final").lstrip()
        return text

    @staticmethod
    def _strip_numeric_prefix(value: str) -> str:
        """
        Remove leading numeric prefixes like '28: Neonatal sepsis' -> 'Neonatal sepsis'.
        """
        trimmed = value.strip()
        match = re.match(r"^\s*\d+\s*[:\-\.)]?\s*(.+)$", trimmed)
        if match:
            return match.group(1).strip()
        return trimmed


# ---------------------------
# Dataset builder
# ---------------------------

class CHAMPSDatasetBuilder:
    """Loads labels and the pre-joined structured subset (builds it if missing)."""

    def __init__(self, config: DataConfig):
        self.config = config
        self._label_list: Optional[List[str]] = None
        self._full_dataset: Optional[pd.DataFrame] = None

    def _ordered_label_columns(self) -> List[str]:
        cfg = self.config
        label_cols = [cfg.id_column, cfg.label_column]
        if cfg.immediate_label_column:
            label_cols.append(cfg.immediate_label_column)
        if cfg.comorbidity_columns:
            label_cols.extend(cfg.comorbidity_columns)
        if cfg.stratify_column:
            label_cols.append(cfg.stratify_column)
        seen = set()
        ordered = []
        for col in label_cols:
            if col in seen:
                continue
            seen.add(col)
            ordered.append(col)
        return ordered

    def _apply_icd_mapping(
        self,
        df: pd.DataFrame,
        target_column: str,
        icd_name_map: Dict[str, str],
        default: str,
    ) -> None:
        if not target_column or target_column not in df.columns:
            return
        if _should_map_icd_column(target_column):
            df[target_column] = _map_icd_codes_to_names(df[target_column], icd_name_map, default)

    def build_features(self) -> tuple[Path, pd.DataFrame]:
        """
        Always (re)build the structured subset before loading data.
        This keeps the features synchronized with source CSVs.
        """
        cfg = self.config
        data_root = cfg.root_path()
        subset_path = data_root.parent.parent / "outputs" / "structured_subset.json"
        subset_path.parent.mkdir(parents=True, exist_ok=True)

        # Collect per-case structured inputs that will be merged into the final dataset.
        final_data: Dict[str, Dict[str, object]] = {
            "narrative": {},
            "structured_va": {},
            "demographics": {},
            "mits": {},
            "clinical_abstraction": {},
        }

        # Step 1: Verbal autopsy (narrative + structured VA)
        va_path = data_root / cfg.narrative_file
        va_dict_path = data_root / "Columns_1_2_CHAMPS_VA_variable_dictionary.csv"
        va_vars, va_label_map = _load_variable_dictionary(va_dict_path)
        va_cols = [cfg.id_column, cfg.narrative_column]
        for col in va_vars:
            if col not in va_cols:
                va_cols.append(col)
        print(f"[Subset] Reading Verbal Autopsy: {va_path}")
        va_df = read_subset(va_path, va_cols, limit=None)
        narrative_col = cfg.narrative_column

        for _, row in va_df.iterrows():
            cid = str(row[cfg.id_column])
            val = row.get(narrative_col)
            final_data["narrative"][cid] = str(val).strip() if not is_missing(val) else ""

            va_items = []
            for col in va_vars:
                if col == cfg.narrative_column:
                    continue
                if col not in row or is_missing(row[col]):
                    continue
                raw_val = row[col]
                if _is_dk_value(raw_val):
                    continue
                if isinstance(raw_val, str) and _is_unknown_value(raw_val):
                    continue
                if isinstance(raw_val, pd.Series):
                    raw_val = next((v for v in raw_val.values if not is_missing(v)), None)
                    if _is_dk_value(raw_val):
                        continue
                    if isinstance(raw_val, str) and _is_unknown_value(raw_val):
                        continue
                question = _lookup_full_field(va_label_map, col) or col
                if _is_age_related_va(col, question):
                    answer = str(raw_val).strip()
                    if not answer:
                        continue
                else:
                    norm = normalize_binary(raw_val)
                    if norm in {"yes", "no"}:
                        answer = norm
                    else:
                        if is_missing(raw_val):
                            continue
                        answer = str(raw_val).strip()
                        if not answer:
                            continue
                if _is_dk_value(answer):
                    continue
                if _is_unknown_value(answer):
                    continue
                va_items.append({"id": col, "question": question, "answer": answer})
            final_data["structured_va"][cid] = va_items

        # Step 2: Demographics
        demo_path = data_root / cfg.demographics_file
        demo_dict_path = data_root / "CHAMPS_demographics_variable_dictionary.csv"
        demo_vars, demo_label_map = _load_variable_dictionary(demo_dict_path)
        demo_cols = [cfg.id_column]
        for col in demo_vars:
            if col not in demo_cols:
                demo_cols.append(col)
        print(f"[Subset] Reading Demographics: {demo_path}")
        demo_df = read_subset(demo_path, demo_cols, limit=None)
        for _, row in demo_df.iterrows():
            cid = str(row[cfg.id_column])
            entries: Dict[str, Any] = {}
            labels: Dict[str, str] = {}
            for col in demo_vars:
                if col not in row or is_missing(row[col]):
                    continue
                if _is_dk_value(row[col]):
                    continue
                if isinstance(row[col], str) and _is_unknown_value(row[col]):
                    continue
                label = _lookup_full_field(demo_label_map, col) or col
                value = row[col]
                entries[col] = value.strip() if isinstance(value, str) else value
                labels[col] = label
            if entries:
                final_data["demographics"][cid] = {"__labels__": labels, **entries}
            else:
                final_data["demographics"][cid] = {}

        # Step 3: MITS (lab + TAC)
        lab_path = data_root / cfg.mits_lab_file
        tac_path = data_root / cfg.mits_tac_file
        lab_cols_pattern = tuple(cfg.mits_lab_prefixes)
        tac_cols_pattern = tuple(cfg.mits_tac_prefixes)

        def process_mits_file(path: Path, prefixes):
            if not path.exists():
                print(f"[WARN] MITS file not found: {path}")
                return {}
            print(f"[Subset] Reading MITS: {path}")
            df = pd.read_csv(path, low_memory=False)
            cols = [c for c in df.columns if any(c.startswith(p) for p in prefixes)]
            res: Dict[str, List[Dict[str, str]]] = {}
            for _, row in df.iterrows():
                cid = str(row["champs_deid"])
                items = []
                for c in cols:
                    val = row.get(c)
                    if not is_missing(val):
                        norm = normalize_binary(val)
                        if norm in {"yes", "no"}:
                            items.append(
                                {
                                    "field": c,
                                    "value": norm,
                                    "description": describe_mits_field(
                                        c, cfg.lab_test_map, cfg.tac_assay_map
                                    ),
                                }
                            )
                if items:
                    res[cid] = items
            return res

        lab_data = process_mits_file(lab_path, lab_cols_pattern)
        tac_data = process_mits_file(tac_path, tac_cols_pattern)

        all_cids = set(lab_data.keys()) | set(tac_data.keys())
        for cid in all_cids:
            final_data["mits"][cid] = {
                "lab_results": lab_data.get(cid, []),
                "tac_results": tac_data.get(cid, []),
            }

        # Step 4: Clinical abstraction (only v1/v2 files)
        abs_dir = cfg.clinical_abstraction_dir
        root: Optional[Path] = None
        if abs_dir:
            root = Path(abs_dir)
            if not root.is_absolute():
                root = data_root / root

        if root:
            print(f"[Subset] Scanning Clinical Abstraction directory: {root}")
        else:
            print("[WARN] Clinical abstraction directory not configured; skipping")

        if root and root.exists():
            target_files = {
                "CHAMPS_deid_child_abstraction_v1.csv",
                "CHAMPS_deid_child_abstraction_v2.csv",
            }
            files: List[str] = []
            for filename in target_files:
                path = root / filename
                if path.exists():
                    files.append(str(path))

            if not files:
                print("[WARN] No clinical abstraction v1/v2 files found.")
            else:
                id_col = "champs_deid"
                text_cols = cfg.clinical_abstraction_text_cols
                case_data_map: Dict[str, Dict[str, str]] = {}
                dict_v1 = _load_abstraction_dictionary(
                    root / "CHAMPS_ChildAbstraction_data_dictionary_v1.csv"
                )
                dict_v2 = _load_abstraction_dictionary(
                    root / "CHAMPS_ChildAbstraction_data_dictionary_v2.csv"
                )
                v1_vars = _load_clinical_variable_list(
                    root / "CHAMPS_child_abstraction_v1_clinical.csv"
                )
                v2_vars = _load_clinical_variable_list(
                    root / "CHAMPS_child_abstraction_v2_clinical.csv"
                )
                v1_norm = {_normalize_field_key(v) for v in v1_vars}
                v2_norm = {_normalize_field_key(v) for v in v2_vars}

                for file in files:
                    is_v1 = "v1" in Path(file).name.lower()
                    dictionary = dict_v1 if is_v1 else dict_v2
                    allowed_norm = v1_norm if is_v1 else v2_norm
                    try:
                        df = pd.read_csv(file, dtype=str, low_memory=False)
                    except Exception as e:
                        print(f"[WARN] Failed to read {file}: {e}")
                        continue
                    id_col_key = _normalize_field_key(id_col)
                    id_col_actual = next(
                        (c for c in df.columns if _normalize_field_key(c) == id_col_key),
                        None,
                    )
                    if not id_col_actual:
                        print(f"[WARN] ID column {id_col} not found in {file}")
                        continue

                    normalized_cols = {
                        _normalize_field_key(col): col for col in df.columns if col != id_col_actual
                    }
                    use_cols = [
                        normalized_cols[norm]
                        for norm in sorted(allowed_norm)
                        if norm in normalized_cols
                    ]
                    if text_cols:
                        use_cols = [c for c in text_cols if c in df.columns]
                    for _, row in df.iterrows():
                        cid = str(row[id_col_actual]).strip()
                        if not cid or cid == "nan":
                            continue
                        entries: Dict[str, str] = case_data_map.get(cid, {})
                        for col in use_cols:
                            val = row.get(col)
                            if is_missing(val):
                                continue
                            cleaned = str(val).strip() if not is_missing(val) else ""
                            if not cleaned or cleaned.lower() == "nan":
                                continue
                            if _is_dk_value(cleaned):
                                continue
                            if _is_unknown_value(cleaned):
                                continue
                            meta = _lookup_field_meta(dictionary, col) or {}
                            label = meta.get("label") or col
                            choices = meta.get("choices") or {}
                            answer = _format_choice_value(cleaned, choices)
                            answer = re.sub(r"\s+", " ", answer).strip()
                            if _is_dk_value(answer):
                                continue
                            if _is_unknown_value(answer):
                                continue
                            if _is_unmapped_code(answer):
                                continue
                            label_key = label
                            if label_key in entries:
                                label_key = f"{label} ({col})"
                            entries[label_key] = answer
                        if entries:
                            case_data_map[cid] = entries

                for cid, entries in case_data_map.items():
                    final_data["clinical_abstraction"][cid] = entries
        elif root:
            print(f"[WARN] Clinical abstraction directory not found: {root}")


        # Step 5: Convert nested dicts to a flat per-case dataframe for merging.
        rows = []
        all_ids = set()
        for section in final_data.values():
            if isinstance(section, dict):
                all_ids.update(section.keys())
        for cid in sorted(all_ids):
            row = {cfg.id_column: cid}
            for section_name, section_dict in final_data.items():
                if isinstance(section_dict, dict):
                    row[section_name] = section_dict.get(cid)
            rows.append(row)
        subset_df = pd.DataFrame(rows)
        return subset_path, subset_df

    def build_labels(self) -> pd.DataFrame:
        cfg = self.config
        # Step 1: Read the requested label columns from the decode file.
        df = pd.read_csv(
            cfg.label_path(),
            usecols=self._ordered_label_columns(),
            dtype=str,
            low_memory=False,
        )

        # Step 2: Normalize column names and drop rows without a label.
        rename_map = {cfg.label_column: "label"}
        if cfg.immediate_label_column:
            rename_map[cfg.immediate_label_column] = "immediate_label"
        df = df.rename(columns=rename_map)
        df["label"] = df["label"].fillna("").astype(str)
        df = df[df["label"].str.strip().str.len().gt(0)]
        if cfg.stratify_column:
            strat_col = cfg.stratify_column
            if strat_col == cfg.label_column:
                df["stratify_label"] = df["label"]
            elif strat_col in df.columns:
                df["stratify_label"] = df[strat_col].fillna("").astype(str)

        # Step 3: Identify comorbidity columns to aggregate.
        available_comorb_cols = []
        if cfg.comorbidity_columns:
            available_comorb_cols = [col for col in cfg.comorbidity_columns if col in df.columns]
        # Step 4: Optionally map ICD codes to names.
        if cfg.map_icd_codes:
            icd_name_map = load_icd_name_map(cfg.root_path())
            if icd_name_map:
                self._apply_icd_mapping(df, "label", icd_name_map, "Others")
                self._apply_icd_mapping(df, "immediate_label", icd_name_map, "")
                for col in available_comorb_cols:
                    self._apply_icd_mapping(df, col, icd_name_map, "")
            else:
                print(
                    "[WARN] ICD name map missing; leaving ICD codes unconverted."
                )

        # Step 5: Consolidate comorbidities into a single list column.
        if cfg.comorbidity_columns:
            df["comorbidity_labels"] = df[available_comorb_cols].apply(
                lambda row: [val.strip() for val in row if isinstance(val, str) and val.strip()],
                axis=1,
            )
            df = df.drop(columns=available_comorb_cols)
        else:
            df["comorbidity_labels"] = [[] for _ in range(len(df))]

        return df

    def load_dataframe(self, limit: Optional[int] = None) -> pd.DataFrame:
        cfg = self.config

        # Step 1: Build features (structured subset).
        subset_path, subset_df = self.build_features()

        # Step 2: Build labels from decode file.
        df = self.build_labels()

        # Step 3: Merge labels with structured inputs.
        print(f"[Dataset] Consuming inputs from structured subset: {subset_path}")

        df = df.merge(subset_df, on=cfg.id_column, how="inner")

        # Step 4: Ensure narrative text exists.
        if "narrative" not in df.columns:
            df["narrative"] = ""
        df["narrative"] = df["narrative"].fillna("").astype(str)

        # Step 5: Apply age-group filters.
        def get_age_group(row):
            val = row.get(cfg.age_group_column)
            if isinstance(val, str):
                return val
            demo = row.get("demographics")
            if isinstance(demo, dict):
                return demo.get(cfg.age_group_column, "")
            return ""

        age_series = df.apply(get_age_group, axis=1).fillna("").astype(str)
        mask = pd.Series(True, index=df.index)
        if cfg.include_age_groups:
            allowed = {g.lower().strip() for g in cfg.include_age_groups}
            mask = age_series.str.strip().str.lower().isin(allowed)
        elif cfg.age_group_keywords:
            keywords = [kw.lower() for kw in cfg.age_group_keywords]
            mask = age_series.str.lower().apply(lambda x: any(kw in x for kw in keywords))
        if cfg.age_group_required:
            mask &= age_series.str.strip().ne("")
        df = df.loc[mask]

        if cfg.max_samples:
            df = df.sample(n=min(cfg.max_samples, len(df)), random_state=cfg.random_seed)

        # Step 6: Enforce minimum narrative length if configured.
        if cfg.text_min_chars and cfg.text_min_chars > 0:
            df = df[df["narrative"].fillna("").astype(str).str.len() >= cfg.text_min_chars]

        # Step 6b: Drop rows missing stratify label if configured.
        if cfg.stratify_column and "stratify_label" in df.columns:
            df = df[df["stratify_label"].fillna("").astype(str).str.strip().ne("")]

        # Step 7: Final cleanup and optional row limit.
        df = df.drop_duplicates(subset=[cfg.id_column])
        df = df.reset_index(drop=True)
        if limit:
            df = df.head(limit)
        self._label_list = sorted(df["label"].unique())
        self._full_dataset = df
        return df

    def build_splits(self, limit: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        df = self.load_dataframe(limit=limit)
        return self._split_dataframe(df)

    def label_list(self) -> List[str]:
        if self._label_list is None:
            raise ValueError("Dataset not loaded yet.")
        return self._label_list

    def full_dataset(self) -> pd.DataFrame:
        if self._full_dataset is None:
            raise ValueError("Final dataset not built yet.")
        return self._full_dataset

    def _split_dataframe(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        cfg = self.config
        test_size = cfg.test_size
        val_size = cfg.val_size
        stratify_series = df.get("stratify_label", df.get("label"))
        label_counts = stratify_series.value_counts() if stratify_series is not None else df["label"].value_counts()
        stratify = None
        if stratify_series is not None and stratify_series.nunique() > 1 and label_counts.min() >= 2:
            stratify = stratify_series

        if test_size + val_size >= 1.0:
            raise ValueError("Sum of val_size and test_size must be < 1.0")

        if test_size + val_size == 0:
            return {"train": df}

        temp_size = test_size + val_size
        train_df, temp_df = train_test_split(
            df,
            test_size=temp_size,
            random_state=cfg.random_seed,
            stratify=stratify,
        )

        relative_test = 0 if temp_size == 0 else test_size / temp_size
        if relative_test == 0:
            val_df = temp_df
            test_df = df.iloc[0:0]
        elif relative_test == 1:
            val_df = df.iloc[0:0]
            test_df = temp_df
        else:
            stratify_temp = None
            if stratify is not None:
                temp_counts = temp_df["label"].value_counts()
                if temp_counts.min() >= 2 and temp_df["label"].nunique() > 1:
                    stratify_temp = temp_df["label"]
            val_df, test_df = train_test_split(
                temp_df,
                test_size=relative_test,
                random_state=cfg.random_seed,
                stratify=stratify_temp,
            )

        result = {"train": train_df, "validation": val_df, "test": test_df}
        return result
